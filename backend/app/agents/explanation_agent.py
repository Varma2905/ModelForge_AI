import logging
import os
from typing import Dict, Any, List, Optional
from app.services.huggingface_service import get_llm
from app.ml.feature_types import map_expanded_coefficients, _is_linear_regression_model

logger = logging.getLogger("regression_studio.agents")

try:
    from langchain_core.messages import SystemMessage, HumanMessage
except ImportError:
    pass

class StatisticalInterpretationAgent:
    def __init__(self):
        self.llm = get_llm()

    async def explain(
        self, metrics: Dict[str, float], stats: Dict[str, Any], features: List[str], target: str,
        numeric_features: Optional[List[str]] = None, categorical_features: Optional[List[str]] = None,
        model_name: str = "",
    ) -> str:
        """
        Interprets OLS statistics and model performance.
        Returns a detailed Markdown analysis of R2, RMSE, coefficients, and feature significance.
        """
        context = f"""
        Evaluation Metrics: {metrics}
        Target Variable: {target}
        Feature Variables: {features}
        Coefficients: {stats.get('coefficients')}
        P-Values: {stats.get('p_values')}
        Standard Errors: {stats.get('standard_errors')}
        F-Statistic: {stats.get('f_statistic')}
        F-Pvalue: {stats.get('f_pvalue')}
        """

        # Applied to BOTH the LLM response and the analytical fallback below
        # — a system-prompt instruction alone isn't a reliable guarantee the
        # LLM actually includes this caveat, so it's prepended deterministically
        # here rather than left to the model's discretion.
        aux_model_note = ""
        if model_name and not _is_linear_regression_model(model_name):
            aux_model_note = (
                f"> **Note:** the coefficient/p-value figures below come from an auxiliary Ordinary Least "
                f"Squares (OLS) fit used for interpretability — they describe correlational structure in the "
                f"data, not the internal mechanics of the trained **{model_name}** model.\n\n"
            )

        if self.llm:
            try:
                system_msg = SystemMessage(content="You are a senior statistical analyst. Interpret the OLS regression summary metrics. Explain R-squared, RMSE, which features are statistically significant based on p-values, and what the coefficients mean in plain English. Format output in clean Markdown.")
                human_msg = HumanMessage(content=f"Interpret these statistical regression results:\n{context}")
                response = await self.llm.ainvoke([system_msg, human_msg])
                return aux_model_note + response.content
            except Exception as e:
                logger.warning(f"StatisticalInterpretationAgent LLM failed: {e}. Falling back to statistical parser.")

        # Analytical fallback
        return aux_model_note + self._generate_analytical_fallback(
            metrics, stats, features, target, numeric_features, categorical_features, model_name,
        )

    def _generate_analytical_fallback(
        self, metrics: Dict[str, float], stats: Dict[str, Any], features: List[str], target: str,
        numeric_features: Optional[List[str]] = None, categorical_features: Optional[List[str]] = None,
        model_name: str = "",
    ) -> str:
        r2 = metrics.get("R2", 0.0)
        adj_r2 = metrics.get("Adjusted R2", 0.0)
        rmse = metrics.get("RMSE", 0.0)
        mae = metrics.get("MAE", 0.0)
        
        coefs = stats.get("coefficients", {})
        p_values = stats.get("p_values", {})
        
        # 1. Interpret R2
        if r2 >= 0.90:
            r2_desc = f"**Excellent fit** ({r2:.1%}). The model explains **{r2:.1%}** of the variance in `{target}`, indicating very high predictive capability."
        elif r2 >= 0.70:
            r2_desc = f"**Strong fit** ({r2:.1%}). The model explains **{r2:.1%}** of the variance in `{target}`. This is typically reliable for decision-making and forecasting."
        elif r2 >= 0.50:
            r2_desc = f"**Moderate fit** ({r2:.1%}). The model explains **{r2:.1%}** of the variance in `{target}`. There is noticeable unexplained variance, indicating potential missing variables or nonlinear relationships."
        else:
            r2_desc = f"**Weak fit** ({r2:.1%}). The model explains only **{r2:.1%}** of the variance in `{target}`. It should not be used in production without incorporating additional features or testing non-linear algorithms."

        # 2. Interpret features significance. Iterates over the ACTUAL
        # coefficient keys (which are now expanded one-hot/scaled pipeline
        # output names like "cat__city_Chennai", not the original selected
        # feature names) via the same shared mapping helper used for the PDF
        # report and charts — a categorical feature produces one coefficient
        # PER CATEGORY relative to its dropped reference category, not one
        # coefficient for the whole column, so iterating `features` directly
        # would silently miss every categorical entry (they'd never match a
        # dict key exactly).
        numeric_feats = numeric_features if numeric_features is not None else features
        categorical_feats = categorical_features or []
        encoded_categorical_names = [
            k[len("cat__"):] for k in coefs if k.startswith("cat__")
        ]
        # coefs and p_values come from the same fitted statsmodels result, so
        # they share the exact same key set in the exact same (insertion)
        # order — mapping both through the identical call and zipping by
        # position is simpler and more robust than trying to reconstruct one
        # dict's raw key from the other's already-stripped display label.
        mapped_coefs = map_expanded_coefficients(
            {k: v for k, v in coefs.items() if k != "const"},
            numeric_feats, categorical_feats, encoded_categorical_names,
        )
        mapped_pvalues = map_expanded_coefficients(
            {k: v for k, v in p_values.items() if k != "const"},
            numeric_feats, categorical_feats, encoded_categorical_names,
        )
        pvalue_by_feature = {e["feature"]: e["value"] for e in mapped_pvalues}

        significant_feats = []
        insignificant_feats = []

        for entry in mapped_coefs:
            feat = entry["feature"]
            source = entry["source_feature"]
            coef_val = entry["value"]
            p_val = pvalue_by_feature.get(feat)

            # statsmodels can leave a coefficient/p-value unresolved (NaN,
            # sanitized to None before storage) on a near-singular design
            # matrix — most commonly Polynomial Regression once feature
            # expansion produces many collinear columns. Report that
            # explicitly rather than crashing on a None comparison/format.
            if p_val is None or coef_val is None:
                insignificant_feats.append(
                    f"- **`{feat}`**: statistical significance could not be computed (likely multicollinearity "
                    f"among the expanded features). Treat this coefficient with caution."
                )
                continue

            # Identify significance
            if p_val <= 0.05:
                direction = "positive" if coef_val >= 0 else "negative"
                significant_feats.append(
                    f"- **`{feat}`** is **statistically significant** (p-value: `{p_val:.4f}` < 0.05). It has a **{direction}** impact "
                    + (
                        f"on `{target}` relative to the baseline category of `{source}`, of **{coef_val:,.2f}**."
                        if source in categorical_feats
                        else f"— holding all other features constant, a 1-unit increase in `{feat}` is associated with an average change of **{coef_val:,.2f}** in `{target}`."
                    )
                )
            else:
                insignificant_feats.append(
                    f"- **`{feat}`** is **not statistically significant** (p-value: `{p_val:.4f}` >= 0.05). We cannot reject the null hypothesis that this feature has no effect on `{target}`. Consider dropping it to simplify the model."
                )

        significant_text = "\n".join(significant_feats) if significant_feats else "No single feature was statistically significant at the 5% confidence level."
        insignificant_text = "\n".join(insignificant_feats) if insignificant_feats else "All selected features are statistically significant."

        # 3. Intercept explanation
        intercept = coefs.get("const", coefs.get("Intercept"))
        if intercept is None:
            intercept_text = (
                f"The baseline intercept could not be reliably estimated (likely multicollinearity among the "
                f"expanded features), so no baseline `{target}` value is reported here."
            )
        else:
            intercept_text = f"The baseline intercept value is **{intercept:,.2f}**, representing the predicted `{target}` if all features are zero (though this baseline configuration may not always represent a realistic physical scenario)."

        markdown_output = f"""### Statistical Interpretation: **Model Results**

#### Fit Quality (R-Squared & Variance)
* {r2_desc}
* **Adjusted R²:** `{adj_r2:.4f}` (accounting for feature count, ensuring features add genuine predictive value).
* **Average Error:** The Root Mean Squared Error (RMSE) is **{rmse:,.2f}**, and the Mean Absolute Error (MAE) is **{mae:,.2f}**. This indicates that the typical difference between the model's prediction and the actual value is approximately **{mae:,.2f}** units.

#### Parameter Signifcance & Relationships
{significant_text}

{f'#### Statistically Insignificant Parameters (p > 0.05)\n{insignificant_text}' if insignificant_feats else ''}

* **Intercept (Constant):** {intercept_text}
"""
        return markdown_output
