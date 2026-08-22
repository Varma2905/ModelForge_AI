import logging
import os
from typing import Dict, Any, List
from app.agents.dataset_agent import get_llm

logger = logging.getLogger("regression_studio.agents")

try:
    from langchain_core.messages import SystemMessage, HumanMessage
except ImportError:
    pass

class StatisticalInterpretationAgent:
    def __init__(self):
        self.llm = get_llm()

    async def explain(self, metrics: Dict[str, float], stats: Dict[str, Any], features: List[str], target: str) -> str:
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

        if self.llm:
            try:
                system_msg = SystemMessage(content="You are a senior statistical analyst. Interpret the OLS regression summary metrics. Explain R-squared, RMSE, which features are statistically significant based on p-values, and what the coefficients mean in plain English. Format output in clean Markdown.")
                human_msg = HumanMessage(content=f"Interpret these statistical regression results:\n{context}")
                response = await self.llm.ainvoke([system_msg, human_msg])
                return response.content
            except Exception as e:
                logger.warning(f"StatisticalInterpretationAgent LLM failed: {e}. Falling back to statistical parser.")

        # Analytical fallback
        return self._generate_analytical_fallback(metrics, stats, features, target)

    def _generate_analytical_fallback(self, metrics: Dict[str, float], stats: Dict[str, Any], features: List[str], target: str) -> str:
        r2 = metrics.get("R2", 0.0)
        adj_r2 = metrics.get("Adjusted R2", 0.0)
        rmse = metrics.get("RMSE", 0.0)
        mae = metrics.get("MAE", 0.0)
        
        coefs = stats.get("coefficients", {})
        p_values = stats.get("p_values", {})
        
        # 1. Interpret R2
        if r2 >= 0.90:
            r2_desc = f"**Excellent fit** ({r2:.1%}). The model explains **{r2:.1f}%** of the variance in `{target}`, indicating very high predictive capability."
        elif r2 >= 0.70:
            r2_desc = f"**Strong fit** ({r2:.1%}). The model explains **{r2:.1f}%** of the variance in `{target}`. This is typically reliable for decision-making and forecasting."
        elif r2 >= 0.50:
            r2_desc = f"**Moderate fit** ({r2:.1%}). The model explains **{r2:.1f}%** of the variance in `{target}`. There is noticeable unexplained variance, indicating potential missing variables or nonlinear relationships."
        else:
            r2_desc = f"**Weak fit** ({r2:.1%}). The model explains only **{r2:.1f}%** of the variance in `{target}`. It should not be used in production without incorporating additional features or testing non-linear algorithms."

        # 2. Interpret features significance
        significant_feats = []
        insignificant_feats = []
        
        for feat in features:
            p_val = p_values.get(feat, 0.5)
            coef_val = coefs.get(feat, 0.0)
            
            # Identify significance
            if p_val <= 0.05:
                direction = "positive" if coef_val >= 0 else "negative"
                significant_feats.append(
                    f"- **`{feat}`** is **statistically significant** (p-value: `{p_val:.4f}` < 0.05). It has a **{direction}** impact. Holding all other features constant, a 1-unit increase in `{feat}` is associated with an average change of **{coef_val:,.2f}** in `{target}`."
                )
            else:
                insignificant_feats.append(
                    f"- **`{feat}`** is **not statistically significant** (p-value: `{p_val:.4f}` >= 0.05). We cannot reject the null hypothesis that this feature has no effect on `{target}`. Consider dropping it to simplify the model."
                )

        significant_text = "\n".join(significant_feats) if significant_feats else "No single feature was statistically significant at the 5% confidence level."
        insignificant_text = "\n".join(insignificant_feats) if insignificant_feats else "All selected features are statistically significant."
        
        # 3. Intercept explanation
        intercept = coefs.get("const", coefs.get("Intercept", 0.0))
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
