import logging
import os
from typing import Dict, Any, List, Optional
from app.services.huggingface_service import get_llm
from app.ml.feature_types import map_expanded_coefficients

logger = logging.getLogger("regression_studio.agents")

try:
    from langchain_core.messages import SystemMessage, HumanMessage
except ImportError:
    pass


class ClassificationInterpretationAgent:
    """
    Classification counterpart to explanation_agent.py's
    StatisticalInterpretationAgent. Kept as a genuinely separate class
    rather than a task_type branch inside the regression agent — the
    regression fallback text is sentence-level regression-specific
    (R²-tiering, "1-unit increase associated with a change in target"
    coefficient framing, intercept-as-baseline-prediction) and none of that
    vocabulary is statistically meaningful for a classifier (no R²; a
    logistic coefficient means "change in log-odds," not "change in the
    target's value").
    """

    def __init__(self):
        self.llm = get_llm()

    async def explain(
        self, metrics: Dict[str, Any], stats: Dict[str, Any], features: List[str], target: str,
        numeric_features: Optional[List[str]] = None, categorical_features: Optional[List[str]] = None,
    ) -> str:
        """
        Interprets classification metrics and Logit/MNLogit statistics.
        Returns a detailed Markdown analysis of Accuracy/Precision/Recall/F1,
        the confusion matrix, and feature significance.
        """
        context = f"""
        Evaluation Metrics: Accuracy={metrics.get('Accuracy')}, Precision={metrics.get('Precision')},
        Recall={metrics.get('Recall')}, F1={metrics.get('F1')}, ROC_AUC={metrics.get('ROC_AUC')},
        Balanced Accuracy={metrics.get('BalancedAccuracy')}, Matthews Correlation Coefficient={metrics.get('MCC')},
        Log Loss={metrics.get('LogLoss')}, Specificity={metrics.get('Specificity')},
        Precision Macro={metrics.get('Precision Macro')}, Recall Macro={metrics.get('Recall Macro')}, F1 Macro={metrics.get('F1 Macro')}
        Classes: {metrics.get('Classes')}
        Confusion Matrix: {metrics.get('ConfusionMatrix')}
        Target Variable: {target}
        Feature Variables: {features}
        Coefficients: {stats.get('coefficients')}
        P-Values: {stats.get('p_values')}
        """

        if self.llm:
            try:
                system_msg = SystemMessage(content="You are a senior data scientist. Interpret these classification model results. Explain accuracy, precision, recall, F1-score, what the confusion matrix reveals about the model's mistakes, and which features are statistically significant based on p-values. Coefficients are on the log-odds scale — explain them as increasing or decreasing the odds of a class, never as a 'change in the target value'. Format output in clean Markdown.")
                human_msg = HumanMessage(content=f"Interpret these classification results:\n{context}")
                response = await self.llm.ainvoke([system_msg, human_msg])
                return response.content
            except Exception as e:
                logger.warning(f"ClassificationInterpretationAgent LLM failed: {e}. Falling back to statistical parser.")

        return self._generate_analytical_fallback(metrics, stats, features, target, numeric_features, categorical_features)

    def _generate_analytical_fallback(
        self, metrics: Dict[str, Any], stats: Dict[str, Any], features: List[str], target: str,
        numeric_features: Optional[List[str]] = None, categorical_features: Optional[List[str]] = None,
    ) -> str:
        accuracy = metrics.get("Accuracy") or 0.0
        precision = metrics.get("Precision") or 0.0
        recall = metrics.get("Recall") or 0.0
        f1 = metrics.get("F1") or 0.0
        roc_auc = metrics.get("ROC_AUC")
        classes = metrics.get("Classes") or []
        conf_matrix = metrics.get("ConfusionMatrix") or []

        coefs = stats.get("coefficients", {})
        p_values = stats.get("p_values", {})

        # 1. Interpret Accuracy — same tiering shape as the regression
        # agent's R² tiers, but keyed to accuracy thresholds since the two
        # metrics aren't on comparable scales for interpretation purposes.
        if accuracy >= 0.90:
            acc_desc = f"**Excellent classifier** ({accuracy:.1%} accuracy). The model correctly classifies **{accuracy:.1%}** of samples, indicating very strong predictive capability."
        elif accuracy >= 0.75:
            acc_desc = f"**Strong classifier** ({accuracy:.1%} accuracy). The model correctly classifies **{accuracy:.1%}** of samples — typically reliable for decision-making."
        elif accuracy >= 0.60:
            acc_desc = f"**Moderate classifier** ({accuracy:.1%} accuracy). There is a meaningful error rate, suggesting missing features or class overlap that's hard to separate with the current feature set."
        else:
            acc_desc = f"**Weak classifier** ({accuracy:.1%} accuracy), only modestly better than guessing for this number of classes. It should not be used in production without additional features or a different algorithm."

        roc_text = f" The ROC-AUC of **{roc_auc:.3f}** measures how well the model ranks positive cases above negative ones across all decision thresholds." if isinstance(roc_auc, (int, float)) else ""

        # 2. Confusion-matrix-driven observation — identify the largest
        # off-diagonal cell (the pair of classes the model confuses most).
        confusion_text = "Confusion matrix data was not available for this model."
        if conf_matrix and classes and len(conf_matrix) == len(classes):
            worst = None
            for i, row in enumerate(conf_matrix):
                for j, count in enumerate(row):
                    if i != j and (worst is None or count > worst[2]):
                        worst = (i, j, count)
            if worst and worst[2] > 0:
                actual_cls, pred_cls, count = classes[worst[0]], classes[worst[1]], worst[2]
                confusion_text = f"The model's most common mistake is misclassifying **`{actual_cls}`** as **`{pred_cls}`** ({count} such cases in the test set) — worth a closer look if this specific confusion is costly for your use case."
            else:
                confusion_text = "The confusion matrix shows no off-diagonal errors on the test set — every test sample was classified correctly."

        # 3. Per-feature significance — same map_expanded_coefficients
        # pattern as the regression agent, but phrased in log-odds terms
        # (a logistic/MNLogit coefficient is the change in log-odds of the
        # target class per unit of the feature, not a change in the target's
        # value, which has no meaning for a categorical target).
        numeric_feats = numeric_features if numeric_features is not None else features
        categorical_feats = categorical_features or []
        encoded_categorical_names = [
            k[len("cat__"):] for k in coefs if k.startswith("cat__")
        ]
        mapped_coefs = map_expanded_coefficients(
            {k: v for k, v in coefs.items() if k != "const" and not k.startswith("const__")},
            numeric_feats, categorical_feats, encoded_categorical_names,
        )
        mapped_pvalues = map_expanded_coefficients(
            {k: v for k, v in p_values.items() if k != "const" and not k.startswith("const__")},
            numeric_feats, categorical_feats, encoded_categorical_names,
        )
        pvalue_by_feature = {e["feature"]: e["value"] for e in mapped_pvalues}

        positive_class = classes[-1] if classes else target

        significant_feats = []
        insignificant_feats = []

        for entry in mapped_coefs:
            feat = entry["feature"]
            source = entry["source_feature"]
            coef_val = entry["value"]
            p_val = pvalue_by_feature.get(feat)

            if p_val is None or coef_val is None:
                insignificant_feats.append(
                    f"- **`{feat}`**: statistical significance could not be computed (likely multicollinearity "
                    f"among the expanded features, or a near-singular design matrix). Treat this coefficient with caution."
                )
                continue

            if p_val <= 0.05:
                direction = "increases" if coef_val >= 0 else "decreases"
                significant_feats.append(
                    f"- **`{feat}`** is **statistically significant** (p-value: `{p_val:.4f}` < 0.05). It **{direction}** the log-odds "
                    + (
                        f"of `{target}` = `{positive_class}` relative to the baseline category of `{source}`, by **{coef_val:,.3f}**."
                        if source in categorical_feats
                        else f"of `{target}` = `{positive_class}` by **{coef_val:,.3f}** per unit increase, holding other features constant."
                    )
                )
            else:
                insignificant_feats.append(
                    f"- **`{feat}`** is **not statistically significant** (p-value: `{p_val:.4f}` >= 0.05). We cannot reject the null hypothesis that this feature has no effect on the classification outcome. Consider dropping it to simplify the model."
                )

        significant_text = "\n".join(significant_feats) if significant_feats else "No single feature was statistically significant at the 5% confidence level."
        insignificant_text = "\n".join(insignificant_feats) if insignificant_feats else "All selected features are statistically significant."

        intercept = coefs.get("const")
        if intercept is None:
            intercept_text = (
                "The baseline log-odds could not be reliably estimated (likely multicollinearity among the "
                "expanded features, or a multiclass fit with per-class intercepts), so no single baseline value is reported here."
            )
        else:
            intercept_text = f"The baseline log-odds is **{intercept:,.3f}**, representing the predicted log-odds of the positive class when all features are zero (this baseline configuration may not always represent a realistic physical scenario)."

        # Additional metrics computed alongside Accuracy/Precision/Recall/F1
        # (see calculate_classification_metrics) — each independently
        # optional, so only the ones this model/data combination actually
        # produced are rendered.
        extra_metric_lines = []
        balanced_accuracy = metrics.get("BalancedAccuracy")
        if isinstance(balanced_accuracy, (int, float)):
            extra_metric_lines.append(
                f"* **Balanced Accuracy:** `{balanced_accuracy:.3f}` — average per-class recall, unaffected by class imbalance (unlike plain Accuracy)."
            )
        mcc = metrics.get("MCC")
        if isinstance(mcc, (int, float)):
            extra_metric_lines.append(
                f"* **Matthews Correlation Coefficient:** `{mcc:.3f}` — a single balanced score from -1 (always wrong) to +1 (always right) that accounts for all four confusion-matrix quadrants, robust to class imbalance."
            )
        log_loss_val = metrics.get("LogLoss")
        if isinstance(log_loss_val, (int, float)):
            extra_metric_lines.append(
                f"* **Log Loss:** `{log_loss_val:.4f}` — penalizes confident-but-wrong predicted probabilities; lower is better."
            )
        specificity = metrics.get("Specificity")
        if isinstance(specificity, (int, float)):
            extra_metric_lines.append(
                f"* **Specificity:** `{specificity:.3f}` — of everything that actually belonged to the negative class, this fraction was correctly identified as negative."
            )
        f1_macro = metrics.get("F1 Macro")
        if isinstance(f1_macro, (int, float)):
            extra_metric_lines.append(
                f"* **F1 Macro:** `{f1_macro:.3f}` — F1 averaged equally across all classes regardless of size, versus the class-frequency-weighted F1-Score above; a large gap between the two suggests the model is underperforming on a rarer class."
            )
        extra_metrics_text = "\n".join(extra_metric_lines)

        markdown_output = f"""### Statistical Interpretation: **Classification Model Results**

#### Fit Quality (Accuracy, Precision, Recall, F1)
* {acc_desc}{roc_text}
* **Precision:** `{precision:.3f}` — of everything the model predicted as a given class, this fraction was actually correct.
* **Recall:** `{recall:.3f}` — of everything that actually belonged to a given class, this fraction was correctly identified.
* **F1-Score:** `{f1:.3f}` — the harmonic mean of precision and recall, a single balanced measure of classification quality.
{extra_metrics_text}

#### Confusion Matrix Insight
{confusion_text}

#### Parameter Significance & Relationships
{significant_text}

{f'#### Statistically Insignificant Parameters (p > 0.05)\n{insignificant_text}' if insignificant_feats else ''}

* **Intercept (Baseline Log-Odds):** {intercept_text}
"""
        return markdown_output
