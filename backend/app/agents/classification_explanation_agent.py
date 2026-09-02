import logging
import os
from typing import Dict, Any, List, Optional
from app.services.huggingface_service import get_llm
from app.ml.feature_types import map_expanded_coefficients, _is_logistic_regression_model

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
        model_name: str = "",
    ) -> str:
        """
        Interprets classification metrics and Logit/MNLogit statistics.
        Returns a detailed Markdown analysis of Accuracy/Precision/Recall/F1,
        the confusion matrix, and feature significance.
        """
        coefs = stats.get("coefficients", {}) or {}
        p_vals = stats.get("p_values", {}) or {}
        
        cat_counts = {}
        for k in coefs.keys():
            if k.startswith("cat__"):
                col_name = None
                if categorical_features:
                    for cf in categorical_features:
                        if k.startswith(f"cat__{cf}_") or k == f"cat__{cf}":
                            col_name = cf
                            break
                if not col_name:
                    parts = k[5:].split("_")
                    col_name = parts[0]
                cat_counts[col_name] = cat_counts.get(col_name, 0) + 1

        high_card_cols = {col for col, count in cat_counts.items() if count > 10}

        filtered_coefs = {}
        filtered_pvals = {}
        omitted_notes = []
        for col in sorted(list(high_card_cols)):
            count = cat_counts[col]
            omitted_notes.append(f"{col} is a high-cardinality categorical feature with {count} unique values. Individual encoded coefficients are omitted from the report to maintain readability.")

        for k, v in coefs.items():
            is_high_card = False
            for col in high_card_cols:
                if k.startswith(f"cat__{col}_") or k == f"cat__{col}":
                    is_high_card = True
                     # Wait, let's keep only top 10 features if we wanted to show top 10
                    break
            if not is_high_card:
                filtered_coefs[k] = v

        for k, v in p_vals.items():
            is_high_card = False
            for col in high_card_cols:
                if k.startswith(f"cat__{col}_") or k == f"cat__{col}":
                    is_high_card = True
                    break
            if not is_high_card:
                filtered_pvals[k] = v

        stats_clean = {
            **stats,
            "coefficients": filtered_coefs,
            "p_values": filtered_pvals,
        }

        omitted_text = "\n".join(omitted_notes)

        class_report = metrics.get("ClassificationReport") or {}
        classes_list = metrics.get("Classes") or []
        # Per-class support (test-set row counts) is the ONLY honest basis for
        # a class-imbalance claim — pass it explicitly so the LLM checks real
        # numbers instead of assuming imbalance whenever performance is weak.
        per_class_lines = []
        for cls in classes_list:
            entry = class_report.get(str(cls)) or {}
            per_class_lines.append(
                f"  - {cls}: precision={entry.get('precision')}, recall={entry.get('recall')}, "
                f"f1-score={entry.get('f1-score')}, support={entry.get('support')}"
            )
        per_class_text = "\n".join(per_class_lines) if per_class_lines else "Not available."

        context = f"""
        Evaluation Metrics: Accuracy={metrics.get('Accuracy')}, Precision={metrics.get('Precision')},
        Recall={metrics.get('Recall')}, F1={metrics.get('F1')}, ROC_AUC={metrics.get('ROC_AUC')},
        Balanced Accuracy={metrics.get('BalancedAccuracy')}, Matthews Correlation Coefficient={metrics.get('MCC')},
        Log Loss={metrics.get('LogLoss')}, Specificity={metrics.get('Specificity')},
        Precision Macro={metrics.get('Precision Macro')}, Recall Macro={metrics.get('Recall Macro')}, F1 Macro={metrics.get('F1 Macro')}
        Classes: {metrics.get('Classes')}
        Per-Class Performance (precision/recall/f1-score/support — support is the real per-class sample count, use it to judge class balance):
{per_class_text}
        Confusion Matrix: {metrics.get('ConfusionMatrix')}
        Target Variable: {target}
        Feature Variables: {features}
        Coefficients (Filtered): {filtered_coefs}
        P-Values (Filtered): {filtered_pvals}
        High-cardinality Categorical Features Omitted: {omitted_text}
        """

        # Applied to BOTH the LLM response and the analytical fallback below
        # — a system-prompt instruction alone isn't a reliable guarantee the
        # LLM actually includes this caveat, so it's prepended deterministically
        # here rather than left to the model's discretion.
        aux_model_note = ""
        if model_name and not _is_logistic_regression_model(model_name):
            aux_model_note = (
                f"> **Note:** the coefficient/p-value figures below come from an auxiliary Logistic/"
                f"Multinomial Logistic Regression fit used for interpretability — they describe correlational "
                f"structure in the data, not the internal mechanics of the trained **{model_name}** model.\n\n"
            )

        if self.llm:
            try:
                system_msg = SystemMessage(content=(
                    "You are a senior data scientist. Interpret these classification model results using ONLY "
                    "the actual numbers provided below — never invent or estimate a value. Your response MUST "
                    "explicitly cover, in this order: (1) overall model performance (accuracy/F1), "
                    "(2) the STRONGEST-performing class, named explicitly, identified by comparing the per-class "
                    "F1-scores provided, (3) the WEAKEST-performing class, named explicitly, identified the same "
                    "way, (4) the most important misclassification pattern(s) from the confusion matrix, "
                    "(5) plausible feature limitations that could explain the weak class's performance, and "
                    "(6) 2-3 practical, specific improvement suggestions. "
                    "Do NOT write generic statements like 'performance is steady across classes' — if the "
                    "per-class F1-scores differ meaningfully, say so explicitly and name which classes. "
                    "Only mention class imbalance if the per-class 'support' values actually show a meaningful "
                    "skew — if support is roughly even across classes, explicitly state that the distribution is "
                    "relatively balanced and that weak performance is more likely due to insufficient predictive "
                    "signal, feature representation, or model configuration, not imbalance. "
                    "Coefficients are on the log-odds scale — explain them as increasing or decreasing the odds "
                    "of a class, never as a 'change in the target value'. Format output in clean Markdown with "
                    "a '## Fit Quality' heading before the overall/strongest/weakest discussion. Do NOT output "
                    "large tables of coefficients or duplicate metric tables."
                ))
                human_msg = HumanMessage(content=f"Interpret these classification results:\n{context}")
                response = await self.llm.ainvoke([system_msg, human_msg])
                return aux_model_note + response.content
            except Exception as e:
                logger.warning(f"ClassificationInterpretationAgent LLM failed: {e}. Falling back to statistical parser.")

        return aux_model_note + self._generate_analytical_fallback(metrics, stats_clean, features, target, numeric_features, categorical_features, omitted_text, model_name)

    def _generate_analytical_fallback(
        self, metrics: Dict[str, Any], stats: Dict[str, Any], features: List[str], target: str,
        numeric_features: Optional[List[str]] = None, categorical_features: Optional[List[str]] = None,
        omitted_text: str = "", model_name: str = "",
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

        # 2.5 Strongest/weakest class by F1-score, plus a real class-balance
        # check from actual per-class support — never assume imbalance, and
        # never claim performance is "steady" when the per-class F1-scores
        # actually differ (see the settings/spec that motivated this: a
        # generic "steady across classes" statement is wrong whenever the
        # weakest class's F1 is meaningfully below the strongest's).
        class_report = metrics.get("ClassificationReport") or {}
        class_perf_text = "Per-class performance data was not available for this model."
        feature_limitation_text = ""
        if class_report and classes:
            f1_by_class = [
                (cls, class_report.get(str(cls), {}).get("f1-score"))
                for cls in classes
                if isinstance(class_report.get(str(cls), {}).get("f1-score"), (int, float))
            ]
            if f1_by_class:
                strongest_cls, strongest_f1 = max(f1_by_class, key=lambda x: x[1])
                weakest_cls, weakest_f1 = min(f1_by_class, key=lambda x: x[1])
                if len(f1_by_class) > 1 and (strongest_f1 - weakest_f1) > 0.05:
                    class_perf_text = (
                        f"Performance is **not uniform across classes**. **`{strongest_cls}`** is the "
                        f"strongest-performing class (F1 = `{strongest_f1:.3f}`), while **`{weakest_cls}`** "
                        f"is the weakest-performing class (F1 = `{weakest_f1:.3f}`) — a gap of "
                        f"`{strongest_f1 - weakest_f1:.3f}`. Treat metrics for `{weakest_cls}` with caution "
                        "in any downstream decision."
                    )
                    feature_limitation_text = (
                        f"The gap between `{strongest_cls}` and `{weakest_cls}` suggests the current feature "
                        f"set may not capture what distinguishes `{weakest_cls}` from the other classes — "
                        "consider adding features specific to that class's behavior, or reviewing whether "
                        f"`{weakest_cls}` samples overlap heavily with another class in feature space."
                    )
                else:
                    class_perf_text = (
                        f"Performance is **relatively even across classes** — the strongest class "
                        f"(**`{strongest_cls}`**, F1 = `{strongest_f1:.3f}`) and weakest class "
                        f"(**`{weakest_cls}`**, F1 = `{weakest_f1:.3f}`) are close enough that no single "
                        "class stands out as a specific weak point."
                    )

            support_by_class = [
                class_report.get(str(cls), {}).get("support")
                for cls in classes
                if isinstance(class_report.get(str(cls), {}).get("support"), (int, float))
            ]
            if len(support_by_class) > 1:
                ratio = max(support_by_class) / max(min(support_by_class), 1)
                if ratio > 3:
                    class_perf_text += (
                        f" The classes are also noticeably imbalanced in the test set (largest class is "
                        f"{ratio:.1f}x the size of the smallest) — this alone can suppress recall on the "
                        "smaller class(es) regardless of feature quality."
                    )
                elif accuracy < 0.75:
                    class_perf_text += (
                        " The class distribution is relatively balanced, so the lower performance is more "
                        "likely related to insufficient predictive signal, feature representation, or model "
                        "configuration rather than class imbalance."
                    )

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

#### Class-Level Performance
{class_perf_text}
{f'* **Possible feature limitation:** {feature_limitation_text}' if feature_limitation_text else ''}

#### Confusion Matrix Insight
{confusion_text}

#### Parameter Significance & Relationships
{significant_text}

{omitted_text}

{f'#### Statistically Insignificant Parameters (p > 0.05)\n{insignificant_text}' if insignificant_feats else ''}

* **Intercept (Baseline Log-Odds):** {intercept_text}

"""
        return markdown_output
