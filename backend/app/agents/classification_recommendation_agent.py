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


class ClassificationRecommendationAgent:
    """Classification counterpart to recommendation_agent.py's
    RecommendationAgent — see classification_explanation_agent.py's
    docstring for why this is a separate class rather than a task_type
    branch inside the regression agent."""

    def __init__(self):
        self.llm = get_llm()

    async def recommend(
        self, model_name: str, metrics: Dict[str, Any], stats: Dict[str, Any], features: List[str], target: str,
        numeric_features: Optional[List[str]] = None, categorical_features: Optional[List[str]] = None,
    ) -> str:
        """
        Provides actionable suggestions to improve classification accuracy,
        handle class imbalance, and try alternate algorithms.
        """
        p_vals = stats.get("p_values", {}) or {}
        cat_counts = {}
        for k in p_vals.keys():
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

        filtered_pvals = {}
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
            "p_values": filtered_pvals,
        }

        class_report = metrics.get("ClassificationReport") or {}
        classes_list = metrics.get("Classes") or []
        support_by_class = {
            str(cls): class_report.get(str(cls), {}).get("support")
            for cls in classes_list
        }

        context = f"""
        Current Model: {model_name}
        Evaluation Metrics: Accuracy={metrics.get('Accuracy')}, Precision={metrics.get('Precision')},
        Recall={metrics.get('Recall')}, F1={metrics.get('F1')}
        Per-Class Support (real test-set sample count per class — the only honest basis for a class-imbalance claim): {support_by_class}
        Confusion Matrix: {metrics.get('ConfusionMatrix')}
        Features Used: {features}
        Target: {target}
        P-Values (Filtered): {filtered_pvals}
        """

        if self.llm:
            try:
                system_msg = SystemMessage(content=(
                    "You are a senior ML advisor. Suggest actionable steps to improve this classifier's "
                    "accuracy, engineer features, and list alternative classification algorithms the user "
                    "should try next. Output in clean Markdown format. "
                    "IMPORTANT: do not assume or automatically claim class imbalance. First check the actual "
                    "per-class support values provided above. Only recommend imbalance-handling techniques "
                    "(class_weight='balanced', oversampling/SMOTE, etc.) if the support values genuinely show "
                    "a meaningful skew (e.g. one class more than ~3x the size of another). If the classes are "
                    "roughly balanced, explicitly state: 'The class distribution is relatively balanced, so "
                    "the low performance is more likely related to insufficient predictive signal, feature "
                    "representation, or model configuration rather than severe class imbalance.' and focus "
                    "your recommendations on feature engineering, model choice, and hyperparameters instead. "
                    "Start your response with the exact heading '## Recommendations' on its own line, since "
                    "the report generator locates this section by that heading."
                ))
                human_msg = HumanMessage(content=f"Provide recommendations based on this classification model run:\n{context}")
                response = await self.llm.ainvoke([system_msg, human_msg])
                return response.content
            except Exception as e:
                logger.warning(f"ClassificationRecommendationAgent LLM failed: {e}. Falling back to analytical suggestions.")

        return self._generate_analytical_fallback(
            model_name, metrics, stats_clean, features, target, numeric_features, categorical_features,
        )

    def _generate_analytical_fallback(
        self, model_name: str, metrics: Dict[str, Any], stats: Dict[str, Any], features: List[str], target: str,
        numeric_features: Optional[List[str]] = None, categorical_features: Optional[List[str]] = None,
    ) -> str:
        accuracy = metrics.get("Accuracy", 0.0)
        precision = metrics.get("Precision", 0.0)
        recall = metrics.get("Recall", 0.0)
        conf_matrix = metrics.get("ConfusionMatrix") or []
        classes = metrics.get("Classes") or []
        p_values = stats.get("p_values", {})

        numeric_feats = numeric_features if numeric_features is not None else features
        categorical_feats = categorical_features or []
        encoded_categorical_names = [k[len("cat__"):] for k in p_values if k.startswith("cat__")]
        mapped_pvalues = map_expanded_coefficients(
            {k: v for k, v in p_values.items() if k != "const" and not k.startswith("const__")},
            numeric_feats, categorical_feats, encoded_categorical_names,
        )
        insignificant_feats = [
            e["feature"] for e in mapped_pvalues
            if isinstance(e["value"], (int, float)) and e["value"] > 0.05
        ]

        # Class-imbalance detection from the real per-class support in the
        # classification report (falls back to confusion-matrix row sums —
        # mathematically the same count, kept only for models trained before
        # ClassificationReport was stored) — a >3x ratio between the largest
        # and smallest class is a common rule-of-thumb imbalance threshold.
        # Never assumed: only flagged when these actual counts warrant it.
        class_report = metrics.get("ClassificationReport") or {}
        class_counts = [
            v for cls in classes
            if isinstance((v := class_report.get(str(cls), {}).get("support")), (int, float))
        ]
        if not class_counts and conf_matrix:
            class_counts = [sum(row) for row in conf_matrix]
        is_imbalanced = bool(class_counts) and len(class_counts) > 1 and max(class_counts) > 3 * max(min(class_counts), 1)
        is_balanced_check_possible = bool(class_counts) and len(class_counts) > 1

        recommendations = ["### Diagnostic Recommendations & Next Steps\n"]

        # 1. Dataset improvements & Feature Engineering
        recommendations.append("#### 1. Feature Engineering & Data Quality")
        if insignificant_feats:
            feats_str = ", ".join([f"`{f}`" for f in insignificant_feats])
            recommendations.append(f"- **Drop Insignificant Features:** Consider dropping {feats_str} because their p-values are above 0.05. Removing them simplifies the model and mitigates overfitting without losing predictive power.")
        else:
            recommendations.append("- **Feature Significance:** Good! All selected features are statistically significant, meaning they contribute relevant information to the classification decision.")

        if is_imbalanced:
            recommendations.append(
                "- **Class Imbalance Detected:** The classes in your test split are noticeably unequal in size "
                f"(largest class support is {max(class_counts) / max(min(class_counts), 1):.1f}x the smallest). "
                "Consider `class_weight=\"balanced\"` (supported by Logistic Regression, SVM, Random Forest, and "
                "similar models), oversampling the minority class (e.g. SMOTE), or evaluating with per-class "
                "precision/recall rather than overall accuracy alone, since accuracy can look deceptively high "
                "on an imbalanced dataset."
            )
        elif is_balanced_check_possible and accuracy < 0.75:
            recommendations.append(
                "- **Class Distribution:** The class distribution is relatively balanced, so the low performance "
                "is more likely related to insufficient predictive signal, feature representation, or model "
                "configuration rather than severe class imbalance — focus on the feature engineering and "
                "algorithm suggestions below rather than imbalance-handling techniques."
            )

        if len(features) < 2:
            recommendations.append("- **Expand Predictors:** You are using only one feature. Consider adding more features to help the model separate the classes.")
        else:
            recommendations.append("- **Interaction Terms:** Try creating interaction terms (e.g. multiplying two features together) if you suspect their joint effect helps distinguish classes.")

        recommendations.append("- **Scaling check:** Distance-based algorithms (KNN, SVM) are sensitive to feature scale — ensure StandardScaler or MinMaxScaler is applied if you haven't already.")

        # 2. Algorithm Suggestions
        recommendations.append("\n#### 2. Alternative Algorithms to Try")
        name_clean = model_name.replace(" ", "").lower()

        if "logistic" in name_clean:
            recommendations.append("- **Random Forest Classifier:** Since your current model is linear, testing a **Random Forest Classifier** is recommended — it captures non-linear decision boundaries and feature interactions automatically.")
            recommendations.append("- **Gradient Boosting / XGBoost:** Often outperforms a single linear model on tabular classification problems with mixed feature types.")
        elif "randomforest" in name_clean or "decisiontree" in name_clean or "gradientboosting" in name_clean or "xgboost" in name_clean or "lightgbm" in name_clean:
            recommendations.append("- **Logistic Regression:** Your current model is a complex tree-based ensemble. Try **Logistic Regression** to see if a simpler, highly interpretable model achieves competitive performance — and gives you readable per-feature log-odds coefficients.")
            recommendations.append("- **Support Vector Machine (SVM):** With an RBF kernel, SVM is robust on complex, non-linearly-separable class boundaries.")
        else:
            recommendations.append("- **Random Forest / Gradient Boosting:** Tree ensembles are broadly strong baselines — worth comparing against your current model.")

        # 3. Optimization suggestions
        recommendations.append("\n#### 3. Model Fine-Tuning")
        if accuracy < 0.70:
            recommendations.append("- **Increase Dataset Size:** Accuracy below 0.70 suggests the model struggles to separate the classes with the current data. Collecting more labeled samples, especially for underrepresented classes, can help.")
            recommendations.append("- **Threshold Tuning:** For binary classification, the default 0.5 probability threshold isn't always optimal — tune it against your precision/recall trade-off.")
        else:
            recommendations.append("- **Hyperparameter Tuning:** Fine-tune model parameters (e.g., `n_estimators`/`max_depth` for tree ensembles, `C` for Logistic Regression/SVM) using GridSearchCV to squeeze out extra accuracy or F1.")

        if precision < recall - 0.15:
            recommendations.append("- **Precision/Recall Trade-off:** Recall is notably higher than precision — the model over-predicts the positive class. Consider raising the decision threshold or adding features that help distinguish false positives.")
        elif recall < precision - 0.15:
            recommendations.append("- **Precision/Recall Trade-off:** Precision is notably higher than recall — the model is missing true positives. Consider lowering the decision threshold or addressing class imbalance.")

        return "\n".join(recommendations)
