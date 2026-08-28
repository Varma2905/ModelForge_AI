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
        context = f"""
        Current Model: {model_name}
        Evaluation Metrics: Accuracy={metrics.get('Accuracy')}, Precision={metrics.get('Precision')},
        Recall={metrics.get('Recall')}, F1={metrics.get('F1')}
        Confusion Matrix: {metrics.get('ConfusionMatrix')}
        Features Used: {features}
        Target: {target}
        P-Values: {stats.get('p_values')}
        """

        if self.llm:
            try:
                system_msg = SystemMessage(content="You are a senior ML advisor. Suggest actionable steps to improve this classifier's accuracy, handle class imbalance, engineer features, and list alternative classification algorithms the user should try next. Output in clean Markdown format.")
                human_msg = HumanMessage(content=f"Provide recommendations based on this classification model run:\n{context}")
                response = await self.llm.ainvoke([system_msg, human_msg])
                return response.content
            except Exception as e:
                logger.warning(f"ClassificationRecommendationAgent LLM failed: {e}. Falling back to analytical suggestions.")

        return self._generate_analytical_fallback(
            model_name, metrics, stats, features, target, numeric_features, categorical_features,
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

        # Class-imbalance detection from the confusion matrix's row sums
        # (actual class counts in the test split) — a >3x ratio between the
        # largest and smallest class is a common rule-of-thumb imbalance
        # threshold worth flagging.
        class_counts = [sum(row) for row in conf_matrix] if conf_matrix else []
        is_imbalanced = bool(class_counts) and max(class_counts) > 3 * max(min(class_counts), 1)

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
                "- **Class Imbalance Detected:** The classes in your test split are noticeably unequal in size. "
                "Consider `class_weight=\"balanced\"` (supported by Logistic Regression, SVM, Random Forest, and "
                "similar models), oversampling the minority class (e.g. SMOTE), or evaluating with per-class "
                "precision/recall rather than overall accuracy alone, since accuracy can look deceptively high "
                "on an imbalanced dataset."
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
