import logging
import os
from typing import Dict, Any, List
from app.agents.dataset_agent import get_llm

logger = logging.getLogger("regression_studio.agents")

try:
    from langchain_core.messages import SystemMessage, HumanMessage
except ImportError:
    pass

class RecommendationAgent:
    def __init__(self):
        self.llm = get_llm()

    async def recommend(self, model_name: str, metrics: Dict[str, float], stats: Dict[str, Any], features: List[str], target: str) -> str:
        """
        Provides actionable suggestions to improve model accuracy, clean data, and try alternate algorithms.
        """
        context = f"""
        Current Model: {model_name}
        Evaluation Metrics: {metrics}
        Features Used: {features}
        Target: {target}
        P-Values: {stats.get('p_values')}
        """

        if self.llm:
            try:
                system_msg = SystemMessage(content="You are a senior ML advisor. Suggest actionable steps to improve the model's accuracy, data quality, feature engineering, and list alternative algorithms the user should try next. Output in clean Markdown format.")
                human_msg = HumanMessage(content=f"Provide recommendations based on this model run:\n{context}")
                response = await self.llm.ainvoke([system_msg, human_msg])
                return response.content
            except Exception as e:
                logger.warning(f"RecommendationAgent LLM failed: {e}. Falling back to analytical suggestions.")

        # Analytical fallback
        return self._generate_analytical_fallback(model_name, metrics, stats, features, target)

    def _generate_analytical_fallback(self, model_name: str, metrics: Dict[str, float], stats: Dict[str, Any], features: List[str], target: str) -> str:
        r2 = metrics.get("R2", 0.0)
        p_values = stats.get("p_values", {})
        
        insignificant_feats = [f for f in features if p_values.get(f, 0.5) > 0.05]
        
        recommendations = ["### Diagnostic Recommendations & Next Steps\n"]
        
        # 1. Dataset improvements & Feature Engineering
        recommendations.append("#### 1. Feature Engineering & Data Quality")
        if insignificant_feats:
            feats_str = ", ".join([f"`{f}`" for f in insignificant_feats])
            recommendations.append(f"- **Drop Insignificant Features:** Consider dropping {feats_str} because their p-values are above 0.05. Removing them simplifies the model and mitigates overfitting without losing predictive power.")
        else:
            recommendations.append("- **Feature Significance:** Good! All selected features are statistically significant, meaning they contribute relevant information to the target variable.")
            
        if len(features) < 2:
            recommendations.append("- **Expand Predictors:** You are using only one feature. Consider adding more features to explain more variance in the target variable.")
        else:
            recommendations.append("- **Interaction Terms:** Try creating interaction terms (e.g. multiplying two features together) if you suspect their joint effect is synergistic.")
            
        recommendations.append("- **Scaling check:** If you haven't scaled features and are testing distance-based algorithms, ensure MinMaxScaler or StandardScaler is applied.")

        # 2. Algorithm Suggestions
        recommendations.append("\n#### 2. Alternative Algorithms to Try")
        name_clean = model_name.replace(" ", "").lower()
        
        if "linear" in name_clean or "ridge" in name_clean or "lasso" in name_clean:
            recommendations.append("- **Random Forest Regression:** Since your current model is linear, testing a **Random Forest Regressor** is recommended. Random Forest can capture non-linear relationships and high-degree feature interactions automatically without manual transformation.")
            recommendations.append("- **Polynomial Regression:** If you suspect curvature but want to keep linear interpretability, try adding **Polynomial Features** (degree=2).")
        elif "randomforest" in name_clean or "decisiontree" in name_clean:
            recommendations.append("- **Ridge / Lasso Regression:** Your current model is a complex tree-based ensemble. Try a regularized linear model like **Ridge** or **Lasso** to see if a simpler, highly interpretable model can achieve competitive performance.")
            recommendations.append("- **Support Vector Regression (SVR):** With an RBF kernel, SVR is highly robust and performs well on complex non-linear problems with fewer samples.")
        else:
            recommendations.append("- **Random Forest Regression:** Ensembles are highly recommended as they reduce variance and model non-linear boundaries smoothly.")
            
        # 3. Optimization suggestions
        recommendations.append("\n#### 3. Model Fine-Tuning")
        if r2 < 0.70:
            recommendations.append("- **Increase Dataset Size:** An R2 of less than 0.70 indicates the model struggles to fit the variance. Collecting more training samples can help trees or high-degree polynomials stabilize.")
            recommendations.append("- **Regularization Tuning:** If using Ridge/Lasso/ElasticNet, perform cross-validation to search for the optimal regularization strength (`alpha`).")
        else:
            recommendations.append("- **Hyperparameter Tuning:** Fine-tune model parameters (e.g., `n_estimators` and `max_depth` for Random Forest) using GridSearchCV to squeeze out extra percentage points of accuracy.")

        return "\n".join(recommendations)
