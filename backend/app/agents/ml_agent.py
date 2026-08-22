import logging
import os
from typing import Dict, Any, Optional
from app.agents.dataset_agent import get_llm

logger = logging.getLogger("regression_studio.agents")

try:
    from langchain_core.messages import SystemMessage, HumanMessage
except ImportError:
    pass

class MLModelExplanationAgent:
    def __init__(self):
        self.llm = get_llm()

    async def explain(self, model_name: str, hyperparameters: Optional[Dict[str, Any]] = None) -> str:
        """
        Explains why the algorithm was selected, how it works, and its advantages/limitations.
        """
        params = hyperparameters or {}
        
        context = f"""
        Model Name: {model_name}
        Hyperparameters: {params}
        """

        if self.llm:
            try:
                system_msg = SystemMessage(content="You are an expert machine learning researcher. Explain the chosen regression algorithm, its mathematical logic, why it fits the problem, and its pros/cons. Output in clean Markdown format.")
                human_msg = HumanMessage(content=f"Explain the following regression model configuration:\n{context}")
                response = await self.llm.ainvoke([system_msg, human_msg])
                return response.content
            except Exception as e:
                logger.warning(f"MLModelExplanationAgent LLM run failed: {e}. Falling back to pre-written explanation.")

        # Analytical fallback
        return self._generate_analytical_fallback(model_name, params)

    def _generate_analytical_fallback(self, model_name: str, params: Dict[str, Any]) -> str:
        name_clean = model_name.replace(" ", "").lower()
        
        explanations = {
            "linear": {
                "title": "Linear Regression (Simple / Multiple)",
                "how_it_works": "Linear Regression models the relationship between dependent and independent variables by fitting a linear equation to observed data. It uses the Ordinary Least Squares (OLS) method, which minimizes the sum of squared residuals (vertical distances between data points and the regression line).",
                "why_selected": "Linear Regression is selected as the baseline model. It assumes a linear relationship between features and the target variable, and is highly interpretable.",
                "pros": [
                    "Extremely fast to train and run.",
                    "Highly interpretable coefficients showing exact feature effects.",
                    "No risk of overfitting on small datasets if assumptions hold."
                ],
                "cons": [
                    "Assumes linear relationships; fails to capture nonlinear dependencies.",
                    "Highly sensitive to outliers.",
                    "Assumes independence between features (suffers from multicollinearity)."
                ]
            },
            "polynomial": {
                "title": "Polynomial Regression",
                "how_it_works": "Polynomial Regression is a form of linear regression where the relationship between the independent variable x and the dependent variable y is modeled as an n-th degree polynomial. It creates polynomial combinations of the features (e.g. x², x³, x₁x₂), and fits a standard linear regression to these expanded features.",
                "why_selected": "Polynomial regression is selected to model non-linear boundaries. It allows linear models to capture curvilinear interactions between features.",
                "pros": [
                    "Can model complex, curved, non-linear relationships.",
                    "Built on top of linear engines, making it computationally accessible.",
                    "Fits a wide range of curvature distributions."
                ],
                "cons": [
                    "Prone to severe overfitting near the edges of the dataset (Runge's phenomenon).",
                    "Extremely sensitive to outliers.",
                    "The number of features grows exponentially with the degree, causing dimensionality issues."
                ]
            },
            "ridge": {
                "title": "Ridge Regression (L2 Regularization)",
                "how_it_works": "Ridge Regression extends linear regression by adding a penalty term proportional to the square of the magnitude of coefficients (L2 penalty) to the OLS loss function. This regularizer shrinks the coefficients toward zero, reducing model complexity and multicollinearity.",
                "why_selected": "Ridge is selected when features are highly correlated (multicollinearity) or when the model is prone to overfitting. It stabilizes coefficients by shrinking their variance.",
                "pros": [
                    "Effectively handles multicollinearity by distributing weight across correlated features.",
                    "Prevents overfitting by penalizing large coefficient values.",
                    "Guarantees a unique stable solution even if features exceed observations."
                ],
                "cons": [
                    "Does not perform feature selection; retains all variables (coefficients shrink but never hit exactly zero).",
                    "Requires scaling of features before training.",
                    "The regularization parameter (alpha) must be carefully tuned."
                ]
            },
            "lasso": {
                "title": "Lasso Regression (L1 Regularization)",
                "how_it_works": "Lasso (Least Absolute Shrinkage and Selection Operator) adds a penalty term proportional to the absolute value of the coefficients (L1 penalty) to the loss function. This penalty forces some coefficient estimates to be exactly zero, effectively performing automatic feature selection.",
                "why_selected": "Lasso is selected when we suspect only a subset of features is truly relevant. It generates sparse models that are easy to interpret.",
                "pros": [
                    "Performs automatic feature selection by zeroing out irrelevant features.",
                    "Creates highly interpretable, simplified models.",
                    "Effective in high-dimensional datasets with many features."
                ],
                "cons": [
                    "If features are highly correlated, Lasso arbitrarily selects one and discards the others.",
                    "May underperform relative to Ridge if many features have small, equal effects.",
                    "Requires scaling of features."
                ]
            },
            "elasticnet": {
                "title": "Elastic Net Regression",
                "how_it_works": "Elastic Net is a regularized regression method that linearly combines both L1 (Lasso) and L2 (Ridge) penalties. It allows tuning the balance between L1 and L2 via the `l1_ratio` parameter, getting the grouping effect of Ridge and the feature selection of Lasso.",
                "why_selected": "Elastic Net is selected when there are multiple correlated features and we want to perform feature selection while maintaining the stability of Ridge.",
                "pros": [
                    "Overcomes Lasso's limitation of selecting arbitrary features by grouping correlated variables.",
                    "Enjoys the double benefits of coefficient shrinkage and feature selection.",
                    "Very robust on datasets with highly correlated variables."
                ],
                "cons": [
                    "More complex to tune because it has two hyperparameters (alpha and l1_ratio).",
                    "Requires feature scaling.",
                    "Slightly higher computational cost than simple Lasso/Ridge."
                ]
            },
            "decisiontree": {
                "title": "Decision Tree Regression",
                "how_it_works": "Decision Trees split the dataset recursively into smaller subsets based on feature thresholds that minimize variance (or Mean Squared Error) within each leaf node. Predictions are made by taking the average of target values in the leaf node matching the input.",
                "why_selected": "Decision Trees are chosen to capture complex non-linear relationships and interactions without requiring manual polynomial feature engineering.",
                "pros": [
                    "Non-parametric; makes no assumptions about data distributions.",
                    "Implicitly captures feature interactions (e.g. 'if Area > 2000 AND Bedrooms > 3').",
                    "Does not require feature scaling and is robust to outliers."
                ],
                "cons": [
                    "Highly prone to overfitting, building trees that are too deep and specific.",
                    "Extremely sensitive to small changes in training data (high variance).",
                    "Cannot extrapolate beyond the range of training data (outputs are step functions)."
                ]
            },
            "randomforest": {
                "title": "Random Forest Regression",
                "how_it_works": "Random Forest is an ensemble method that trains multiple decision trees in parallel using bagging (bootstrap aggregating) and random feature selection. The final prediction is the average of predictions from all individual trees, which drastically reduces model variance.",
                "why_selected": "Random Forest is a highly powerful non-linear regressor. It is selected for its high accuracy, stability, and resistance to overfitting.",
                "pros": [
                    "Excellent accuracy on complex, non-linear datasets.",
                    "Resistant to overfitting due to tree averaging.",
                    "Provides feature importance metrics based on variance reduction."
                ],
                "cons": [
                    "Acts as a black box; harder to interpret than simple linear regression.",
                    "Slow to train and predict on very large scale environments.",
                    "Like decision trees, it cannot extrapolate outside the training bounds."
                ]
            },
            "svr": {
                "title": "Support Vector Regression (SVR)",
                "how_it_works": "SVR fits a tube of width epsilon around the data points. It tries to fit as many points as possible inside the tube while minimizing coefficient sizes. Using the 'kernel trick' (e.g., RBF kernel), it maps inputs to high-dimensional spaces to find linear solutions for non-linear problems.",
                "why_selected": "SVR is chosen for its robustness to outliers and ability to model complex non-linear relationships in low-to-medium sized datasets.",
                "pros": [
                    "Highly effective in high-dimensional spaces.",
                    "Robust to outliers because it only cares about support vector points near the margins.",
                    "Kernels allow modeling of highly complex, non-linear boundaries."
                ],
                "cons": [
                    "Extremely sensitive to feature scaling; outputs will be wrong without it.",
                    "Does not scale well to large datasets (O(n³) complexity).",
                    "Hard to interpret and tune (requires choosing C, epsilon, and kernel)."
                ]
            }
        }
        
        # Match model key
        matched_model = None
        for key in explanations:
            if key in name_clean:
                matched_model = explanations[key]
                break
                
        if not matched_model:
            # Default to linear if no match
            matched_model = explanations["linear"]
            
        pros_list = "\n".join([f"- {p}" for p in matched_model["pros"]])
        cons_list = "\n".join([f"- {c}" for c in matched_model["cons"]])
        
        markdown_output = f"""### Model Methodology: **{matched_model['title']}**

#### How the Model Works
{matched_model['how_it_works']}

#### Rationale for Selection
{matched_model['why_selected']}

#### Advantages (Pros)
{pros_list}

#### Limitations (Cons)
{cons_list}
"""
        return markdown_output
