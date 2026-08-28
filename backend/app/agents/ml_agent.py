import logging
import os
from typing import Dict, Any, List, Optional
from app.services.huggingface_service import get_llm

logger = logging.getLogger("regression_studio.agents")

try:
    from langchain_core.messages import SystemMessage, HumanMessage
except ImportError:
    pass

class MLModelExplanationAgent:
    def __init__(self):
        self.llm = get_llm()

    async def explain(
        self, model_name: str, hyperparameters: Optional[Dict[str, Any]] = None,
        model_type: str = "regression",
    ) -> str:
        """
        Explains why the algorithm was selected, how it works, and its advantages/limitations.
        """
        params = hyperparameters or {}
        task_word = "classification" if model_type == "classification" else "clustering" if model_type == "clustering" else "regression"

        context = f"""
        Model Name: {model_name}
        Hyperparameters: {params}
        """

        if self.llm:
            try:
                system_msg = SystemMessage(content=f"You are an expert machine learning researcher. Explain the chosen {task_word} algorithm, its mathematical logic, why it fits the problem, and its pros/cons. Output in clean Markdown format.")
                human_msg = HumanMessage(content=f"Explain the following {task_word} model configuration:\n{context}")
                response = await self.llm.ainvoke([system_msg, human_msg])
                return response.content
            except Exception as e:
                logger.warning(f"MLModelExplanationAgent LLM run failed: {e}. Falling back to pre-written explanation.")

        # Analytical fallback
        return self._generate_analytical_fallback(model_name, params, model_type)

    def _generate_analytical_fallback(self, model_name: str, params: Dict[str, Any], model_type: str = "regression") -> str:
        # Strips hyphens too (not just spaces) — needed for clustering names
        # like "K-Means"/"K-Medoids" to match clustering_models.py's own
        # identical normalization; harmless for regression/classification
        # names, none of which contain a hyphen.
        name_clean = model_name.replace(" ", "").replace("-", "").lower()

        if model_type == "classification":
            return self._classification_fallback(name_clean)
        if model_type == "clustering":
            return self._clustering_fallback(name_clean)
        return self._regression_fallback(name_clean)

    def _regression_fallback(self, name_clean: str) -> str:
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
            },
            "gradientboosting": {
                "title": "Gradient Boosting Regression",
                "how_it_works": "Gradient Boosting builds an ensemble of shallow decision trees sequentially, where each new tree is trained to correct the residual errors of the ensemble built so far, weighted by a learning rate.",
                "why_selected": "Selected for its high predictive accuracy on tabular data — it typically outperforms a single tree or a Random Forest when carefully tuned.",
                "pros": [
                    "Often achieves higher accuracy than bagging-based ensembles like Random Forest.",
                    "Handles mixed numerical and categorical (one-hot encoded) features well.",
                    "Flexible — many loss functions and regularization options."
                ],
                "cons": [
                    "More sensitive to hyperparameters (learning_rate, n_estimators, max_depth) than Random Forest.",
                    "Sequential training makes it slower to fit than parallel ensembles.",
                    "Can overfit if the number of boosting rounds is too high without early stopping."
                ]
            },
            "xgboost": {
                "title": "XGBoost Regressor",
                "how_it_works": "XGBoost is an optimized, regularized implementation of gradient boosting that adds L1/L2 penalties on leaf weights, uses a second-order (Newton) approximation of the loss, and includes built-in handling for missing values.",
                "why_selected": "Selected as a state-of-the-art gradient boosting engine — a common top performer on structured/tabular regression problems.",
                "pros": [
                    "Regularization reduces overfitting compared to plain gradient boosting.",
                    "Highly optimized for speed, including parallel tree construction.",
                    "Strong out-of-the-box performance on tabular data."
                ],
                "cons": [
                    "More hyperparameters to tune than simpler models.",
                    "Less interpretable than linear models.",
                    "Can still overfit small datasets without careful regularization."
                ]
            },
            "lightgbm": {
                "title": "LightGBM Regressor",
                "how_it_works": "LightGBM is a gradient boosting framework that grows trees leaf-wise (rather than level-wise) and uses histogram-based binning of feature values, making it fast and memory-efficient on large datasets.",
                "why_selected": "Selected for training speed and scalability — a strong choice when the dataset is large or has high-cardinality categorical features.",
                "pros": [
                    "Very fast training, especially on large datasets.",
                    "Lower memory usage than many gradient boosting implementations.",
                    "Handles high-cardinality categorical features (via one-hot encoding here) efficiently."
                ],
                "cons": [
                    "Leaf-wise growth can overfit on small datasets if max_depth isn't constrained.",
                    "More sensitive to noisy data than level-wise boosting.",
                    "Less interpretable than linear models."
                ]
            },
            "catboost": {
                "title": "CatBoost Regressor",
                "how_it_works": "CatBoost is a gradient boosting framework built around ordered boosting (which avoids the target leakage that naive gradient boosting can suffer from) and a native categorical-feature encoding scheme, reducing the need for manual preprocessing tuning.",
                "why_selected": "Selected for strong out-of-the-box accuracy with minimal hyperparameter tuning, particularly on datasets with many categorical features.",
                "pros": [
                    "Strong default performance with little tuning required.",
                    "Ordered boosting reduces overfitting/target leakage compared to naive gradient boosting.",
                    "Robust handling of categorical features."
                ],
                "cons": [
                    "Slower to train than LightGBM on very large datasets.",
                    "Less interpretable than linear models.",
                    "Smaller community/ecosystem than XGBoost or LightGBM."
                ]
            },
            "knn": {
                "title": "K-Nearest Neighbors (KNN) Regression",
                "how_it_works": "KNN predicts a target value by averaging the target values of the k closest training points in feature space, measured by a distance metric (typically Euclidean).",
                "why_selected": "Selected as a simple, non-parametric baseline that makes no assumptions about the underlying relationship between features and target.",
                "pros": [
                    "Simple to understand, with no training/fitting step required.",
                    "Naturally captures local, non-linear patterns.",
                    "Few hyperparameters (mainly the number of neighbors, k)."
                ],
                "cons": [
                    "Prediction is slow on large datasets (must search all training points).",
                    "Very sensitive to feature scaling and irrelevant features.",
                    "Suffers from the curse of dimensionality with many features."
                ]
            },
            "huber": {
                "title": "Huber Regression",
                "how_it_works": "Huber Regression uses a loss function that behaves like squared error for small residuals and like absolute error for large residuals, making it a linear model that's robust to outliers.",
                "why_selected": "Selected when the dataset likely contains outliers that would otherwise distort a plain OLS fit.",
                "pros": [
                    "Robust to outliers compared to ordinary least squares.",
                    "Still produces interpretable linear coefficients.",
                    "Combines the smoothness of squared loss near zero with robustness of absolute loss at the tails."
                ],
                "cons": [
                    "Requires tuning the epsilon threshold that separates the two loss regimes.",
                    "Still assumes a linear relationship between features and target.",
                    "Less commonly supported/understood than plain linear regression."
                ]
            },
            "bayesian": {
                "title": "Bayesian Ridge Regression",
                "how_it_works": "Bayesian Ridge places a probabilistic prior over the regression coefficients and the noise, estimating a posterior distribution rather than a single point estimate — effectively an automatically-regularized linear model.",
                "why_selected": "Selected when a probabilistic, automatically-regularized linear model is preferred, without manually tuning a regularization strength.",
                "pros": [
                    "Regularization strength is learned from the data, not manually tuned.",
                    "Provides a natural notion of prediction uncertainty.",
                    "Robust to overfitting on small-to-medium datasets."
                ],
                "cons": [
                    "Still fundamentally a linear model — can't capture non-linear relationships.",
                    "Slower to fit than plain Ridge/Linear Regression.",
                    "Less commonly used, so less familiar to most practitioners."
                ]
            },
            "quantile": {
                "title": "Quantile Regression",
                "how_it_works": "Quantile Regression estimates a specific conditional quantile (e.g. the median, or the 90th percentile) of the target variable, rather than its conditional mean like OLS.",
                "why_selected": "Selected when the goal is to model a specific quantile of the target's distribution — for example, to understand the range or worst-case of predictions, not just the average.",
                "pros": [
                    "Models the full conditional distribution, not just the mean.",
                    "Robust to outliers, especially for quantiles away from the extremes.",
                    "Useful for risk analysis and prediction intervals."
                ],
                "cons": [
                    "Only estimates one quantile per fit — multiple quantiles need multiple models.",
                    "Less familiar/interpretable to non-technical stakeholders than a mean prediction.",
                    "Still assumes a linear relationship at the chosen quantile."
                ]
            }
        }
        # Mirrors app/ml/regression_models.py::get_regression_model()'s own
        # exact `name_clean in [...]` alias groups precisely — a loose
        # substring match (e.g. "xgboost" in "xgbregressor") silently misses
        # several real aliases (like "xgbregressor" itself, or
        # "kneighborsregressor"), which previously made every unmatched
        # model default to "Linear Regression".
        aliases = {
            "linear": ["linearregression", "linear", "multiplelinear", "multiplelinearregression"],
            "polynomial": ["polynomialregression", "polynomial"],
            "ridge": ["ridge", "ridgeregression"],
            "lasso": ["lasso", "lassoregression"],
            "elasticnet": ["elasticnet", "elasticnetregression"],
            "decisiontree": ["decisiontree", "decisiontreeregressor", "decisiontreeregression"],
            "randomforest": ["randomforest", "randomforestregressor", "randomforestregression"],
            "svr": ["svr", "supportvectorregression", "supportvectorregressor"],
            "gradientboosting": ["gradientboosting", "gradientboostingregressor", "gradientboostingregression"],
            "xgboost": ["xgboost", "xgbregressor", "xgboostregressor", "xgboostregression"],
            "lightgbm": ["lightgbm", "lgbmregressor", "lightgbmregressor", "lightgbmregression"],
            "catboost": ["catboost", "catboostregressor", "catboostregression"],
            "knn": ["knn", "kneighborsregressor", "knnregression", "kneighbors"],
            "huber": ["huber", "huberregressor", "huberregression"],
            "bayesian": ["bayesianridge", "bayesianregression", "bayesian"],
            "quantile": ["quantileregression", "quantileregressor", "quantile"],
        }
        return self._render_explanation(explanations, aliases, name_clean, default_key="linear")

    def _classification_fallback(self, name_clean: str) -> str:
        explanations = {
            "logistic": {
                "title": "Logistic Regression",
                "how_it_works": "Logistic Regression models the log-odds of a class as a linear combination of the features, then converts that to a probability via the sigmoid (or softmax, for multiple classes) function. Coefficients are fit by maximum likelihood estimation.",
                "why_selected": "Selected as an interpretable baseline classifier — coefficients directly show how each feature shifts the odds of a given class.",
                "pros": [
                    "Fast to train and highly interpretable coefficients (log-odds).",
                    "Outputs well-calibrated class probabilities, not just hard labels.",
                    "Works well when classes are linearly (or near-linearly) separable."
                ],
                "cons": [
                    "Assumes a linear decision boundary between classes.",
                    "Struggles with complex, non-linear class boundaries without feature engineering.",
                    "Sensitive to multicollinearity among features."
                ]
            },
            "decisiontree": {
                "title": "Decision Tree Classifier",
                "how_it_works": "A Decision Tree Classifier splits the dataset recursively based on feature thresholds that best separate the classes (e.g. by Gini impurity or entropy). Predictions are the majority class in the matching leaf node.",
                "why_selected": "Selected to capture non-linear decision boundaries and feature interactions without manual feature engineering, with a fully interpretable decision path.",
                "pros": [
                    "Easy to visualize and interpret — the decision path is a set of if/else rules.",
                    "Handles non-linear class boundaries and feature interactions naturally.",
                    "No feature scaling required."
                ],
                "cons": [
                    "Prone to overfitting, especially with deep trees on small datasets.",
                    "High variance — small changes in training data can produce a very different tree.",
                    "Biased toward features with more distinct values / one-hot-encoded high-cardinality columns."
                ]
            },
            "randomforest": {
                "title": "Random Forest Classifier",
                "how_it_works": "Random Forest trains many decision trees on bootstrapped samples of the data and random feature subsets, then predicts by majority vote across all trees — substantially reducing the variance of a single tree.",
                "why_selected": "Selected as a strong, robust all-rounder classifier that resists overfitting better than a single decision tree.",
                "pros": [
                    "High accuracy on a wide range of classification problems.",
                    "Resistant to overfitting due to averaging across many trees.",
                    "Provides feature importance estimates."
                ],
                "cons": [
                    "Less interpretable than a single decision tree or logistic regression.",
                    "Slower to train and predict than simpler models, especially with many trees.",
                    "Can still overfit noisy datasets if trees are too deep."
                ]
            },
            "gradientboosting": {
                "title": "Gradient Boosting Classifier",
                "how_it_works": "Gradient Boosting builds an ensemble of shallow trees sequentially, each one correcting the classification errors of the ensemble so far, guided by the gradient of the loss function.",
                "why_selected": "Selected for its typically strong accuracy on structured/tabular classification problems, often outperforming bagging-based ensembles.",
                "pros": [
                    "Often achieves higher accuracy than Random Forest when well-tuned.",
                    "Flexible — supports different loss functions for binary and multiclass problems.",
                    "Handles mixed numerical/categorical (one-hot encoded) features well."
                ],
                "cons": [
                    "More hyperparameter-sensitive than Random Forest.",
                    "Sequential training is slower to fit than parallel ensembles.",
                    "Can overfit without careful tuning of learning rate and number of estimators."
                ]
            },
            "xgboost": {
                "title": "XGBoost Classifier",
                "how_it_works": "XGBoost is a regularized, optimized gradient boosting implementation that adds L1/L2 penalties on leaf weights and uses a second-order loss approximation, applied here to a classification objective (log loss / softmax).",
                "why_selected": "Selected as a state-of-the-art gradient boosting classifier — a frequent top performer on structured/tabular classification tasks.",
                "pros": [
                    "Regularization reduces overfitting versus plain gradient boosting.",
                    "Highly optimized for speed via parallel tree construction.",
                    "Strong out-of-the-box accuracy on tabular data."
                ],
                "cons": [
                    "More hyperparameters to tune than simpler classifiers.",
                    "Less interpretable than logistic regression.",
                    "Can overfit small datasets without careful regularization."
                ]
            },
            "lightgbm": {
                "title": "LightGBM Classifier",
                "how_it_works": "LightGBM grows trees leaf-wise using histogram-based binning of feature values, making it fast and memory-efficient — applied here to a classification objective.",
                "why_selected": "Selected for training speed and scalability on larger datasets or high-cardinality categorical features.",
                "pros": [
                    "Very fast training, especially on large datasets.",
                    "Lower memory usage than many gradient boosting implementations.",
                    "Efficient with high-cardinality categorical (one-hot encoded) features."
                ],
                "cons": [
                    "Leaf-wise growth can overfit small datasets if max_depth isn't constrained.",
                    "More sensitive to noisy labels than level-wise boosting.",
                    "Less interpretable than logistic regression."
                ]
            },
            "svm": {
                "title": "Support Vector Machine (SVM)",
                "how_it_works": "SVM finds the hyperplane that maximizes the margin between classes. Using the 'kernel trick' (e.g. RBF), it can map inputs into a higher-dimensional space to find a separating boundary for non-linearly-separable classes.",
                "why_selected": "Selected for its strong performance on complex, non-linearly-separable class boundaries, particularly on small-to-medium datasets.",
                "pros": [
                    "Effective in high-dimensional feature spaces.",
                    "Kernels allow modeling highly complex, non-linear decision boundaries.",
                    "Robust when there's a clear margin of separation between classes."
                ],
                "cons": [
                    "Extremely sensitive to feature scaling.",
                    "Does not scale well to large datasets (training cost grows quickly with sample size).",
                    "Hyperparameters (C, kernel, gamma) require careful tuning."
                ]
            },
            "knn": {
                "title": "K-Nearest Neighbors (KNN) Classifier",
                "how_it_works": "KNN classifies a point by majority vote among the k closest training points in feature space, measured by a distance metric (typically Euclidean).",
                "why_selected": "Selected as a simple, non-parametric baseline classifier that makes no assumptions about the shape of the decision boundary.",
                "pros": [
                    "Simple to understand, with no explicit training/fitting step.",
                    "Naturally captures local, non-linear decision boundaries.",
                    "Few hyperparameters (mainly the number of neighbors, k)."
                ],
                "cons": [
                    "Prediction is slow on large datasets (must search all training points).",
                    "Very sensitive to feature scaling and irrelevant features.",
                    "Suffers from the curse of dimensionality with many features."
                ]
            },
            "naivebayes": {
                "title": "Naive Bayes (Gaussian)",
                "how_it_works": "Gaussian Naive Bayes applies Bayes' theorem assuming features are conditionally independent given the class, and that each feature follows a normal distribution within each class.",
                "why_selected": "Selected as a fast, simple probabilistic baseline classifier, particularly effective when the independence assumption is a reasonable approximation.",
                "pros": [
                    "Extremely fast to train and predict, even on large datasets.",
                    "Works well with relatively little training data.",
                    "Outputs class probabilities directly."
                ],
                "cons": [
                    "The feature-independence assumption is rarely exactly true, which can hurt accuracy.",
                    "Less accurate than ensemble methods on complex, correlated feature sets.",
                    "Assumes numeric features are normally distributed within each class."
                ]
            },
            "catboost": {
                "title": "CatBoost Classifier",
                "how_it_works": "CatBoost is a gradient boosting framework built around ordered boosting (which avoids the target leakage naive gradient boosting can suffer from) and a native categorical-feature encoding scheme.",
                "why_selected": "Selected for strong out-of-the-box accuracy with minimal hyperparameter tuning, particularly on datasets with many categorical features.",
                "pros": [
                    "Strong default performance with little tuning required.",
                    "Ordered boosting reduces overfitting/target leakage.",
                    "Robust handling of categorical features."
                ],
                "cons": [
                    "Slower to train than LightGBM on very large datasets.",
                    "Less interpretable than logistic regression.",
                    "Smaller community/ecosystem than XGBoost or LightGBM."
                ]
            },
            "extratrees": {
                "title": "Extra Trees Classifier",
                "how_it_works": "Extremely Randomized Trees builds an ensemble of decision trees like Random Forest, but additionally randomizes the split thresholds within each candidate feature, rather than searching for the locally optimal split.",
                "why_selected": "Selected as a fast, low-variance ensemble alternative to Random Forest — the extra randomization typically reduces overfitting further.",
                "pros": [
                    "Faster to train than Random Forest (no per-split threshold search).",
                    "Lower variance / less prone to overfitting than a single tree.",
                    "Provides feature importance estimates."
                ],
                "cons": [
                    "Slightly higher bias than Random Forest in some cases.",
                    "Less interpretable than a single decision tree or logistic regression.",
                    "Can still overfit very noisy datasets."
                ]
            },
            "adaboost": {
                "title": "AdaBoost Classifier",
                "how_it_works": "AdaBoost trains a sequence of weak learners (shallow trees by default), reweighting the training samples after each round so the next learner focuses more on the examples the ensemble so far got wrong.",
                "why_selected": "Selected as a simple, well-understood boosting method with few hyperparameters to tune.",
                "pros": [
                    "Few hyperparameters to tune.",
                    "Often improves substantially on a single weak learner.",
                    "Provides feature importance estimates."
                ],
                "cons": [
                    "Sensitive to noisy data and outliers (they get progressively upweighted).",
                    "Generally outperformed by gradient boosting (XGBoost/LightGBM/CatBoost) on tabular data.",
                    "Sequential training is slower than parallel ensembles."
                ]
            }
        }
        # Mirrors app/ml/classification_models.py::get_classification_model()'s
        # own exact `name_clean in [...]` alias groups precisely — see the
        # regression fallback's identical comment above for why exact
        # membership (not substring) matching matters.
        aliases = {
            "logistic": ["logisticregression", "logistic"],
            "decisiontree": ["decisiontree", "decisiontreeclassifier", "decisiontreeclassification"],
            "randomforest": ["randomforest", "randomforestclassifier", "randomforestclassification"],
            "extratrees": ["extratrees", "extratreesclassifier", "extratreesclassification"],
            "adaboost": ["adaboost", "adaboostclassifier", "adaboostclassification"],
            "gradientboosting": ["gradientboosting", "gradientboostingclassifier", "gradientboostingclassification"],
            "xgboost": ["xgboost", "xgbclassifier", "xgboostclassifier", "xgboostclassification"],
            "lightgbm": ["lightgbm", "lgbmclassifier", "lightgbmclassifier", "lightgbmclassification"],
            "catboost": ["catboost", "catboostclassifier", "catboostclassification"],
            "svm": ["svm", "supportvectormachine", "svc"],
            "knn": ["knn", "kneighborsclassifier", "knnclassification", "kneighbors"],
            "naivebayes": ["naivebayes", "gaussiannb", "nb"],
        }
        return self._render_explanation(explanations, aliases, name_clean, default_key="logistic")

    def _clustering_fallback(self, name_clean: str) -> str:
        explanations = {
            "kmeans": {
                "title": "K-Means",
                "how_it_works": "K-Means partitions data into k clusters by iteratively assigning each point to its nearest centroid, then recomputing each centroid as the mean of its assigned points, until assignments stop changing.",
                "why_selected": "Selected as a fast, well-understood baseline for finding roughly spherical, similarly-sized clusters.",
                "pros": [
                    "Fast and scales well to large datasets.",
                    "Simple to understand and interpret (each cluster has a clear centroid).",
                    "Works well when clusters are roughly spherical and similar in size."
                ],
                "cons": [
                    "Requires choosing k (the number of clusters) in advance.",
                    "Assumes spherical, similarly-sized clusters — struggles with elongated or unevenly-sized ones.",
                    "Sensitive to feature scaling and outliers."
                ]
            },
            "kmedoids": {
                "title": "K-Medoids",
                "how_it_works": "Like K-Means, but each cluster is centered on an actual data point (a medoid) rather than a computed mean, chosen to minimize total distance to other points in its cluster.",
                "why_selected": "Selected when robustness to outliers matters more than K-Means' speed — a medoid can't be dragged away from the data the way a mean can.",
                "pros": [
                    "More robust to outliers than K-Means (centroids are real data points).",
                    "Works with any distance metric, not just Euclidean.",
                    "Cluster centers are directly interpretable as real examples."
                ],
                "cons": [
                    "Slower than K-Means, especially on large datasets.",
                    "Still requires choosing k in advance.",
                    "Can converge to a different result depending on the initial medoids chosen."
                ]
            },
            "agglomerative": {
                "title": "Agglomerative Clustering",
                "how_it_works": "Builds a hierarchy of clusters bottom-up: each point starts as its own cluster, and the two closest clusters are repeatedly merged (by a chosen linkage rule) until only k remain.",
                "why_selected": "Selected to reveal a full hierarchy of groupings, not just one flat partition — useful when clusters may nest within each other.",
                "pros": [
                    "Produces a full dendrogram — the merge hierarchy at every level, not just one k.",
                    "No assumption of spherical clusters (depends on the linkage method chosen).",
                    "Deterministic — no random initialization to vary between runs."
                ],
                "cons": [
                    "Computationally expensive on large datasets (distance matrix grows quadratically).",
                    "Once two points are merged into a cluster, that merge can't be undone.",
                    "Choice of linkage method (ward/complete/average/single) meaningfully changes the result."
                ]
            },
            "dbscan": {
                "title": "DBSCAN",
                "how_it_works": "Groups points that are densely packed together (within `eps` distance of at least `min_samples` neighbors), and marks points that don't belong to any dense region as noise.",
                "why_selected": "Selected because the number of clusters doesn't need to be specified in advance, and it naturally identifies outliers as noise rather than forcing every point into a cluster.",
                "pros": [
                    "Doesn't require specifying the number of clusters upfront.",
                    "Naturally identifies outliers/noise points.",
                    "Can find arbitrarily-shaped (non-spherical) clusters."
                ],
                "cons": [
                    "Sensitive to the eps/min_samples parameters — poor choices produce one giant cluster or all noise.",
                    "Struggles when clusters have very different densities.",
                    "Doesn't scale as well as K-Means to very large datasets."
                ]
            },
            "optics": {
                "title": "OPTICS",
                "how_it_works": "Similar to DBSCAN, but orders points by reachability distance to build a density profile that can extract clusters of varying density from the same run, instead of using one fixed density threshold.",
                "why_selected": "Selected over DBSCAN when the dataset likely contains clusters of meaningfully different densities that a single eps value couldn't capture together.",
                "pros": [
                    "Handles clusters of varying density in the same dataset, unlike DBSCAN's single eps.",
                    "Still naturally identifies noise points.",
                    "Less sensitive to the exact parameter choice than DBSCAN."
                ],
                "cons": [
                    "More computationally expensive than DBSCAN.",
                    "Results (the reachability plot) are less immediately interpretable than a flat cluster assignment.",
                    "Still requires tuning min_samples."
                ]
            },
            "gaussianmixture": {
                "title": "Gaussian Mixture Model",
                "how_it_works": "Models the data as a mixture of several Gaussian (normal) distributions, fit via Expectation-Maximization — each point gets a probability of belonging to each component rather than one hard assignment.",
                "why_selected": "Selected when clusters may overlap or have elliptical (not just spherical) shapes, and a probabilistic membership is more informative than a hard assignment.",
                "pros": [
                    "Provides soft (probabilistic) cluster membership, not just a hard label.",
                    "Can model elliptical clusters, not just spherical ones like K-Means.",
                    "Has a principled way to compare different numbers of components (BIC/AIC)."
                ],
                "cons": [
                    "Still requires choosing the number of components in advance.",
                    "Sensitive to initialization — can converge to a poor local optimum.",
                    "Assumes each cluster is genuinely Gaussian-shaped, which may not hold."
                ]
            },
            "spectral": {
                "title": "Spectral Clustering",
                "how_it_works": "Builds a similarity graph between points, then uses the eigenvectors of that graph's Laplacian to project the data into a new space where a simpler algorithm (typically K-Means) can separate non-convex clusters.",
                "why_selected": "Selected when clusters have complex, non-convex shapes that K-Means' spherical assumption can't separate.",
                "pros": [
                    "Can find clusters of complex, non-convex shapes.",
                    "Doesn't assume any particular cluster shape the way K-Means/GMM do.",
                    "Works well on graph-structured or connectivity-based data."
                ],
                "cons": [
                    "Computationally expensive on large datasets (eigendecomposition of an n×n matrix).",
                    "Requires choosing the number of clusters in advance.",
                    "Sensitive to the choice of similarity/affinity measure."
                ]
            },
            "fuzzycmeans": {
                "title": "Fuzzy C-Means",
                "how_it_works": "Like K-Means, but instead of assigning each point to exactly one cluster, it computes a degree of membership for every point in every cluster, allowing points to partially belong to multiple clusters.",
                "why_selected": "Selected when cluster boundaries are expected to genuinely overlap, and a hard single-cluster assignment would lose information.",
                "pros": [
                    "Captures genuine ambiguity — a point can partially belong to more than one cluster.",
                    "Often more robust than K-Means when clusters overlap.",
                    "Membership degrees provide extra information beyond a hard label."
                ],
                "cons": [
                    "Still requires choosing the number of clusters in advance.",
                    "Slower to converge than K-Means.",
                    "Sensitive to the fuzziness parameter and initialization."
                ]
            }
        }
        aliases = {
            "kmeans": ["kmeans", "kmeanclustering", "kmeansclustering"],
            "kmedoids": ["kmedoids", "kmedoid"],
            "agglomerative": ["agglomerative", "agglomerativeclustering", "hierarchical", "hierarchicalclustering"],
            "dbscan": ["dbscan"],
            "optics": ["optics"],
            "gaussianmixture": ["gaussianmixture", "gmm", "gaussianmixturemodel"],
            "spectral": ["spectral", "spectralclustering"],
            "fuzzycmeans": ["fuzzycmeans", "fcm", "fuzzycmean"],
        }
        return self._render_explanation(explanations, aliases, name_clean, default_key="kmeans")

    @staticmethod
    def _render_explanation(
        explanations: Dict[str, Dict[str, Any]],
        aliases: Dict[str, List[str]],
        name_clean: str,
        default_key: str,
    ) -> str:
        matched_key = default_key
        for key, alts in aliases.items():
            if name_clean in alts:
                matched_key = key
                break
        matched_model = explanations[matched_key]

        pros_list = "\n".join([f"- {p}" for p in matched_model["pros"]])
        cons_list = "\n".join([f"- {c}" for c in matched_model["cons"]])

        return f"""### Model Methodology: **{matched_model['title']}**

#### How the Model Works
{matched_model['how_it_works']}

#### Rationale for Selection
{matched_model['why_selected']}

#### Advantages (Pros)
{pros_list}

#### Limitations (Cons)
{cons_list}
"""
