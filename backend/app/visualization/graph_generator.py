import logging
import os
import matplotlib
# Use the non-interactive Agg backend to prevent GUI rendering issues on the server
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

# Set seaborn style for rich aesthetics
sns.set_theme(style="darkgrid")
plt.rcParams.update({
    'figure.facecolor': '#1e1e24',
    'axes.facecolor': '#2a2a35',
    'text.color': '#f8f9fa',
    'axes.labelcolor': '#f8f9fa',
    'xtick.color': '#adb5bd',
    'ytick.color': '#adb5bd',
    'axes.edgecolor': '#495057',
    'grid.color': '#495057',
    'font.family': 'sans-serif'
})

logger = logging.getLogger("regression_studio.visualization")

STATIC_GRAPHS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "static", "graphs"))

def ensure_graph_dir(model_id: str) -> str:
    path = os.path.join(STATIC_GRAPHS_DIR, model_id)
    os.makedirs(path, exist_ok=True)
    return path

def generate_dataset_graphs(df: pd.DataFrame, target: Optional[str], model_id: str) -> Dict[str, str]:
    """
    Generates dataset-specific visualization graphs:
    1. Distribution histograms for all numerical columns.
    2. Box plots of numerical columns to show outliers.
    3. Correlation heatmap.
    """
    graph_dir = ensure_graph_dir(model_id)
    paths = {}
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        return paths
        
    # 1. Histogram (Distributions)
    try:
        n_cols = len(numeric_cols)
        fig, axes = plt.subplots(
            nrows=(n_cols + 1) // 2, 
            ncols=2, 
            figsize=(12, 4 * ((n_cols + 1) // 2)),
            squeeze=False
        )
        axes = axes.flatten()
        
        for idx, col in enumerate(numeric_cols):
            sns.histplot(df[col], kde=True, ax=axes[idx], color="#6366f1", edgecolor="#4f46e5")
            axes[idx].set_title(f"Distribution of {col}", fontsize=12, color="#f8f9fa")
            axes[idx].set_xlabel("")
            
        # Hide unused subplots
        for idx in range(n_cols, len(axes)):
            fig.delaxes(axes[idx])
            
        plt.tight_layout()
        hist_path = os.path.join(graph_dir, "distribution.png")
        plt.savefig(hist_path, dpi=150, facecolor='#1e1e24')
        plt.close()
        paths["distribution"] = hist_path
    except Exception as e:
        logger.warning(f"Failed to generate distribution graph: {e}")
        
    # 2. Box Plot (Outliers)
    try:
        plt.figure(figsize=(10, 6))
        # Scale/normalize numeric data temporarily for comparison on a single box plot
        df_num = df[numeric_cols]
        df_norm = (df_num - df_num.mean()) / df_num.std()
        
        sns.boxplot(data=df_norm, orient="h", palette="crest")
        plt.title("Box Plot of Normalized Numerical Features (Outlier Check)", fontsize=14, color="#f8f9fa")
        plt.xlabel("Standard Deviations from Mean")
        plt.tight_layout()
        
        box_path = os.path.join(graph_dir, "boxplot.png")
        plt.savefig(box_path, dpi=150, facecolor='#1e1e24')
        plt.close()
        paths["boxplot"] = box_path
    except Exception as e:
        logger.warning(f"Failed to generate box plot: {e}")

    # 3. Correlation Heatmap
    try:
        if len(numeric_cols) > 1:
            plt.figure(figsize=(8, 6))
            corr = df[numeric_cols].corr()
            sns.heatmap(
                corr, 
                annot=True, 
                cmap="coolwarm", 
                vmin=-1, 
                vmax=1, 
                fmt=".2f",
                linewidths=0.5,
                annot_kws={"size": 10},
                cbar_kws={"shrink": 0.8}
            )
            plt.title("Feature Correlation Heatmap", fontsize=14, color="#f8f9fa")
            plt.tight_layout()
            
            heatmap_path = os.path.join(graph_dir, "heatmap.png")
            plt.savefig(heatmap_path, dpi=150, facecolor='#1e1e24')
            plt.close()
            paths["correlation_heatmap"] = heatmap_path
    except Exception as e:
        logger.warning(f"Failed to generate heatmap: {e}")
        
    return paths

def generate_regression_graphs(
    y_actual: np.ndarray, 
    y_predicted: np.ndarray, 
    features: List[str], 
    coefficients: Dict[str, float],
    model_id: str
) -> Dict[str, str]:
    """
    Generates regression-specific visualization graphs:
    1. Actual vs. Predicted scatter plot (with 45-degree reference line).
    2. Residual plot (residuals vs. predicted).
    3. Error Distribution histogram.
    4. Feature Importance bar chart (if coefficients/feature importances are available).
    """
    graph_dir = ensure_graph_dir(model_id)
    paths = {}
    
    y_act = np.array(y_actual).flatten()
    y_pred = np.array(y_predicted).flatten()
    residuals = y_act - y_pred
    
    # 1. Actual vs. Predicted
    try:
        plt.figure(figsize=(8, 6))
        plt.scatter(y_pred, y_act, color="#3b82f6", alpha=0.7, edgecolors="#1d4ed8")
        
        # Draw 45-degree ideal fit line
        lims = [
            np.min([plt.xlim()[0], plt.ylim()[0]]),
            np.max([plt.xlim()[1], plt.ylim()[1]])
        ]
        plt.plot(lims, lims, color="#ef4444", linestyle="--", linewidth=2, label="Ideal Fit (y = x)")
        
        plt.title("Actual vs. Predicted Values", fontsize=14, color="#f8f9fa")
        plt.xlabel("Predicted Values", fontsize=12)
        plt.ylabel("Actual Values", fontsize=12)
        plt.legend(loc="upper left")
        plt.tight_layout()
        
        act_vs_pred_path = os.path.join(graph_dir, "actual_vs_predicted.png")
        plt.savefig(act_vs_pred_path, dpi=150, facecolor='#1e1e24')
        plt.close()
        paths["actual_vs_predicted"] = act_vs_pred_path
    except Exception as e:
        logger.warning(f"Failed to generate actual vs predicted graph: {e}")
        
    # 2. Residual Plot
    try:
        plt.figure(figsize=(8, 6))
        plt.scatter(y_pred, residuals, color="#10b981", alpha=0.7, edgecolors="#047857")
        plt.axhline(y=0, color="#ef4444", linestyle="-", linewidth=2)
        
        plt.title("Residual Plot (Residuals vs. Predictions)", fontsize=14, color="#f8f9fa")
        plt.xlabel("Predicted Values", fontsize=12)
        plt.ylabel("Residuals (Actual - Predicted)", fontsize=12)
        plt.tight_layout()
        
        residual_path = os.path.join(graph_dir, "residuals.png")
        plt.savefig(residual_path, dpi=150, facecolor='#1e1e24')
        plt.close()
        paths["residuals"] = residual_path
    except Exception as e:
        logger.warning(f"Failed to generate residual plot: {e}")

    # 3. Error Distribution (Residual Histogram)
    try:
        plt.figure(figsize=(8, 6))
        sns.histplot(residuals, kde=True, color="#f59e0b", edgecolor="#d97706")
        plt.axvline(x=0, color="#ef4444", linestyle="--", linewidth=1.5)
        
        plt.title("Error Distribution (Residuals Histogram)", fontsize=14, color="#f8f9fa")
        plt.xlabel("Residual Value", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.tight_layout()
        
        error_dist_path = os.path.join(graph_dir, "error_distribution.png")
        plt.savefig(error_dist_path, dpi=150, facecolor='#1e1e24')
        plt.close()
        paths["error_distribution"] = error_dist_path
    except Exception as e:
        logger.warning(f"Failed to generate error distribution graph: {e}")

    # 4. Feature Importance / Coefficients Bar Chart
    try:
        # Filter coefficients to exclude intercept (const)
        feat_coefs = {k: v for k, v in coefficients.items() if k != "const" and k in features}
        
        if feat_coefs:
            plt.figure(figsize=(8, 6))
            
            # Sort features by absolute coefficient size
            sorted_coefs = sorted(feat_coefs.items(), key=lambda x: abs(x[1]), reverse=True)
            feat_names = [x[0] for x in sorted_coefs]
            feat_vals = [x[1] for x in sorted_coefs]
            
            # Determine color based on positive/negative impact
            colors = ["#3b82f6" if val >= 0 else "#ef4444" for val in feat_vals]
            
            sns.barplot(x=feat_vals, y=feat_names, palette=colors if len(colors) == len(feat_names) else None, hue=feat_names, legend=False)
            plt.axvline(x=0, color="#adb5bd", linestyle="-", linewidth=1)
            
            plt.title("Feature Impact (Coefficient / Importance Size)", fontsize=14, color="#f8f9fa")
            plt.xlabel("Effect Size (Coefficient Value)", fontsize=12)
            plt.ylabel("Features", fontsize=12)
            plt.tight_layout()
            
            importance_path = os.path.join(graph_dir, "feature_importance.png")
            plt.savefig(importance_path, dpi=150, facecolor='#1e1e24')
            plt.close()
            paths["feature_importance"] = importance_path
    except Exception as e:
        logger.warning(f"Failed to generate feature importance graph: {e}")

    return paths
