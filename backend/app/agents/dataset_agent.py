import logging
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

from app.services.huggingface_service import get_llm

logger = logging.getLogger("regression_studio.agents")

# Try importing LangChain's provider-agnostic message schema. This is the
# only LangChain dependency left in the agents — the actual LLM call goes
# through HuggingFaceChatModel (see app/services/huggingface_service.py),
# which implements just enough of the ainvoke()/astream() shape that these
# plain SystemMessage/HumanMessage objects keep working unchanged.
try:
    from langchain_core.messages import SystemMessage, HumanMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

class DatasetAnalysisAgent:
    def __init__(self):
        self.llm = get_llm()

    async def analyze(self, df_summary: Dict[str, Any], raw_data: Optional[pd.DataFrame] = None) -> str:
        """
        Analyzes dataset shape, data types, missing value percentages, and correlates columns.
        Returns a detailed Markdown analysis.
        """
        # Prepare the dataset summary context
        name = df_summary.get("name", "dataset")
        rows = df_summary.get("rows", 0)
        cols_count = df_summary.get("columns", 0)
        col_names = df_summary.get("column_names", [])
        data_types = df_summary.get("data_types", {})
        
        # Calculate missing values if raw_data is provided
        missing_info = {}
        if raw_data is not None:
            missing_info = raw_data.isna().sum().to_dict()
        else:
            missing_info = {col: 0 for col in col_names}
            
        context = f"""
        Dataset Name: {name}
        Total Rows: {rows}
        Total Columns: {cols_count}
        Column Names: {', '.join(col_names)}
        Data Types: {data_types}
        Missing Values Count: {missing_info}
        """

        if self.llm:
            try:
                system_msg = SystemMessage(content="You are a senior data scientist. Analyze the dataset structure, missing values, column configurations, and suggest data preprocessing techniques. Keep it concise, professional, and formatted in Markdown.")
                human_msg = HumanMessage(content=f"Here is the dataset summary:\n{context}\nPlease analyze it.")
                response = await self.llm.ainvoke([system_msg, human_msg])
                return response.content
            except Exception as e:
                logger.warning(f"DatasetAnalysisAgent LLM run failed: {e}. Falling back to analytical summary.")

        # Analytical fallback
        return self._generate_analytical_fallback(name, rows, cols_count, col_names, data_types, missing_info)

    def _generate_analytical_fallback(self, name: str, rows: int, cols_count: int, col_names: List[str], data_types: Dict[str, str], missing_info: Dict[str, int]) -> str:
        # Categorize columns
        num_cols = []
        cat_cols = []
        for col in col_names:
            t = str(data_types.get(col, "")).lower()
            if "int" in t or "float" in t or "num" in t:
                num_cols.append(col)
            else:
                cat_cols.append(col)
                
        missing_summary = []
        has_missing = False
        for col, count in missing_info.items():
            if count > 0:
                has_missing = True
                pct = (count / rows) * 100
                missing_summary.append(f"- **{col}**: {count} missing values ({pct:.1f}%)")
                
        missing_text = "\n".join(missing_summary) if has_missing else "No missing values detected."
        
        analysis = f"""### Dataset Analysis: **{name}**

* **Data Scale:** The dataset contains **{rows} records** and **{cols_count} features**.
* **Variable Breakdowns:**
  * Numerical Columns ({len(num_cols)}): {', '.join([f'`{c}`' for c in num_cols])}
  * Categorical/Text Columns ({len(cat_cols)}): {', '.join([f'`{c}`' for c in cat_cols]) if cat_cols else "None detected."}

#### Data Quality & Cleanliness
* **Missing Data:** {missing_text}
* **Data Volume:** {f"The dataset volume of {rows} rows is sufficient for training linear models, though tree-based ensemble models perform better with larger sample sizes." if rows > 100 else f"The dataset is relatively small ({rows} rows). Models trained on this dataset might be prone to overfitting; simple regularized models (like Ridge or Lasso) are recommended."}

#### Suggested Preprocessing Pipeline
1. {f"**Missing Value Imputation:** Missing values in {', '.join([f'`{c}`' for c in missing_info if missing_info[c] > 0])} must be handled. For numerical columns, **Mean** or **Median** replacement is recommended. For categorical features, use **Mode** replacement or remove the corresponding rows." if has_missing else "**Missing Values:** No imputation needed as dataset is complete."}
2. **Duplicate Check:** Ensure duplicate rows are removed to avoid inflating R2 scores falsely during splitting.
3. **Outlier Filtering:** Use the **IQR method** to filter extreme outliers in columns like {', '.join([f'`{c}`' for c in num_cols[:2]])} to improve linear model stability.
4. **Feature Scaling:** Since features have different scales and units, applying **StandardScaler** is highly recommended before training regularized regressions (Lasso/Ridge) or SVR.
"""
        return analysis
