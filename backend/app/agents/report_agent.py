import logging
import os
from typing import Dict, Any, List
from app.agents.dataset_agent import get_llm

logger = logging.getLogger("regression_studio.agents")

try:
    from langchain_core.messages import SystemMessage, HumanMessage
except ImportError:
    pass

class ReportAgent:
    def __init__(self):
        self.llm = get_llm()

    async def generate_full_report(
        self,
        dataset_name: str,
        model_name: str,
        metrics: Dict[str, float],
        dataset_analysis: str,
        model_explanation: str,
        stats_explanation: str,
        recommendations: str
    ) -> str:
        """
        Aggregates and compiles the reports from all agents into a unified, styled markdown report.
        """
        r2 = metrics.get("R2", 0.0)
        rmse = metrics.get("RMSE", 0.0)
        
        header = f"""# AI REGRESSION STUDIO EXECUTIVE REPORT
**Dataset:** {dataset_name} | **Model Trained:** {model_name}
**Key Performance Indicators:** R² Score = `{r2:.4f}` | RMSE = `{rmse:,.2f}`

---

## Executive Summary
This document provides a comprehensive report of the machine learning analysis carried out on the dataset `{dataset_name}` using a `{model_name}` algorithm. The model achieved a prediction accuracy (R²) of `{r2:.2f}`. Detailed explanations of the dataset structure, model methodology, statistical calculations, and engineering recommendations are compiled below by the AI Agent Network.
"""

        full_content = f"""{header}

---

{dataset_analysis}

---

{model_explanation}

---

{stats_explanation}

---

{recommendations}
"""

        # If LLM is available, we can optionally ask it to rewrite/enhance the executive summary
        if self.llm:
            try:
                system_msg = SystemMessage(content="You are a chief AI officer. Write a professional, executive summary (2-3 paragraphs) that outlines the main findings of the regression model, highlights key takeaways, and introduces the detailed sections. Return only the Markdown executive summary.")
                human_msg = HumanMessage(content=f"Write an executive summary based on the following metrics and reports:\nModel: {model_name}\nMetrics: {metrics}\nDataset Analysis: {dataset_analysis[:500]}...\nStatistical Analysis: {stats_explanation[:500]}...")
                response = await self.llm.ainvoke([system_msg, human_msg])
                
                # Replace the placeholder executive summary with the LLM-generated one
                exec_summary_llm = response.content
                full_content = f"""# AI REGRESSION STUDIO EXECUTIVE REPORT
**Dataset:** {dataset_name} | **Model Trained:** {model_name}
**Key Performance Indicators:** R² Score = `{r2:.4f}` | RMSE = `{rmse:,.2f}`

---

## Executive Summary
{exec_summary_llm}

---

{dataset_analysis}

---

{model_explanation}

---

{stats_explanation}

---

{recommendations}
"""
            except Exception as e:
                logger.warning(f"ReportAgent LLM executive summary failed: {e}. Keeping default executive summary.")
                
        return full_content
