import logging
import os
from typing import Dict, Any, List
from app.services.huggingface_service import get_llm

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
        recommendations: str,
        model_type: str = "regression",
    ) -> str:
        """
        Aggregates and compiles the reports from all agents into a unified, styled markdown report.
        """
        is_classification = model_type == "classification"
        if is_classification:
            accuracy = metrics.get("Accuracy", 0.0)
            f1 = metrics.get("F1", 0.0)
            kpi_line = f"Accuracy = `{accuracy:.2%}` | F1-Score = `{f1:.4f}`"
            summary_metric_sentence = f"The model achieved a classification accuracy of `{accuracy:.2%}`."
        else:
            r2 = metrics.get("R2", 0.0)
            rmse = metrics.get("RMSE", 0.0)
            kpi_line = f"R² Score = `{r2:.4f}` | RMSE = `{rmse:,.2f}`"
            summary_metric_sentence = f"The model achieved a prediction accuracy (R²) of `{r2:.2f}`."

        report_title = "MODELFORGE AI STUDIO EXECUTIVE REPORT"

        header = f"""# {report_title}
**Dataset:** {dataset_name} | **Model Trained:** {model_name}
**Key Performance Indicators:** {kpi_line}

---

## Executive Summary
This document provides a comprehensive report of the machine learning analysis carried out on the dataset `{dataset_name}` using a `{model_name}` algorithm. {summary_metric_sentence} Detailed explanations of the dataset structure, model methodology, statistical calculations, and engineering recommendations are compiled below by the AI Agent Network.
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
                task_word = "classification" if is_classification else "regression"
                system_msg = SystemMessage(content=f"You are a chief AI officer. Write a professional, executive summary (2-3 paragraphs) that outlines the main findings of the {task_word} model, highlights key takeaways, and introduces the detailed sections. Return only the Markdown executive summary.")
                human_msg = HumanMessage(content=f"Write an executive summary based on the following metrics and reports:\nModel: {model_name}\nMetrics: {metrics}\nDataset Analysis: {dataset_analysis[:500]}...\nStatistical Analysis: {stats_explanation[:500]}...")
                response = await self.llm.ainvoke([system_msg, human_msg])

                # Replace the placeholder executive summary with the LLM-generated one
                exec_summary_llm = response.content
                full_content = f"""# {report_title}
**Dataset:** {dataset_name} | **Model Trained:** {model_name}
**Key Performance Indicators:** {kpi_line}

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
