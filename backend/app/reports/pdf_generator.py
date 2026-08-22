import os
import re
from typing import Dict, Any, List
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

def clean_markdown_for_pdf(text: str) -> str:
    """
    Converts simple markdown syntax (bold, italic, list markers) into
    ReportLab-compatible XML-like tags (<b>, <i>, etc.) and strips unhandled tags.
    """
    # 1. Escape HTML special characters to prevent ReportLab XML parser crashes
    text = text.replace("&", "&amp;")
    text = text.replace("<", "&lt;")
    text = text.replace(">", "&gt;")
    
    # 2. Replace headers
    text = re.sub(r'^###\s+(.*?)$', r'<b><font color="#1e3a8a">\1</font></b>', text, flags=re.MULTILINE)
    text = re.sub(r'^####\s+(.*?)$', r'<b><font color="#475569">\1</font></b>', text, flags=re.MULTILINE)
    text = re.sub(r'^##\s+(.*?)$', r'<b><font size="14" color="#1e3a8a">\1</font></b>', text, flags=re.MULTILINE)
    text = re.sub(r'^#\s+(.*?)$', r'<b><font size="16" color="#1e3a8a">\1</font></b>', text, flags=re.MULTILINE)
    
    # 3. Replace bold (only asterisks to avoid variable name underscore clashes)
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
    
    # 4. Replace italic (only asterisks)
    text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)
    
    # 5. Replace inline code blocks
    text = re.sub(r'`(.*?)`', r'<font face="Courier">\1</font>', text)
    
    # 6. Convert lists to bullets
    text = re.sub(r'^\s*-\s+(.*?)$', r'• \1', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\*\s+(.*?)$', r'• \1', text, flags=re.MULTILINE)
    
    return text

def build_pdf_report(
    model_id: str,
    model_info: Dict[str, Any],
    graph_paths: Dict[str, str],
    ai_report_markdown: str,
    output_pdf_path: str
) -> str:
    """
    Generates a structured multi-page PDF report using ReportLab.
    """
    # Ensure parent directories exist
    os.makedirs(os.path.dirname(output_pdf_path), exist_ok=True)
    
    doc = SimpleDocTemplate(
        output_pdf_path,
        pagesize=letter,
        rightMargin=54, leftMargin=54, topMargin=54, bottomMargin=54
    )
    
    styles = getSampleStyleSheet()
    
    # Custom styles to maintain professional typography
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        textColor=colors.HexColor('#1e3a8a'),
        spaceAfter=15
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#475569'),
        spaceAfter=25
    )
    
    h1_style = ParagraphStyle(
        'SectionHeader',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=15,
        leading=19,
        textColor=colors.HexColor('#1e3a8a'),
        spaceBefore=15,
        spaceAfter=10,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'ReportBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor('#1e293b'),
        spaceAfter=8
    )
    
    table_cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor('#1e293b')
    )
    
    table_cell_header_style = ParagraphStyle(
        'TableCellHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=11,
        textColor=colors.white
    )

    story = []

    # --- PAGE 1: TITLE & METRICS OVERVIEW ---
    story.append(Paragraph("AI REGRESSION STUDIO", title_style))
    story.append(Paragraph(f"Analytical Report for Model ID: {model_id} | Generated on behalf of User Analysis History", subtitle_style))
    story.append(Spacer(1, 10))

    # 1. Dataset & Model Specifications Table
    story.append(Paragraph("1. Configuration and Dataset Details", h1_style))

    # The model doc's `split` field stores fractions (test_size/val_size), not
    # train/test percentages — compute the real percentages instead of reading
    # keys that don't exist (which previously always fell back to hardcoded 80/20).
    split_cfg = model_info.get("split", {}) or {}
    test_pct = round(split_cfg.get("test_size", 0.2) * 100)
    val_pct = round(split_cfg.get("val_size", 0.0) * 100)
    train_pct = 100 - test_pct - val_pct
    split_label = f"{train_pct}% Train / {test_pct}% Test"
    if val_pct > 0:
        split_label += f" / {val_pct}% Validation"

    # `preprocessing` isn't stored on the model doc itself — the caller (report_routes.py)
    # merges it in from the associated dataset doc's preprocessing_config before calling this.
    preprocessing_cfg = model_info.get("preprocessing", {}) or {}

    config_data = [
        [
            Paragraph("<b>Dataset Name</b>", table_cell_style),
            Paragraph(str(model_info.get("dataset_name", "N/A")), table_cell_style),
            Paragraph("<b>Selected Algorithm</b>", table_cell_style),
            Paragraph(str(model_info.get("model", "N/A")), table_cell_style)
        ],
        [
            Paragraph("<b>Target Variable</b>", table_cell_style),
            Paragraph(str(model_info.get("target", "N/A")), table_cell_style),
            Paragraph("<b>Features Count</b>", table_cell_style),
            Paragraph(str(len(model_info.get("features", []))), table_cell_style)
        ],
        [
            Paragraph("<b>Dataset Split Ratio</b>", table_cell_style),
            Paragraph(split_label, table_cell_style),
            Paragraph("<b>Missing Value Strategy</b>", table_cell_style),
            Paragraph(str(preprocessing_cfg.get("missing", "none")), table_cell_style)
        ],
        [
            Paragraph("<b>Outlier Filtering</b>", table_cell_style),
            Paragraph(str(preprocessing_cfg.get("outlier", "none")), table_cell_style),
            Paragraph("<b>Feature Scaling</b>", table_cell_style),
            Paragraph(str(preprocessing_cfg.get("scaling", "none")), table_cell_style)
        ]
    ]
    
    t_config = Table(config_data, colWidths=[120, 130, 120, 130])
    t_config.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f8fafc')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#e2e8f0')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(t_config)
    story.append(Spacer(1, 15))
    
    # 2. Evaluation Metrics Table
    story.append(Paragraph("2. Model Evaluation Performance", h1_style))
    metrics = model_info.get("metrics", {})
    metrics_data = [
        [Paragraph("Metric", table_cell_header_style), Paragraph("Value", table_cell_header_style), Paragraph("Description", table_cell_header_style)]
    ]
    
    descriptions = {
        "R2": "R-squared coefficient of determination. Represents percentage of explained variance.",
        "Adjusted R2": "Adjusted R-squared. Adjusts for feature counts to prevent overestimation.",
        "MSE": "Mean Squared Error. Measures average squared difference between predictions and targets.",
        "RMSE": "Root Mean Squared Error. Shows average prediction error in target units.",
        "MAE": "Mean Absolute Error. Average absolute size of prediction residuals."
    }
    
    for metric_name, metric_val in metrics.items():
        desc = descriptions.get(metric_name, "Model evaluation parameter.")
        metrics_data.append([
            Paragraph(f"<b>{metric_name}</b>", table_cell_style),
            Paragraph(f"{metric_val:,.4f}" if isinstance(metric_val, (int, float)) else str(metric_val), table_cell_style),
            Paragraph(desc, table_cell_style)
        ])
        
    t_metrics = Table(metrics_data, colWidths=[100, 100, 300])
    t_metrics.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1e3a8a')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#1e3a8a')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(t_metrics)
    story.append(Spacer(1, 15))
    
    # 3. Statsmodels Coefficient Table
    story.append(Paragraph("3. Statistical Parameter Analysis", h1_style))
    stats = model_info.get("statistical_analysis", {})
    coefs = stats.get("coefficients", {})
    p_values = stats.get("p_values", {})
    std_errors = stats.get("standard_errors", {})
    t_stats = stats.get("t_statistics", {})
    
    stats_rows = [
        [
            Paragraph("Variable", table_cell_header_style), 
            Paragraph("Coefficient", table_cell_header_style), 
            Paragraph("Std. Error", table_cell_header_style), 
            Paragraph("t-Statistic", table_cell_header_style), 
            Paragraph("p-Value", table_cell_header_style),
            Paragraph("Significant (p &lt; 0.05)", table_cell_header_style)
        ]
    ]
    
    for var_name, coef in coefs.items():
        p_val = p_values.get(var_name, 1.0)
        std_err = std_errors.get(var_name, 0.0)
        t_stat = t_stats.get(var_name, 0.0)
        sig = "YES" if p_val < 0.05 else "NO"
        
        stats_rows.append([
            Paragraph(f"<b>{var_name}</b>", table_cell_style),
            Paragraph(f"{coef:,.4f}", table_cell_style),
            Paragraph(f"{std_err:,.4f}", table_cell_style),
            Paragraph(f"{t_stat:,.4f}", table_cell_style),
            Paragraph(f"{p_val:.4f}", table_cell_style),
            Paragraph(f"<font color='{'#10b981' if sig == 'YES' else '#ef4444'}'><b>{sig}</b></font>", table_cell_style)
        ])
        
    t_stats_table = Table(stats_rows, colWidths=[120, 75, 75, 75, 75, 80])
    t_stats_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#475569')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#475569')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(t_stats_table)
    
    story.append(PageBreak()) # Move to next page for visualizations
    
    # --- PAGE 2: VISUALIZATION PLOTS ---
    story.append(Paragraph("4. Visualization Plots", h1_style))
    story.append(Spacer(1, 5))
    
    # Embed graphs side-by-side or stacked
    # Stacked pairs inside tables to prevent overflowing margins
    embeds = []
    
    if "actual_vs_predicted" in graph_paths and os.path.exists(graph_paths["actual_vs_predicted"]):
        embeds.append(Image(graph_paths["actual_vs_predicted"], width=230, height=172))
    if "residuals" in graph_paths and os.path.exists(graph_paths["residuals"]):
        embeds.append(Image(graph_paths["residuals"], width=230, height=172))
    if "error_distribution" in graph_paths and os.path.exists(graph_paths["error_distribution"]):
        embeds.append(Image(graph_paths["error_distribution"], width=230, height=172))
    if "feature_importance" in graph_paths and os.path.exists(graph_paths["feature_importance"]):
        embeds.append(Image(graph_paths["feature_importance"], width=230, height=172))
        
    if len(embeds) >= 2:
        # Build 2x2 grid table
        grid_data = [
            [embeds[0], embeds[1] if len(embeds) > 1 else ""],
            [embeds[2] if len(embeds) > 2 else "", embeds[3] if len(embeds) > 3 else ""]
        ]
        t_grid = Table(grid_data, colWidths=[250, 250])
        t_grid.setStyle(TableStyle([
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('BOTTOMPADDING', (0,0), (-1,-1), 10),
        ]))
        story.append(t_grid)
    elif len(embeds) == 1:
        story.append(embeds[0])
        
    story.append(PageBreak()) # Move to AI explanation
    
    # --- PAGE 3+: AI EXPLANATIONS ---
    story.append(Paragraph("5. AI Agent Detailed Explanations", h1_style))
    story.append(Spacer(1, 10))
    
    # Split the AI markdown by headings and append paragraphs
    raw_paragraphs = ai_report_markdown.split('\n')
    current_text_block = []
    
    for line in raw_paragraphs:
        line_clean = line.strip()
        if not line_clean:
            # Empty line, output current accumulated block as paragraph
            if current_text_block:
                full_para = " ".join(current_text_block)
                cleaned_para = clean_markdown_for_pdf(full_para)
                story.append(Paragraph(cleaned_para, body_style))
                current_text_block = []
        elif line_clean.startswith('#') or line_clean.startswith('---'):
            # Clear text block first
            if current_text_block:
                full_para = " ".join(current_text_block)
                cleaned_para = clean_markdown_for_pdf(full_para)
                story.append(Paragraph(cleaned_para, body_style))
                current_text_block = []
                
            if line_clean.startswith('---'):
                story.append(Spacer(1, 8))
                # Horizontal line representation in PDF
                t_hr = Table([['']], colWidths=[500])
                t_hr.setStyle(TableStyle([
                    ('LINEBELOW', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
                    ('BOTTOMPADDING', (0,0), (-1,-1), 5),
                ]))
                story.append(t_hr)
                story.append(Spacer(1, 8))
            else:
                cleaned_header = clean_markdown_for_pdf(line_clean)
                story.append(Paragraph(cleaned_header, h1_style))
        else:
            current_text_block.append(line_clean)
            
    # Dump remaining text block
    if current_text_block:
        full_para = " ".join(current_text_block)
        cleaned_para = clean_markdown_for_pdf(full_para)
        story.append(Paragraph(cleaned_para, body_style))
        
    # Build document
    doc.build(story)
    
    return output_pdf_path
