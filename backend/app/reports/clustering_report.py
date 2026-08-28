import os
import re
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

from app.reports.pdf_generator import _report_styles, clean_markdown_for_pdf, _grid_layout

def validate_clustering_report(model_info: Dict[str, Any]) -> None:
    """
    Performs strict consistency checks before PDF generation.
    Fails with ValueError if any mismatch is found.
    """
    # 1. Dataset rows in overview vs. model results
    dataset_profile = model_info.get("dataset_profile")
    total_rows = model_info.get("total_rows")
    if dataset_profile and total_rows is not None:
        if dataset_profile.get("total_rows") != total_rows:
            raise ValueError(
                f"Validation Error: Dataset row count mismatch. "
                f"Overview has {dataset_profile.get('total_rows')}, but model was trained on {total_rows}."
            )

    # 2. Selected feature count vs actual features list
    features = model_info.get("features", [])
    if not features:
        raise ValueError("Validation Error: No features selected for clustering.")

    # 3. Cluster count in metrics vs unique predicted cluster labels
    metrics = model_info.get("metrics", {})
    cluster_sizes = model_info.get("cluster_sizes", {})
    metric_cluster_count = metrics.get("ClusterCount")
    
    # Exclude noise points (-1) from the count of actual clusters
    real_clusters = [k for k in cluster_sizes.keys() if k != "-1"]
    actual_cluster_count = len(real_clusters)
    
    if metric_cluster_count is not None and metric_cluster_count != actual_cluster_count:
         raise ValueError(
             f"Validation Error: Cluster count mismatch. "
             f"Metrics specify {metric_cluster_count} clusters, but cluster sizes have {actual_cluster_count} real clusters."
         )

    # 4. Cluster sizes sum == total dataset rows
    sum_sizes = sum(cluster_sizes.values())
    if total_rows is not None and sum_sizes != total_rows:
        raise ValueError(
            f"Validation Error: Sum of cluster sizes ({sum_sizes}) does not match total dataset rows ({total_rows})."
        )


def _clean_feature_name(name: str) -> str:
    """Clean feature names to render nicely in PDF without malformed characters."""
    cleaned = name.strip()
    # Remove characters like ■, ■1, etc.
    cleaned = re.sub(r'[\u25a0\u25ae\u25ac\u25ad\u25af]+', '', cleaned)
    return cleaned


def build_clustering_pdf_report(
    model_id: str,
    model_info: Dict[str, Any],
    graph_paths: Dict[str, str],
    ai_report_markdown: str,
    output_pdf_path: str
) -> str:
    """
    Generates a structured task-specific multi-page PDF report for Unsupervised Clustering.
    """
    # 1. Run consistency validation
    validate_clustering_report(model_info)

    # Ensure parent directories exist
    os.makedirs(os.path.dirname(output_pdf_path), exist_ok=True)
    
    doc = SimpleDocTemplate(
        output_pdf_path,
        pagesize=letter,
        rightMargin=54, leftMargin=54, topMargin=54, bottomMargin=54
    )
    
    s = _report_styles()
    title_style = s["title"]
    subtitle_style = s["subtitle"]
    h1_style = s["h1"]
    body_style = s["body"]
    table_cell_style = s["cell"]
    table_cell_header_style = s["cell_header"]

    story = []
    section_num = 1

    def next_section(title: str) -> str:
        nonlocal section_num
        heading = f"{section_num}. {title}"
        section_num += 1
        return heading

    # --- COVER/TITLE PAGE ---
    story.append(Paragraph("MODELFORGE AI STUDIO", title_style))
    story.append(Paragraph(f"Unsupervised Clustering Analysis Report | Model ID: {model_id}", subtitle_style))
    story.append(Spacer(1, 10))

    dataset_profile = model_info.get("dataset_profile")
    total_features = dataset_profile.get("total_columns", 0) if dataset_profile else len(model_info.get("features", []))

    # --- 1. DATASET OVERVIEW ---
    story.append(Paragraph(next_section("Dataset Overview"), h1_style))
    if dataset_profile:
        overview_rows = [
            [
                Paragraph("<b>Dataset Name</b>", table_cell_style),
                Paragraph(_clean_feature_name(str(model_info.get("dataset_name", "N/A"))), table_cell_style),
                Paragraph("<b>Total Samples (Rows)</b>", table_cell_style),
                Paragraph(f"{dataset_profile.get('total_rows', 0):,}", table_cell_style),
            ],
            [
                Paragraph("<b>Total Features Available</b>", table_cell_style),
                Paragraph(str(dataset_profile.get("total_columns", 0)), table_cell_style),
                Paragraph("<b>Target Variable</b>", table_cell_style),
                Paragraph("<font color='#ef4444'><b>None / Not Applicable (Unsupervised)</b></font>", table_cell_style),
            ],
            [
                Paragraph("<b>Numerical Columns</b>", table_cell_style),
                Paragraph(str(dataset_profile.get("numerical_count", 0)), table_cell_style),
                Paragraph("<b>Categorical Columns</b>", table_cell_style),
                Paragraph(str(dataset_profile.get("categorical_count", 0)), table_cell_style),
            ],
            [
                Paragraph("<b>Missing Values</b>", table_cell_style),
                Paragraph(f"{dataset_profile.get('missing_values_total', 0):,}", table_cell_style),
                Paragraph("<b>Duplicate Rows</b>", table_cell_style),
                Paragraph(f"{dataset_profile.get('duplicate_rows', 0):,}", table_cell_style),
            ],
        ]
        t_overview = Table(overview_rows, colWidths=[120, 130, 120, 130])
        t_overview.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8fafc')),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#e2e8f0')),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        story.append(t_overview)
    else:
        story.append(Paragraph("Dataset profile information not available.", body_style))
    story.append(Spacer(1, 15))

    # --- 2. FEATURE SELECTION ---
    story.append(Paragraph(next_section("Feature Selection & Exclusion"), h1_style))
    selected_features = model_info.get("features", [])
    
    # Calculate excluded columns
    excluded_features = []
    if dataset_profile:
        all_cols_dict = {f["name"]: f for f in dataset_profile.get("features", [])}
        for name, info in all_cols_dict.items():
            if name not in selected_features:
                reason = "Not selected by the user."
                if info["kind"] == "datetime":
                    reason = "Excluded because the column is date/time values."
                elif info.get("is_identifier"):
                    reason = "Excluded because the column appears to be an identifier rather than a meaningful analytical feature."
                excluded_features.append((name, reason))

    selected_count_text = f"<b>{len(selected_features)} of {total_features}</b> available features were selected for clustering."
    story.append(Paragraph(selected_count_text, body_style))
    story.append(Spacer(1, 6))

    feature_table_data = [
        [Paragraph("Feature Name", table_cell_header_style), Paragraph("Status", table_cell_header_style), Paragraph("Notes / Reason", table_cell_header_style)]
    ]
    for sf in selected_features:
        feature_table_data.append([
            Paragraph(f"<b>{_clean_feature_name(sf)}</b>", table_cell_style),
            Paragraph("<font color='#10b981'><b>Selected</b></font>", table_cell_style),
            Paragraph("Used as analytical feature for grouping calculation.", table_cell_style)
        ])
    for ef_name, reason in excluded_features[:15]: # Show up to 15 exclusions for space
        feature_table_data.append([
            Paragraph(_clean_feature_name(ef_name), table_cell_style),
            Paragraph("<font color='#ef4444'><b>Excluded</b></font>", table_cell_style),
            Paragraph(reason, table_cell_style)
        ])

    t_features = Table(feature_table_data, colWidths=[150, 90, 260])
    t_features.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#475569')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#e2e8f0')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(t_features)
    story.append(Spacer(1, 15))

    # --- 3. PREPROCESSING ---
    story.append(Paragraph(next_section("Preprocessing Pipeline"), h1_style))
    preprocessing_cfg = model_info.get("preprocessing", {}) or {}
    
    numerical_features = model_info.get("numerical_features", [])
    categorical_features = model_info.get("categorical_features", [])

    pre_rows = [
        [Paragraph("<b>Numerical Scaling</b>", table_cell_style), Paragraph(f"StandardScaler (Scaling is distance-sensitive)" if preprocessing_cfg.get("scaling", "standard") == "standard" else str(preprocessing_cfg.get("scaling", "StandardScaler")), table_cell_style)],
        [Paragraph("<b>Categorical Encoding</b>", table_cell_style), Paragraph("OneHotEncoder(handle_unknown='ignore', drop='first')" if categorical_features else "None (No categorical columns selected)", table_cell_style)],
        [Paragraph("<b>Missing Value Strategy</b>", table_cell_style), Paragraph(str(preprocessing_cfg.get("missing", "median imputation")), table_cell_style)],
        [Paragraph("<b>Outlier Removal Strategy</b>", table_cell_style), Paragraph(str(preprocessing_cfg.get("outlier", "None")), table_cell_style)],
    ]
    t_pre = Table(pre_rows, colWidths=[150, 350])
    t_pre.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f8fafc')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#e2e8f0')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(t_pre)

    if categorical_features:
        story.append(Spacer(1, 6))
        warn_msg = (
            "<b>Warning on Categorical Columns:</b> High-dimensional one-hot encoded variables can create a poor distance representation "
            "for centroid- or density-based clustering models. Use results cautiously."
        )
        story.append(Paragraph(warn_msg, body_style))
    story.append(Spacer(1, 15))

    # --- 4. CLUSTERING CONFIGURATION ---
    story.append(Paragraph(next_section("Clustering Configuration"), h1_style))
    hparams = model_info.get("hyperparameters", {}) or {}
    hparam_str = ", ".join(f"{k}: {v}" for k, v in hparams.items())
    
    config_rows = [
        [Paragraph("<b>Clustering Algorithm</b>", table_cell_style), Paragraph(str(model_info.get("model", "N/A")), table_cell_style)],
        [Paragraph("<b>Hyperparameters Used</b>", table_cell_style), Paragraph(hparam_str if hparam_str else "Default settings", table_cell_style)],
        [Paragraph("<b>Distance/Linkage Metrics</b>", table_cell_style), Paragraph("Euclidean / Ward Linkage" if "agglomerative" in str(model_info.get("model")).lower() else "Euclidean Distance", table_cell_style)]
    ]
    t_config = Table(config_rows, colWidths=[150, 350])
    t_config.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f8fafc')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#e2e8f0')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(t_config)
    story.append(Spacer(1, 15))

    # --- 5. CLUSTER QUALITY METRICS ---
    story.append(Paragraph(next_section("Cluster Quality Metrics"), h1_style))
    metrics = model_info.get("metrics", {})
    silhouette = metrics.get("Silhouette")
    ch_index = metrics.get("CalinskiHarabasz")
    db_index = metrics.get("DaviesBouldin")
    noise_points = metrics.get("NoisePoints")

    def _fmt_metric(val):
        return f"{val:.4f}" if isinstance(val, (int, float)) else "N/A"

    metrics_rows = [
        [Paragraph("Metric", table_cell_header_style), Paragraph("Value", table_cell_header_style), Paragraph("Interpretation", table_cell_header_style)],
        [
            Paragraph("<b>Silhouette Score</b>", table_cell_style),
            Paragraph(_fmt_metric(silhouette), table_cell_style),
            Paragraph("Ranges from -1 to +1. Closer to +1 indicates clear, well-separated cluster divisions.", table_cell_style)
        ],
        [
            Paragraph("<b>Davies-Bouldin Index</b>", table_cell_style),
            Paragraph(_fmt_metric(db_index), table_cell_style),
            Paragraph("Lower scores indicate better clustering partitions (closer to 0 is optimal).", table_cell_style)
        ],
        [
            Paragraph("<b>Calinski-Harabasz Index</b>", table_cell_style),
            Paragraph(_fmt_metric(ch_index), table_cell_style),
            Paragraph("Higher scores denote dense and well-separated clusters.", table_cell_style)
        ]
    ]
    if noise_points is not None:
        metrics_rows.append([
            Paragraph("<b>Noise Points Detected</b>", table_cell_style),
            Paragraph(str(noise_points), table_cell_style),
            Paragraph("Points labeled as outliers (-1) and excluded from metric scores.", table_cell_style)
        ])

    t_metrics = Table(metrics_rows, colWidths=[150, 80, 270])
    t_metrics.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1e3a8a')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#1e3a8a')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(t_metrics)
    
    # Cautious metric summary
    story.append(Spacer(1, 6))
    if isinstance(silhouette, (int, float)):
        if silhouette >= 0.5:
            interp_word = "strong cluster separation"
        elif silhouette >= 0.25:
            interp_word = "moderate cluster separation with some overlap"
        else:
            interp_word = "weak clustering structure and significant overlap"
        story.append(Paragraph(f"<i>Metric Summary:</i> The computed Silhouette value of {silhouette:.3f} indicates {interp_word}.", body_style))
    story.append(PageBreak())

    # --- 6. CLUSTER DISTRIBUTION ---
    story.append(Paragraph(next_section("Cluster Distribution"), h1_style))
    cluster_sizes = model_info.get("cluster_sizes", {})
    total_samples = sum(cluster_sizes.values())

    dist_rows = [
        [Paragraph("Cluster Label", table_cell_header_style), Paragraph("Sample Count", table_cell_header_style), Paragraph("Percentage", table_cell_header_style)]
    ]
    for lbl, count in sorted(cluster_sizes.items(), key=lambda x: int(x[0]) if x[0] != 'Noise' and x[0] != '-1' else 999):
        pct = (count / total_samples * 100) if total_samples > 0 else 0
        lbl_display = "Noise / Outliers" if lbl in ('-1', 'Noise') else f"Cluster {lbl}"
        dist_rows.append([
            Paragraph(lbl_display, table_cell_style),
            Paragraph(f"{count:,}", table_cell_style),
            Paragraph(f"{pct:.1f}%", table_cell_style),
        ])

    t_dist = Table(dist_rows, colWidths=[180, 160, 160])
    t_dist.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#475569')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#cbd5e1')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(t_dist)
    story.append(Spacer(1, 15))

    # --- 7. CLUSTER PROFILES ---
    story.append(Paragraph(next_section("Cluster Feature Profiles"), h1_style))
    cluster_profiles = model_info.get("cluster_profiles", [])
    
    for profile in cluster_profiles:
        c_lbl = profile.get("cluster")
        c_samples = profile.get("samples", 0)
        c_pct = (c_samples / total_samples * 100) if total_samples > 0 else 0
        
        lbl_display = "Noise / Outliers" if c_lbl == 'Noise' else f"Cluster {c_lbl}"
        story.append(Paragraph(f"<b>{lbl_display} ({c_samples:,} samples, {c_pct:.1f}%)</b>", body_style))
        
        prof_rows = [
            [Paragraph("Feature Name", table_cell_header_style), Paragraph("Mean Value", table_cell_header_style), Paragraph("Median Value", table_cell_header_style)]
        ]
        
        means = profile.get("feature_means", {})
        medians = profile.get("feature_medians", {})
        
        for feat in selected_features:
            if feat in numerical_features:
                prof_rows.append([
                    Paragraph(_clean_feature_name(feat), table_cell_style),
                    Paragraph(f"{means.get(feat, 0.0):,.4f}" if isinstance(means.get(feat), (int, float)) else "N/A", table_cell_style),
                    Paragraph(f"{medians.get(feat, 0.0):,.4f}" if isinstance(medians.get(feat), (int, float)) else "N/A", table_cell_style),
                ])
                
        t_prof = Table(prof_rows, colWidths=[200, 150, 150])
        t_prof.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#f1f5f9')),
            ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#cbd5e1')),
            ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
            ('TOPPADDING', (0,0), (-1,-1), 4),
            ('BOTTOMPADDING', (0,0), (-1,-1), 4),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ]))
        story.append(t_prof)
        story.append(Spacer(1, 10))
    story.append(PageBreak())

    # --- 8. VISUALIZATIONS ---
    story.append(Paragraph(next_section("Visualizations"), h1_style))
    embeds = []
    
    # PCA Plot
    if graph_paths.get("pca_cluster_plot") and os.path.exists(graph_paths["pca_cluster_plot"]):
        embeds.append(Image(graph_paths["pca_cluster_plot"], width=230, height=172))
    # Size Plot
    if graph_paths.get("cluster_sizes") and os.path.exists(graph_paths["cluster_sizes"]):
        embeds.append(Image(graph_paths["cluster_sizes"], width=230, height=172))
    # Silhouette Plot
    if graph_paths.get("silhouette_plot") and os.path.exists(graph_paths["silhouette_plot"]):
        embeds.append(Image(graph_paths["silhouette_plot"], width=230, height=172))
    # Dendrogram Plot
    if graph_paths.get("dendrogram_plot") and os.path.exists(graph_paths["dendrogram_plot"]):
        embeds.append(Image(graph_paths["dendrogram_plot"], width=230, height=172))
    # Elbow Plot
    if graph_paths.get("elbow_plot") and os.path.exists(graph_paths["elbow_plot"]):
        embeds.append(Image(graph_paths["elbow_plot"], width=230, height=172))

    if embeds:
        story.extend(_grid_layout(embeds))
    else:
        story.append(Paragraph("No dynamic clustering visualization graphs available.", body_style))
    story.append(Spacer(1, 15))

    # --- 9. CLUSTER INTERPRETATION (Key Distinguishing Features) ---
    story.append(Paragraph(next_section("Cluster Interpretation & Key Characteristics"), h1_style))
    
    has_characteristics = False
    for profile in cluster_profiles:
        c_lbl = profile.get("cluster")
        chars = profile.get("important_characteristics", [])
        if chars:
            has_characteristics = True
            lbl_display = "Noise / Outliers" if c_lbl == 'Noise' else f"Cluster {c_lbl}"
            story.append(Paragraph(f"<b>{lbl_display} Key Distinguishing Features:</b>", body_style))
            for char in chars:
                clean_char = _clean_feature_name(char)
                story.append(Paragraph(f"• {clean_char}", body_style))
            story.append(Spacer(1, 8))

    if not has_characteristics:
        story.append(Paragraph("No significant feature differences were identified dynamically against the dataset average.", body_style))
    story.append(Spacer(1, 15))

    # --- 10. AI INSIGHTS ---
    story.append(Paragraph(next_section("AI Agent Detailed Explanations"), h1_style))
    story.append(Spacer(1, 5))
    
    # Process and append AI markdown paragraphs
    raw_paragraphs = ai_report_markdown.split('\n')
    current_text_block = []
    
    for line in raw_paragraphs:
        line_clean = line.strip()
        if not line_clean:
            if current_text_block:
                full_para = " ".join(current_text_block)
                cleaned_para = clean_markdown_for_pdf(full_para)
                story.append(Paragraph(cleaned_para, body_style))
                current_text_block = []
        elif line_clean.startswith('#') or line_clean.startswith('---'):
            if current_text_block:
                full_para = " ".join(current_text_block)
                cleaned_para = clean_markdown_for_pdf(full_para)
                story.append(Paragraph(cleaned_para, body_style))
                current_text_block = []
                
            if line_clean.startswith('---'):
                story.append(Spacer(1, 8))
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
        elif line_clean.startswith('- ') or line_clean.startswith('* '):
            if current_text_block:
                full_para = " ".join(current_text_block)
                cleaned_para = clean_markdown_for_pdf(full_para)
                story.append(Paragraph(cleaned_para, body_style))
                current_text_block = []
            bullet_text = line_clean[2:].strip()
            cleaned_bullet = clean_markdown_for_pdf(bullet_text)
            story.append(Paragraph(f"• {cleaned_bullet}", body_style))
        else:
            current_text_block.append(line_clean)
            
    if current_text_block:
        full_para = " ".join(current_text_block)
        cleaned_para = clean_markdown_for_pdf(full_para)
        story.append(Paragraph(cleaned_para, body_style))
        
    story.append(Spacer(1, 15))

    # --- 11. RECOMMENDATIONS ---
    story.append(Paragraph(next_section("Strategic Recommendations"), h1_style))
    recs = [
        "Use cluster groups to target distinct profiles individually rather than applying a single uniform strategy.",
        "Compare stability of clusters by running alternative algorithms (e.g. comparing K-Means against DBSCAN/OPTICS to ensure divisions are stable).",
        "Consider collecting additional analytical attributes that directly describe behavior to further refine grouping boundaries."
    ]
    for rec in recs:
        story.append(Paragraph(f"• {rec}", body_style))
    story.append(Spacer(1, 15))

    # --- 12. LIMITATIONS ---
    story.append(Paragraph(next_section("Methodological Limitations"), h1_style))
    lims = [
        "Unsupervised clustering indicates correlations and natural patterns, not causal relations.",
        "Centroid-based algorithms (like K-Means) assume spherical clusters of similar size and density, which might not reflect complex real-world data structures.",
        "One-hot encoded categorical variables scale differently than continuous ones, potentially biasing distance metrics."
    ]
    for lim in lims:
        story.append(Paragraph(f"• {lim}", body_style))

    # Build document
    doc.build(story)
    return output_pdf_path
