import json
import logging
import os
import re
from typing import Dict, Any, List
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

from app.ml.feature_types import map_expanded_coefficients

logger = logging.getLogger("regression_studio.pdf_generator")

# The base-14 PDF fonts (Helvetica etc.) only cover WinAnsi/Latin-1 — no
# Greek letters, no Unicode superscript/subscript block. The Model Equation
# section needs β, Σ, ŷ and <sub>/<super> tags to render correctly, so it
# uses DejaVu Sans instead, reusing the copy matplotlib already ships with
# (it's matplotlib's own default font) rather than adding a new dependency.
_MATH_FONT_REGULAR = "Helvetica"
_MATH_FONT_BOLD = "Helvetica-Bold"
_math_fonts_registered = False


def _register_math_fonts() -> None:
    global _math_fonts_registered, _MATH_FONT_REGULAR, _MATH_FONT_BOLD
    if _math_fonts_registered:
        return
    _math_fonts_registered = True
    try:
        import matplotlib
        ttf_dir = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
        pdfmetrics.registerFont(TTFont("DejaVuSans", os.path.join(ttf_dir, "DejaVuSans.ttf")))
        pdfmetrics.registerFont(TTFont("DejaVuSans-Bold", os.path.join(ttf_dir, "DejaVuSans-Bold.ttf")))
        pdfmetrics.registerFont(TTFont("DejaVuSans-Oblique", os.path.join(ttf_dir, "DejaVuSans-Oblique.ttf")))
        pdfmetrics.registerFont(TTFont("DejaVuSans-BoldOblique", os.path.join(ttf_dir, "DejaVuSans-BoldOblique.ttf")))
        pdfmetrics.registerFontFamily(
            "DejaVuSans", normal="DejaVuSans", bold="DejaVuSans-Bold",
            italic="DejaVuSans-Oblique", boldItalic="DejaVuSans-BoldOblique",
        )
        _MATH_FONT_REGULAR = "DejaVuSans"
        _MATH_FONT_BOLD = "DejaVuSans-Bold"
    except Exception as e:
        logger.warning(f"Could not register DejaVu Sans for math notation; falling back to Helvetica (Greek/Unicode math glyphs will not render): {e}")

def clean_markdown_for_pdf(text: str) -> str:
    """
    Converts simple markdown syntax (headers, bold, italic, inline code) into
    ReportLab-compatible XML-like tags (<b>, <i>, etc.). Expects list markers
    ("- "/"* ") to already be stripped by the caller — see the AI-explanation
    rendering loop in build_pdf_report, which renders each bullet as its own
    Paragraph rather than passing raw "* " markers through here. Doing that
    conversion on a per-paragraph blob (as this used to) let the *italic*
    regex below pair one bullet's leading "*" with the next bullet's leading
    "*", producing overlapping/unbalanced <i> tags that crashed ReportLab's
    XML parser.
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
    
    # 4. Replace italic (only asterisks). Excludes commas/newlines from the
    # matched span — without this, LLM-generated prose using literal
    # asterisks for multiplication in a list (e.g. "a*b, a*c, b*c") lets the
    # non-greedy match span across unrelated pairs ("*b, a*" here), producing
    # unbalanced <i> tags that crash ReportLab's XML parser. Genuine italic
    # phrases essentially never contain a bare comma, so this is a safe
    # trade-off: crash prevention over converting that rare edge case.
    text = re.sub(r'\*([^*\n,]+?)\*', r'<i>\1</i>', text)
    
    # 5. Replace inline code blocks
    text = re.sub(r'`(.*?)`', r'<font face="Courier">\1</font>', text)
    
    return text


def _report_styles() -> Dict[str, ParagraphStyle]:
    """Shared paragraph styles used by both the model report and the
    database query export, so the two PDFs read as one visual family."""
    _register_math_fonts()
    base = getSampleStyleSheet()
    return {
        "eq_header": ParagraphStyle(
            "EquationHeader", parent=base["Heading3"], fontName=_MATH_FONT_BOLD,
            fontSize=11, leading=15, textColor=colors.HexColor("#1e3a8a"),
            spaceBefore=10, spaceAfter=6, keepWithNext=True,
        ),
        "eq_body": ParagraphStyle(
            "EquationBody", parent=base["Normal"], fontName=_MATH_FONT_REGULAR,
            fontSize=9.5, leading=15, textColor=colors.HexColor("#1e293b"), spaceAfter=6,
        ),
        "eq_formula": ParagraphStyle(
            "EquationFormula", parent=base["Normal"], fontName=_MATH_FONT_REGULAR,
            fontSize=11, leading=17, textColor=colors.HexColor("#0f172a"), spaceAfter=8,
            leftIndent=16,
        ),
        "title": ParagraphStyle(
            "DocTitle", parent=base["Heading1"], fontName="Helvetica-Bold",
            fontSize=24, leading=28, textColor=colors.HexColor("#1e3a8a"), spaceAfter=15,
        ),
        "subtitle": ParagraphStyle(
            "DocSubTitle", parent=base["Normal"], fontName="Helvetica-Oblique",
            fontSize=10, leading=14, textColor=colors.HexColor("#475569"), spaceAfter=25,
        ),
        "h1": ParagraphStyle(
            "SectionHeader", parent=base["Heading2"], fontName="Helvetica-Bold",
            fontSize=15, leading=19, textColor=colors.HexColor("#1e3a8a"),
            spaceBefore=15, spaceAfter=10, keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "ReportBody", parent=base["Normal"], fontName="Helvetica",
            fontSize=9.5, leading=14, textColor=colors.HexColor("#1e293b"), spaceAfter=8,
        ),
        "cell": ParagraphStyle(
            "TableCell", parent=base["Normal"], fontName="Helvetica",
            fontSize=8.5, leading=11, textColor=colors.HexColor("#1e293b"),
        ),
        "cell_header": ParagraphStyle(
            "TableCellHeader", parent=base["Normal"], fontName="Helvetica-Bold",
            fontSize=8.5, leading=11, textColor=colors.white,
        ),
    }


# A high-cardinality categorical column (e.g. a customer-ID-like column
# with hundreds/thousands of categories) one-hot-expands into one
# coefficient PER CATEGORY — rendering every single one would produce a
# report hundreds of pages long. Cap the table to the largest-magnitude
# coefficients (the ones actually worth reading) and say so explicitly
# rather than silently dropping the rest.
_MAX_STATS_TABLE_ROWS = 30


def _build_stats_table(
    coefs, p_values, std_errors, t_stats,
    numeric_features, categorical_features, encoded_categorical_names,
    table_cell_style, table_cell_header_style, body_style,
) -> List:
    """Builds the Variable/Coefficient/Std.Error/t-Statistic/p-Value/
    Significant table shared by the regression "Statistical Parameter
    Analysis" section and the classification analysis section below.

    `coefs`/`p_values`/`std_errors`/`t_stats` are keyed by the fitted
    pipeline's EXPANDED output names (e.g. "cat__id_customer_customer_482",
    or "{feature}__class_{k}" for a flattened multiclass fit) — one entry
    per one-hot category, not one per original selected column. Rendering
    those raw keys directly (the old behavior) produced a table with one row
    per category of every categorical feature, which for a high-cardinality
    column like a customer/product ID balloons the report to hundreds of
    pages of rows like "cat__id_customer_customer_1__class_5". Mapping
    through the same map_expanded_coefficients() helper used for the
    interactive chart/PNG/AI explanation gives real, attributable display
    labels ("city = Chennai") instead, and the table is additionally capped
    to the largest-magnitude coefficients so it stays report-sized.
    """
    encoded_categorical_names = encoded_categorical_names or [
        k[len("cat__"):] for k in coefs if k.startswith("cat__")
    ]
    strip_const = lambda d: {k: v for k, v in d.items() if k != "const" and not k.startswith("const__")}
    mapped_coefs = map_expanded_coefficients(strip_const(coefs), numeric_features, categorical_features, encoded_categorical_names)
    mapped_p = map_expanded_coefficients(strip_const(p_values), numeric_features, categorical_features, encoded_categorical_names)
    mapped_se = map_expanded_coefficients(strip_const(std_errors), numeric_features, categorical_features, encoded_categorical_names)
    mapped_t = map_expanded_coefficients(strip_const(t_stats), numeric_features, categorical_features, encoded_categorical_names)
    # coefs/p_values/std_errors/t_stats come from the same fitted
    # statsmodels result, so they share the exact same key set in the exact
    # same order — zipping the four parallel mapped lists by position is
    # simpler and more robust than re-deriving one dict's raw key from
    # another's already-stripped display label.
    p_by_idx = [e["value"] for e in mapped_p]
    se_by_idx = [e["value"] for e in mapped_se]
    t_by_idx = [e["value"] for e in mapped_t]

    total_rows = len(mapped_coefs)
    # Largest-magnitude coefficients first — the ones actually worth a
    # reader's attention — then cap to a report-sized page count.
    order = sorted(range(total_rows), key=lambda i: abs(mapped_coefs[i]["value"]) if mapped_coefs[i]["value"] is not None else -1, reverse=True)
    shown_order = order[:_MAX_STATS_TABLE_ROWS]

    def _fmt(value, spec):
        # statsmodels returns NaN (sanitized to None before storage) for
        # coefficients/p-values it can't resolve on a near-singular design
        # matrix — common with Polynomial Regression / multiclass Logit once
        # feature expansion produces many collinear columns. Render "N/A"
        # instead of crashing PDF generation on a None value.
        return format(value, spec) if isinstance(value, (int, float)) else "N/A"

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
    for i in shown_order:
        var_name = mapped_coefs[i]["feature"]
        coef = mapped_coefs[i]["value"]
        p_val = p_by_idx[i] if i < len(p_by_idx) else None
        std_err = se_by_idx[i] if i < len(se_by_idx) else None
        t_stat = t_by_idx[i] if i < len(t_by_idx) else None
        sig = "YES" if isinstance(p_val, (int, float)) and p_val < 0.05 else ("N/A" if p_val is None else "NO")
        sig_color = "#10b981" if sig == "YES" else ("#94a3b8" if sig == "N/A" else "#ef4444")

        stats_rows.append([
            Paragraph(f"<b>{var_name}</b>", table_cell_style),
            Paragraph(_fmt(coef, ",.4f"), table_cell_style),
            Paragraph(_fmt(std_err, ",.4f"), table_cell_style),
            Paragraph(_fmt(t_stat, ",.4f"), table_cell_style),
            Paragraph(_fmt(p_val, ".4f"), table_cell_style),
            Paragraph(f"<font color='{sig_color}'><b>{sig}</b></font>", table_cell_style)
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

    flow: List = [t_stats_table]
    if total_rows > len(shown_order):
        flow.append(Spacer(1, 6))
        flow.append(Paragraph(
            f"Showing the {len(shown_order)} largest-magnitude coefficients of {total_rows} total "
            "(one-hot-expanded categories are collapsed to real feature names, but a high-cardinality "
            "categorical column can still produce more rows than fit in a report).",
            body_style,
        ))
    return flow


# Feature Summary can be as wide as the dataset has columns — capped for
# the same "never let a report balloon to hundreds of pages" reason as the
# stats coefficient table.
_MAX_FEATURE_SUMMARY_ROWS = 40


def _build_dataset_overview_section(profile: Dict[str, Any], model_info: Dict[str, Any], styles: Dict[str, ParagraphStyle]) -> List:
    """Compact, dataset-independent overview table — every value is read
    from `profile` (computed by report_routes.py from the ACTUAL uploaded
    dataframe, see dataset_profile.py) or model_info, never hard-coded.
    Deliberately never prints dataset rows themselves."""
    table_cell_style = styles["cell"]
    flow: List = []

    rows = [
        [
            Paragraph("<b>Dataset Name</b>", table_cell_style),
            Paragraph(str(model_info.get("dataset_name", "N/A")), table_cell_style),
            Paragraph("<b>Total Rows</b>", table_cell_style),
            Paragraph(f"{profile.get('total_rows', 0):,}", table_cell_style),
        ],
        [
            Paragraph("<b>Total Columns</b>", table_cell_style),
            Paragraph(str(profile.get("total_columns", 0)), table_cell_style),
            Paragraph("<b>Target Variable</b>", table_cell_style),
            Paragraph(str(model_info.get("target", "N/A")), table_cell_style),
        ],
        [
            Paragraph("<b>Numerical Features</b>", table_cell_style),
            Paragraph(str(profile.get("numerical_count", 0)), table_cell_style),
            Paragraph("<b>Categorical Features</b>", table_cell_style),
            Paragraph(str(profile.get("categorical_count", 0)), table_cell_style),
        ],
        [
            Paragraph("<b>Datetime Columns</b>", table_cell_style),
            Paragraph(str(profile.get("datetime_count", 0)), table_cell_style),
            Paragraph("<b>Input Features Used</b>", table_cell_style),
            Paragraph(str(len(model_info.get("features", []) or [])), table_cell_style),
        ],
        [
            Paragraph("<b>Missing Values</b>", table_cell_style),
            Paragraph(f"{profile.get('missing_values_total', 0):,}", table_cell_style),
            Paragraph("<b>Duplicate Rows</b>", table_cell_style),
            Paragraph(f"{profile.get('duplicate_rows', 0):,}", table_cell_style),
        ],
    ]
    t = Table(rows, colWidths=[115, 135, 115, 135])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8fafc')),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#e2e8f0')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    flow.append(t)
    return flow


def _build_feature_summary_section(profile: Dict[str, Any], model_info: Dict[str, Any], styles: Dict[str, ParagraphStyle]) -> List:
    """Per-column summary table keyed by ORIGINAL dataset column names only
    (never the one-hot-expanded pipeline output names) — see `profile`'s
    `features` list, built by dataset_profile.py from classify_columns(),
    the same column-typing helper the rest of the app already uses."""
    table_cell_style = styles["cell"]
    table_cell_header_style = styles["cell_header"]
    body_style = styles["body"]
    flow: List = []

    target = model_info.get("target")
    all_features = profile.get("features", []) or []
    shown = all_features[:_MAX_FEATURE_SUMMARY_ROWS]

    rows = [[
        Paragraph(h, table_cell_header_style)
        for h in ("Feature Name", "Data Type", "Role", "Missing %", "Unique Values")
    ]]
    for f in shown:
        role = "Target" if f["name"] == target else "Feature"
        rows.append([
            Paragraph(f["name"], table_cell_style),
            Paragraph(f["kind"].capitalize(), table_cell_style),
            Paragraph(role, table_cell_style),
            Paragraph(f"{f['missing_pct']:.1f}%", table_cell_style),
            Paragraph(f"{f['unique_count']:,}", table_cell_style),
        ])

    t = Table(rows, colWidths=[150, 85, 65, 75, 100])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#475569')),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#475569')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    flow.append(t)

    if len(all_features) > len(shown):
        flow.append(Spacer(1, 6))
        flow.append(Paragraph(
            f"Showing {len(shown)} of {len(all_features)} dataset columns.",
            body_style,
        ))
    return flow


def _build_classification_analysis_section(model_info: Dict[str, Any], styles: Dict[str, ParagraphStyle]) -> List:
    """Classification counterpart to the regression "Statistical Parameter
    Analysis" section: a confusion matrix table, per-class precision/
    recall/F1 from the classification report, and the same coefficients/
    p-values table used for regression (reusing _build_stats_table — the
    stats dict is deliberately kept in the same OLS-shaped key layout, see
    classification_evaluation.py, so no separate rendering logic is needed)."""
    table_cell_style = styles["cell"]
    table_cell_header_style = styles["cell_header"]
    body_style = styles["body"]
    flow: List = []

    metrics = model_info.get("metrics", {}) or {}
    classes = [str(c) for c in (metrics.get("Classes") or model_info.get("classes") or [])]
    conf_matrix = metrics.get("ConfusionMatrix") or []

    if conf_matrix and classes and len(conf_matrix) == len(classes):
        header_row = [Paragraph("Actual \\ Predicted", table_cell_header_style)] + [
            Paragraph(c, table_cell_header_style) for c in classes
        ]
        cm_rows = [header_row]
        for i, row in enumerate(conf_matrix):
            cm_rows.append(
                [Paragraph(f"<b>{classes[i]}</b>", table_cell_style)]
                + [Paragraph(str(v), table_cell_style) for v in row]
            )
        col_width = min(90, 400 // max(len(classes), 1))
        t_cm = Table(cm_rows, colWidths=[130] + [col_width] * len(classes))
        t_cm.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e3a8a')),
            ('BACKGROUND', (0, 1), (0, -1), colors.HexColor('#1e3a8a')),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#1e3a8a')),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        flow.append(Paragraph("Confusion Matrix (rows = actual class, columns = predicted class)", body_style))
        flow.append(Spacer(1, 6))
        flow.append(t_cm)
        flow.append(Spacer(1, 12))

    class_report = metrics.get("ClassificationReport") or {}
    per_class_rows = [
        [
            Paragraph("Class", table_cell_header_style),
            Paragraph("Precision", table_cell_header_style),
            Paragraph("Recall", table_cell_header_style),
            Paragraph("F1-Score", table_cell_header_style),
            Paragraph("Support", table_cell_header_style),
        ]
    ]
    for cls in classes:
        entry = class_report.get(cls) or {}
        per_class_rows.append([
            Paragraph(str(cls), table_cell_style),
            Paragraph(f"{entry.get('precision', 0):.3f}" if isinstance(entry.get("precision"), (int, float)) else "N/A", table_cell_style),
            Paragraph(f"{entry.get('recall', 0):.3f}" if isinstance(entry.get("recall"), (int, float)) else "N/A", table_cell_style),
            Paragraph(f"{entry.get('f1-score', 0):.3f}" if isinstance(entry.get("f1-score"), (int, float)) else "N/A", table_cell_style),
            Paragraph(str(int(entry["support"])) if isinstance(entry.get("support"), (int, float)) else "N/A", table_cell_style),
        ])
    if len(per_class_rows) > 1:
        t_per_class = Table(per_class_rows, colWidths=[140, 90, 90, 90, 90])
        t_per_class.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#475569')),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#475569')),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        flow.append(Paragraph("Per-Class Performance", body_style))
        flow.append(Spacer(1, 6))
        flow.append(t_per_class)
        flow.append(Spacer(1, 12))

    stats = model_info.get("statistical_analysis", {}) or {}
    features = model_info.get("features", []) or []
    categorical_features = model_info.get("categorical_features", []) or []
    numeric_features = model_info.get("numerical_features", features) or features
    flow.append(Paragraph("Coefficients (log-odds scale)", body_style))
    flow.append(Spacer(1, 6))
    flow.extend(_build_stats_table(
        stats.get("coefficients", {}), stats.get("p_values", {}),
        stats.get("standard_errors", {}), stats.get("t_statistics", {}),
        numeric_features, categorical_features, None,
        table_cell_style, table_cell_header_style, body_style,
    ))

    return flow


def _grid_layout(embeds: List) -> List:
    """Lays out a list of ReportLab Image flowables 2-per-row (stacked
    inside a Table to prevent overflowing the page margins) — shared by
    every image section (dataset EDA plots, model performance plots) so
    they all read as one consistent visual family."""
    if not embeds:
        return []
    if len(embeds) == 1:
        return [embeds[0]]
    rows = [embeds[i:i + 2] for i in range(0, len(embeds), 2)]
    grid_data = [row + [""] * (2 - len(row)) for row in rows]
    t_grid = Table(grid_data, colWidths=[250, 250])
    t_grid.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
    ]))
    return [t_grid]


def _is_linear_regression_model(model_name: str) -> bool:
    """Matches the same OLS-family name variants get_regression_model()
    accepts (app/ml/regression_models.py), so the equation section below
    only renders for genuine unregularized Linear Regression — not Ridge/
    Lasso/ElasticNet/Polynomial/tree/SVR models, whose math it doesn't
    describe."""
    name_clean = (model_name or "").replace(" ", "").lower()
    return name_clean in ("linearregression", "linear", "multiplelinear", "multiplelinearregression")


def _build_linear_regression_equation_section(model_info: Dict[str, Any], styles: Dict[str, ParagraphStyle]) -> List:
    """Builds the 'Model Equation' section for Linear Regression reports.

    Sections A-D are fixed OLS theory — identical for every Linear
    Regression report regardless of dataset. The final block is the only
    dataset-dependent part: it pulls the actual fitted coefficients,
    sample counts, and metrics from model_info so the report clearly
    separates general theory from this specific trained model's results.
    """
    eq_header = styles["eq_header"]
    eq_body = styles["eq_body"]
    eq_formula = styles["eq_formula"]
    table_cell_style = styles["cell"]
    table_cell_header_style = styles["cell_header"]

    features = model_info.get("features", []) or []
    categorical_features = model_info.get("categorical_features", []) or []
    numeric_features = model_info.get("numerical_features", features) or features
    is_simple = len(features) == 1
    flow: List = []

    # --- A. Model Equation ---
    flow.append(Paragraph("A. Model Equation", eq_header))
    if is_simple:
        flow.append(Paragraph("Simple Linear Regression — one independent feature x:", eq_body))
        flow.append(Paragraph("ŷ = β<sub>0</sub> + β<sub>1</sub>x", eq_formula))
        flow.append(Paragraph(
            "Where <b>ŷ</b> is the predicted value of the target variable, <b>x</b> is the input "
            "feature, <b>β<sub>0</sub></b> is the intercept, and <b>β<sub>1</sub></b> is the "
            "coefficient (slope) of the feature.",
            eq_body,
        ))
        flow.append(Paragraph(
            "β<sub>1</sub> represents the expected change in the predicted target for a one-unit "
            "increase in x, keeping the model structure fixed.",
            eq_body,
        ))
    else:
        flow.append(Paragraph(f"Multiple Linear Regression — {len(features)} independent features:", eq_body))
        shown = min(len(features), 3)
        terms = " + ".join(f"β<sub>{i+1}</sub>x<sub>{i+1}</sub>" for i in range(shown))
        if len(features) > shown:
            terms += " + ... + β<sub>p</sub>x<sub>p</sub>"
        flow.append(Paragraph(f"ŷ = β<sub>0</sub> + {terms}", eq_formula))
        flow.append(Paragraph("In vector notation:", eq_body))
        flow.append(Paragraph("ŷ = x<super>T</super>β", eq_formula))
        flow.append(Paragraph(
            "x = [1, x<sub>1</sub>, x<sub>2</sub>, ..., x<sub>p</sub>]<super>T</super>"
            "&nbsp;&nbsp;&nbsp;β = [β<sub>0</sub>, β<sub>1</sub>, β<sub>2</sub>, ..., β<sub>p</sub>]<super>T</super>",
            eq_formula,
        ))
        flow.append(Paragraph("The first element of β represents the intercept term.", eq_body))

    if categorical_features:
        flow.append(Paragraph(
            "<i>Note on categorical features:</i> the simplified equation above shows one term per "
            "selected column for readability. In the actual fitted model, each categorical feature "
            f"({', '.join(categorical_features)}) is expanded into multiple one-hot indicator terms "
            "(one per category, minus a dropped baseline category) — see the coefficient table below "
            "for the real per-category terms actually estimated.",
            eq_body,
        ))

    # --- B. Objective / Loss Function ---
    flow.append(Paragraph("B. Objective / Loss Function", eq_header))
    flow.append(Paragraph(
        "Linear Regression using Ordinary Least Squares (OLS) estimates the coefficients by "
        "minimizing the sum of squared residuals across the n training observations:",
        eq_body,
    ))
    flow.append(Paragraph(
        "RSS = Σ<sub>i=1</sub><super>n</super> (y<sub>i</sub> - ŷ<sub>i</sub>)<super>2</super>",
        eq_formula,
    ))
    flow.append(Paragraph(
        "The objective is to minimize RSS with respect to β<sub>0</sub>, β<sub>1</sub>, ..., β<sub>p</sub>. "
        "The Mean Squared Error (MSE) is the average of the same squared errors:",
        eq_body,
    ))
    flow.append(Paragraph(
        "MSE = (1/n) Σ<sub>i=1</sub><super>n</super> (y<sub>i</sub> - ŷ<sub>i</sub>)<super>2</super>",
        eq_formula,
    ))
    flow.append(Paragraph(
        "<i>Note:</i> RSS/SSE (the raw sum of squared errors) and MSE (its average) are closely "
        "related but not the same quantity — MSE is RSS divided by n.",
        eq_body,
    ))

    # --- C. Parameter Estimation — OLS ---
    flow.append(Paragraph("C. Parameter Estimation — Ordinary Least Squares", eq_header))
    flow.append(Paragraph(
        "In matrix notation, y = Xβ + ε, where X is the design matrix and ε is the error vector. "
        "The classical closed-form OLS estimator is:",
        eq_body,
    ))
    flow.append(Paragraph(
        "β<super>^</super> = (X<super>T</super>X)<super>-1</super>X<super>T</super>y",
        eq_formula,
    ))
    flow.append(Paragraph(
        "when X<super>T</super>X is invertible. In practice, implementations such as scikit-learn's "
        "LinearRegression do not directly compute this matrix inverse — they use a numerically "
        "stable linear algebra solver, equivalent to the Moore-Penrose pseudoinverse:",
        eq_body,
    ))
    flow.append(Paragraph("β<super>^</super> = X<super>+</super>y", eq_formula))
    flow.append(Paragraph(
        "This pseudoinverse form stays well-defined even when X<super>T</super>X is singular or "
        "poorly conditioned due to multicollinearity among the features.",
        eq_body,
    ))
    flow.append(Paragraph(
        "<i>OLS vs. Gradient Descent:</i> the closed-form solution above is what scikit-learn's "
        "LinearRegression actually computes for this report — it is exact and requires no learning "
        "rate or iteration count. Gradient Descent is an alternative, iterative way to reach the same "
        "optimum: it starts from initial coefficients and repeatedly updates them in the direction "
        "that reduces MSE, β &#8592; β - α&#8711;MSE(β), until convergence. It scales better to very "
        "large or high-dimensional datasets where forming and inverting X<super>T</super>X is "
        "expensive, but it is not used here since the closed-form solver is both exact and fast at "
        "this dataset's scale.",
        eq_body,
    ))

    # --- D. Hyperparameters ---
    flow.append(Paragraph("D. Hyperparameters", eq_header))
    flow.append(Paragraph(
        "Standard OLS Linear Regression has no regularization-strength hyperparameter (λ) — its "
        "coefficients are estimated directly from the training data by minimizing RSS. "
        "Implementation-level options such as <font face=\"Courier\">fit_intercept</font>, "
        "<font face=\"Courier\">copy_X</font>, <font face=\"Courier\">positive</font>, and "
        "<font face=\"Courier\">n_jobs</font> may exist, but these control implementation "
        "behavior — they are not hyperparameters of the mathematical OLS objective itself.",
        eq_body,
    ))
    flow.append(Paragraph(
        "For contrast, regularized variants add a penalty term controlled by λ: Ridge minimizes "
        "RSS + λΣ<sub>j</sub>β<sub>j</sub><super>2</super>, and Lasso minimizes "
        "RSS + λΣ<sub>j</sub>|β<sub>j</sub>|. Neither applies to the plain OLS model used here.",
        eq_body,
    ))

    # --- Actual trained model (data-driven, never hardcoded) ---
    flow.append(Paragraph("Actual Trained Model — This Report's Dataset", eq_header))
    flow.append(Paragraph(
        "The equations above describe the general Linear Regression model. The values below are "
        "the real coefficients and metrics fitted on your uploaded dataset, not illustrative examples.",
        eq_body,
    ))

    stats = model_info.get("statistical_analysis", {}) or {}
    coefs = stats.get("coefficients", {}) or {}
    intercept = coefs.get("const", coefs.get("Intercept"))
    metrics = model_info.get("metrics", {}) or {}
    total_rows = model_info.get("total_rows")
    train_rows = model_info.get("train_rows")
    test_rows = model_info.get("test_rows")

    # Models trained before train_rows/test_rows were persisted on the
    # document fall back to a percentage-based estimate from the split config.
    if (train_rows is None or test_rows is None) and total_rows:
        split_cfg = model_info.get("split", {}) or {}
        test_rows = round(total_rows * split_cfg.get("test_size", 0.2))
        train_rows = total_rows - test_rows

    summary_rows = [
        [Paragraph("<b>Model</b>", table_cell_style), Paragraph(str(model_info.get("model", "N/A")), table_cell_style)],
        [Paragraph("<b>Features</b>", table_cell_style), Paragraph(", ".join(features) or "N/A", table_cell_style)],
        [Paragraph("<b>Target</b>", table_cell_style), Paragraph(str(model_info.get("target", "N/A")), table_cell_style)],
        [Paragraph("<b>Training samples</b>", table_cell_style), Paragraph(str(train_rows) if train_rows is not None else "N/A", table_cell_style)],
        [Paragraph("<b>Test samples</b>", table_cell_style), Paragraph(str(test_rows) if test_rows is not None else "N/A", table_cell_style)],
        [Paragraph("<b>Intercept (β0)</b>", table_cell_style), Paragraph(f"{intercept:,.6f}" if isinstance(intercept, (int, float)) else "N/A", table_cell_style)],
    ]
    t_summary = Table(summary_rows, colWidths=[150, 350])
    t_summary.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f8fafc')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#e2e8f0')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    flow.append(t_summary)
    flow.append(Spacer(1, 8))

    # coefs is keyed by the fitted pipeline's expanded output names (e.g.
    # "cat__city_Chennai" — one entry per one-hot category, not one per
    # original selected column), so a direct `coefs.get(feat)` per original
    # feature name would silently render "N/A" for every categorical
    # feature even though a real fitted coefficient exists under a
    # different key. Map through the same shared helper used for the
    # interactive chart/PNG/AI explanation so this table shows the actual
    # per-category terms instead of misrepresenting them as missing.
    encoded_categorical_names = [k[len("cat__"):] for k in coefs if k.startswith("cat__")]
    mapped_coefs = map_expanded_coefficients(
        {k: v for k, v in coefs.items() if k != "const"},
        numeric_features, categorical_features, encoded_categorical_names,
    )
    coef_rows = [[Paragraph("Feature", table_cell_header_style), Paragraph("Coefficient (β)", table_cell_header_style)]]
    for entry in mapped_coefs:
        val = entry["value"]
        coef_rows.append([
            Paragraph(entry["feature"], table_cell_style),
            Paragraph(f"{val:,.6f}" if isinstance(val, (int, float)) else "N/A", table_cell_style),
        ])
    t_coef = Table(coef_rows, colWidths=[250, 250])
    t_coef.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1e3a8a')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#1e3a8a')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    flow.append(t_coef)
    flow.append(Spacer(1, 8))

    eval_rows = [[Paragraph("Metric", table_cell_header_style), Paragraph("Value", table_cell_header_style)]]
    for key, label in (("R2", "R²"), ("MAE", "MAE"), ("MSE", "MSE"), ("RMSE", "RMSE")):
        val = metrics.get(key)
        eval_rows.append([
            Paragraph(label, table_cell_style),
            Paragraph(f"{val:,.4f}" if isinstance(val, (int, float)) else "N/A", table_cell_style),
        ])
    t_eval = Table(eval_rows, colWidths=[250, 250])
    t_eval.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#475569')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#475569')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    flow.append(t_eval)

    return flow


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
    if model_info.get("model_type") == "clustering":
        from app.reports.clustering_report import build_clustering_pdf_report
        return build_clustering_pdf_report(
            model_id=model_id,
            model_info=model_info,
            graph_paths=graph_paths,
            ai_report_markdown=ai_report_markdown,
            output_pdf_path=output_pdf_path
        )

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
    # Section headings are numbered off a running counter rather than
    # hardcoded literals, since the Model Equation section below only
    # appears for Linear Regression reports — a fixed "3.", "4." etc. would
    # desync between Linear and non-Linear reports otherwise.
    section_num = 1

    def next_section(title: str) -> str:
        nonlocal section_num
        heading = f"{section_num}. {title}"
        section_num += 1
        return heading

    # --- PAGE 1: TITLE & METRICS OVERVIEW ---
    story.append(Paragraph("MODELFORGE AI STUDIO", title_style))
    story.append(Paragraph(f"Analytical Report for Model ID: {model_id} | Generated on behalf of User Analysis History", subtitle_style))
    story.append(Spacer(1, 10))

    # 0. Dataset Overview + Feature Summary — compact, dataset-independent
    # sections computed from the actual uploaded dataframe (see
    # report_routes.py / dataset_profile.py), never a row/column dump.
    # `dataset_profile` is optional so a report still renders (minus these
    # two sections) if profiling fails for any reason rather than crashing
    # the whole PDF.
    dataset_profile = model_info.get("dataset_profile")
    if dataset_profile:
        story.append(Paragraph(next_section("Dataset Overview"), h1_style))
        story.extend(_build_dataset_overview_section(dataset_profile, model_info, s))
        story.append(Spacer(1, 15))

        story.append(Paragraph(next_section("Feature Summary"), h1_style))
        story.extend(_build_feature_summary_section(dataset_profile, model_info, s))
        story.append(Spacer(1, 15))

    # 1. Dataset & Model Specifications Table
    story.append(Paragraph(next_section("Model Configuration"), h1_style))

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
    story.append(Paragraph(next_section("Model Evaluation Performance"), h1_style))
    metrics = model_info.get("metrics", {})
    metrics_data = [
        [Paragraph("Metric", table_cell_header_style), Paragraph("Value", table_cell_header_style), Paragraph("Description", table_cell_header_style)]
    ]
    
    descriptions = {
        "R2": "R-squared coefficient of determination. Represents percentage of explained variance.",
        "Adjusted R2": "Adjusted R-squared. Adjusts for feature counts to prevent overestimation.",
        "MSE": "Mean Squared Error. Measures average squared difference between predictions and targets.",
        "RMSE": "Root Mean Squared Error. Shows average prediction error in target units.",
        "MAE": "Mean Absolute Error. Average absolute size of prediction residuals.",
        "MAPE": "Mean Absolute Percentage Error. Average error as a percentage of the actual value.",
        "MSLE": "Mean Squared Logarithmic Error. Penalizes underprediction more than overprediction; undefined if any actual or predicted value is negative.",
        "RMSLE": "Root Mean Squared Logarithmic Error. Square root of MSLE, in a log-scale error unit.",
        "Median Absolute Error": "Median of the absolute prediction errors. Like MAE but robust to outlier residuals.",
        "Max Error": "The single largest absolute prediction error observed on the test set — a worst-case bound, not an average.",
        "Explained Variance": "Share of the target's variance the model accounts for. Similar to R², but not penalized by a systematic prediction bias (offset).",
        "Pearson Correlation": "Linear correlation between actual and predicted values, from -1 to 1. Undefined for a constant array.",
        "Spearman Correlation": "Rank correlation between actual and predicted values, from -1 to 1 — captures monotonic (not necessarily linear) agreement. Undefined for a constant array.",
        "Accuracy": "Fraction of test samples classified correctly.",
        "Precision": "Of everything predicted as a given class, the fraction that was actually correct.",
        "Recall": "Of everything that actually belonged to a given class, the fraction correctly identified.",
        "F1": "Harmonic mean of precision and recall — a single balanced classification quality measure.",
        "ROC_AUC": "Area under the ROC curve. Measures how well the model ranks positive cases above negative ones.",
        "BalancedAccuracy": "Average of per-class recall. Unlike plain Accuracy, not inflated by a large majority class.",
        "MCC": "Matthews Correlation Coefficient. A single balanced score from -1 (always wrong) to +1 (always right), accounting for all four confusion-matrix quadrants.",
        "LogLoss": "Penalizes confident-but-wrong predicted probabilities. Lower is better; requires the model to output probabilities.",
        "Specificity": "True negative rate — of everything that actually belonged to the negative class, the fraction correctly identified as negative. Binary classification only.",
        "Precision Macro": "Precision averaged equally across all classes regardless of class size.",
        "Recall Macro": "Recall averaged equally across all classes regardless of class size.",
        "F1 Macro": "F1-score averaged equally across all classes regardless of class size — a large gap from the weighted F1 suggests underperformance on a rarer class.",
        "Precision Weighted": "Precision averaged across classes, weighted by each class's frequency in the test set.",
        "Recall Weighted": "Recall averaged across classes, weighted by each class's frequency in the test set.",
        "F1 Weighted": "F1-score averaged across classes, weighted by each class's frequency in the test set.",
    }
    # ConfusionMatrix / Classes / ClassificationReport are structured data
    # (lists/dicts), not single scalar values — they get their own tables in
    # the classification analysis section below rather than being rendered
    # (unreadably) as a single row's `str(metric_val)` here.
    _NON_SCALAR_METRIC_KEYS = {"ConfusionMatrix", "Classes", "ClassificationReport"}

    for metric_name, metric_val in metrics.items():
        if metric_name in _NON_SCALAR_METRIC_KEYS:
            continue
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

    # 2.5 Model Equation — deterministic (non-AI-generated) OLS theory plus
    # this model's actual fitted coefficients. Only rendered for genuine
    # Linear Regression; other algorithms' math isn't described by this
    # section. Classification never gets this section at all — there's no
    # OLS-style closed-form equation narrative for a classifier (its
    # coefficients are covered by the classification analysis section
    # below instead).
    is_classification_model = model_info.get("model_type", "regression") == "classification"
    if (
        not is_classification_model
        and model_info.get("source") != "huggingface"
        and _is_linear_regression_model(model_info.get("model", ""))
    ):
        story.append(Paragraph(next_section("Model Equation"), h1_style))
        story.extend(_build_linear_regression_equation_section(model_info, s))
        story.append(Spacer(1, 15))

    # 3. Statsmodels Coefficient Table — or, for a Hugging Face-sourced model
    # (never fit locally via OLS), a provenance block instead.
    if model_info.get("source") == "huggingface":
        story.append(Paragraph(next_section("Model Provenance — Hugging Face"), h1_style))
        hf_model_id = model_info.get("hf_model_id") or model_info.get("model", "N/A")
        version_warning = model_info.get("version_warning")

        provenance_rows = [
            [Paragraph("<b>Hugging Face Model ID</b>", table_cell_style), Paragraph(str(hf_model_id), table_cell_style)],
            [Paragraph("<b>Execution Mode</b>", table_cell_style), Paragraph(str(model_info.get("execution_mode", "inference_only")), table_cell_style)],
            [Paragraph("<b>Source</b>", table_cell_style), Paragraph("huggingface.co — pretrained, inference-only", table_cell_style)],
        ]
        if version_warning:
            provenance_rows.append([
                Paragraph("<b>Version Notice</b>", table_cell_style),
                Paragraph(f"<font color='#b45309'>{version_warning}</font>", table_cell_style),
            ])

        t_provenance = Table(provenance_rows, colWidths=[150, 350])
        t_provenance.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f8fafc')),
            ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#e2e8f0')),
            ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
            ('TOPPADDING', (0,0), (-1,-1), 6),
            ('BOTTOMPADDING', (0,0), (-1,-1), 6),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ]))
        story.append(t_provenance)
        story.append(Spacer(1, 8))
        story.append(Paragraph(
            "This model was executed via a pretrained Hugging Face pipeline rather than fit locally — "
            "coefficient-level statistical analysis (p-values, t-statistics, standard errors) does not apply.",
            body_style,
        ))
    elif is_classification_model:
        story.append(Paragraph(next_section("Classification Analysis"), h1_style))
        story.extend(_build_classification_analysis_section(model_info, s))
    else:
        story.append(Paragraph(next_section("Statistical Parameter Analysis"), h1_style))
        stats = model_info.get("statistical_analysis", {})
        features = model_info.get("features", []) or []
        categorical_features = model_info.get("categorical_features", []) or []
        numeric_features = model_info.get("numerical_features", features) or features
        story.extend(_build_stats_table(
            stats.get("coefficients", {}), stats.get("p_values", {}),
            stats.get("standard_errors", {}), stats.get("t_statistics", {}),
            numeric_features, categorical_features, None,
            table_cell_style, table_cell_header_style, body_style,
        ))

    story.append(PageBreak()) # Move to next page for visualizations

    # --- DATASET VISUALIZATIONS (exploratory data analysis) — generated
    # from the ORIGINAL dataset columns only (see generate_dataset_graphs),
    # never the expanded one-hot matrix. Independent of model_type: every
    # dataset gets a distribution/outlier/correlation view regardless of
    # whether the trained model is a regressor or classifier.
    dataset_embeds: List[Any] = []
    for key in ("distribution", "boxplot", "correlation_heatmap", "target_distribution", "categorical_frequency"):
        if key in graph_paths and graph_paths[key] and os.path.exists(graph_paths[key]):
            dataset_embeds.append(Image(graph_paths[key], width=230, height=172))

    if dataset_embeds:
        story.append(Paragraph(next_section("Exploratory Data Analysis"), h1_style))
        story.append(Spacer(1, 5))
        story.extend(_grid_layout(dataset_embeds))
        story.append(Spacer(1, 10))
    else:
        story.append(Paragraph(next_section("Exploratory Data Analysis"), h1_style))
        story.append(Paragraph(
            "No dataset visualizations were available to generate for this dataset/feature selection.",
            body_style,
        ))

    story.append(PageBreak())

    # --- MODEL PERFORMANCE VISUALIZATIONS — model-result-specific plots,
    # correct set picked by model_type so a classification report never
    # renders a regression-only plot (Actual vs Predicted / Residuals) and
    # vice versa. See graph_generator.py's generate_regression_graphs() /
    # generate_classification_graphs() for the actual key sets produced.
    story.append(Paragraph(next_section("Model Performance Visualizations"), h1_style))
    story.append(Spacer(1, 5))

    model_embeds: List[Any] = []
    unavailable_notes: List[str] = []

    if is_classification_model:
        if "confusion_matrix" in graph_paths and os.path.exists(graph_paths["confusion_matrix"]):
            model_embeds.append(Image(graph_paths["confusion_matrix"], width=230, height=172))
        if "roc_curve" in graph_paths and os.path.exists(graph_paths["roc_curve"]):
            model_embeds.append(Image(graph_paths["roc_curve"], width=230, height=172))
        else:
            unavailable_notes.append(
                "ROC curve not available for this configuration (requires predicted class probabilities)."
            )
        if "precision_recall_curve" in graph_paths and os.path.exists(graph_paths["precision_recall_curve"]):
            model_embeds.append(Image(graph_paths["precision_recall_curve"], width=230, height=172))
        else:
            unavailable_notes.append(
                "Precision-Recall curve not available for this configuration (requires predicted class probabilities)."
            )
        if "feature_importance" in graph_paths and os.path.exists(graph_paths["feature_importance"]):
            model_embeds.append(Image(graph_paths["feature_importance"], width=230, height=172))
        else:
            unavailable_notes.append("Feature importance unavailable for this selected model.")
    else:
        if "actual_vs_predicted" in graph_paths and os.path.exists(graph_paths["actual_vs_predicted"]):
            model_embeds.append(Image(graph_paths["actual_vs_predicted"], width=230, height=172))
        if "residuals" in graph_paths and os.path.exists(graph_paths["residuals"]):
            model_embeds.append(Image(graph_paths["residuals"], width=230, height=172))
        if "error_distribution" in graph_paths and os.path.exists(graph_paths["error_distribution"]):
            model_embeds.append(Image(graph_paths["error_distribution"], width=230, height=172))
        if "feature_importance" in graph_paths and os.path.exists(graph_paths["feature_importance"]):
            model_embeds.append(Image(graph_paths["feature_importance"], width=230, height=172))
        else:
            unavailable_notes.append("Feature importance unavailable for this selected model.")

    if model_embeds:
        story.extend(_grid_layout(model_embeds))
    if unavailable_notes:
        story.append(Spacer(1, 6))
        for note in unavailable_notes:
            story.append(Paragraph(note, body_style))

    story.append(PageBreak()) # Move to AI explanation
    
    # --- PAGE 3+: AI EXPLANATIONS ---
    story.append(Paragraph(next_section("AI Agent Detailed Explanations"), h1_style))
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
        elif line_clean.startswith('- ') or line_clean.startswith('* '):
            # Flush any accumulated prose first, then render this bullet as
            # its own Paragraph — joining multiple bullet lines into one
            # blob (the old behavior) let clean_markdown_for_pdf's *italic*
            # regex pair one bullet's "*" with the next bullet's "*".
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
            
    # Dump remaining text block
    if current_text_block:
        full_para = " ".join(current_text_block)
        cleaned_para = clean_markdown_for_pdf(full_para)
        story.append(Paragraph(cleaned_para, body_style))
        
    # Build document
    doc.build(story)

    return output_pdf_path
