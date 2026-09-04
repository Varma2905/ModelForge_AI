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

from app.ml.feature_types import (
    map_expanded_coefficients,
    _is_linear_regression_model,
    _is_logistic_regression_model,
)

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

    # 2. Pull inline code spans out into placeholders BEFORE any bold/italic
    # conversion runs. Inline code content is never re-parsed for markdown
    # emphasis by any real markdown renderer, and skipping this step is
    # exactly what used to crash PDF generation: LLM-generated code like
    # `int(0.8*len(df))` uses "*" for multiplication, and with the code
    # spans still inline, the italic regex below would happily pair that
    # "*" with an unrelated "*" inside a LATER, separate code span (e.g.
    # two adjacent `...` snippets joined by "<br>"), producing an <i> tag
    # that opens inside one code span and closes inside another. Once both
    # spans are then independently wrapped in their own <font> tags, the
    # <i>/<font> tag pairs cross each other — invalid XML nesting ReportLab
    # rejects with "saw </font> instead of expected </i>". Extracting code
    # spans first means the "*" characters inside them are never visible to
    # the italic regex at all.
    code_spans: List[str] = []

    def _stash_code(match: "re.Match") -> str:
        code_spans.append(match.group(1))
        return f"\x00CODE{len(code_spans) - 1}\x00"

    text = re.sub(r'`(.*?)`', _stash_code, text)

    # 3. Replace headers
    text = re.sub(r'^###\s+(.*?)$', r'<b><font color="#1e3a8a">\1</font></b>', text, flags=re.MULTILINE)
    text = re.sub(r'^####\s+(.*?)$', r'<b><font color="#475569">\1</font></b>', text, flags=re.MULTILINE)
    text = re.sub(r'^##\s+(.*?)$', r'<b><font size="14" color="#1e3a8a">\1</font></b>', text, flags=re.MULTILINE)
    text = re.sub(r'^#\s+(.*?)$', r'<b><font size="16" color="#1e3a8a">\1</font></b>', text, flags=re.MULTILINE)

    # 4. Replace bold (only asterisks to avoid variable name underscore clashes)
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)

    # 5. Replace italic (only asterisks). Excludes commas/newlines from the
    # matched span — without this, LLM-generated prose using literal
    # asterisks for multiplication in a list (e.g. "a*b, a*c, b*c") lets the
    # non-greedy match span across unrelated pairs ("*b, a*" here), producing
    # unbalanced <i> tags that crash ReportLab's XML parser. Genuine italic
    # phrases essentially never contain a bare comma, so this is a safe
    # trade-off: crash prevention over converting that rare edge case. Code
    # spans are already stashed out at this point (step 2), so a "*" used
    # for multiplication inside one can no longer reach this regex at all.
    text = re.sub(r'\*([^*\n,]+?)\*', r'<i>\1</i>', text)

    # 6. Restore the stashed code spans as <font> tags now that bold/italic
    # conversion is done, so their content can never be split across an
    # emphasis tag boundary.
    for i, code in enumerate(code_spans):
        text = text.replace(f"\x00CODE{i}\x00", f'<font face="Courier">{code}</font>')

    return text


def _safe_paragraph(text: str, style: ParagraphStyle) -> Paragraph:
    """Constructs a ReportLab Paragraph from `text` (assumed to already have
    passed through clean_markdown_for_pdf), falling back to a plain-text
    (fully re-escaped, no inline markup at all) Paragraph if ReportLab's own
    XML-like parser still rejects the result — a defense-in-depth backstop
    against any markdown/tag-nesting edge case clean_markdown_for_pdf
    doesn't anticipate, since this renders free-form LLM-generated text.
    One malformed paragraph should degrade to plain text, not fail the
    entire report."""
    try:
        return Paragraph(text, style)
    except Exception as e:
        logger.warning(f"Paragraph markup rejected by ReportLab, falling back to plain text: {e}")
        plain = re.sub(r"<[^>]+>", "", text)
        plain = plain.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
        plain = plain.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        return Paragraph(plain, style)


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


MAX_FEATURES_IN_REPORT = 10
MAX_CATEGORIES_DISPLAYED = 10
MAX_TABLE_ROWS = 10

_MAX_STATS_TABLE_ROWS = MAX_TABLE_ROWS



def _build_stats_table(
    coefs, p_values, std_errors, t_stats,
    numeric_features, categorical_features, encoded_categorical_names,
    table_cell_style, table_cell_header_style, body_style,
    model_info: Dict[str, Any] = None,
    disclaimer_text: str = None,
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

    high_cardinality_features = {}
    if model_info and "dataset_profile" in model_info:
        for f in model_info["dataset_profile"].get("features", []):
            if f.get("kind") == "categorical" and f.get("unique_count", 0) > MAX_CATEGORIES_DISPLAYED:
                high_cardinality_features[f["name"]] = f["unique_count"]

    filtered_coefs = []
    filtered_p = []
    filtered_se = []
    filtered_t = []
    omitted_features = set()

    for i, entry in enumerate(mapped_coefs):
        src_feat = entry.get("source_feature")
        if src_feat in high_cardinality_features:
            omitted_features.add(src_feat)
        else:
            filtered_coefs.append(entry)
            filtered_p.append(p_by_idx[i] if i < len(p_by_idx) else None)
            filtered_se.append(se_by_idx[i] if i < len(se_by_idx) else None)
            filtered_t.append(t_by_idx[i] if i < len(t_by_idx) else None)

    total_rows = len(filtered_coefs)
    # Largest-magnitude coefficients first — the ones actually worth a
    # reader's attention — then cap to a report-sized page count.
    order = sorted(range(total_rows), key=lambda i: abs(filtered_coefs[i]["value"]) if filtered_coefs[i]["value"] is not None else -1, reverse=True)
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
        var_name = filtered_coefs[i]["feature"]
        coef = filtered_coefs[i]["value"]
        p_val = filtered_p[i] if i < len(filtered_p) else None
        std_err = filtered_se[i] if i < len(filtered_se) else None
        t_stat = filtered_t[i] if i < len(filtered_t) else None
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

    flow: List = []
    if disclaimer_text:
        flow.append(Paragraph(f"<i>{disclaimer_text}</i>", body_style))
        flow.append(Spacer(1, 6))
    if total_rows > 0:
        t_stats_table = Table(stats_rows, colWidths=[120, 75, 75, 75, 75, 80])
        t_stats_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#475569')),
            ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#475569')),
            ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
            ('TOPPADDING', (0,0), (-1,-1), 4),
            ('BOTTOMPADDING', (0,0), (-1,-1), 4),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ]))
        flow.append(t_stats_table)

    for feat in sorted(list(omitted_features)):
        unique_val = high_cardinality_features[feat]
        flow.append(Spacer(1, 4))
        flow.append(Paragraph(
            f"<i><b>{feat}</b> is a high-cardinality categorical feature with {unique_val} unique values. "
            "Individual encoded coefficients are omitted from the report to maintain readability.</i>",
            body_style,
        ))

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
_MAX_FEATURE_SUMMARY_ROWS = MAX_TABLE_ROWS



def _raw_feature_count(model_info: Dict[str, Any]) -> int:
    """Raw, user-selected feature count — prefers the persisted
    `feature_counts.raw_selected` (see training_routes.py/classify_routes.py/
    cluster_routes.py train_model) and falls back to len(features) for any
    model trained before that field existed."""
    feature_counts = model_info.get("feature_counts") or {}
    if isinstance(feature_counts.get("raw_selected"), int):
        return feature_counts["raw_selected"]
    return len(model_info.get("features", []) or [])


def _encoded_feature_count_display(model_info: Dict[str, Any]) -> str:
    """One-hot-expanded/final model input dimension count — distinct from
    `_raw_feature_count` above so a report can show both instead of
    conflating "features selected" with "columns actually fed to the
    model" (see feature_counts persisted at train time). "N/A" for any
    model trained before this field existed, rather than guessing."""
    feature_counts = model_info.get("feature_counts") or {}
    encoded = feature_counts.get("encoded_final")
    return str(encoded) if isinstance(encoded, int) else "N/A"


def _build_data_quality_notices(model_info: Dict[str, Any], styles: Dict[str, ParagraphStyle]) -> List:
    """Non-fatal Data Quality Notices panel: surfaces warning-severity
    validation issues (report_validation.py, run in report_routes.py before
    this file is ever called) plus the target-leakage/identifier-column
    warnings persisted at train time (feature_types.py's
    detect_target_leakage/detect_identifier_target). Renders nothing when
    there's nothing to say — same "no empty sections" convention as the
    rest of this file's optional blocks."""
    body_style = styles["body"]
    warning_issues = [
        i for i in (model_info.get("validation_issues") or [])
        if i.get("severity") == "warning"
    ]
    leakage_warnings = model_info.get("leakage_warnings") or []
    identifier_warnings = model_info.get("identifier_warnings") or []

    notices: List[str] = []
    notices.extend(i["message"] for i in warning_issues)
    notices.extend(identifier_warnings)
    notices.extend(w["detail"] if isinstance(w, dict) else str(w) for w in leakage_warnings)

    if not notices:
        return []

    flow: List = []
    flow.append(Paragraph("<b>Data Quality Notices</b>", body_style))
    flow.append(Spacer(1, 4))
    for notice in notices:
        flow.append(Paragraph(f"• <font color='#b45309'>{notice}</font>", body_style))
    flow.append(Spacer(1, 12))
    return flow


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
            Paragraph(str(_raw_feature_count(model_info)), table_cell_style),
        ],
        [
            Paragraph("<b>Missing Values</b>", table_cell_style),
            Paragraph(f"{profile.get('missing_values_total', 0):,}", table_cell_style),
            Paragraph("<b>Duplicate Rows</b>", table_cell_style),
            Paragraph(f"{profile.get('duplicate_rows', 0):,}", table_cell_style),
        ],
        [
            Paragraph("<b>Encoded/Model Input Dimensions</b>", table_cell_style),
            Paragraph(_encoded_feature_count_display(model_info), table_cell_style),
            Paragraph("", table_cell_style),
            Paragraph("", table_cell_style),
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
    real_model_name = model_info.get("model", "")
    is_logistic = _is_logistic_regression_model(real_model_name)
    section_title = (
        "Coefficients (log-odds scale)" if is_logistic
        else "Additional Statistical Analysis (Auxiliary Model: Logistic Regression)"
    )
    disclaimer = None
    if not is_logistic:
        disclaimer = (
            f"This section fits an auxiliary Logistic/Multinomial Logistic Regression model to the same "
            f"data to provide interpretable coefficients on the log-odds scale. It is a supplementary "
            f"statistical view, not the actual trained {real_model_name or 'model'} used for the "
            "predictions and metrics above."
        )
    flow.append(Paragraph(section_title, body_style))
    flow.append(Spacer(1, 6))
    flow.extend(_build_stats_table(
        stats.get("coefficients", {}), stats.get("p_values", {}),
        stats.get("standard_errors", {}), stats.get("t_statistics", {}),
        numeric_features, categorical_features, None,
        table_cell_style, table_cell_header_style, body_style,
        model_info=model_info,
        disclaimer_text=disclaimer,
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


# _is_linear_regression_model / _is_logistic_regression_model now live in
# app.ml.feature_types (imported above) so the explanation agents can share
# them too, instead of each module keeping its own copy of the name-matching
# logic.


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


def _build_classification_ai_insights_text(model_info: Dict[str, Any]) -> List[str]:
    """Computes the 'AI Agent Insights' narrative DIRECTLY from real
    metrics — overall performance, strongest/weakest class by F1, key
    misclassifications, feature limitations, class-balance check, and
    improvement suggestions. Deliberately does NOT depend on parsing
    free-form LLM markdown (which may use any heading structure the model
    chooses) — that fragile substring-matching is what previously fell back
    to a generic "performance is steady across classes" sentence whenever
    the LLM's own prose didn't happen to contain a matching heading. This
    function is always accurate and never generic when the per-class
    metrics actually differ, and never claims class imbalance unless the
    real per-class support numbers support it.
    """
    metrics = model_info.get("metrics", {}) or {}
    accuracy = metrics.get("Accuracy")
    classes = [str(c) for c in (metrics.get("Classes") or [])]
    class_report = metrics.get("ClassificationReport") or {}
    conf_matrix = metrics.get("ConfusionMatrix") or []

    # Every line below uses **markdown** bold (never a raw <b> tag) so it can
    # go through clean_markdown_for_pdf() like the rest of this file — class
    # names come from the user's own uploaded data and must be HTML-escaped
    # before any ReportLab markup is applied, or a class label containing
    # "&"/"<"/">" would crash the PDF's XML parser.
    lines: List[str] = []

    # 1. Overall model performance
    if isinstance(accuracy, (int, float)):
        if accuracy >= 0.90:
            lines.append(f"**Overall performance:** the model performs excellently, correctly classifying {accuracy:.1%} of test samples.")
        elif accuracy >= 0.75:
            lines.append(f"**Overall performance:** the model performs well, correctly classifying {accuracy:.1%} of test samples.")
        elif accuracy >= 0.60:
            lines.append(f"**Overall performance:** the model performs moderately, correctly classifying only {accuracy:.1%} of test samples — there is meaningful room for improvement.")
        else:
            lines.append(f"**Overall performance:** the model performs weakly, correctly classifying only {accuracy:.1%} of test samples.")
    else:
        lines.append("**Overall performance:** accuracy was not available for this model.")

    # 2. Strongest / weakest class by F1-score — never a generic "steady
    # across classes" statement when the real per-class F1 values differ.
    f1_by_class = [
        (cls, class_report.get(cls, {}).get("f1-score"))
        for cls in classes
        if isinstance(class_report.get(cls, {}).get("f1-score"), (int, float))
    ]
    weakest_cls = None
    f1_gap = 0.0
    if f1_by_class:
        strongest_cls, strongest_f1 = max(f1_by_class, key=lambda x: x[1])
        weakest_cls, weakest_f1 = min(f1_by_class, key=lambda x: x[1])
        f1_gap = strongest_f1 - weakest_f1
        if len(f1_by_class) > 1 and f1_gap > 0.05:
            lines.append(
                f"**Strongest class:** {strongest_cls} (F1 = {strongest_f1:.3f}). "
                f"**Weakest class:** {weakest_cls} (F1 = {weakest_f1:.3f}) — a real gap of "
                f"{f1_gap:.3f}, not noise. Predictions for {weakest_cls} should be treated with extra caution."
            )
        else:
            lines.append(
                f"**Class-level performance** is relatively even: the strongest class is "
                f"{strongest_cls} (F1 = {strongest_f1:.3f}) and the weakest is {weakest_cls} "
                f"(F1 = {weakest_f1:.3f})."
            )

    # 3. Important misclassifications, straight from the confusion matrix.
    if conf_matrix and classes and len(conf_matrix) == len(classes):
        confusions = []
        for i, row in enumerate(conf_matrix):
            for j, count in enumerate(row):
                if i != j and count > 0:
                    confusions.append((classes[i], classes[j], count))
        confusions.sort(key=lambda x: x[2], reverse=True)
        if confusions:
            top = confusions[:3]
            conf_text = "; ".join(f"{a} misclassified as {p} ({c} cases)" for a, p, c in top)
            lines.append(f"**Important misclassifications:** {conf_text}.")
        else:
            lines.append("**Important misclassifications:** none — the confusion matrix shows zero errors on the test set.")

    # 4. Possible feature limitations — grounded in the actual F1 gap /
    # overall accuracy, not a boilerplate sentence.
    if weakest_cls and f1_gap > 0.05:
        lines.append(
            f"**Possible feature limitations:** the gap in performance for {weakest_cls} suggests the "
            "current feature set may not fully capture what distinguishes it from the other classes — "
            f"consider adding features specific to {weakest_cls}'s behavior, or reviewing whether its "
            "samples overlap heavily with another class in feature space."
        )
    elif isinstance(accuracy, (int, float)) and accuracy < 0.75:
        lines.append(
            "**Possible feature limitations:** the relatively low overall accuracy suggests the current "
            "feature set may not carry enough signal to separate the classes reliably — consider adding "
            "more discriminative features."
        )

    # 5. Class balance — only claimed when the real per-class support
    # numbers actually show a meaningful skew; otherwise say so explicitly.
    support_by_class = [
        class_report.get(cls, {}).get("support")
        for cls in classes
        if isinstance(class_report.get(cls, {}).get("support"), (int, float))
    ]
    if len(support_by_class) > 1:
        ratio = max(support_by_class) / max(min(support_by_class), 1)
        if ratio > 3:
            lines.append(
                f"**Class distribution:** the classes are noticeably imbalanced in the test set (the "
                f"largest class is {ratio:.1f}x the size of the smallest), which can suppress recall on the "
                "smaller class(es) regardless of feature quality."
            )
        elif isinstance(accuracy, (int, float)) and accuracy < 0.75:
            lines.append(
                "**Class distribution:** the class distribution is relatively balanced, so the low "
                "performance is more likely related to insufficient predictive signal, feature "
                "representation, or model configuration rather than severe class imbalance."
            )

    # 6. Practical, specific improvement suggestions.
    suggestions = []
    if weakest_cls:
        suggestions.append(
            f"collect more labeled examples for {weakest_cls}, or engineer features that specifically help "
            "separate it from the class(es) it is most often confused with"
        )
    if isinstance(accuracy, (int, float)) and accuracy < 0.75:
        suggestions.append("try a different algorithm and compare per-class F1, not just overall accuracy")
    suggestions.append("review the Confusion Matrix and Per-Class Performance sections above for the exact numbers behind these observations")
    if suggestions:
        lines.append("**Practical improvement suggestions:** " + "; ".join(suggestions) + ".")

    return lines


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

    is_classification_model = model_info.get("model_type", "regression") == "classification"
    if is_classification_model:
        story = []
        
        # --- PAGE 1: TITLE & COVER & EXECUTIVE SUMMARY ---
        story.append(Paragraph("MODELFORGE AI STUDIO", title_style))
        story.append(Paragraph(f"Classification Analytical Report | Model ID: {model_id}", subtitle_style))
        story.append(Spacer(1, 20))
        story.append(Paragraph("Executive Summary", h1_style))
        
        exec_summary_text = ""
        if "## Executive Summary" in ai_report_markdown:
            parts = ai_report_markdown.split("## Executive Summary")
            if len(parts) > 1:
                content_after = parts[1]
                next_heading_idx = content_after.find("\n## ")
                if next_heading_idx != -1:
                    exec_summary_text = content_after[:next_heading_idx].strip()
                else:
                    exec_summary_text = content_after.strip()
        
        if not exec_summary_text:
            accuracy = model_info.get("metrics", {}).get("Accuracy", 0.0)
            exec_summary_text = (
                f"This document provides a comprehensive report of the machine learning classification analysis "
                f"carried out using a <b>{model_info.get('model', 'N/A')}</b> algorithm. "
                f"The model achieved a classification accuracy of <b>{accuracy:.2%}</b>. "
                f"Detailed explanations of the dataset structure, model methodology, statistical calculations, "
                f"and engineering recommendations are compiled below by the AI Agent Network."
            )
        
        for line in exec_summary_text.split('\n'):
            line_clean = line.strip()
            if not line_clean:
                continue
            if line_clean.startswith('- ') or line_clean.startswith('* '):
                story.append(_safe_paragraph(f"• {clean_markdown_for_pdf(line_clean[2:])}", body_style))
            else:
                story.append(_safe_paragraph(clean_markdown_for_pdf(line_clean), body_style))

        story.append(Spacer(1, 10))
        story.extend(_build_data_quality_notices(model_info, s))
        story.append(PageBreak())

        # --- PAGE 2: DATASET OVERVIEW ---
        dataset_profile = model_info.get("dataset_profile")
        if dataset_profile:
            story.append(Paragraph(next_section("Dataset Overview"), h1_style))
            story.extend(_build_dataset_overview_section(dataset_profile, model_info, s))
            story.append(Spacer(1, 15))
            
            story.append(Paragraph(next_section("Feature Summary (Top Columns)"), h1_style))
            story.extend(_build_feature_summary_section(dataset_profile, model_info, s))
            story.append(Spacer(1, 15))
            story.append(PageBreak())

        # --- PAGE 3: DATA QUALITY & PREPROCESSING ---
        story.append(Paragraph(next_section("Data Quality & Preprocessing"), h1_style))
        preprocessing_cfg = model_info.get("preprocessing", {}) or {}
        dq_rows = [
            [
                Paragraph("<b>Total Rows</b>", table_cell_style),
                Paragraph(f"{dataset_profile.get('total_rows', 0):,}" if dataset_profile else "N/A", table_cell_style),
                Paragraph("<b>Total Columns</b>", table_cell_style),
                Paragraph(f"{dataset_profile.get('total_columns', 0):,}" if dataset_profile else "N/A", table_cell_style),
            ],
            [
                Paragraph("<b>Total Missing Values</b>", table_cell_style),
                Paragraph(f"{dataset_profile.get('missing_values_total', 0):,}" if dataset_profile else "N/A", table_cell_style),
                Paragraph("<b>Duplicate Rows</b>", table_cell_style),
                Paragraph(f"{dataset_profile.get('duplicate_rows', 0):,}" if dataset_profile else "N/A", table_cell_style),
            ],
            [
                Paragraph("<b>Missing Value Strategy</b>", table_cell_style),
                Paragraph(str(preprocessing_cfg.get("missing", "none")), table_cell_style),
                Paragraph("<b>Outlier Filtering</b>", table_cell_style),
                Paragraph(str(preprocessing_cfg.get("outlier", "none")), table_cell_style),
            ],
            [
                Paragraph("<b>Feature Scaling</b>", table_cell_style),
                Paragraph(str(preprocessing_cfg.get("scaling", "none")), table_cell_style),
                Paragraph("<b>Encoded Target</b>", table_cell_style),
                Paragraph(str(model_info.get("target", "N/A")), table_cell_style),
            ]
        ]
        t_dq = Table(dq_rows, colWidths=[120, 130, 120, 130])
        t_dq.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f8fafc')),
            ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#e2e8f0')),
            ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
            ('TOPPADDING', (0,0), (-1,-1), 6),
            ('BOTTOMPADDING', (0,0), (-1,-1), 6),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ]))
        story.append(t_dq)
        story.append(PageBreak())

        # --- PAGE 4: MODEL CONFIGURATION ---
        story.append(Paragraph(next_section("Model Configuration"), h1_style))
        split_cfg = model_info.get("split", {}) or {}
        test_pct = round(split_cfg.get("test_size", 0.2) * 100)
        val_pct = round(split_cfg.get("val_size", 0.0) * 100)
        train_pct = 100 - test_pct - val_pct
        split_label = f"{train_pct}% Train / {test_pct}% Test"
        if val_pct > 0:
            split_label += f" / {val_pct}% Validation"

        config_data = [
            [
                Paragraph("<b>Selected Algorithm</b>", table_cell_style),
                Paragraph(str(model_info.get("model", "N/A")), table_cell_style),
                Paragraph("<b>Features Used</b>", table_cell_style),
                Paragraph(str(len(model_info.get("features", []))), table_cell_style)
            ],
            [
                Paragraph("<b>Dataset Split Ratio</b>", table_cell_style),
                Paragraph(split_label, table_cell_style),
                Paragraph("<b>Hyperparameters</b>", table_cell_style),
                Paragraph(str(model_info.get("hyperparameters", "default")), table_cell_style)
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
        story.append(PageBreak())

        # --- PAGE 5-6: MODEL EVALUATION METRICS ---
        story.append(Paragraph(next_section("Model Evaluation Metrics"), h1_style))
        metrics = model_info.get("metrics", {})
        metrics_data = [
            [Paragraph("Metric", table_cell_header_style), Paragraph("Value", table_cell_header_style), Paragraph("Description", table_cell_header_style)]
        ]
        for m_name in ["Accuracy", "Precision", "Recall", "F1", "BalancedAccuracy", "ROC_AUC", "LogLoss", "MCC", "Specificity"]:
            if m_name in metrics and metrics[m_name] is not None:
                desc = descriptions.get(m_name, "Model evaluation metric.")
                metrics_data.append([
                    Paragraph(f"<b>{m_name}</b>", table_cell_style),
                    Paragraph(f"{metrics[m_name]:,.4f}" if isinstance(metrics[m_name], (int, float)) else str(metrics[m_name]), table_cell_style),
                    Paragraph(desc, table_cell_style)
                ])
        t_metrics = Table(metrics_data, colWidths=[120, 80, 300])
        t_metrics.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1e3a8a')),
            ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#1e3a8a')),
            ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
            ('TOPPADDING', (0,0), (-1,-1), 5),
            ('BOTTOMPADDING', (0,0), (-1,-1), 5),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ]))
        story.append(t_metrics)
        story.append(PageBreak())

        # --- PAGE 7: CONFUSION MATRIX ---
        story.append(Paragraph(next_section("Confusion Matrix"), h1_style))
        classes = [str(c) for c in (metrics.get("Classes") or model_info.get("classes") or [])]
        conf_matrix = metrics.get("ConfusionMatrix") or []
        if conf_matrix and classes and len(conf_matrix) == len(classes):
            if len(classes) <= MAX_CATEGORIES_DISPLAYED:
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
                    ('BOX', (0, 0), (-1,-1), 1, colors.HexColor('#1e3a8a')),
                    ('INNERGRID', (0, 0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
                    ('TOPPADDING', (0, 0), (-1,-1), 5),
                    ('BOTTOMPADDING', (0, 0), (-1,-1), 5),
                    ('VALIGN', (0, 0), (-1,-1), 'MIDDLE'),
                ]))
                story.append(t_cm)
            else:
                story.append(Paragraph(
                    f"<i>Confusion matrix table is omitted from the report as the target variable has {len(classes)} classes. "
                    "Please refer to the interactive dashboard for class-level prediction details.</i>",
                    body_style,
                ))
            story.append(Spacer(1, 15))
        
        if "confusion_matrix" in graph_paths and os.path.exists(graph_paths["confusion_matrix"]):
            story.append(Image(graph_paths["confusion_matrix"], width=230, height=172))
        story.append(PageBreak())

        # --- PAGE 8: PER-CLASS PERFORMANCE ---
        story.append(Paragraph(next_section("Per-Class Performance"), h1_style))
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
        shown_classes = classes[:MAX_TABLE_ROWS]
        for cls in shown_classes:
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
            story.append(t_per_class)
            
            if len(classes) > MAX_TABLE_ROWS:
                story.append(Spacer(1, 4))
                story.append(Paragraph(
                    f"<i>Showing top {MAX_TABLE_ROWS} of {len(classes)} classes. Remaining class metrics are omitted from the report to maintain readability.</i>",
                    body_style,
                ))
            story.append(Spacer(1, 15))

        story.append(PageBreak())

        # --- PAGES 9-10: EXPLORATORY DATA ANALYSIS / VISUALIZATIONS ---
        # Target/Class Distribution now lives here (moved from the
        # Per-Class Performance page above) alongside the other genuinely
        # relevant EDA plots — Numerical Feature Distribution, Feature vs
        # Target (classification-only, see ensure_graphs), and Correlation
        # Heatmap. Each is embedded only if it actually exists (graceful
        # skip, never an empty/broken chart) — see generate_dataset_graphs.
        story.append(Paragraph(next_section("Exploratory Data Analysis"), h1_style))
        dataset_embeds = []
        for key in ("target_distribution", "distribution", "feature_vs_target", "correlation_heatmap", "boxplot", "categorical_frequency"):
            if key in graph_paths and graph_paths[key] and os.path.exists(graph_paths[key]):
                dataset_embeds.append(Image(graph_paths[key], width=230, height=172))
        if dataset_embeds:
            story.extend(_grid_layout(dataset_embeds))
        else:
            story.append(Paragraph("No exploratory visualizations could be generated for this dataset/feature combination.", body_style))
        story.append(PageBreak())

        # --- PAGE 11: FEATURE IMPORTANCE / EXPLAINABILITY ---
        # Uses `visualizations.feature_importance` — computed from the
        # ACTUAL trained sklearn model's own `feature_importances_` (tree
        # ensembles) or `coef_` (linear models), see
        # classification_evaluation.py's extract_feature_importance(). That
        # function already returns None, honestly, for models with neither
        # attribute (KNN, Gaussian Naive Bayes, non-linear-kernel SVM) — so
        # this section only ever shows a table backed by the real model,
        # never a fabricated one, and never the auxiliary statsmodels-fit
        # coefficients used elsewhere purely for p-value significance.
        story.append(Paragraph(next_section("Feature Importance & Explainability"), h1_style))
        visualizations = model_info.get("visualizations") or {}
        fi = visualizations.get("feature_importance") if isinstance(visualizations, dict) else None
        fi_features = (fi or {}).get("features") or []
        fi_values = (fi or {}).get("importance") or []

        if fi_features and fi_values:
            top_n = min(10, len(fi_features))
            story.append(Paragraph(
                f"Top {top_n} Feature Importance — computed from this model's own trained parameters.",
                body_style,
            ))
            story.append(Spacer(1, 6))
            fi_rows = [[
                Paragraph("Rank", table_cell_header_style),
                Paragraph("Feature", table_cell_header_style),
                Paragraph("Importance", table_cell_header_style),
            ]]
            for idx, (feat, val) in enumerate(zip(fi_features[:top_n], fi_values[:top_n]), start=1):
                fi_rows.append([
                    Paragraph(str(idx), table_cell_style),
                    Paragraph(str(feat), table_cell_style),
                    Paragraph(f"{val:,.4f}" if isinstance(val, (int, float)) else "N/A", table_cell_style),
                ])
            t_fi = Table(fi_rows, colWidths=[50, 300, 150])
            t_fi.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e3a8a')),
                ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#1e3a8a')),
                ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
                ('TOPPADDING', (0, 0), (-1, -1), 5),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ]))
            story.append(t_fi)
        else:
            story.append(Paragraph("Feature importance is not directly available for this model.", body_style))
        story.append(PageBreak())

        # --- PAGE 12: ERROR ANALYSIS ---
        story.append(Paragraph(next_section("Error Analysis"), h1_style))
        error_analysis_text = ""
        if conf_matrix and classes and len(conf_matrix) == len(classes):
            worst_confusions = []
            for i, row in enumerate(conf_matrix):
                for j, count in enumerate(row):
                    if i != j and count > 0:
                        worst_confusions.append((classes[i], classes[j], count))
            worst_confusions.sort(key=lambda x: x[2], reverse=True)
            if worst_confusions:
                error_analysis_text = "Based on the confusion matrix, here are the most confused classes:\n\n"
                for act, pred, cnt in worst_confusions[:3]:
                    error_analysis_text += f"- Class **`{act}`** was misclassified as **`{pred}`** {cnt} times.\n"
            else:
                error_analysis_text = "The confusion matrix shows zero errors on the test set. The model classifies all test samples correctly.\n"
        
        for line in error_analysis_text.split('\n'):
            if line.strip():
                story.append(_safe_paragraph(clean_markdown_for_pdf(line), body_style))
        story.append(PageBreak())

        # --- PAGE 13: AI INSIGHTS ---
        # Built deterministically from real metrics (see
        # _build_classification_ai_insights_text) rather than parsed out of
        # free-form LLM markdown — the LLM's own heading structure isn't
        # guaranteed, and a failed parse used to silently fall back to a
        # generic "performance is steady across classes" sentence even when
        # the real per-class metrics differed meaningfully.
        story.append(Paragraph(next_section("AI Agent Insights"), h1_style))
        for line in _build_classification_ai_insights_text(model_info):
            story.append(_safe_paragraph(clean_markdown_for_pdf(line), body_style))
            story.append(Spacer(1, 4))
        story.append(PageBreak())

        # --- PAGE 14: RECOMMENDATIONS ---
        story.append(Paragraph(next_section("Recommendations"), h1_style))
        recs_text = ""
        if "## Diagnostic Recommendations" in ai_report_markdown:
            parts = ai_report_markdown.split("## Diagnostic Recommendations")
            recs_text = parts[1].strip()
        elif "## Recommendations" in ai_report_markdown:
            parts = ai_report_markdown.split("## Recommendations")
            recs_text = parts[1].strip()
        
        if not recs_text:
            recs_text = (
                "- Try tree-based ensembling or hyperparameter fine-tuning.\n"
                "- Address class imbalance if the distribution is highly skewed."
            )
            
        for line in recs_text.split('\n'):
            line_clean = line.strip()
            if not line_clean or line_clean.startswith('|') or line_clean.startswith('+--'):
                continue
            if line_clean.startswith('- ') or line_clean.startswith('* '):
                story.append(_safe_paragraph(f"• {clean_markdown_for_pdf(line_clean[2:])}", body_style))
            else:
                story.append(_safe_paragraph(clean_markdown_for_pdf(line_clean), body_style))
                
        doc.build(story)
        return output_pdf_path

    # --- PAGE 1: TITLE & METRICS OVERVIEW ---

    story.append(Paragraph("MODELFORGE AI STUDIO", title_style))
    story.append(Paragraph(f"Analytical Report for Model ID: {model_id} | Generated on behalf of User Analysis History", subtitle_style))
    story.append(Spacer(1, 10))
    story.extend(_build_data_quality_notices(model_info, s))

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
        real_model_name = model_info.get("model", "")
        is_linear = _is_linear_regression_model(real_model_name)
        section_title = (
            "Statistical Parameter Analysis" if is_linear
            else "Additional Statistical Analysis (Auxiliary Model: OLS)"
        )
        disclaimer = None
        if not is_linear:
            disclaimer = (
                f"This section fits an auxiliary Ordinary Least Squares (OLS) model to the same data to "
                f"provide interpretable coefficients, standard errors, and p-values. It is a supplementary "
                f"statistical view, not the actual trained {real_model_name or 'model'} used for the "
                "predictions and metrics above."
            )
        story.append(Paragraph(next_section(section_title), h1_style))
        stats = model_info.get("statistical_analysis", {})
        features = model_info.get("features", []) or []
        categorical_features = model_info.get("categorical_features", []) or []
        numeric_features = model_info.get("numerical_features", features) or features
        story.extend(_build_stats_table(
            stats.get("coefficients", {}), stats.get("p_values", {}),
            stats.get("standard_errors", {}), stats.get("t_statistics", {}),
            numeric_features, categorical_features, None,
            table_cell_style, table_cell_header_style, body_style,
            model_info=model_info,
            disclaimer_text=disclaimer,
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
                story.append(_safe_paragraph(cleaned_para, body_style))
                current_text_block = []
        elif line_clean.startswith('#') or line_clean.startswith('---'):
            # Clear text block first
            if current_text_block:
                full_para = " ".join(current_text_block)
                cleaned_para = clean_markdown_for_pdf(full_para)
                story.append(_safe_paragraph(cleaned_para, body_style))
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
                story.append(_safe_paragraph(cleaned_header, h1_style))
        elif line_clean.startswith('- ') or line_clean.startswith('* '):
            # Flush any accumulated prose first, then render this bullet as
            # its own Paragraph — joining multiple bullet lines into one
            # blob (the old behavior) let clean_markdown_for_pdf's *italic*
            # regex pair one bullet's "*" with the next bullet's "*".
            if current_text_block:
                full_para = " ".join(current_text_block)
                cleaned_para = clean_markdown_for_pdf(full_para)
                story.append(_safe_paragraph(cleaned_para, body_style))
                current_text_block = []
            bullet_text = line_clean[2:].strip()
            cleaned_bullet = clean_markdown_for_pdf(bullet_text)
            story.append(_safe_paragraph(f"• {cleaned_bullet}", body_style))
        else:
            current_text_block.append(line_clean)
            
    # Dump remaining text block
    if current_text_block:
        full_para = " ".join(current_text_block)
        cleaned_para = clean_markdown_for_pdf(full_para)
        story.append(_safe_paragraph(cleaned_para, body_style))
        
    # Build document
    doc.build(story)

    return output_pdf_path
