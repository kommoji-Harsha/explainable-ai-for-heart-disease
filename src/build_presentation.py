import os
import pptx
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE
from pptx.dml.color import RGBColor

# Colors (Hex, no '#')
NAVY = RGBColor(0x21, 0x29, 0x5C)       # Dark background
PRIMARY = RGBColor(0x06, 0x5A, 0x82)    # Deep blue
SECONDARY = RGBColor(0x1C, 0x72, 0x93)  # Teal
ACCENT = RGBColor(0xD9, 0x8E, 0x04)     # Amber / Accent
LIGHT_BG = RGBColor(0xF7, 0xFA, 0xFC)   # Light neutral background
WHITE = RGBColor(0xFF, 0xFF, 0xFF)      # White
TEXT_DARK = RGBColor(0x1F, 0x29, 0x37)  # Dark text
TEXT_MUTED = RGBColor(0x64, 0x74, 0x8B) # Muted text
RED = RGBColor(0xDC, 0x26, 0x26)        # Alarm red
GREEN = RGBColor(0x16, 0xA3, 0x4A)      # Success green

FONT_HEADING = "Cambria"
FONT_BODY = "Calibri"


def create_presentation():
    prs = Presentation()
    prs.slide_width = Inches(13.33)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    def add_blank_slide(is_dark=False):
        slide = prs.slides.add_slide(blank_layout)
        bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.33), Inches(7.5))
        bg.fill.solid()
        bg.fill.fore_color.rgb = NAVY if is_dark else LIGHT_BG
        bg.line.fill.background()
        return slide

    def add_header(slide, kicker, title, is_dark=False):
        txBox = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.73), Inches(1.1))
        tf = txBox.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

        p1 = tf.paragraphs[0]
        p1.text = kicker.upper()
        p1.font.size = Pt(11)
        p1.font.bold = True
        p1.font.name = FONT_BODY
        p1.font.color.rgb = ACCENT if is_dark else SECONDARY

        p2 = tf.add_paragraph()
        p2.text = title
        p2.font.size = Pt(26)
        p2.font.bold = True
        p2.font.name = FONT_HEADING
        p2.font.color.rgb = WHITE if is_dark else TEXT_DARK
        p2.space_before = Pt(4)

    def add_page_number(slide, num, total=17, is_dark=False):
        txBox = slide.shapes.add_textbox(Inches(12.0), Inches(7.0), Inches(1.0), Inches(0.4))
        tf = txBox.text_frame
        p = tf.paragraphs[0]
        p.text = f"{num}/{total}"
        p.alignment = PP_ALIGN.RIGHT
        p.font.size = Pt(10)
        p.font.name = FONT_BODY
        p.font.color.rgb = SECONDARY if is_dark else TEXT_MUTED

    def add_card(slide, left, top, width, height, bg_color=WHITE, border_color=None):
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height))
        card.fill.solid()
        card.fill.fore_color.rgb = bg_color
        if border_color:
            card.line.color.rgb = border_color
            card.line.width = Pt(1.5)
        else:
            card.line.fill.background()
        return card

    def add_icon(slide, left, top, circle_color, icon_str, icon_size=18):
        circle = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(left), Inches(top), Inches(0.45), Inches(0.45))
        circle.fill.solid()
        circle.fill.fore_color.rgb = circle_color
        circle.line.fill.background()

        tx = slide.shapes.add_textbox(Inches(left), Inches(top - 0.02), Inches(0.45), Inches(0.45))
        tf = tx.text_frame
        p = tf.paragraphs[0]
        p.text = icon_str
        p.alignment = PP_ALIGN.CENTER
        p.font.size = Pt(icon_size)

    # -------------------------------------------------------------
    # SLIDE 1: Title (Dark)
    # -------------------------------------------------------------
    s1 = add_blank_slide(is_dark=True)
    card1 = add_card(s1, 1.0, 1.2, 11.33, 5.1, bg_color=NAVY)

    txBox = s1.shapes.add_textbox(Inches(1.2), Inches(1.8), Inches(10.93), Inches(3.8))
    tf = txBox.text_frame
    tf.word_wrap = True

    p0 = tf.paragraphs[0]
    p0.text = "CLINICAL MACHINE LEARNING RESEARCH"
    p0.font.size = Pt(12)
    p0.font.bold = True
    p0.font.color.rgb = ACCENT
    p0.font.name = FONT_BODY

    p1 = tf.add_paragraph()
    p1.text = "Optimized, Explainable & Reliable\nEnsemble Framework"
    p1.font.size = Pt(38)
    p1.font.bold = True
    p1.font.color.rgb = WHITE
    p1.font.name = FONT_HEADING
    p1.space_before = Pt(12)

    p2 = tf.add_paragraph()
    p2.text = "Extending Mienye & Jere (2024) with rigorous reliability engineering, multi-site fairness analysis, genuine cross-population validation, and layered explainability."
    p2.font.size = Pt(15)
    p2.font.color.rgb = LIGHT_BG
    p2.font.name = FONT_BODY
    p2.space_before = Pt(16)

    p3 = tf.add_paragraph()
    p3.text = "BTech Final Year Project — Research & Engineering Report"
    p3.font.size = Pt(12)
    p3.font.bold = True
    p3.font.color.rgb = SECONDARY
    p3.font.name = FONT_BODY
    p3.space_before = Pt(28)

    # -------------------------------------------------------------
    # SLIDE 2: Agenda (Light)
    # -------------------------------------------------------------
    s2 = add_blank_slide(is_dark=False)
    add_header(s2, "OVERVIEW", "What We'll Cover")
    add_page_number(s2, 2)

    agenda_items = [
        ("🎯", "1. Motivation & Reference", "Why 90%+ accuracy alone fails clinical trust & our foundation"),
        ("🌐", "2. Datasets & Cohorts", "Combined 920-patient 4-site cohort & Framingham validation"),
        ("⚡", "3. Methodology Pipeline", "Optuna tuning, soft-voting ensemble & leak-free nested CV"),
        ("🚀", "4. Our Contributions", "Nested CV, calibration, fairness, agreement, external testing"),
        ("🔬", "5. Methodological Rigor", "3 real bugs caught and fixed during rigorous pipeline review"),
        ("📊", "6. Results & Conclusion", "Key findings, generalization performance, and deployed deliverables")
    ]

    for idx, (icon, title, desc) in enumerate(agenda_items):
        col = idx % 3
        row = idx // 3
        left = 0.8 + col * 3.95
        top = 1.7 + row * 2.5
        add_card(s2, left, top, 3.7, 2.2)
        add_icon(s2, left + 0.3, top + 0.3, PRIMARY, icon)

        tx = s2.shapes.add_textbox(Inches(left + 0.3), Inches(top + 0.9), Inches(3.1), Inches(1.1))
        tf = tx.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

        p1 = tf.paragraphs[0]
        p1.text = title
        p1.font.size = Pt(15)
        p1.font.bold = True
        p1.font.name = FONT_HEADING
        p1.font.color.rgb = TEXT_DARK

        p2 = tf.add_paragraph()
        p2.text = desc
        p2.font.size = Pt(12)
        p2.font.name = FONT_BODY
        p2.font.color.rgb = TEXT_MUTED
        p2.space_before = Pt(6)

    # -------------------------------------------------------------
    # SLIDE 3: Motivation (Light)
    # -------------------------------------------------------------
    s3 = add_blank_slide(is_dark=False)
    add_header(s3, "CLINICAL MACHINE LEARNING", "Why This Project Matters")
    add_page_number(s3, 3)

    # Left Box
    add_card(s3, 0.8, 1.7, 5.6, 5.0)
    tx = s3.shapes.add_textbox(Inches(1.1), Inches(2.0), Inches(5.0), Inches(4.4))
    tf = tx.text_frame
    tf.word_wrap = True

    p = tf.paragraphs[0]
    p.text = "Beyond Standard Accuracy"
    p.font.size = Pt(20)
    p.font.bold = True
    p.font.name = FONT_HEADING
    p.font.color.rgb = PRIMARY

    p2 = tf.add_paragraph()
    p2.text = "In clinical medicine, high accuracy alone is insufficient. A model reporting 90%+ accuracy can still fail catastrophically if its predictions are overconfident, biased against demographic subgroups, or unable to generalize to new hospital populations."
    p2.font.size = Pt(14)
    p2.font.name = FONT_BODY
    p2.font.color.rgb = TEXT_DARK
    p2.space_before = Pt(12)

    p3 = tf.add_paragraph()
    p3.text = "This project asks what else must be true for a heart disease prediction model to be genuinely deployable and reliable in clinical practice."
    p3.font.size = Pt(14)
    p3.font.name = FONT_BODY
    p3.font.color.rgb = TEXT_MUTED
    p3.space_before = Pt(12)

    # Right Column: 4 Small Cards
    q_items = [
        ("🎯", "Is Its Confidence Honest?", "Probability Calibration: Ensuring 80% risk really means 8 out of 10 patients have disease."),
        ("⚖️", "Is It Fair Across Patients?", "Subgroup Fairness: Evaluating performance across age, sex, and hospital site cohorts."),
        ("🌐", "Does It Work Beyond One Hospital?", "External Validation: Testing generalization on unseen multi-center populations."),
        ("🔍", "Can We Explain Its Reasoning?", "Layered Explainability: SHAP, LIME, and counterfactual what-if reasoning.")
    ]

    for idx, (icon, title, desc) in enumerate(q_items):
        top = 1.7 + idx * 1.25
        add_card(s3, 6.7, top, 5.8, 1.15)
        add_icon(s3, 6.9, top + 0.2, SECONDARY, icon)

        tx = s3.shapes.add_textbox(Inches(7.5), Inches(top + 0.15), Inches(4.8), Inches(0.85))
        tf = tx.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

        p1 = tf.paragraphs[0]
        p1.text = title
        p1.font.size = Pt(14)
        p1.font.bold = True
        p1.font.name = FONT_HEADING
        p1.font.color.rgb = TEXT_DARK

        p2 = tf.add_paragraph()
        p2.text = desc
        p2.font.size = Pt(11)
        p2.font.name = FONT_BODY
        p2.font.color.rgb = TEXT_MUTED
        p2.space_before = Pt(2)

    # -------------------------------------------------------------
    # SLIDE 4: Reference Paper (Light)
    # -------------------------------------------------------------
    s4 = add_blank_slide(is_dark=False)
    add_header(s4, "LITERATURE BASE", "Our Reference Paper")
    add_page_number(s4, 4)

    # Left Primary Card
    add_card(s4, 0.8, 1.7, 5.6, 5.0, bg_color=PRIMARY)
    tx = s4.shapes.add_textbox(Inches(1.1), Inches(2.0), Inches(5.0), Inches(4.4))
    tf = tx.text_frame
    tf.word_wrap = True

    p = tf.paragraphs[0]
    p.text = "REFERENCE PAPER FOUNDATION"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = ACCENT

    p1 = tf.add_paragraph()
    p1.text = "Optimized Ensemble Learning Approach with Explainable AI for Improved Heart Disease Prediction"
    p1.font.size = Pt(18)
    p1.font.bold = True
    p1.font.name = FONT_HEADING
    p1.font.color.rgb = WHITE
    p1.space_before = Pt(8)

    p2 = tf.add_paragraph()
    p2.text = "Authors: Ibomoiye Domor Mienye & Nobert Jere\nJournal: Information (MDPI), July 2024"
    p2.font.size = Pt(13)
    p2.font.name = FONT_BODY
    p2.font.color.rgb = LIGHT_BG
    p2.space_before = Pt(12)

    p3 = tf.add_paragraph()
    p3.text = "Method Summary: Combines Random Forest, XGBoost, and AdaBoost into an ensemble with Bayesian (Optuna) hyperparameter tuning and SHAP feature explanations, evaluated on Cleveland and Framingham datasets."
    p3.font.size = Pt(13)
    p3.font.name = FONT_BODY
    p3.font.color.rgb = WHITE
    p3.space_before = Pt(14)

    # Right Card: The Gap We Address
    add_card(s4, 6.7, 1.7, 5.8, 5.0)
    tx = s4.shapes.add_textbox(Inches(7.0), Inches(2.0), Inches(5.2), Inches(4.4))
    tf = tx.text_frame
    tf.word_wrap = True

    p = tf.paragraphs[0]
    p.text = "The Gaps We Address"
    p.font.size = Pt(20)
    p.font.bold = True
    p.font.name = FONT_HEADING
    p.font.color.rgb = TEXT_DARK

    gaps = [
        "No nested cross-validation (vulnerable to hyperparameter leak)",
        "No probability calibration assessment (Brier / ECE)",
        "No subgroup or multi-site fairness breakdown",
        "Single-hospital training data (Cleveland N=303 only)",
        "No genuinely external population generalizability test",
        "No explanation-stability or model disagreement checking"
    ]

    for g in gaps:
        p_g = tf.add_paragraph()
        p_g.text = f"❌  {g}"
        p_g.font.size = Pt(12)
        p_g.font.name = FONT_BODY
        p_g.font.color.rgb = TEXT_DARK
        p_g.space_before = Pt(10)

    # -------------------------------------------------------------
    # SLIDE 5: Datasets (Light)
    # -------------------------------------------------------------
    s5 = add_blank_slide(is_dark=False)
    add_header(s5, "DATA ARCHITECTURE", "Two Cohorts: Train on Many Sites, Validate on a New One")
    add_page_number(s5, 5)

    # Left Card
    add_card(s5, 0.8, 1.7, 5.6, 5.0)
    add_icon(s5, 1.1, 2.0, PRIMARY, "🏥")
    tx = s5.shapes.add_textbox(Inches(1.7), Inches(1.95), Inches(4.5), Inches(4.5))
    tf = tx.text_frame
    tf.word_wrap = True

    p = tf.paragraphs[0]
    p.text = "TRAINING COHORT"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = SECONDARY

    p1 = tf.add_paragraph()
    p1.text = "920 Patients"
    p1.font.size = Pt(36)
    p1.font.bold = True
    p1.font.name = FONT_HEADING
    p1.font.color.rgb = PRIMARY
    p1.space_before = Pt(2)

    p2 = tf.add_paragraph()
    p2.text = "Combined 4-Site UCI Heart Disease Dataset"
    p2.font.size = Pt(14)
    p2.font.bold = True
    p2.font.color.rgb = TEXT_DARK
    p2.space_before = Pt(6)

    p3 = tf.add_paragraph()
    p3.text = "• Cleveland, USA: N = 303\n• Hungarian Institute: N = 294\n• Switzerland: N = 123\n• VA Long Beach, CA: N = 200\n\n13 clinical features. Source: UCI Repository."
    p3.font.size = Pt(13)
    p3.font.color.rgb = TEXT_MUTED
    p3.space_before = Pt(8)

    # Right Card
    add_card(s5, 6.7, 1.7, 5.8, 5.0)
    add_icon(s5, 7.0, 2.0, ACCENT, "🌐")
    tx = s5.shapes.add_textbox(Inches(7.6), Inches(1.95), Inches(4.6), Inches(4.5))
    tf = tx.text_frame
    tf.word_wrap = True

    p = tf.paragraphs[0]
    p.text = "EXTERNAL VALIDATION COHORT"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = ACCENT

    p1 = tf.add_paragraph()
    p1.text = "4,240 Patients"
    p1.font.size = Pt(36)
    p1.font.bold = True
    p1.font.name = FONT_HEADING
    p1.font.color.rgb = TEXT_DARK
    p1.space_before = Pt(2)

    p2 = tf.add_paragraph()
    p2.text = "Framingham Heart Study Cohort"
    p2.font.size = Pt(14)
    p2.font.bold = True
    p2.font.color.rgb = TEXT_DARK
    p2.space_before = Pt(6)

    p3 = tf.add_paragraph()
    p3.text = "A completely independent epidemiological population cohort, never seen during model training or hyperparameter tuning.\n\nHarmonized onto 5 common clinical features. Source: NHLBI Framingham Study."
    p3.font.size = Pt(13)
    p3.font.color.rgb = TEXT_MUTED
    p3.space_before = Pt(8)

    # -------------------------------------------------------------
    # SLIDE 6: Methodology Pipeline (Light)
    # -------------------------------------------------------------
    s6 = add_blank_slide(is_dark=False)
    add_header(s6, "SYSTEM ARCHITECTURE", "End-to-End Modeling Pipeline")
    add_page_number(s6, 6)

    steps = [
        ("1. Preprocessing", "Clean, impute median/mode, scale & encode"),
        ("2. Base Models", "Random Forest, XGBoost, AdaBoost"),
        ("3. Optuna Tuning", "Bayesian hyperparameter search"),
        ("4. Soft Ensemble", "Weighted vote across 3 models"),
        ("5. Nested CV", "Leak-free 5-outer / 3-inner CV"),
        ("6. Explainability", "SHAP, LIME & counterfactuals")
    ]

    for idx, (stitle, sdesc) in enumerate(steps):
        left = 0.8 + idx * 1.98
        add_card(s6, left, 1.8, 1.85, 3.8)

        tx = s6.shapes.add_textbox(Inches(left + 0.1), Inches(2.0), Inches(1.65), Inches(3.4))
        tf = tx.text_frame
        tf.word_wrap = True

        p1 = tf.paragraphs[0]
        p1.text = f"0{idx+1}"
        p1.font.size = Pt(22)
        p1.font.bold = True
        p1.font.color.rgb = SECONDARY

        p2 = tf.add_paragraph()
        p2.text = stitle
        p2.font.size = Pt(13)
        p2.font.bold = True
        p2.font.name = FONT_HEADING
        p2.font.color.rgb = TEXT_DARK
        p2.space_before = Pt(8)

        p3 = tf.add_paragraph()
        p3.text = sdesc
        p3.font.size = Pt(11)
        p3.font.color.rgb = TEXT_MUTED
        p3.space_before = Pt(8)

    # Bottom Banner Card
    add_card(s6, 0.8, 5.9, 11.73, 0.9, bg_color=PRIMARY)
    tx = s6.shapes.add_textbox(Inches(1.0), Inches(6.0), Inches(11.33), Inches(0.7))
    tf = tx.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "🔒 Leak-Free Evaluation: All preprocessing and Optuna tuning take place strictly inside inner CV folds — hyperparameter selection never touches the held-out test fold."
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = WHITE

    # -------------------------------------------------------------
    # SLIDE 7: Contributions Overview (Dark Divider)
    # -------------------------------------------------------------
    s7 = add_blank_slide(is_dark=True)
    add_header(s7, "CORE CONTRIBUTIONS", "What We Added Beyond the Reference Paper", is_dark=True)
    add_page_number(s7, 7, is_dark=True)

    contribs = [
        ("🔒", "Leak-Free Nested CV", "5-fold outer / 3-fold inner cross-validation preventing optimistic bias"),
        ("🎯", "Probability Calibration", "Platt Scaling & Isotonic Regression to ensure honest confidence"),
        ("⚖️", "Multi-Site Fairness", "Disaggregated subgroup evaluation across sex, age, and 4 hospital sites"),
        ("🤝", "Model Agreement Scoring", "★ Original Contribution: Per-patient consensus & uncertainty flagging", True),
        ("🌐", "Cross-Dataset Validation", "External generalizability testing on unseen Framingham cohort (N=4,240)"),
        ("🔬", "Explanation Depth", "LIME comparison, counterfactual what-ifs, and SHAP stability scoring"),
        ("💻", "Deployed Web App", "Interactive SaaS clinical decision support interface on Streamlit Cloud")
    ]

    for idx, item in enumerate(contribs):
        icon, title, desc = item[0], item[1], item[2]
        is_highlight = len(item) > 3 and item[3]

        col = idx % 3 if idx < 6 else 1
        row = idx // 3
        left = 0.8 + col * 3.95
        top = 1.7 + row * 1.65

        if idx == 6:
            left = 4.75
            width = 3.8
        else:
            width = 3.7

        add_card(s7, left, top, width, 1.5, bg_color=ACCENT if is_highlight else NAVY, border_color=ACCENT if is_highlight else SECONDARY)
        add_icon(s7, left + 0.2, top + 0.2, NAVY if is_highlight else SECONDARY, icon)

        tx = s7.shapes.add_textbox(Inches(left + 0.75), Inches(top + 0.15), Inches(width - 0.85), Inches(1.2))
        tf = tx.text_frame
        tf.word_wrap = True

        p1 = tf.paragraphs[0]
        p1.text = title
        p1.font.size = Pt(13)
        p1.font.bold = True
        p1.font.name = FONT_HEADING
        p1.font.color.rgb = NAVY if is_highlight else WHITE

        p2 = tf.add_paragraph()
        p2.text = desc
        p2.font.size = Pt(10.5)
        p2.font.color.rgb = NAVY if is_highlight else LIGHT_BG
        p2.space_before = Pt(3)

    tx_foot = s7.shapes.add_textbox(Inches(0.8), Inches(6.8), Inches(11.73), Inches(0.4))
    tf_f = tx_foot.text_frame
    p_f = tf_f.paragraphs[0]
    p_f.text = "★ Original contribution — not present in the reference paper or related 2025 literature."
    p_f.font.size = Pt(11)
    p_f.font.bold = True
    p_f.font.color.rgb = ACCENT

    # -------------------------------------------------------------
    # SLIDE 8: Reliability (Light)
    # -------------------------------------------------------------
    s8 = add_blank_slide(is_dark=False)
    add_header(s8, "RELIABILITY ENGINEERING", "Honest Evaluation & Trustworthy Probabilities")
    add_page_number(s8, 8)

    # Left Card
    add_card(s8, 0.8, 1.7, 5.6, 5.0)
    tx = s8.shapes.add_textbox(Inches(1.1), Inches(2.0), Inches(5.0), Inches(4.4))
    tf = tx.text_frame
    tf.word_wrap = True

    p = tf.paragraphs[0]
    p.text = "Methodological Framework"
    p.font.size = Pt(18)
    p.font.bold = True
    p.font.name = FONT_HEADING
    p.font.color.rgb = PRIMARY

    p1 = tf.add_paragraph()
    p1.text = "1. Nested Cross-Validation"
    p1.font.size = Pt(14)
    p1.font.bold = True
    p1.font.color.rgb = TEXT_DARK
    p1.space_before = Pt(10)

    p2 = tf.add_paragraph()
    p2.text = "Hyperparameter optimization happens strictly inside outer training folds, never touching held-out test folds. Prevents optimistic evaluation bias."
    p2.font.size = Pt(12)
    p2.font.color.rgb = TEXT_MUTED
    p2.space_before = Pt(4)

    p3 = tf.add_paragraph()
    p3.text = "2. Probability Calibration"
    p3.font.size = Pt(14)
    p3.font.bold = True
    p3.font.color.rgb = TEXT_DARK
    p3.space_before = Pt(14)

    p4 = tf.add_paragraph()
    p4.text = "Evaluates Expected Calibration Error (ECE) and Brier Score Loss. Ensures a predicted 80% risk translates to 80% true disease incidence."
    p4.font.size = Pt(12)
    p4.font.color.rgb = TEXT_MUTED
    p4.space_before = Pt(4)

    # Right Card: Bug & Fix
    add_card(s8, 6.7, 1.7, 5.8, 5.0, bg_color=NAVY)
    tx = s8.shapes.add_textbox(Inches(7.0), Inches(2.0), Inches(5.2), Inches(4.4))
    tf = tx.text_frame
    tf.word_wrap = True

    p = tf.paragraphs[0]
    p.text = "🐛 A BUG WE FOUND & FIXED"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = ACCENT

    p1 = tf.add_paragraph()
    p1.text = "Calibration Leakage"
    p1.font.size = Pt(20)
    p1.font.bold = True
    p1.font.name = FONT_HEADING
    p1.font.color.rgb = WHITE
    p1.space_before = Pt(6)

    p2 = tf.add_paragraph()
    p2.text = "Calibrating models directly on their own training set predictions caused severe in-sample probability distortion, making out-of-fold calibration worse instead of better."
    p2.font.size = Pt(13)
    p2.font.color.rgb = LIGHT_BG
    p2.space_before = Pt(10)

    p3 = tf.add_paragraph()
    p3.text = "FIX: Applied a 75/25 model-fit / calibration-fit out-of-sample split within outer training folds."
    p3.font.size = Pt(13)
    p3.font.bold = True
    p3.font.color.rgb = ACCENT
    p3.space_before = Pt(12)

    p4 = tf.add_paragraph()
    p4.text = "• AdaBoost ECE: 0.181 → 0.050 (3.6× Improvement)\n• Soft Ensemble ECE: 0.102 → 0.069"
    p4.font.size = Pt(14)
    p4.font.bold = True
    p4.font.color.rgb = WHITE
    p4.space_before = Pt(12)

    # -------------------------------------------------------------
    # SLIDE 9: Fairness (Light)
    # -------------------------------------------------------------
    s9 = add_blank_slide(is_dark=False)
    add_header(s9, "FAIRNESS & SUBGROUP ANALYSIS", "Fairness Across Patients & Hospitals")
    add_page_number(s9, 9)

    add_card(s9, 0.8, 1.7, 11.73, 5.0)
    tx = s9.shapes.add_textbox(Inches(1.1), Inches(1.9), Inches(11.13), Inches(4.6))
    tf = tx.text_frame
    tf.word_wrap = True

    p = tf.paragraphs[0]
    p.text = "FINDING: THE SWITZERLAND SITE ANOMALY"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = ACCENT

    p1 = tf.add_paragraph()
    p1.text = "High Accuracy Can Hide a Model That Is Barely Discriminating At All"
    p1.font.size = Pt(20)
    p1.font.bold = True
    p1.font.name = FONT_HEADING
    p1.font.color.rgb = TEXT_DARK
    p1.space_before = Pt(4)

    # 4 Stat Boxes
    stats = [
        ("92.7%", "Disease Prevalence", TEXT_DARK),
        ("91.1%", "Accuracy", PRIMARY),
        ("97.4%", "Recall (Sensitivity)", GREEN),
        ("11.1%", "Specificity (TNR)", RED)
    ]

    for idx, (val, label, color) in enumerate(stats):
        left_s = 1.1 + idx * 2.8
        add_card(s9, left_s, 3.1, 2.5, 1.5, bg_color=LIGHT_BG)
        tx_s = s9.shapes.add_textbox(Inches(left_s), Inches(3.2), Inches(2.5), Inches(1.3))
        tf_s = tx_s.text_frame
        tf_s.word_wrap = True

        p_v = tf_s.paragraphs[0]
        p_v.text = val
        p_v.alignment = PP_ALIGN.CENTER
        p_v.font.size = Pt(28)
        p_v.font.bold = True
        p_v.font.name = FONT_HEADING
        p_v.font.color.rgb = color

        p_l = tf_s.add_paragraph()
        p_l.text = label
        p_l.alignment = PP_ALIGN.CENTER
        p_l.font.size = Pt(11)
        p_l.font.bold = True
        p_l.font.color.rgb = TEXT_MUTED

    p_note = tf.add_paragraph()
    p_note.text = "Clinical Insight: Because 92.7% of patients in the Switzerland cohort had heart disease, a naive model predicting 'disease' almost constantly achieves 91.1% accuracy and 97.4% recall. However, its 11.1% specificity reveals it fails to identify healthy patients. Disaggregating metrics by site was essential to surface this prevalence bias."
    p_note.font.size = Pt(12)
    p_note.font.color.rgb = TEXT_DARK
    p_note.space_before = Pt(105)

    # -------------------------------------------------------------
    # SLIDE 10: Model Agreement (Light - Original Contribution)
    # -------------------------------------------------------------
    s10 = add_blank_slide(is_dark=False)
    add_header(s10, "ORIGINAL CONTRIBUTION", "Per-Patient Model Agreement Scoring")
    add_page_number(s10, 10)

    # Left Card
    add_card(s10, 0.8, 1.7, 5.6, 5.0)
    tx = s10.shapes.add_textbox(Inches(1.1), Inches(1.9), Inches(5.0), Inches(4.5))
    tf = tx.text_frame
    tf.word_wrap = True

    p = tf.paragraphs[0]
    p.text = "★ NOT IN THE REFERENCE PAPER"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = ACCENT

    p1 = tf.add_paragraph()
    p1.text = "Flagging Uncertainty at Prediction Time"
    p1.font.size = Pt(18)
    p1.font.bold = True
    p1.font.name = FONT_HEADING
    p1.font.color.rgb = TEXT_DARK
    p1.space_before = Pt(4)

    p2 = tf.add_paragraph()
    p2.text = "When individual base classifiers (RF, XGBoost, AdaBoost) disagree sharply on a patient, silently averaging their probabilities conceals clinical uncertainty."
    p2.font.size = Pt(13)
    p2.font.color.rgb = TEXT_MUTED
    p2.space_before = Pt(8)

    p3 = tf.add_paragraph()
    p3.text = "Agreement Thresholds (Std Dev σ):"
    p3.font.size = Pt(13)
    p3.font.bold = True
    p3.font.color.rgb = TEXT_DARK
    p3.space_before = Pt(12)

    p4 = tf.add_paragraph()
    p4.text = "• High Agreement: σ < 0.05\n• Moderate Agreement: 0.05 ≤ σ < 0.15\n• Low Agreement: σ ≥ 0.15 (Generates Warning Flag)"
    p4.font.size = Pt(12)
    p4.font.color.rgb = TEXT_DARK
    p4.space_before = Pt(4)

    # Right Card
    add_card(s10, 6.7, 1.7, 5.8, 5.0, bg_color=NAVY)
    tx = s10.shapes.add_textbox(Inches(7.0), Inches(2.0), Inches(5.2), Inches(4.4))
    tf = tx.text_frame
    tf.word_wrap = True

    p = tf.paragraphs[0]
    p.text = "VALIDATED BY THE DATA"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = ACCENT

    p1 = tf.add_paragraph()
    p1.text = "Accuracy Disaggregated by Agreement Level"
    p1.font.size = Pt(18)
    p1.font.bold = True
    p1.font.name = FONT_HEADING
    p1.font.color.rgb = WHITE
    p1.space_before = Pt(4)

    bars = [
        ("High Agreement (σ < 0.05)", "89.3% Accuracy", GREEN),
        ("Moderate Agreement (0.05–0.15)", "67.6% Accuracy", ACCENT),
        ("Low Agreement (σ ≥ 0.15)", "60.0% Accuracy", RED)
    ]

    for idx, (label, acc, color) in enumerate(bars):
        p_b = tf.add_paragraph()
        p_b.text = f"{label}\n➔ {acc}"
        p_b.font.size = Pt(14)
        p_b.font.bold = True
        p_b.font.color.rgb = color
        p_b.space_before = Pt(12)

    p_sum = tf.add_paragraph()
    p_sum.text = "Empirical Proof: High agreement predictions are 29.3% more accurate than low agreement predictions, validating model variance as a real-time confidence signal."
    p_sum.font.size = Pt(12)
    p_sum.font.color.rgb = LIGHT_BG
    p_sum.space_before = Pt(14)

    # -------------------------------------------------------------
    # SLIDE 11: External Validation (Light)
    # -------------------------------------------------------------
    s11 = add_blank_slide(is_dark=False)
    add_header(s11, "EXTERNAL GENERALIZABILITY", "Does It Generalize Beyond One Hospital?")
    add_page_number(s11, 11)

    # Table Card
    add_card(s11, 0.8, 1.6, 11.73, 3.6)

    # Table creation
    x, y, cx, cy = Inches(1.0), Inches(1.8), Inches(11.33), Inches(3.2)
    shape = s11.shapes.add_table(5, 4, x, y, cx, cy)
    table = shape.table

    headers = ["Model", "Cleveland CV ROC-AUC", "Combined 4-Site CV ROC-AUC", "Framingham External ROC-AUC"]
    for col_idx, h in enumerate(headers):
        cell = table.cell(0, col_idx)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = PRIMARY
        p = cell.text_frame.paragraphs[0]
        p.font.size = Pt(12)
        p.font.bold = True
        p.font.color.rgb = WHITE
        p.font.name = FONT_HEADING

    rows_data = [
        ["Random Forest", "73.04%", "74.99%", "67.84%"],
        ["XGBoost", "72.71%", "74.89%", "69.30%"],
        ["AdaBoost", "69.73%", "74.73%", "68.53%"],
        ["Soft Ensemble", "72.95%", "75.29%", "68.82%"]
    ]

    for row_idx, r_data in enumerate(rows_data, 1):
        for col_idx, val in enumerate(r_data):
            cell = table.cell(row_idx, col_idx)
            cell.text = val
            p = cell.text_frame.paragraphs[0]
            p.font.size = Pt(12)
            p.font.name = FONT_BODY
            p.font.color.rgb = TEXT_DARK
            if col_idx == 0:
                p.font.bold = True

    # Callout Banner
    add_card(s11, 0.8, 5.4, 11.73, 1.4, bg_color=NAVY)
    tx = s11.shapes.add_textbox(Inches(1.0), Inches(5.5), Inches(11.33), Inches(1.2))
    tf = tx.text_frame
    tf.word_wrap = True

    p = tf.paragraphs[0]
    p.text = "🐛 HIDDEN BUG CAUGHT DURING EXTERNAL VALIDATION:"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = ACCENT

    p1 = tf.add_paragraph()
    p1.text = "One candidate model initially showed 77.7% accuracy on Framingham — yet caught zero real disease cases due to majority class prediction in a low-prevalence cohort (15.2%). Fixed by adding Recall, F1, and PR-AUC metrics."
    p1.font.size = Pt(12)
    p1.font.color.rgb = WHITE
    p1.space_before = Pt(4)

    # -------------------------------------------------------------
    # SLIDE 12: Explanation Depth (Light)
    # -------------------------------------------------------------
    s12 = add_blank_slide(is_dark=False)
    add_header(s12, "EXPLANATION DEPTH", "Explaining Every Prediction, Two Independent Ways")
    add_page_number(s12, 12)

    cards = [
        ("80%", "SHAP vs. LIME Overlap@5", "High Local Explanation Consensus", "Two independent, fundamentally different explainers (TreeExplainer vs LimeTabular) share 80% of top-5 risk features on sample predictions."),
        ("0.78", "Kendall's τ Stability", "Fold-to-Fold Ranking Stability", "Top-5 SHAP feature rankings remain stable across outer CV fold data splits, confirming explanations reflect true patterns rather than split noise."),
        ("3–4", "Counterfactual What-Ifs", "Minimal Realistic Modifications", "Generates actionable feature shifts (e.g., reducing oldpeak) needed to flip high risk below 0.50 threshold, tagged with explicit medical disclaimers.")
    ]

    for idx, (stat, subtitle, title, desc) in enumerate(cards):
        left = 0.8 + idx * 3.95
        add_card(s12, left, 1.7, 3.7, 5.0)

        tx = s12.shapes.add_textbox(Inches(left + 0.2), Inches(1.9), Inches(3.3), Inches(4.5))
        tf = tx.text_frame
        tf.word_wrap = True

        p0 = tf.paragraphs[0]
        p0.text = stat
        p0.font.size = Pt(42)
        p0.font.bold = True
        p0.font.name = FONT_HEADING
        p0.font.color.rgb = PRIMARY

        p1 = tf.add_paragraph()
        p1.text = subtitle
        p1.font.size = Pt(12)
        p1.font.bold = True
        p1.font.color.rgb = SECONDARY
        p1.space_before = Pt(2)

        p2 = tf.add_paragraph()
        p2.text = title
        p2.font.size = Pt(15)
        p2.font.bold = True
        p2.font.name = FONT_HEADING
        p2.font.color.rgb = TEXT_DARK
        p2.space_before = Pt(12)

        p3 = tf.add_paragraph()
        p3.text = desc
        p3.font.size = Pt(12)
        p3.font.color.rgb = TEXT_MUTED
        p3.space_before = Pt(8)

    # -------------------------------------------------------------
    # SLIDE 13: Final Deliverable (Light)
    # -------------------------------------------------------------
    s13 = add_blank_slide(is_dark=False)
    add_header(s13, "DEPLOYABLE DELIVERABLES", "From Research Notebook to a Usable Tool")
    add_page_number(s13, 13)

    # Left Panel
    add_card(s13, 0.8, 1.7, 5.6, 5.0, bg_color=PRIMARY)
    tx = s13.shapes.add_textbox(Inches(1.1), Inches(2.0), Inches(5.0), Inches(4.4))
    tf = tx.text_frame
    tf.word_wrap = True

    p = tf.paragraphs[0]
    p.text = "💻 COMMAND LINE INTERFACE"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = ACCENT

    p1 = tf.add_paragraph()
    p1.text = "src/predict.py"
    p1.font.size = Pt(22)
    p1.font.bold = True
    p1.font.name = FONT_HEADING
    p1.font.color.rgb = WHITE
    p1.space_before = Pt(4)

    p2 = tf.add_paragraph()
    p2.text = "• Instant single-patient risk inference from terminal\n• Outputs probability %, risk category & agreement flag\n• Lists top SHAP contributing features\n• Concludes with mandatory clinical disclaimer\n• Supports raw JSON output for API pipelines"
    p2.font.size = Pt(13)
    p2.font.color.rgb = LIGHT_BG
    p2.space_before = Pt(12)

    # Right Panel
    add_card(s13, 6.7, 1.7, 5.8, 5.0, bg_color=NAVY)
    tx = s13.shapes.add_textbox(Inches(7.0), Inches(2.0), Inches(5.2), Inches(4.4))
    tf = tx.text_frame
    tf.word_wrap = True

    p = tf.paragraphs[0]
    p.text = "🌐 SAAS CLINICAL WEB APP"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = ACCENT

    p1 = tf.add_paragraph()
    p1.text = "app.py (Streamlit Cloud)"
    p1.font.size = Pt(22)
    p1.font.bold = True
    p1.font.name = FONT_HEADING
    p1.font.color.rgb = WHITE
    p1.space_before = Pt(4)

    p2 = tf.add_paragraph()
    p2.text = "• Modern SaaS health-tech UI styling (custom CSS)\n• Multi-column card input grid for 13 parameters\n• Circular risk probability gauge visual\n• Horizontal model agreement probability comparison\n• Plotly SHAP feature impact chart\n• Deployed on Streamlit Community Cloud"
    p2.font.size = Pt(13)
    p2.font.color.rgb = LIGHT_BG
    p2.space_before = Pt(12)

    # -------------------------------------------------------------
    # SLIDE 14: Methodological Rigor (Dark Divider)
    # -------------------------------------------------------------
    s14 = add_blank_slide(is_dark=True)
    add_header(s14, "METHODOLOGICAL RIGOR", "Three Real Bugs We Found — And Why That's a Good Sign", is_dark=True)
    add_page_number(s14, 14, is_dark=True)

    bugs = [
        ("🐛 1. Calibration Leak", "Calibrating models directly on training set predictions caused severe probability distortion.", "FIX: Implemented out-of-sample 75/25 split within outer training folds. ECE improved up to 3.6×."),
        ("🐛 2. Hidden Class Imbalance", "A candidate model hit 77.7% accuracy on Framingham while catching 0 real disease cases.", "FIX: Expanded metrics to include Recall, Specificity, F1, and PR-AUC to catch prevalence bias."),
        ("🐛 3. Silent Data Corruption", "Unrecorded cholesterol/BP values were stored as literal 0s in non-Cleveland sites.", "FIX: Implemented zero-as-missing conversion for non-Cleveland continuous attributes.")
    ]

    for idx, (btitle, bdesc, bfix) in enumerate(bugs):
        left = 0.8 + idx * 3.95
        add_card(s14, left, 1.8, 3.7, 4.8, bg_color=WHITE)

        tx = s14.shapes.add_textbox(Inches(left + 0.2), Inches(2.0), Inches(3.3), Inches(4.4))
        tf = tx.text_frame
        tf.word_wrap = True

        p1 = tf.paragraphs[0]
        p1.text = btitle
        p1.font.size = Pt(15)
        p1.font.bold = True
        p1.font.name = FONT_HEADING
        p1.font.color.rgb = RED

        p2 = tf.add_paragraph()
        p2.text = bdesc
        p2.font.size = Pt(12)
        p2.font.color.rgb = TEXT_DARK
        p2.space_before = Pt(10)

        p3 = tf.add_paragraph()
        p3.text = bfix
        p3.font.size = Pt(12)
        p3.font.bold = True
        p3.font.color.rgb = SECONDARY
        p3.space_before = Pt(14)

    # -------------------------------------------------------------
    # SLIDE 15: Results Summary (Light)
    # -------------------------------------------------------------
    s15 = add_blank_slide(is_dark=False)
    add_header(s15, "PERFORMANCE SUMMARY", "Key Results at a Glance")
    add_page_number(s15, 15)

    # Table Card
    add_card(s15, 0.8, 1.6, 11.73, 3.6)

    x, y, cx, cy = Inches(1.0), Inches(1.8), Inches(11.33), Inches(3.2)
    shape = s15.shapes.add_table(5, 4, x, y, cx, cy)
    table = shape.table

    headers = ["Model", "Combined 4-Site CV Accuracy", "Combined 4-Site CV ROC-AUC", "Framingham External ROC-AUC"]
    for col_idx, h in enumerate(headers):
        cell = table.cell(0, col_idx)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = PRIMARY
        p = cell.text_frame.paragraphs[0]
        p.font.size = Pt(12)
        p.font.bold = True
        p.font.color.rgb = WHITE
        p.font.name = FONT_HEADING

    rows_data = [
        ["Random Forest", "81.5%", "89.5%", "67.8%"],
        ["XGBoost", "82.1%", "88.6%", "69.3%"],
        ["AdaBoost", "81.0%", "88.2%", "68.5%"],
        ["Soft Ensemble", "81.8%", "89.7%", "68.8%"]
    ]

    for row_idx, r_data in enumerate(rows_data, 1):
        for col_idx, val in enumerate(r_data):
            cell = table.cell(row_idx, col_idx)
            cell.text = val
            p = cell.text_frame.paragraphs[0]
            p.font.size = Pt(12)
            p.font.name = FONT_BODY
            p.font.color.rgb = TEXT_DARK
            if col_idx == 0:
                p.font.bold = True

    # Bottom Notes Card
    add_card(s15, 0.8, 5.4, 11.73, 1.4)
    tx = s15.shapes.add_textbox(Inches(1.0), Inches(5.5), Inches(11.33), Inches(1.2))
    tf = tx.text_frame
    tf.word_wrap = True

    p = tf.paragraphs[0]
    p.text = "• Internal performance remains highly robust (~82% accuracy, ~89.7% ROC-AUC) after expanding cohort from 303 to 920 patients across 4 sites."
    p.font.size = Pt(12)
    p.font.color.rgb = TEXT_DARK

    p1 = tf.add_paragraph()
    p1.text = "• External ROC-AUC holds reasonably (~69%) on a completely unseen population cohort — demonstrating genuine cross-population generalization."
    p1.font.size = Pt(12)
    p1.font.color.rgb = TEXT_DARK
    p1.space_before = Pt(4)

    # -------------------------------------------------------------
    # SLIDE 16: Conclusion (Light)
    # -------------------------------------------------------------
    s16 = add_blank_slide(is_dark=False)
    add_header(s16, "SUMMARY & FUTURE DIRECTION", "Conclusion & Future Work")
    add_page_number(s16, 16)

    # Left Panel
    add_card(s16, 0.8, 1.7, 5.6, 5.0)
    tx = s16.shapes.add_textbox(Inches(1.1), Inches(1.9), Inches(5.0), Inches(4.5))
    tf = tx.text_frame
    tf.word_wrap = True

    p = tf.paragraphs[0]
    p.text = "Project Summary"
    p.font.size = Pt(18)
    p.font.bold = True
    p.font.name = FONT_HEADING
    p.font.color.rgb = PRIMARY

    p1 = tf.add_paragraph()
    p1.text = "A rigorous extension of an existing ensemble framework — through nested validation, calibration, fairness analysis, multi-site data, genuine external validation, and an original agreement mechanism — surfaces real issues a straightforward reproduction never would have caught."
    p1.font.size = Pt(13)
    p1.font.color.rgb = TEXT_DARK
    p1.space_before = Pt(8)

    # Quote Box
    add_card(s16, 1.1, 4.3, 5.0, 2.1, bg_color=NAVY)
    tx_q = s16.shapes.add_textbox(Inches(1.25), Inches(4.45), Inches(4.7), Inches(1.8))
    tf_q = tx_q.text_frame
    tf_q.word_wrap = True

    p_q = tf_q.paragraphs[0]
    p_q.text = "“The story isn't that everything worked — it's that we could tell when it didn't.”"
    p_q.font.size = Pt(14)
    p_q.font.bold = True
    p_q.font.name = FONT_HEADING
    p_q.font.color.rgb = ACCENT

    # Right Panel: Future Work
    add_card(s16, 6.7, 1.7, 5.8, 5.0)
    tx = s16.shapes.add_textbox(Inches(7.0), Inches(1.9), Inches(5.2), Inches(4.5))
    tf = tx.text_frame
    tf.word_wrap = True

    p = tf.paragraphs[0]
    p.text = "Future Work Directions"
    p.font.size = Pt(18)
    p.font.bold = True
    p.font.name = FONT_HEADING
    p.font.color.rgb = TEXT_DARK

    future_items = [
        ("🔑", "Broader Deployment", "Implement clinician auth, EHR integration & PDF report exporting."),
        ("🌐", "Additional Cohorts", "Validate on additional international cohorts (e.g. UK Biobank)."),
        ("🔬", "Deeper Stability Analysis", "Extend explanation stability metrics across neural & GAM architectures."),
        ("📄", "Publication", "Package calibration leakage & specificity findings for journal submission.")
    ]

    for icon, f_title, f_desc in future_items:
        p_f = tf.add_paragraph()
        p_f.text = f"{icon}  {f_title}"
        p_f.font.size = Pt(13)
        p_f.font.bold = True
        p_f.font.color.rgb = SECONDARY
        p_f.space_before = Pt(10)

        p_fd = tf.add_paragraph()
        p_fd.text = f_desc
        p_fd.font.size = Pt(11)
        p_fd.font.color.rgb = TEXT_MUTED
        p_fd.space_before = Pt(1)

    # -------------------------------------------------------------
    # SLIDE 17: Closing (Dark)
    # -------------------------------------------------------------
    s17 = add_blank_slide(is_dark=True)
    card17 = add_card(s17, 1.0, 1.2, 11.33, 5.1, bg_color=NAVY)

    txBox = s17.shapes.add_textbox(Inches(1.2), Inches(2.2), Inches(10.93), Inches(3.2))
    tf = txBox.text_frame
    tf.word_wrap = True

    p0 = tf.paragraphs[0]
    p0.text = "THANK YOU"
    p0.alignment = PP_ALIGN.CENTER
    p0.font.size = Pt(44)
    p0.font.bold = True
    p0.font.name = FONT_HEADING
    p0.font.color.rgb = WHITE

    p1 = tf.add_paragraph()
    p1.text = "Questions & Discussion"
    p1.alignment = PP_ALIGN.CENTER
    p1.font.size = Pt(22)
    p1.font.bold = True
    p1.font.color.rgb = ACCENT
    p1.space_before = Pt(12)

    p2 = tf.add_paragraph()
    p2.text = "Optimized, Explainable & Reliable Ensemble Framework for Heart Disease Prediction"
    p2.alignment = PP_ALIGN.CENTER
    p2.font.size = Pt(13)
    p2.font.color.rgb = LIGHT_BG
    p2.space_before = Pt(36)

    os.makedirs("presentations", exist_ok=True)
    out_path = "presentations/project_presentation.pptx"
    prs.save(out_path)
    print(f"Presentation saved successfully to {out_path}")


if __name__ == "__main__":
    create_presentation()
