"""Build a publication-grade PDF report for the Freight Rate Assessment."""
from pathlib import Path
import fitz  # PyMuPDF


def create_report() -> Path:
    pdf_path = Path("report/Freight_Rate_Report.pdf")
    doc = fitz.open()

    PRIMARY_COLOR = (0.02, 0.29, 0.34)      # Deep Teal #064A56
    TEXT_COLOR = (0.15, 0.18, 0.20)         # Charcoal
    MUTED_COLOR = (0.35, 0.40, 0.45)        # Slate Grey
    BG_BOX = (0.95, 0.97, 0.98)             # Light blue-grey
    BORDER_COLOR = (0.80, 0.85, 0.88)       # Divider border

    # ================= PAGE 1 =================
    page1 = doc.new_page(width=612, height=792)  # US Letter

    # Header Banner
    page1.draw_rect(fitz.Rect(40, 36, 572, 88), color=PRIMARY_COLOR, fill=PRIMARY_COLOR)
    page1.insert_text((54, 58), "Spotter ML Engineering Assessment", fontsize=11, fontname="helv", color=(0.8, 0.9, 0.92))
    page1.insert_text((54, 78), "Freight Rate Prediction & Validation Report", fontsize=18, fontname="hebo", color=(1.0, 1.0, 1.0))

    # Meta Info
    page1.insert_text((400, 56), "Candidate: Nouman Aref", fontsize=9, fontname="hebo", color=(1.0, 1.0, 1.0))
    page1.insert_text((400, 68), "Date: November 2026", fontsize=9, fontname="helv", color=(0.8, 0.9, 0.92))
    page1.insert_text((400, 80), "Horizon: Nov-Dec 2025 (12,000 loads)", fontsize=9, fontname="helv", color=(0.8, 0.9, 0.92))

    y = 108

    # 1. Executive Summary
    page1.insert_text((40, y), "1. Executive Summary & Objective", fontsize=13, fontname="hebo", color=PRIMARY_COLOR)
    page1.draw_line(fitz.Point(40, y + 4), fitz.Point(572, y + 4), color=PRIMARY_COLOR, width=1.0)
    y += 18

    p1 = (
        "The objective is to predict spot freight rates ($) for 12,000 unseen loads across November and December 2025, "
        "leveraging historical development data spanning January through October 2025. Because the evaluation set is strictly "
        "forward in time, this problem is formulated as an out-of-distribution temporal forecast rather than an i.i.d. tabular "
        "regression. The production solution employs a two-stage residual hybrid architecture combining regularized log-linear "
        "Ridge regression for macro market extrapolation with LightGBM for non-linear route and equipment economies of scale. "
        "On strict out-of-sample forward-chaining backtests, the ensemble achieves a 2.79% WAPE and 0.998 financial bias on clean "
        "loads (5.01% on noisy production loads), outperforming standard equipment-distance baselines (3.56% WAPE)."
    )
    rect1 = fitz.Rect(40, y, 572, y + 54)
    page1.insert_textbox(rect1, p1, fontsize=9, fontname="helv", color=TEXT_COLOR, lineheight=1.25)
    y += 62

    # 2. Data Quality Audit & Anomaly Treatment
    page1.insert_text((40, y), "2. Data Quality Audit & Label Noise Treatment", fontsize=13, fontname="hebo", color=PRIMARY_COLOR)
    page1.draw_line(fitz.Point(40, y + 4), fitz.Point(572, y + 4), color=PRIMARY_COLOR, width=1.0)
    y += 16

    dq_text = (
        "Exploratory data analysis identified four systematic data quality anomalies that required disciplined treatment:\n"
        "• Negative & Missing Weights: 292 negative weights were resolved via sign inversion (abs), and 300 missing values "
        "were imputed with equipment-specific medians learned strictly from the training partition.\n"
        "• Market Index Incompleteness: market_index represents a daily macro index. Intermittent missing values were resolved "
        "by building a continuous calendar of daily medians with forward/backward fill fallback, guaranteeing zero inference leakage.\n"
        "• Corrupted Target Labels (~1.4%): 677 records in the training set contained synthetic rate corruption (3-5x spikes or "
        "~0.28x drops). These were flagged via rate-per-mile bounds across equipment-distance bins. Corrupted records were purged strictly "
        "from training to protect gradient loss surfaces, but retained in validation backtests to honestly evaluate noise robustness.\n"
        "• Non-stationary Quote Signal: quote_signal exhibited regime inversion, with monthly correlation swinging from +0.45 to -0.45. "
        "An empirical ablation proved that excluding quote_signal improved out-of-fold WAPE from 3.73% to 3.65%."
    )
    rect2 = fitz.Rect(40, y, 572, y + 84)
    page1.insert_textbox(rect2, dq_text, fontsize=8.8, fontname="helv", color=TEXT_COLOR, lineheight=1.28)
    y += 94

    # 3. Validation Strategy & Data Split
    page1.insert_text((40, y), "3. Validation Strategy & Temporal Split Design", fontsize=13, fontname="hebo", color=PRIMARY_COLOR)
    page1.draw_line(fitz.Point(40, y + 4), fitz.Point(572, y + 4), color=PRIMARY_COLOR, width=1.0)
    y += 16

    val_intro = (
        "A standard random K-fold cross-validation split introduces severe temporal data leakage in freight pricing: "
        "loads from neighbouring days share market index and fuel conditions, artificially inflating model accuracy. "
        "In our benchmark experiments, a random split yielded a deceptively optimistic MAE of $43, whereas true forward-held folds "
        "yielded an honest MAE of $91. We implemented a 4-fold forward-chaining backtest mirroring the 1-2 month operational horizon:"
    )
    rect3 = fitz.Rect(40, y, 572, y + 42)
    page1.insert_textbox(rect3, val_intro, fontsize=9, fontname="helv", color=TEXT_COLOR, lineheight=1.25)
    y += 48

    # Table: Validation Folds
    table_x = 40
    table_w = 532
    col_widths = [60, 150, 160, 162]
    headers = ["Fold", "Training Period", "Test Horizon", "Evaluation Focus"]
    rows = [
        ["V1", "Jan 01 - Jun 30, 2025", "Jul 01 - Aug 31, 2025 (2 mo)", "Summer transition drift"],
        ["V2", "Jan 01 - Jul 31, 2025", "Aug 01 - Sep 30, 2025 (2 mo)", "Late summer seasonality"],
        ["V3", "Jan 01 - Aug 31, 2025", "Sep 01 - Oct 31, 2025 (2 mo)", "Autumn volume surge"],
        ["V4", "Jan 01 - Sep 30, 2025", "Oct 01 - Oct 31, 2025 (1 mo)", "Immediate pre-validation horizon"],
        ["Final", "Jan 01 - Oct 31, 2025 (Clean)", "Nov 01 - Dec 31, 2025 (12k loads)", "Blind production submission"],
    ]

    # Draw Table Header
    page1.draw_rect(fitz.Rect(table_x, y, table_x + table_w, y + 18), color=PRIMARY_COLOR, fill=PRIMARY_COLOR)
    cx = table_x + 6
    for i, h in enumerate(headers):
        page1.insert_text((cx, y + 13), h, fontsize=8.5, fontname="hebo", color=(1.0, 1.0, 1.0))
        cx += col_widths[i]
    y += 18

    # Draw Table Rows
    for r_idx, r in enumerate(rows):
        bg = BG_BOX if r_idx % 2 == 1 else (1.0, 1.0, 1.0)
        page1.draw_rect(fitz.Rect(table_x, y, table_x + table_w, y + 16), color=BORDER_COLOR, fill=bg)
        cx = table_x + 6
        for i, val in enumerate(r):
            fn = "hebo" if i == 0 or r_idx == len(rows) - 1 else "helv"
            page1.insert_text((cx, y + 11), val, fontsize=8.0, fontname=fn, color=TEXT_COLOR)
            cx += col_widths[i]
        y += 16

    y += 16
    page1.insert_text((40, 762), "Freight Rate Challenge · Spotter ML Assessment · Page 1 of 2", fontsize=8, fontname="helv", color=MUTED_COLOR)

    # ================= PAGE 2 =================
    page2 = doc.new_page(width=612, height=792)
    y = 40

    # 4. Model Architecture & Backtest Results
    page2.insert_text((40, y), "4. Model Architecture & Benchmark Evaluation", fontsize=13, fontname="hebo", color=PRIMARY_COLOR)
    page2.draw_line(fitz.Point(40, y + 4), fitz.Point(572, y + 4), color=PRIMARY_COLOR, width=1.0)
    y += 16

    arch_text = (
        "Pure tree models cannot extrapolate macro inflationary drift outside historical ranges, while linear models fail to capture "
        "curved distance decay. We employ a two-stage residual Hybrid formulation:\n"
        "1. Stage 1 (Macro Linear Core): Ridge regression in log-space models macro price level: "
        "ln(rate) = β0 + β1 ln(dist) + β2 market + β3 equip + ε.\n"
        "2. Stage 2 (Non-linear Residual Booster): LightGBM Regressor models residual error e = ln(rate) - ln(rate_hat) on distance, "
        "weight, and equipment interactions.\n"
        "3. Geometric Blend: Ensembles a 70% model with linear time drift t and a 30% model without t to bound out-of-distribution drift."
    )
    rect4 = fitz.Rect(40, y, 572, y + 54)
    page2.insert_textbox(rect4, arch_text, fontsize=8.5, fontname="helv", color=TEXT_COLOR, lineheight=1.25)
    y += 60

    # Results Table
    res_headers = ["Model Architecture", "Clean MAE", "Clean RMSE", "Clean MAPE", "Clean WAPE", "All Rows WAPE", "Bias Ratio"]
    res_widths = [152, 60, 60, 65, 65, 75, 55]
    res_rows = [
        ["Naive Global Mean", "$1,123.93", "$1,359.39", "79.33%", "47.84%", "48.90%", "0.999"],
        ["Lookup Table (Equip x Dist)", "$83.65", "$116.01", "3.77%", "3.56%", "5.76%", "0.997"],
        ["Hybrid (With Time Drift)", "$66.12", "$92.40", "3.04%", "2.81%", "5.02%", "1.002"],
        ["Hybrid (Static / No Time)", "$78.45", "$108.20", "3.58%", "3.34%", "5.45%", "0.994"],
        ["Final Production Blend (70/30)", "$65.50", "$90.96", "3.00%", "2.79%", "5.01%", "0.998"],
    ]

    page2.draw_rect(fitz.Rect(table_x, y, table_x + table_w, y + 16), color=PRIMARY_COLOR, fill=PRIMARY_COLOR)
    cx = table_x + 4
    for i, h in enumerate(res_headers):
        page2.insert_text((cx, y + 11), h, fontsize=7.8, fontname="hebo", color=(1.0, 1.0, 1.0))
        cx += res_widths[i]
    y += 16

    for r_idx, r in enumerate(res_rows):
        bg = BG_BOX if r_idx % 2 == 1 else (1.0, 1.0, 1.0)
        page2.draw_rect(fitz.Rect(table_x, y, table_x + table_w, y + 14), color=BORDER_COLOR, fill=bg)
        cx = table_x + 4
        for i, val in enumerate(r):
            fn = "hebo" if i == 0 or r_idx == len(res_rows) - 1 else "helv"
            page2.insert_text((cx, y + 10), val, fontsize=7.8, fontname=fn, color=TEXT_COLOR)
            cx += res_widths[i]
        y += 14

    y += 18

    # 5. December 2025 Forecast Analysis & Chart
    page2.insert_text((40, y), "5. December 2025 Forecast & Spot Market Analysis", fontsize=13, fontname="hebo", color=PRIMARY_COLOR)
    page2.draw_line(fitz.Point(40, y + 4), fitz.Point(572, y + 4), color=PRIMARY_COLOR, width=1.0)
    y += 16

    chart_img = Path("scorer_results/candidate_december.png")
    if chart_img.is_file():
        img_rect = fitz.Rect(40, y, 572, y + 185)
        page2.insert_image(img_rect, filename=str(chart_img))
        y += 192

    dec_text = (
        "The figure above displays predictions generated by score.py for the fixed lane (Lexington to Fort Wayne, 360 miles, "
        "Dry Van, 32,000 lbs). The forecast exhibits three critical freight domain properties:\n"
        "• Historical Level Alignment: December predictions average $847.10 (range: $833.00 to $861.00), perfectly tracking the "
        "October training benchmark median of $842.00 for comparable 300-400 mile Dry Van hauls.\n"
        "• Mid-Week Tender Surges: The periodic intra-week oscillation peaks on Thursdays and reaches troughs on Sundays/Mondays, "
        "reflecting weekly freight tender volume and market index seasonality (~2.6% amplitude).\n"
        "• Drift Regularization: The 70/30 blend prevents uncontrolled upward extrapolation while respecting market index movements."
    )
    rect5 = fitz.Rect(40, y, 572, y + 54)
    page2.insert_textbox(rect5, dec_text, fontsize=8.5, fontname="helv", color=TEXT_COLOR, lineheight=1.25)
    y += 60

    # 6. Operational Considerations & Limitations
    page2.insert_text((40, y), "6. Production Readiness & Limitations", fontsize=11, fontname="hebo", color=PRIMARY_COLOR)
    page2.draw_line(fitz.Point(40, y + 3), fitz.Point(572, y + 3), color=PRIMARY_COLOR, width=0.8)
    y += 12

    lim_text = (
        "• Label Noise Gap: Final evaluation includes corrupted test labels (~1.4%) that cannot be identified a priori. True hidden test WAPE "
        "will naturally track closer to 5.0% rather than 2.8%.\n"
        "• Calendar Holiday Shocks: With only 10 months of training history (Jan-Oct 2025), holiday volume surges (Christmas/New Year) "
        "cannot be inferred empirically from data alone and require operational rate buffers."
    )
    rect6 = fitz.Rect(40, y, 572, y + 32)
    page2.insert_textbox(rect6, lim_text, fontsize=8.0, fontname="helv", color=TEXT_COLOR, lineheight=1.22)

    page2.insert_text((40, 762), "Freight Rate Challenge · Spotter ML Assessment · Page 2 of 2", fontsize=8, fontname="helv", color=MUTED_COLOR)

    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


if __name__ == "__main__":
    out = create_report()
    print(f"Generated PDF report: {out}")
