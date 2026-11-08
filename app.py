"""Ứng dụng Streamlit tra cứu kết quả mô phỏng tồn kho hàng tươi sống theo lead time.

Ứng dụng KHÔNG chạy lại mô hình hay mô phỏng. Mọi con số đọc trực tiếp từ các bảng kết quả đã xuất
bởi code/perishable_inventory_stress_test.py (thư mục app_data/), nên khớp với báo cáo.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

DATA = Path(__file__).parent / "app_data"

SHOCK_LABEL = {"none": "Không trễ", "moderate": "Trễ vừa (15%)", "high": "Trễ cao (30%)"}
POLICY_SHORT = {
    "P1 Safety stock kinh điển": "P1 Safety stock kinh điển",
    "P2 Newsvendor tĩnh (MA28)": "P2 Newsvendor tĩnh",
    "P3 Quantile động (GBM)": "P3 Quantile động (GBM)",
    "Đối chứng: MA28 chọn tau": "Đối chứng: MA28 chọn tau",
    "Đối chứng: GBM tại q*": "Đối chứng: GBM tại q*",
    "Đối chứng: GBM tau tối ưu ex-post": "Đối chứng: GBM tau ex-post",
}
POLICY_ORDER = list(POLICY_SHORT)

st.set_page_config(page_title="Tồn kho hàng tươi sống & lead time", layout="wide")


@st.cache_data
def load(name):
    return pd.read_csv(DATA / name)


def pct(x, d=1):
    return "—" if pd.isna(x) else f"{x * 100:.{d}f}%"


st.title("Chính sách tồn kho hàng tươi sống khi lead time thay đổi")
st.caption("Tra cứu kết quả mô phỏng đã chạy sẵn (Favorita, 6 nhóm hàng tươi sống × 12 cửa hàng). "
           "Không tính lại trong ứng dụng.")

with st.sidebar:
    st.header("Kịch bản")
    L = st.selectbox("Lead time trung bình (ngày)", [1, 2, 3, 4], index=1)
    shock = st.selectbox("Mức trễ giao hàng", list(SHOCK_LABEL), index=1,
                         format_func=lambda k: SHOCK_LABEL[k])
    st.caption("Hạn sử dụng cơ sở (×1) cho các bảng ở tab 1. Tab 2 và tab 3 có lựa chọn riêng.")

tab1, tab2, tab3, tab4 = st.tabs(
    ["So sánh chính sách", "Rút ngắn lead time hay thêm đệm", "Độ nhạy hạn sử dụng", "Chất lượng dự báo"])

# ---------------------------------------------------------------- Tab 1
with tab1:
    t = load("sim_table_policies_by_L_shock.csv")
    d = t[(t.shock == shock) & (t.Lm == L)].copy()
    ci = load("sim_inference_bootstrap.csv")
    ci = ci[(ci.kind == "level") & (ci.scenario == f"{shock}|L={L}")]

    def ci_text(policy, metric):
        r = ci[(ci.comparison == policy) & (ci.metric == metric)]
        if r.empty:
            return "—"
        r = r.iloc[0]
        return f"[{r.ci_low * 100:.1f}; {r.ci_high * 100:.1f}]"

    d["order"] = d.policy.map({p: i for i, p in enumerate(POLICY_ORDER)})
    d = d.sort_values("order")
    show = pd.DataFrame({
        "Chính sách": d.policy.map(POLICY_SHORT),
        "Tỷ lệ chi phí": d.cost_rate.map(pct),
        "KTC 95% chi phí": [ci_text(p, "cost_rate") for p in d.policy],
        "Mức phục vụ (fill)": d.fill_rate.map(pct),
        "Hao hụt": d.waste_rate.map(pct),
        "Số ngày tồn": d.stock_days.map(lambda x: f"{x:.2f}"),
    })
    st.subheader(f"Lead time {L} ngày, {SHOCK_LABEL[shock].lower()}")
    st.dataframe(show, hide_index=True, width="stretch")
    st.bar_chart(d.set_index(d.policy.map(POLICY_SHORT))["cost_rate"], horizontal=True)
    st.caption("KTC 95% lấy từ bootstrap hai chiều (cửa hàng × tuần); chỉ phản ánh dao động mẫu, "
               "không gồm việc chọn τ, cú sốc lead time hay các giả định chi phí. "
               "Các dòng đối chứng không có KTC nếu không nằm trong bảng bootstrap.")

    cmp_ = load("sim_inference_bootstrap.csv")
    cmp_ = cmp_[(cmp_.kind == "policy") & (cmp_.scenario == f"{shock}|L={L}") & (cmp_.metric == "cost_rate")]
    if not cmp_.empty:
        out = pd.DataFrame({
            "So sánh (chi phí)": cmp_.comparison,
            "Chênh lệch": cmp_.estimate.map(lambda x: f"{x * 100:+.1f} điểm %"),
            "KTC 95%": [f"[{a * 100:+.1f}; {b * 100:+.1f}]" for a, b in zip(cmp_.ci_low, cmp_.ci_high)],
            "Khác 0?": ["Có" if (a > 0 or b < 0) else "Không (KTC chứa 0)" for a, b in zip(cmp_.ci_low, cmp_.ci_high)],
        })
        st.markdown("**P3 so với các chính sách còn lại** (âm = P3 rẻ hơn)")
        st.dataframe(out, hide_index=True, width="stretch")
    st.image(str(DATA / "fig12_cost_by_lead_time.png"), caption="Chi phí theo lead time (từ báo cáo)")

# ---------------------------------------------------------------- Tab 2
with tab2:
    sb = load("sim_table_shorten_vs_buffer.csv")
    c1, c2 = st.columns(2)
    mult = c1.selectbox("Hệ số hạn sử dụng", [0.5, 1.0, 1.5], index=1, format_func=lambda m: f"×{m}")
    target = c2.selectbox("Mức phục vụ mục tiêu", [0.90, 0.93, 0.95], index=0, format_func=lambda m: f"{m:.0%}")
    st.subheader(f"Giữ lead time {L} ngày bằng đòn bẩy rẻ nhất, hay rút ngắn đi 1 ngày?")
    if L == 1:
        st.info("Ở lead time 1 ngày không còn bước rút ngắn nào để so sánh. Hãy chọn L ≥ 2 ở thanh bên.")
    else:
        r = sb[(sb.shock == shock) & (sb.Lm == L) & (sb.mult == mult) & (np.isclose(sb.fill_target, target))]
        if r.empty:
            st.warning("Không có dòng dữ liệu cho tổ hợp này.")
        else:
            r = r.iloc[0]
            a, b, c = st.columns(3)
            a.metric("Chỉ nâng τ (tồn kho an toàn)",
                     "Không đạt mục tiêu" if pd.isna(r.cost_raise_tau) else pct(r.cost_raise_tau, 2))
            b.metric("Thêm đệm thời gian",
                     "Không đạt mục tiêu" if pd.isna(r.cost_buffer) else pct(r.cost_buffer, 2))
            c.metric(f"Rút ngắn xuống L = {L - 1}",
                     "Không đạt mục tiêu" if pd.isna(r.cost_shorter_L) else pct(r.cost_shorter_L, 2))
            if pd.notna(r.gain_shorten_1day):
                st.markdown(
                    f"Rút ngắn 1 ngày làm tỷ lệ chi phí giảm **{pct(r.gain_shorten_1day, 2)}**, "
                    f"tức **{pct(r.gain_pct_of_cost, 0)}** chi phí của đòn bẩy rẻ nhất. "
                    "Đây là mức tối đa đáng chi cho việc rút ngắn, tính theo cùng đơn vị tỷ lệ chi phí.")
            else:
                st.markdown("Chưa tính được mức tiết kiệm vì một trong hai vế không đạt mục tiêu phục vụ.")
            if L == 2:
                st.warning("Bước L = 2 → 1 bị phóng đại: ở L = 1 cú sốc về sớm bị chặn tối thiểu 1 ngày. "
                           "Kiểm tra chỉ-về-trễ cho thấy khoảng 2/3 mức nhảy là artefact. Nên đọc kết quả L ≥ 3 chắc chắn hơn.")
    st.image(str(DATA / "fig14_shorten_vs_buffer.png"), caption="Rút ngắn lead time và thêm đệm (từ báo cáo)")

# ---------------------------------------------------------------- Tab 3
with tab3:
    st.subheader("Hạn sử dụng ảnh hưởng thế nào đến từng chính sách")
    st.caption("Ca cơ sở: lead time 2 ngày, trễ vừa. Hệ số nhân vào hạn sử dụng gốc của từng nhóm.")
    s = load("sim_table_sensitivity_shelf_life.csv")
    s["Chính sách"] = s.policy.map(POLICY_SHORT).fillna(s.policy)
    view = pd.DataFrame({
        "Chính sách": s["Chính sách"], "Hạn sử dụng": s.mult.map(lambda m: f"×{m}"),
        "Tỷ lệ chi phí": s.cost_rate.map(pct), "Mức phục vụ (fill)": s.fill_rate.map(pct),
        "Hao hụt": s.waste_rate.map(pct), "Số ngày tồn": s.stock_days.map(lambda x: f"{x:.2f}"),
    })
    st.dataframe(view, hide_index=True, width="stretch")
    st.image(str(DATA / "fig13_waste_vs_shelf_life.png"), caption="Hao hụt theo hạn sử dụng (từ báo cáo)")

    st.subheader("Hao hụt mô phỏng so với số liệu USDA ERS")
    w = load("sim_table_waste_vs_ers.csv").rename(columns={
        "family": "Nhóm hàng", "0.5": "Hạn ×0,5", "1.0": "Hạn ×1", "1.5": "Hạn ×1,5", "ERS_retail_loss": "ERS (bán lẻ)"})
    for col in ["Hạn ×0,5", "Hạn ×1", "Hạn ×1,5", "ERS (bán lẻ)"]:
        w[col] = w[col].map(pct)
    st.dataframe(w, hide_index=True, width="stretch")
    st.caption("Hao hụt mô phỏng chỉ là cận dưới so với số liệu thực tế: nhiều nhóm gần như không hao hụt "
               "trong mô phỏng (hạn dài hơn thời gian bán), trong khi ERS ghi 7–12%.")

# ---------------------------------------------------------------- Tab 4
with tab4:
    st.subheader("Dự báo phân vị: GBM so với các mô hình cơ sở")
    fm = load("forecast_metrics.csv")
    metric = st.radio("Chỉ số", ["WAPE", "wQL", "calib_err"], horizontal=True,
                      format_func=lambda m: {"WAPE": "WAPE (sai số tuyệt đối có trọng số)",
                                             "wQL": "wQL (pinball có trọng số)",
                                             "calib_err": "Sai lệch hiệu chuẩn"}[m])
    piv = fm.pivot(index="model", columns="L", values=metric)
    piv.columns = [f"L={c}" for c in piv.columns]
    st.dataframe(piv.map(lambda x: pct(x, 2) if metric != "calib_err" else f"{x:.3f}"),
                 width="stretch")
    st.caption("Mục tiêu dự báo là tổng cầu L ngày, nên sai số giảm theo L chủ yếu do cộng gộp, "
               "không phải vì dự báo xa dễ hơn.")
    c1, c2 = st.columns(2)
    c1.image(str(DATA / "fig06_forecast_accuracy_by_L.png"), caption="Độ chính xác theo L")
    c2.image(str(DATA / "fig07_forecast_calibration.png"), caption="Hiệu chuẩn")

    with st.expander("Điều không nên kết luận từ các kết quả này"):
        st.markdown(
            "- GBM tốt hơn rõ về wQL, nhưng chi phí chỉ thấp hơn đối chứng \"MA28 chọn τ\" một phần nhỏ; "
            "phần lớn lợi ích đến từ việc chọn τ theo chi phí.\n"
            "- Ở 4/6 nhóm hàng, P3 chọn τ tại trần 0,95 của lưới, nên chi phí P3 ở các nhóm này là cận trên.\n"
            "- Lead time là biến kịch bản (stress test), không phải số đo thật; hạn sử dụng, biên lợi nhuận và "
            "chi phí giữ hàng là giả định.\n"
            "- Hao hụt mô phỏng thấp hơn số liệu ERS; dữ liệu quan sát là doanh số, không phải cầu thực khi hết hàng.")
