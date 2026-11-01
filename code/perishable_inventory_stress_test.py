"""
Tồn kho hàng tươi sống: stress test lead time trên bộ dữ liệu Favorita (Kaggle Store Sales).

Toàn bộ pipeline nằm trong MỘT file, chạy từ trên xuống dưới trong PyCharm (chuột phải > Run).
Dữ liệu: https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data
Đặt các file CSV của Kaggle vào ../Dataset (so với vị trí file này). Kết quả ghi vào ../outputs.
"""
import hashlib
import inspect
import pickle
import time
from pathlib import Path

import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats

SHOW_FIGS = False
if not SHOW_FIGS:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# BƯỚC 0: CẤU HÌNH
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "Dataset"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

SEED = 42
START_DATE = "2015-06-01"          # trước 06/2015 PRODUCE gần như không có doanh số; tỷ lệ khuyến mãi cũng còn trôi mạnh
MAX_STORE_ZERO_DAYS = 10           # một cửa hàng được phép có tối đa ngần này ngày doanh số = 0 kể từ START_DATE
MAX_SERIES_ZERO_SHARE = 0.10       # mọi family của cửa hàng được giữ phải có bán ở >= 90% số ngày mở cửa
N_STORES = 12
# Sáu family có một nhóm thực phẩm đơn lẻ để đối chiếu hạn sử dụng (Food Keeper) và hao hụt (USDA ERS).
# MEATS, DELI và PREPARED FOODS gồm nhiều mặt hàng khác nhau nên không có một hạn sử dụng đại diện, vì vậy không đưa vào.
FAMILIES = ["PRODUCE", "BREAD/BAKERY", "POULTRY", "DAIRY", "EGGS", "SEAFOOD"]

# Hạn sử dụng (ngày). Căn cứ: FMI & Cornell, "The Food Keeper" (thời gian bảo quản lạnh tại nhà),
# nên hạn sử dụng tại cửa hàng tính từ lúc nhập có thể ngắn hơn; vì vậy giá trị cơ sở được thử thêm với SHELF_LIFE_MULT.
SHELF_LIFE = pd.DataFrame({
    "shelf_life_base": [5, 3, 2, 7, 21, 2],
    "foodkeeper_evidence": [
        "leaf lettuce 3-7d, spinach 1-2d, tomatoes 2-3d, carrots 3wk, broccoli 3-5d",
        "commercial bread 2-4d on shelf, cakes 1-2d",
        "chicken parts 1-2d refrigerated",
        "milk 1wk, yogurt 7-14d",
        "eggs in shell 3-5wk",
        "raw fish 1-2d refrigerated"],
    # USDA ERS EIB-121 (Buzby, Wells & Hyman, 2014): hao hụt khâu bán lẻ, % nguồn cung thực phẩm, năm 2010
    "ers_retail_loss_pct": [10, 12, 4, 11, 7, 8],
    "ers_category": ["fresh vegetables", "grain products", "poultry", "dairy products",
                     "eggs", "fish and seafood"],
}, index=FAMILIES)
SHELF_LIFE_MULT = [0.5, 1.0, 1.5]

# Kinh tế đơn vị, chuẩn hoá giá vốn c = 1. Không có nguồn cho biên lợi nhuận theo family: dùng giả định + phân tích độ nhạy.
GROSS_MARGIN = {"base": 0.30, "grid": [0.20, 0.30, 0.45]}       # giá bán p = c / (1 - biên)
CARRY_RATE_ANNUAL = {"base": 0.25, "grid": [0.10, 0.25, 0.30]}  # chi phí giữ hàng mỗi đơn vị-ngày = c * tỷ lệ / 365
SALVAGE_VALUE = 0.0                                              # hàng hết hạn bị huỷ, không thu hồi được gì

# Lead time L (số ngày từ lúc đặt hàng đến lúc hàng sẵn sàng, tối thiểu 1). Favorita không có dữ liệu lead time,
# nên L là biến kịch bản chứ không phải giá trị ước lượng.
LEAD_TIME_MEAN = [1, 2, 3, 4]
# Với xác suất p, lead time lệch 1 ngày (sớm hoặc muộn với khả năng như nhau); L không thấp hơn 1.
LEAD_TIME_SHOCK_PROB = {"none": 0.00, "moderate": 0.15, "high": 0.30}
SERVICE_LEVEL_CLASSIC = 0.95

# Thiết kế dự báo. Mục tiêu là tổng cầu L ngày tới (L = lead time, tối đa 4 ngày + 1 ngày trễ).
HORIZONS = [1, 2, 3, 4, 5]
TAUS = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
TUNE_START = "2017-01-10"          # cửa sổ kiểm định siêu tham số (91 ngày), kết thúc ngay trước WF_START
WF_START = "2017-04-11"            # walk-forward bắt đầu; 35 ngày đầu là giai đoạn khởi động cho mô phỏng tồn kho
EVAL_START = "2017-05-16"          # cửa sổ đánh giá ngoài mẫu = 92 ngày cuối của dữ liệu
REFIT_EVERY = 28                   # huấn luyện lại mô hình mỗi 28 ngày (cửa sổ mở rộng)
RESID_WINDOW = 84                  # số ngày phần dư dùng để đổi dự báo điểm của baseline thành phân vị
USE_CACHE = True                   # dùng lại kết quả đã fit nếu code + cấu hình không đổi (outputs/cache)
N_JOBS = 8

plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3})


def save_fig(name):
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{name}.png", bbox_inches="tight")
    if SHOW_FIGS:
        plt.show()
    plt.close()


def banner(text):
    print("\n" + "=" * 80 + f"\n{text}\n" + "=" * 80)


# ----------------------------------------------------------------------------
# BƯỚC 1: NẠP DỮ LIỆU VÀ KIỂM TRA
# ----------------------------------------------------------------------------
def load_data():
    banner("BƯỚC 1: NẠP DỮ LIỆU VÀ KIỂM TRA")
    train = pd.read_csv(DATA_DIR / "train.csv", parse_dates=["date"],
                        dtype={"store_nbr": "int16", "sales": "float32", "onpromotion": "int32"})
    stores = pd.read_csv(DATA_DIR / "stores.csv")
    holidays = pd.read_csv(DATA_DIR / "holidays_events.csv", parse_dates=["date"])
    oil = pd.read_csv(DATA_DIR / "oil.csv", parse_dates=["date"])

    all_days = pd.date_range(train.date.min(), train.date.max())
    missing = all_days.difference(train.date.unique())
    print(f"train: {train.shape[0]:,} dòng | {train.date.min().date()} -> {train.date.max().date()}")
    print(f"số cửa hàng: {train.store_nbr.nunique()} | số family: {train.family.nunique()} | "
          f"số chuỗi: {train.groupby(['store_nbr', 'family']).ngroups}")
    print(f"ngày lịch bị thiếu: {[d.strftime('%Y-%m-%d') for d in missing]}")
    print(f"giá trị null: {int(train.isna().sum().sum())} | doanh số âm: {int((train.sales < 0).sum())}")
    print(f"giá dầu thiếu: {int(oil.dcoilwtico.isna().sum())} / {len(oil)} ngày")
    return train, stores, holidays, oil


# ----------------------------------------------------------------------------
# BƯỚC 2: CHỌN PHẠM VI (cửa hàng x nhóm hàng tươi sống)
# ----------------------------------------------------------------------------
def select_scope(train, stores):
    banner("BƯỚC 2: CHỌN PHẠM VI NGHIÊN CỨU")
    t = train[train.date >= START_DATE]
    store_day = t.groupby(["store_nbr", "date"]).sales.sum().unstack(0)
    zero_days = (store_day == 0).sum().rename("zero_days")
    first_sale = store_day.apply(lambda s: s[s > 0].index.min()).rename("first_sale")
    audit = stores.set_index("store_nbr").join(zero_days).join(first_sale)

    print(f"số ngày doanh số = 0 của mỗi cửa hàng từ {START_DATE} (tổng {len(store_day)} ngày):")
    print(audit.zero_days.describe().round(1).to_string())
    store_ok = audit[audit.zero_days <= MAX_STORE_ZERO_DAYS]
    excluded = audit[audit.zero_days > MAX_STORE_ZERO_DAYS]
    print(f"\nLọc A (mở muộn / đóng cửa dài): giữ {len(store_ok)}, loại {len(excluded)}")
    print(excluded[["city", "type", "zero_days", "first_sale"]].to_string())

    open_days = store_day.gt(0).stack()
    open_days = open_days[open_days].index
    t = t.set_index(["date", "store_nbr"])
    n_before = len(t)
    t = t[t.index.isin(open_days)].reset_index()
    print(f"\nsố dòng ngày đóng cửa bị bỏ (tổng doanh số cửa hàng = 0, đóng cửa không phải là cầu): {n_before - len(t):,}")

    zero_share = (t[t.family.isin(FAMILIES)].groupby(["store_nbr", "family"]).sales
                  .apply(lambda s: (s == 0).mean()).unstack())
    series_ok = zero_share.max(axis=1) <= MAX_SERIES_ZERO_SHARE
    print(f"\nLọc B (mọi family đều hoạt động, tỷ lệ ngày bằng 0 <= {MAX_SERIES_ZERO_SHARE:.0%}): "
          f"số cửa hàng không đạt theo từng family:")
    print((zero_share > MAX_SERIES_ZERO_SHARE).sum().to_string())
    eligible = store_ok[store_ok.index.isin(series_ok[series_ok].index)]
    print(f"số cửa hàng đủ điều kiện sau lọc A và B: {len(eligible)}")

    rng = np.random.default_rng(SEED)
    share = eligible.type.value_counts(normalize=True) * N_STORES
    quota = share.astype(int).clip(lower=1)
    while quota.sum() < N_STORES:
        quota[(share - quota).idxmax()] += 1
    while quota.sum() > N_STORES:
        quota[quota.idxmax()] -= 1
    picked = []
    for stype, n in quota.items():
        pool = eligible[eligible.type == stype].index.to_numpy()
        picked += list(rng.choice(pool, size=min(n, len(pool)), replace=False))
    picked = sorted(int(s) for s in picked)
    print(f"\nđã chọn {len(picked)} cửa hàng (phân tầng theo loại cửa hàng, seed={SEED}): {picked}")
    print(audit.loc[picked, ["city", "state", "type", "cluster"]].to_string())

    scope = t[t.store_nbr.isin(picked) & t.family.isin(FAMILIES)].copy()
    print(f"\nphạm vi: {len(picked)} cửa hàng x {len(FAMILIES)} family = {len(picked) * len(FAMILIES)} chuỗi | "
          f"{len(scope):,} dòng | {scope.date.min().date()} -> {scope.date.max().date()}")
    audit.loc[picked].to_csv(OUT_DIR / "scope_stores.csv")
    return scope, picked, audit


def summarise_scope(scope):
    banner("BƯỚC 2b: TÓM TẮT CÁC CHUỖI TRONG PHẠM VI")
    g = scope.groupby("family").agg(
        series=("store_nbr", "nunique"),
        mean_daily_sales=("sales", "mean"),
        zero_share=("sales", lambda s: (s == 0).mean()),
        promo_share=("onpromotion", lambda s: (s > 0).mean()),
    )
    # CV tính trong từng chuỗi (cửa hàng x family) rồi lấy trung bình, cùng định nghĩa với fig05; không gộp các cửa hàng
    g["cv"] = scope.groupby(["family", "store_nbr"]).sales.agg(lambda s: s.std() / s.mean()).groupby("family").mean()
    g = g[["series", "mean_daily_sales", "zero_share", "cv", "promo_share"]].round(3)
    print(g.to_string())
    frac = (scope.groupby("family").sales.apply(lambda s: (s % 1 != 0).mean())).round(3)
    print("\ntỷ lệ doanh số không nguyên (bán theo cân, kg):", frac.to_dict())
    g.to_csv(OUT_DIR / "scope_family_summary.csv")
    return g


# ----------------------------------------------------------------------------
# BƯỚC 3: BIỂU ĐỒ KHÁM PHÁ DỮ LIỆU (EDA)
# ----------------------------------------------------------------------------
def eda_figures(train, scope, picked):
    banner("BƯỚC 3: BIỂU ĐỒ EDA")

    promo = train.groupby("date").onpromotion.apply(lambda s: (s > 0).mean())
    fig, ax = plt.subplots(figsize=(9, 3.2))
    promo.rolling(28, min_periods=1).mean().plot(ax=ax)
    ax.axvline(pd.Timestamp(START_DATE), color="red", ls="--", label=f"phạm vi bắt đầu {START_DATE}")
    ax.set(title="Tỷ lệ chuỗi có khuyến mãi (trung bình 28 ngày, toàn bộ dữ liệu)",
           xlabel="Ngày", ylabel="Tỷ lệ")
    ax.legend()
    save_fig("fig01_promo_share_drift")

    d = scope.groupby(["family", "date"]).sales.sum().reset_index()
    d["dow"] = d.date.dt.dayofweek
    prof = d.groupby(["family", "dow"]).sales.mean().unstack(0)
    prof = prof / prof.mean()
    fig, ax = plt.subplots(figsize=(8, 3.6))
    prof.plot(marker="o", ax=ax)
    ax.set(title="Hồ sơ theo thứ trong tuần (1.0 = trung bình của family)",
           xlabel="Thứ trong tuần (0 = Thứ Hai ... 6 = Chủ Nhật)", ylabel="Chỉ số")
    ax.legend(ncol=3, fontsize=8)
    save_fig("fig02_dow_profile")

    m = scope.groupby(["family", pd.Grouper(key="date", freq="MS")]).sales.mean().unstack(0)
    days = scope.groupby(pd.Grouper(key="date", freq="MS")).date.nunique()
    m = m[days >= 28]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    (m / m.mean()).plot(ax=ax)
    ax.set(title="Chỉ số theo tháng của doanh số trung bình ngày, theo family (1.0 = trung bình, chỉ tháng đủ ngày)",
           xlabel="Tháng", ylabel="Chỉ số")
    ax.legend(ncol=3, fontsize=8)
    save_fig("fig03_monthly_index")

    s0 = picked[0]
    ex = scope[(scope.store_nbr == s0) & (scope.date >= scope.date.max() - pd.Timedelta(days=120))]
    fig, axes = plt.subplots(len(FAMILIES), 1, figsize=(9, 1.7 * len(FAMILIES)), sharex=True)
    for ax, fam in zip(axes, FAMILIES):
        e = ex[ex.family == fam].set_index("date")
        ax.plot(e.index, e.sales, lw=1)
        ax.set_ylabel(fam, rotation=0, ha="right", fontsize=8)
    axes[0].set_title(f"120 ngày gần nhất, cửa hàng {s0}")
    axes[-1].set_xlabel("Ngày")
    save_fig("fig04_example_series")

    cv = scope.groupby(["family", "store_nbr"]).sales.agg(lambda s: s.std() / s.mean()).unstack(0)
    fig, ax = plt.subplots(figsize=(8, 3.4))
    cv.boxplot(ax=ax)
    ax.set(title="Hệ số biến thiên của doanh số ngày theo family (mỗi điểm là một cửa hàng)",
           ylabel="Hệ số biến thiên")
    plt.setp(ax.get_xticklabels(), fontsize=8)
    save_fig("fig05_cv_by_family")
    print(f"đã lưu 5 biểu đồ vào {OUT_DIR}")


def driver_analysis(pn, oil):
    """Tương quan hạng (Spearman) giữa độ lệch doanh số so với xu hướng và các yếu tố giải thích, theo family.
    Cả hai vế đều loại xu hướng 28 ngày (trung bình trượt căn giữa); doanh số còn loại thêm hiệu ứng thứ trong tuần."""
    banner("BƯỚC 3b: QUAN HỆ GIỮA DOANH SỐ VÀ CÁC YẾU TỐ GIẢI THÍCH")
    cal, fam = pn["cal"], pn["fam_of"]
    sdf = pd.DataFrame(pn["S"], index=cal)
    trend = lambda x: x.rolling(28, center=True, min_periods=14).mean()
    rel = sdf / trend(sdf)
    rel = rel / rel.groupby(cal.dayofweek).transform("mean")
    o = oil.set_index("date").dcoilwtico.reindex(cal).ffill().bfill()
    drivers = ["Khuyến mãi (độ phủ)", "Lễ (theo địa phương)", "Sự kiện quốc gia", "Ngày lương (15, cuối tháng)", "Giá dầu"]
    corr, rows = {}, []
    for f in FAMILIES:
        cols = np.where(fam == f)[0]
        y = rel.iloc[:, cols].mean(axis=1)
        x = pd.DataFrame({
            "Khuyến mãi (độ phủ)": pd.Series((pn["LP"][:, cols] > 0).mean(axis=1), index=cal),
            "Lễ (theo địa phương)": pd.Series(pn["H"][:, cols].mean(axis=1), index=cal),
            "Sự kiện quốc gia": pd.Series(pn["EV"][:, 0], index=cal),
            "Ngày lương (15, cuối tháng)": pd.Series(pn["PD"][:, 0], index=cal),
            "Giá dầu": o})
        x["Khuyến mãi (độ phủ)"] -= trend(x["Khuyến mãi (độ phủ)"])
        x["Giá dầu"] = x["Giá dầu"] / trend(x["Giá dầu"]) - 1
        corr[f] = {k: y.corr(x[k], method="spearman") for k in drivers}
        r, p = rel.iloc[:, cols].to_numpy(), pn["LP"][:, cols] > 0
        ok = ~np.isnan(r)
        rows.append(dict(family=f, promo_day_share=p[ok].mean(), promo_lift=r[ok & p].mean() / r[ok & ~p].mean()))
    C = pd.DataFrame(corr).T
    print("Tương quan hạng Spearman giữa độ lệch doanh số (đã loại xu hướng và thứ trong tuần) và từng yếu tố:")
    print(C.round(2).to_string())
    C.to_csv(OUT_DIR / "eda_driver_correlation.csv")

    fam_sales = sdf.T.groupby(fam).sum().T
    yoy = fam_sales.loc[EVAL_START:].mean() / fam_sales.loc["2016-05-16":"2016-08-15"].mean() - 1
    tab = pd.DataFrame(rows).set_index("family")
    tab["yoy_growth_eval_window"] = yoy
    tab = tab.loc[FAMILIES]
    print("\nHiệu ứng khuyến mãi (doanh số tương đối ngày có khuyến mãi / ngày không, cùng chuỗi) và tăng trưởng so với "
          "cùng kỳ 2016 (cửa sổ đánh giá):")
    print(tab.round(3).to_string())
    tab.to_csv(OUT_DIR / "eda_promo_lift_yoy.csv")

    fig, ax = plt.subplots(figsize=(8, 3.6))
    im = ax.imshow(C.to_numpy(float), cmap="RdBu_r", vmin=-0.5, vmax=0.5, aspect="auto")
    ax.set_xticks(range(C.shape[1]), C.columns, fontsize=8, rotation=15)
    ax.set_yticks(range(C.shape[0]), C.index, fontsize=8)
    ax.grid(False)
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            ax.text(j, i, f"{C.iat[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="Spearman")
    ax.set_title("Tương quan giữa độ lệch doanh số và các yếu tố giải thích (đã loại xu hướng 28 ngày, thứ trong tuần)",
                 fontsize=9)
    save_fig("fig05b_driver_correlation")


# ----------------------------------------------------------------------------
# BƯỚC 4: BẢNG GIẢ ĐỊNH
# ----------------------------------------------------------------------------
def assumptions_report():
    banner("BƯỚC 4: CÁC GIẢ ĐỊNH")
    m, r = GROSS_MARGIN["base"], CARRY_RATE_ANNUAL["base"]
    p = 1 / (1 - m)
    cu = p - 1
    co = 1 - SALVAGE_VALUE + r / 365
    print(f"Kinh tế đơn vị (c = 1): giá bán p = {p:.3f} | chi phí thiếu hàng Cu = p - c = {cu:.3f} | "
          f"chi phí thừa hàng Co = c - salvage + giữ hàng 1 ngày = {co:.3f}")
    print(f"Critical ratio newsvendor 1 kỳ tại biên cơ sở: Cu / (Cu + Co) = {cu / (cu + co):.3f}")
    print("Critical ratio theo biên lợi nhuận:", {mg: round((mg / (1 - mg)) / ((mg / (1 - mg)) + co), 3)
                                                  for mg in GROSS_MARGIN["grid"]})
    print(f"\nHệ số hạn sử dụng được thử: {SHELF_LIFE_MULT}")
    print(f"Kịch bản lead time: trung bình {LEAD_TIME_MEAN} x xác suất trễ/sớm {LEAD_TIME_SHOCK_PROB}")
    print(f"Mức phục vụ của safety stock kinh điển: {SERVICE_LEVEL_CLASSIC}")
    print("\nBảng family:")
    print(SHELF_LIFE[["shelf_life_base", "ers_retail_loss_pct", "ers_category"]].to_string())
    SHELF_LIFE.to_csv(OUT_DIR / "assumptions_family_table.csv")


# ----------------------------------------------------------------------------
# BƯỚC 5: DỰ BÁO - dựng bảng dữ liệu và đặc trưng
# ----------------------------------------------------------------------------
CAT_FEATS = ["dow", "family_code", "store_code"]
FUTURE_COVS = ["promo_rel", "promo0", "hol_L", "event_L", "payday_L"]   # lịch khuyến mãi/lễ/ngày lương biết trước
FEATS = (["lag%d" % j for j in range(7)] + ["lag13", "lag20", "lag27", "ma7", "ma14", "ma56", "sd7", "sd28",
                                            "wp_ratio", "sn7_ratio"]
         + FUTURE_COVS + ["quake0", "dow", "family_code", "store_code"])
GBM_FIXED = dict(learning_rate=0.05, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0)
BASELINES = {"Seasonal naive 7": "b_sn7", "Weekly profile 8w": "b_wp", "MA28": "b_ma28", "Naive": "b_naive"}


def shift_back(A, m):
    """Giá trị quan sát được m ngày trước ngày gốc (NaN nếu chưa có)."""
    out = np.full(A.shape, np.nan)
    if m == 0:
        return np.asarray(A, dtype=float).copy()
    out[m:] = A[:-m]
    return out


def shift_fwd(A, k):
    """Giá trị k ngày sau ngày gốc (NaN nếu vượt quá cuối dữ liệu)."""
    out = np.full(A.shape, np.nan)
    out[:-k] = A[k:]
    return out


def fwd_mean(A, L):
    return np.mean([shift_fwd(A, k) for k in range(1, L + 1)], axis=0)


def weekly_profile(S, k, n_weeks=8, min_obs=4):
    """Trung bình doanh số cùng thứ của n_weeks tuần gần nhất cho ngày t+k, chỉ dùng dữ liệu đến hết ngày t."""
    stack = np.stack([shift_back(S, 7 * j - k) for j in range(1, n_weeks + 1)])
    cnt = (~np.isnan(stack)).sum(axis=0)
    mean = np.nansum(stack, axis=0) / np.maximum(cnt, 1)
    mean[cnt < min_obs] = np.nan
    return mean


def build_panel(scope, stores, holidays):
    """Mảng rộng (ngày x chuỗi) trên lịch đầy đủ. Ngày cửa hàng đóng cửa để NaN (không phải cầu)."""
    cal = pd.date_range(scope.date.min(), scope.date.max())
    wide = scope.set_index(["date", "store_nbr", "family"])
    S_df = wide.sales.unstack(["store_nbr", "family"]).reindex(cal)
    P_df = wide.onpromotion.unstack(["store_nbr", "family"]).reindex(cal).fillna(0)
    store_of = np.array([c[0] for c in S_df.columns])
    fam_of = np.array([c[1] for c in S_df.columns])
    T, N = S_df.shape
    S = S_df.to_numpy(float)
    scale = S_df.rolling(28, min_periods=14).mean().to_numpy()

    base = {}
    for j in list(range(7)) + [13, 20, 27]:
        base["lag%d" % j] = shift_back(S, j) / scale
    for name, w, mp in [("ma7", 7, 4), ("ma14", 14, 7), ("ma56", 56, 28)]:
        base[name] = S_df.rolling(w, min_periods=mp).mean().to_numpy() / scale
    for name, w in [("sd7", 7), ("sd28", 28)]:
        base[name] = S_df.rolling(w, min_periods=4).std().to_numpy() / scale

    LP = np.log1p(P_df.to_numpy(float))
    lp28 = pd.DataFrame(LP).rolling(28, min_periods=7).mean().to_numpy()

    assert holidays.transferred.dtype == bool
    hol = holidays[holidays.type.isin(["Holiday", "Transfer", "Additional", "Bridge"]) & ~holidays.transferred]
    is_quake = holidays.description.str.contains("Terremoto")
    ev_dates = holidays[(holidays.type == "Event") & ~is_quake & (holidays.locale == "National")].date.unique()
    st = stores.set_index("store_nbr")
    H = np.zeros((T, N))
    for j, sn in enumerate(store_of):
        m = ((hol.locale == "National")
             | ((hol.locale == "Regional") & (hol.locale_name == st.loc[sn, "state"]))
             | ((hol.locale == "Local") & (hol.locale_name == st.loc[sn, "city"])))
        H[:, j] = cal.isin(hol.loc[m, "date"])
    bcast = lambda v: np.repeat(np.asarray(v, float)[:, None], N, axis=1)
    EV = bcast(cal.isin(ev_dates))
    QK = bcast(cal.isin(holidays[is_quake].date.unique()))
    PD = bcast((cal.day == 15) | cal.is_month_end)

    return dict(cal=cal, S=S, T=T, N=N, scale=scale, base=base, LP=LP, lp28=lp28, H=H, EV=EV, QK=QK, PD=PD,
                store_of=store_of, fam_of=fam_of, columns=S_df.columns,
                wp={k: weekly_profile(S, k) for k in range(1, max(HORIZONS) + 1)},
                sn7={k: shift_back(S, 7 - k) for k in range(1, max(HORIZONS) + 1)})


def make_L_frame(pn, L):
    """Mỗi dòng = một (ngày gốc, chuỗi). Mục tiêu = cầu các ngày t+1..t+L chia cho (L * scale_t)."""
    T, N, S, sc = pn["T"], pn["N"], pn["S"], pn["scale"]
    ks = range(1, L + 1)
    Y = np.sum([shift_fwd(S, k) for k in ks], axis=0)
    wp = np.sum([pn["wp"][k] for k in ks], axis=0)
    sn = np.sum([pn["sn7"][k] for k in ks], axis=0)
    d = dict(pn["base"])
    d["wp_ratio"] = wp / (L * sc)
    d["sn7_ratio"] = sn / (L * sc)
    d["promo_rel"] = fwd_mean(pn["LP"], L) - pn["lp28"]
    d["promo0"] = pn["LP"] - pn["lp28"]
    d["hol_L"] = fwd_mean(pn["H"], L)
    d["event_L"] = fwd_mean(pn["EV"], L)
    d["payday_L"] = fwd_mean(pn["PD"], L)
    d["quake0"] = pn["QK"]
    d["y"], d["target"], d["scale"] = Y, Y / (L * sc), sc
    d["b_naive"], d["b_ma28"], d["b_sn7"], d["b_wp"] = L * S, L * sc, sn, wp
    di = np.repeat(np.arange(T)[:, None], N, axis=1)
    si = np.repeat(np.arange(N)[None, :], T, axis=0)
    d["di"], d["si"] = di, si
    d["dow"] = np.repeat(pn["cal"].dayofweek.to_numpy()[:, None], N, axis=1)
    d["family_code"] = pd.Series(pn["fam_of"]).astype("category").cat.codes.to_numpy()[si]
    d["store_code"] = pd.Series(pn["store_of"]).astype("category").cat.codes.to_numpy()[si]
    d["win_ok"] = di + L <= T - 1
    return pd.DataFrame({k: v.ravel() for k, v in d.items()})


def wf_blocks(T, cal):
    blocks, s = [], cal.get_loc(pd.Timestamp(WF_START))
    while s <= T - 2:
        e = min(s + REFIT_EVERY - 1, T - 2)
        blocks.append((s, e))
        s = e + 1
    return blocks


def pinball(y, q, tau):
    d = y - q
    return np.maximum(tau * d, (tau - 1) * d)


def fit_gbm(X, y, tau, params):
    m = lgb.LGBMRegressor(objective="quantile", alpha=tau, random_state=SEED, n_jobs=1, verbose=-1,
                          deterministic=True, force_row_wise=True, **GBM_FIXED, **params)
    m.fit(X, y, categorical_feature=CAT_FEATS)
    return m


def train_mask(fr, last_di):
    return ((fr.di <= last_di) & fr.target.notna() & (fr.scale > 0)).to_numpy()


def tune_gbm(frames, pn):
    """Lưới nhỏ trên cửa sổ kiểm định nằm trước giai đoạn walk-forward; chọn xong thì khoá lại."""
    banner("BƯỚC 5b: CHỌN SIÊU THAM SỐ (cửa sổ kiểm định nằm trước walk-forward)")
    cal = pn["cal"]
    v0, v1 = cal.get_loc(pd.Timestamp(TUNE_START)), cal.get_loc(pd.Timestamp(WF_START)) - 1
    grid = [dict(num_leaves=nl, min_child_samples=mc, n_estimators=ne)
            for nl in (7, 15, 31) for mc in (50, 200) for ne in (150, 400)]
    tune_L, tune_taus = (1, 3), (0.3, 0.5, 0.9)
    rows = []
    for params in grid:
        losses = []
        for L in tune_L:
            fr = frames[L]
            tr = fr[train_mask(fr, v0 - 1 - L)]
            va = fr[((fr.di >= v0) & (fr.di <= v1) & fr.win_ok & fr.y.notna() & (fr.scale > 0)).to_numpy()]
            models = Parallel(n_jobs=N_JOBS, prefer="threads")(
                delayed(fit_gbm)(tr[FEATS], tr.target, t, params) for t in tune_taus)
            for t, m in zip(tune_taus, models):
                q = m.predict(va[FEATS]) * L * va.scale.to_numpy()
                losses.append(2 * pinball(va.y.to_numpy(), q, t).sum() / va.y.sum())
        rows.append({**params, "wQL": float(np.mean(losses))})
    res = pd.DataFrame(rows).sort_values("wQL").reset_index(drop=True)
    print(res.round(4).to_string())
    res.to_csv(OUT_DIR / "forecast_tuning.csv", index=False)
    best = {k: int(res.loc[0, k]) for k in ("num_leaves", "min_child_samples", "n_estimators")}
    print(f"\nsiêu tham số đã khoá: {best}")
    return best


def gbm_walk_forward(frames, pn, params, taus, horizons, feats):
    cal = pn["cal"]
    out = {L: np.full((len(frames[L]), len(taus)), np.nan) for L in horizons}
    imp = {}
    for (s, e) in wf_blocks(pn["T"], cal):
        t0 = time.perf_counter()
        for L in horizons:
            fr = frames[L]
            tr = fr[train_mask(fr, s - 1 - L)]
            pr_mask = ((fr.di >= s) & (fr.di <= e) & fr.win_ok & (fr.scale > 0)).to_numpy()
            pr = fr[pr_mask]
            models = Parallel(n_jobs=N_JOBS, prefer="threads")(
                delayed(fit_gbm)(tr[feats], tr.target, t, params) for t in taus)
            for j, m in enumerate(models):
                out[L][pr_mask, j] = m.predict(pr[feats])
            if 0.5 in taus:
                imp[L] = pd.Series(models[taus.index(0.5)].booster_.feature_importance("gain"), index=feats)
        print(f"  khối {cal[s].date()} -> {cal[e].date()} đã fit ({len(taus) * len(horizons)} mô hình, "
              f"{time.perf_counter() - t0:.0f}s)")
    for L in horizons:
        q = out[L] * (L * frames[L].scale.to_numpy())[:, None]
        out[L] = np.clip(np.sort(q, axis=1), 0, None)      # sắp xếp lại để các phân vị không cắt nhau
    return out, imp


def baseline_quantiles(frames, pn, col, taus):
    """Dự báo điểm + phân vị thực nghiệm của phần dư RESID_WINDOW ngày gần nhất của chính chuỗi đó."""
    out = {}
    for L, fr in frames.items():
        pt, y = fr[col].to_numpy(), fr.y.to_numpy()
        di, si, ok = fr.di.to_numpy(), fr.si.to_numpy(), fr.win_ok.to_numpy()
        Q = np.full((len(fr), len(taus)), np.nan)
        for (s, e) in wf_blocks(pn["T"], pn["cal"]):
            m = (di <= s - 1 - L) & (di > s - 1 - L - RESID_WINDOW) & ~np.isnan(pt) & ~np.isnan(y)
            rq = (pd.DataFrame({"si": si[m], "r": (y - pt)[m]}).groupby("si").r.quantile(taus).unstack())
            idx = np.where((di >= s) & (di <= e) & ok)[0]
            Q[idx] = pt[idx, None] + rq.reindex(si[idx]).to_numpy()
        out[L] = np.clip(np.sort(Q, axis=1), 0, None)
    return out


def cached(name, sig, fn):
    path = OUT_DIR / "cache" / f"{name}.pkl"
    if USE_CACHE and path.exists():
        with open(path, "rb") as f:
            s, obj = pickle.load(f)
        if s == sig:
            print(f"  [cache] {name}: dùng lại kết quả cũ")
            return obj
    obj = fn()
    path.parent.mkdir(exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump((sig, obj), f)
    return obj


def code_signature(*objs):
    text = "".join(inspect.getsource(o) if callable(o) else repr(o) for o in objs)
    return hashlib.md5(text.encode()).hexdigest()


# ----------------------------------------------------------------------------
# BƯỚC 5c: ĐÁNH GIÁ DỰ BÁO
# ----------------------------------------------------------------------------
def wql(y, Q, taus):
    return float(np.mean([2 * pinball(y, Q[:, j], t).sum() / y.sum() for j, t in enumerate(taus)]))


def dm_test(loss_a, loss_b, lag):
    """Kiểm định Diebold-Mariano trên loss theo từng ngày gốc; thống kê dương = mô hình B có loss thấp hơn."""
    d = np.asarray(loss_a) - np.asarray(loss_b)
    n, dc = len(d), d - d.mean()
    lrv = dc @ dc / n + 2 * sum((1 - k / (lag + 1)) * (dc[k:] @ dc[:-k]) / n for k in range(1, lag + 1))
    stat = d.mean() / np.sqrt(lrv / n)
    return float(stat), float(2 * stats.t.sf(abs(stat), n - 1))


def evaluate_forecasts(frames, preds, pn, taus):
    banner("BƯỚC 5c: ĐÁNH GIÁ NGOÀI MẪU")
    cal = pn["cal"]
    e0 = cal.get_loc(pd.Timestamp(EVAL_START))
    names = list(preds)
    j50 = taus.index(0.5)
    pt_rows, cov_rows, fam_rows, dm_rows = [], [], [], []
    samples = {}
    for L in HORIZONS:
        fr = frames[L]
        m = ((fr.di >= e0) & fr.win_ok & fr.y.notna()).to_numpy().copy()
        for n in names:
            m &= ~np.isnan(preds[n][L]).any(axis=1)
        y, fam, di = fr.y.to_numpy()[m], pn["fam_of"][fr.si.to_numpy()[m]], fr.di.to_numpy()[m]
        samples[L] = (m, y)
        per_day = {}
        for n in names:
            Q = preds[n][L][m]
            pt = Q[:, j50] if n == "GBM" else fr[BASELINES[n]].to_numpy()[m]
            cov = (y[:, None] <= Q).mean(axis=0)
            pt_rows.append(dict(model=n, L=L, n_obs=len(y), WAPE=np.abs(y - pt).sum() / y.sum(),
                                bias=(pt - y).sum() / y.sum(), wQL=wql(y, Q, taus),
                                calib_err=np.abs(cov - np.array(taus)).mean()))
            for t, c in zip(taus, cov):
                cov_rows.append(dict(model=n, L=L, tau=t, coverage=c))
            for f in FAMILIES:
                mf = fam == f
                fam_rows.append(dict(model=n, L=L, family=f, wQL=wql(y[mf], Q[mf], taus),
                                     WAPE=np.abs(y[mf] - pt[mf]).sum() / y[mf].sum()))
            loss = np.mean([pinball(y, Q[:, j], t) for j, t in enumerate(taus)], axis=0)
            per_day[n] = pd.Series(loss).groupby(di).sum()
        for n in names:
            if n != "GBM":
                stat, p = dm_test(per_day[n].to_numpy(), per_day["GBM"].to_numpy(), lag=max(L, 7))
                dm_rows.append(dict(L=L, baseline=n, DM_stat=stat, p_value=p,
                                    gbm_loss_reduction=1 - per_day["GBM"].sum() / per_day[n].sum()))
    point = pd.DataFrame(pt_rows)
    cov = pd.DataFrame(cov_rows)
    fam = pd.DataFrame(fam_rows)
    dm = pd.DataFrame(dm_rows)
    for name, df in [("forecast_metrics", point), ("forecast_calibration", cov),
                     ("forecast_by_family", fam), ("forecast_dm_test", dm)]:
        df.to_csv(OUT_DIR / f"{name}.csv", index=False)
    print("Sai số điểm (WAPE, càng thấp càng tốt) và sai số phân vị (wQL, càng thấp càng tốt), "
          f"cửa sổ đánh giá {EVAL_START} -> {cal[-1].date()}:")
    for col in ("WAPE", "wQL", "calib_err"):
        print(f"\n{col}:\n{point.pivot(index='model', columns='L', values=col).round(3).to_string()}")
    print("\nDiebold-Mariano (GBM so với baseline, pinball loss theo ngày gốc):")
    print(dm.round(4).to_string(index=False))
    print(f"\nwQL theo family, L=2:\n{fam[fam.L == 2].pivot(index='model', columns='family', values='wQL').round(3).to_string()}")
    return point, cov, fam, dm, samples


def forecast_figures(frames, preds, pn, point, cov, fam, imp, taus, picked):
    cal = pn["cal"]
    palette = {"GBM": "tab:red", "Seasonal naive 7": "tab:blue", "Weekly profile 8w": "tab:green",
               "MA28": "tab:gray", "Naive": "tab:olive"}

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, col, title in zip(axes, ["WAPE", "wQL"], ["Sai số điểm (WAPE)", "Sai số phân vị có trọng số (wQL)"]):
        for n, g in point.groupby("model"):
            ax.plot(g.L, g[col], marker="o", label=n, color=palette[n])
        ax.set(title=title, xlabel="L (số ngày cầu cộng dồn)", ylabel=col, xticks=HORIZONS)
    axes[0].legend(fontsize=8)
    save_fig("fig06_forecast_accuracy_by_L")

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    for ax, L in zip(axes, (1, 3, 5)):
        for n, g in cov[cov.L == L].groupby("model"):
            ax.plot(g.tau, g.coverage, marker="o", ms=3, label=n, color=palette[n])
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set(title=f"L = {L}", xlabel="Mức phân vị danh nghĩa")
    axes[0].set_ylabel("Tỷ lệ cầu thực tế nằm dưới phân vị dự báo")
    axes[0].legend(fontsize=7)
    save_fig("fig07_forecast_calibration")

    f2 = fam[fam.L == 2].pivot(index="family", columns="model", values="wQL")
    fig, ax = plt.subplots(figsize=(9, 3.6))
    f2[list(palette)].plot.bar(ax=ax, color=list(palette.values()), width=0.8)
    ax.set(title="wQL theo family, L = 2", xlabel="", ylabel="wQL")
    plt.setp(ax.get_xticklabels(), rotation=0, fontsize=8)
    ax.legend(fontsize=7)
    save_fig("fig08_forecast_wql_by_family")

    fr, fam_name = frames[1], "PRODUCE"
    sid = [i for i, c in enumerate(pn["columns"]) if c == (picked[0], fam_name)][0]
    m = ((fr.si == sid) & (fr.di >= cal.get_loc(pd.Timestamp(EVAL_START))) & fr.win_ok).to_numpy()
    d = cal[fr.di.to_numpy()[m]]
    Q = preds["GBM"][1][m]
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.fill_between(d, Q[:, taus.index(0.1)], Q[:, taus.index(0.9)], alpha=0.3, label="GBM khoảng P10-P90")
    ax.plot(d, Q[:, taus.index(0.5)], color="tab:red", lw=1, label="GBM trung vị")
    ax.plot(d, fr.y.to_numpy()[m], color="k", lw=0.8, label="thực tế")
    ax.set(title=f"Dự báo cầu ngày kế tiếp, cửa hàng {picked[0]} / {fam_name}",
           xlabel="Ngày", ylabel="Doanh số")
    ax.legend(fontsize=8, ncol=3)
    save_fig("fig09_forecast_fan_example")

    g = imp[2].sort_values().tail(15)
    fig, ax = plt.subplots(figsize=(7, 4))
    (g / imp[2].sum()).plot.barh(ax=ax)
    ax.set(title="Độ quan trọng của đặc trưng (tỷ trọng gain), mô hình trung vị, L = 2, khối walk-forward cuối",
           xlabel="Tỷ trọng gain")
    save_fig("fig10_feature_importance")
    print("đã lưu biểu đồ dự báo fig06-fig10")


def ablation_future_covariates(frames, pn, params, preds_full, taus):
    """Biết trước lịch khuyến mãi / lễ / ngày lương có giúp dự báo không? Fit lại khi bỏ các đặc trưng đó."""
    abl_taus, abl_L = [0.3, 0.5, 0.9], [1, 3]
    feats = [f for f in FEATS if f not in FUTURE_COVS]
    q, _ = gbm_walk_forward(frames, pn, params, abl_taus, abl_L, feats)
    e0 = pn["cal"].get_loc(pd.Timestamp(EVAL_START))
    rows = []
    for L in abl_L:
        fr = frames[L]
        m = ((fr.di >= e0) & fr.win_ok & fr.y.notna()).to_numpy() & ~np.isnan(q[L]).any(axis=1)  # mảng mới
        y = fr.y.to_numpy()[m]
        full = preds_full[L][m][:, [taus.index(t) for t in abl_taus]]
        rows.append(dict(L=L, wQL_full=wql(y, full, abl_taus), wQL_no_future_covariates=wql(y, q[L][m], abl_taus)))
    res = pd.DataFrame(rows)
    res["relative_gain_of_covariates"] = 1 - res.wQL_full / res.wQL_no_future_covariates
    res.to_csv(OUT_DIR / "forecast_ablation.csv", index=False)
    print("\nAblation (tau = 0.3 / 0.5 / 0.9):")
    print(res.round(4).to_string(index=False))
    return res


def save_forecasts(frames, preds, pn, taus):
    cal = pn["cal"]
    w0 = cal.get_loc(pd.Timestamp(WF_START))
    parts = []
    for L in HORIZONS:
        fr = frames[L]
        m = ((fr.di >= w0) & fr.win_ok).to_numpy()
        d = pd.DataFrame(preds["GBM"][L][m], columns=[f"q{int(round(t * 100)):02d}" for t in taus]).round(3)
        d.insert(0, "L", L)
        d.insert(0, "family", pn["fam_of"][fr.si.to_numpy()[m]])
        d.insert(0, "store_nbr", pn["store_of"][fr.si.to_numpy()[m]])
        d.insert(0, "date", cal[fr.di.to_numpy()[m]])
        d["y"] = fr.y.to_numpy()[m]
        d["ma28_daily"] = fr.scale.to_numpy()[m]
        parts.append(d)
    out = pd.concat(parts, ignore_index=True)
    out.to_csv(OUT_DIR / "forecast_gbm_quantiles.csv.gz", index=False)
    print(f"đã lưu {len(out):,} dự báo phân vị GBM -> forecast_gbm_quantiles.csv.gz")


def forecasting(scope, stores, holidays, picked):
    banner("BƯỚC 5: DỰ BÁO - dựng bảng dữ liệu và đặc trưng")
    pn = build_panel(scope, stores, holidays)
    frames = {L: make_L_frame(pn, L) for L in HORIZONS}
    cal = pn["cal"]
    print(f"lịch {cal[0].date()} -> {cal[-1].date()} ({pn['T']} ngày) | {pn['N']} chuỗi | "
          f"{len(FEATS)} đặc trưng: {FEATS}")
    print(f"tỷ lệ NaN của doanh số ngày (ngày cửa hàng đóng cửa, 25/12): {np.isnan(pn['S']).mean():.4f}")
    blocks = wf_blocks(pn["T"], cal)
    print("các khối walk-forward:", [(str(cal[s].date()), str(cal[e].date())) for s, e in blocks])
    print(f"cửa sổ chọn siêu tham số {TUNE_START} -> {(cal[cal.get_loc(pd.Timestamp(WF_START)) - 1]).date()} | "
          f"khởi động {WF_START} -> {(cal[cal.get_loc(pd.Timestamp(EVAL_START)) - 1]).date()} | "
          f"đánh giá {EVAL_START} -> {cal[-1].date()}")
    for L in (1, 5):
        fr = frames[L]
        print(f"  L={L}: {len(fr):,} dòng, có mục tiêu {int(fr.target.notna().sum()):,}, "
              f"tỷ lệ mục tiêu trung vị {fr.target.median():.2f}, P5-P95 "
              f"{fr.target.quantile(.05):.2f}-{fr.target.quantile(.95):.2f}")

    cfg = (HORIZONS, TAUS, TUNE_START, WF_START, EVAL_START, REFIT_EVERY, RESID_WINDOW, SEED, FEATS, GBM_FIXED,
           START_DATE, N_STORES, picked)
    core = (shift_back, shift_fwd, fwd_mean, weekly_profile, build_panel, make_L_frame, wf_blocks, fit_gbm,
            train_mask, tune_gbm, gbm_walk_forward, baseline_quantiles)
    sig = code_signature(*core, cfg)

    best = cached("gbm_tuning", sig, lambda: tune_gbm(frames, pn))
    banner("BƯỚC 5a: GBM WALK-FORWARD (hàm mục tiêu quantile, một mô hình chung cho mỗi L và tau)")
    gbm, imp = cached("gbm_walk_forward", sig + str(best),
                      lambda: gbm_walk_forward(frames, pn, best, TAUS, HORIZONS, FEATS))
    preds = {"GBM": gbm}
    for n, col in BASELINES.items():
        preds[n] = baseline_quantiles(frames, pn, col, TAUS)
    point, cov, fam, dm, samples = evaluate_forecasts(frames, preds, pn, TAUS)
    forecast_figures(frames, preds, pn, point, cov, fam, imp, TAUS, picked)
    cached("gbm_ablation", sig + str(best) + "abl",
           lambda: ablation_future_covariates(frames, pn, best, gbm, TAUS))
    save_forecasts(frames, preds, pn, TAUS)
    return pn, frames, preds, sig


# ----------------------------------------------------------------------------
# BƯỚC 6: MÔ PHỎNG TỒN KHO (FIFO, hạn sử dụng, lead time kịch bản)
# ----------------------------------------------------------------------------
SIM_REPS = 5                       # số lần bốc ngẫu nhiên cú sốc lead time (kịch bản có trễ/sớm)
CALIB_SKIP = 7                     # bỏ 7 ngày đầu khi chọn tau: tồn kho khởi tạo chưa ổn định
BASE_SCEN = dict(Lm=2, shock="moderate", mult=1.0)
FILL_TARGETS = [0.90, 0.93, 0.95]
SHOCK_VI = {"none": "không lệch", "moderate": "lệch vừa (15%)", "high": "lệch cao (30%)"}
QTY = ["demand", "sold", "lost", "waste", "recv", "hold"]
BUFFER_MODELS = {"GBM+0.5": 0.5, "GBM+1": 1.0}    # đòn bẩy "đệm thời gian": mức mục tiêu = (1-w)*S(L) + w*S(L+1)
P1, P2, P3 = "P1 Safety stock kinh điển", "P2 Newsvendor tĩnh (MA28)", "P3 Quantile động (GBM)"
REF_GBM_QSTAR, REF_MA28_CAL, REF_ORACLE = ("Đối chứng: GBM tại q*", "Đối chứng: MA28 chọn tau",
                                          "Đối chứng: GBM tau tối ưu ex-post")


def shelf_days(mult):
    base = SHELF_LIFE.shelf_life_base.to_numpy()
    return np.maximum(np.floor(base * mult + 0.5), 1).astype(int)


def simulate(S, D, K, Lm, delta, j0):
    """Mô phỏng từng ngày cho M chuỗi song song (mỗi cột là một chuỗi, hoặc một cặp chuỗi-tau).

    S[j]: mức tồn kho mục tiêu (order-up-to) tính cuối ngày j; D[j]: cầu của ngày j; K: hạn sử dụng (ngày);
    delta[j]: độ lệch lead time của đơn đặt cuối ngày j. Hàng đặt cuối ngày j về đầu ngày j + max(1, Lm + delta).
    Bán theo FIFO, hàng chạm hạn bị huỷ cuối ngày, hết hàng thì mất khách (không backlog).
    """
    T, M = S.shape
    idx = np.arange(M)
    inv = np.zeros((M, int(K.max())))
    inv[:, 0] = S[j0]
    pipe = np.zeros((T + Lm + 3, M))
    transit = np.zeros(M)
    days = np.arange(j0 + 1, T)
    out = {k: np.zeros((len(days), M)) for k in QTY}
    for n, j in enumerate(days):
        arr = pipe[j]
        inv[:, 0] += arr
        transit -= arr
        out["recv"][n] = arr + (S[j0] if n == 0 else 0)
        d = D[j]
        rev = inv[:, ::-1]
        before = np.cumsum(rev, axis=1) - rev
        take = np.minimum(rev, np.maximum(d[:, None] - before, 0))
        inv -= take[:, ::-1]
        out["demand"][n] = d
        out["sold"][n] = take.sum(1)
        out["lost"][n] = d - out["sold"][n]
        out["waste"][n] = inv[idx, K - 1]
        inv[idx, K - 1] = 0
        hold = inv.sum(1)
        out["hold"][n] = hold
        inv[:, 1:] = inv[:, :-1].copy()
        inv[:, 0] = 0
        if j <= T - 2:
            q = np.maximum(S[j] - (hold + transit), 0)
            pipe[j + np.maximum(1, Lm + delta[j]), idx] += q
            transit += q
    assert np.allclose(out["recv"].sum(0), out["sold"].sum(0) + out["waste"].sum(0) + hold), "hàng không bảo toàn"
    return out


def classical_levels(frames, pn, L):
    """Chính sách 1 (safety stock kinh điển): S = dự báo MA28 + bias + z * sigma, sigma = độ lệch chuẩn phần dư L ngày."""
    T, N = pn["T"], pn["N"]
    fr = frames[L]
    pt = fr["b_ma28"].to_numpy().reshape(T, N)
    resid = shift_back(fr.y.to_numpy().reshape(T, N) - pt, L)
    roll = pd.DataFrame(resid).rolling(RESID_WINDOW, min_periods=28)
    z = stats.norm.ppf(SERVICE_LEVEL_CLASSIC)
    return np.clip(pt + roll.mean().to_numpy() + z * roll.std().to_numpy(), 0, None)


def shock_delta(U, prob):
    return np.where(U < prob / 2, -1, np.where(U > 1 - prob / 2, 1, 0))


def run_scenario(Lm, shock, mult, rep, ctx):
    pn, D, levels = ctx["pn"], ctx["D"], ctx["levels"]
    T, N, nt = pn["T"], pn["N"], len(TAUS)
    j0, e0 = ctx["j0"], ctx["e0"]
    delta = shock_delta(ctx["U"][rep], LEAD_TIME_SHOCK_PROB[shock])
    K = shelf_days(mult)[[FAMILIES.index(f) for f in pn["fam_of"]]]
    days = np.arange(j0 + 1, T)
    wins = {"calib": (days >= j0 + 1 + CALIB_SKIP) & (days < e0), "eval": days >= e0}
    parts = []
    for model, S in levels[Lm].items():
        sweep = S.ndim == 3
        S2 = S.reshape(T, N * nt) if sweep else S
        rep_n = nt if sweep else 1
        res = simulate(S2, np.repeat(D, rep_n, axis=1), np.repeat(K, rep_n), Lm,
                       np.repeat(delta, rep_n, axis=1), j0)
        fam = np.repeat(pn["fam_of"], rep_n)
        tau = np.tile(TAUS, N) if sweep else np.full(N, np.nan)
        for w, mask in wins.items():
            df = pd.DataFrame({k: res[k][mask].sum(0) for k in QTY})
            df["family"], df["tau"] = fam, tau
            g = df.groupby(["family", "tau"], dropna=False)[QTY].sum().reset_index()
            g.insert(0, "window", w)
            g.insert(0, "model", model)
            parts.append(g)
    out = pd.concat(parts, ignore_index=True)
    for k, v in dict(rep=rep, mult=mult, shock=shock, Lm=Lm).items():
        out.insert(0, k, v)
    return out


def build_context(pn, frames, preds):
    T, N = pn["T"], pn["N"]
    cal = pn["cal"]
    j0, e0 = cal.get_loc(pd.Timestamp(WF_START)), cal.get_loc(pd.Timestamp(EVAL_START))
    D = np.nan_to_num(pn["S"], nan=0.0)
    levels = {}
    for Lm in LEAD_TIME_MEAN:
        # vài ngày cuối chưa đủ cầu thực L ngày sau nên dự báo bị bỏ trống; đơn đặt lúc đó hầu như về sau kỳ mô phỏng,
        # nên giữ nguyên mức mục tiêu của ngày gần nhất (ffill)
        ff = lambda A: pd.DataFrame(A.reshape(T, -1)).ffill().to_numpy().reshape(A.shape)
        gbm_L = ff(preds["GBM"][Lm].reshape(T, N, len(TAUS)))
        gbm_L1 = ff(preds["GBM"][Lm + 1].reshape(T, N, len(TAUS)))
        levels[Lm] = {"Classical": classical_levels(frames, pn, Lm), "GBM": gbm_L,
                      **{name: (1 - w) * gbm_L + w * gbm_L1 for name, w in BUFFER_MODELS.items()},
                      "MA28": ff(preds["MA28"][Lm].reshape(T, N, len(TAUS)))}
        for name, S in levels[Lm].items():
            assert not np.isnan(S[j0:T - 1]).any(), f"mức mục tiêu bị NaN: {name}, L={Lm}"
    U = np.random.default_rng(SEED).random((SIM_REPS, T, N))
    return dict(pn=pn, D=D, levels=levels, U=U, j0=j0, e0=e0)


def run_simulations(pn, frames, preds):
    banner("BƯỚC 6: MÔ PHỎNG TỒN KHO - chạy lưới kịch bản")
    ctx = build_context(pn, frames, preds)
    jobs = [(Lm, sh, mu, r) for sh, p in LEAD_TIME_SHOCK_PROB.items() for r in range(1 if p == 0 else SIM_REPS)
            for Lm in LEAD_TIME_MEAN for mu in SHELF_LIFE_MULT]
    t0 = time.perf_counter()
    res = Parallel(n_jobs=N_JOBS, prefer="threads")(delayed(run_scenario)(*j, ctx) for j in jobs)
    phys = pd.concat(res, ignore_index=True)
    print(f"đã chạy {len(jobs)} kịch bản x 5 mô hình mức mục tiêu ({time.perf_counter() - t0:.0f}s), "
          f"{len(phys):,} dòng kết quả vật lý")
    return phys


def economics(phys, margin, carry):
    p = 1 / (1 - margin)
    out = phys.copy()
    out["cost"] = (p - 1) * out.lost + (1 - SALVAGE_VALUE) * out.waste + carry / 365 * out.hold
    return out


def critical_ratio(margin, carry):
    cu, co = margin / (1 - margin), 1 - SALVAGE_VALUE + carry / 365
    return cu / (cu + co)


KEYS = ["Lm", "shock", "mult", "rep", "family"]


def best_tau(df, model):
    g = df[df.model == model]
    return g.loc[g.groupby(KEYS).cost.idxmin(), KEYS + ["tau"]]


def policy_results(phys, margin, carry):
    """Gom kết quả giai đoạn đánh giá theo từng chính sách; tau của chính sách 3 chọn trên giai đoạn khởi động."""
    ec = economics(phys, margin, carry)
    cal_df, ev = ec[ec.window == "calib"], ec[ec.window == "eval"]
    tau_static = min(TAUS, key=lambda x: abs(x - critical_ratio(margin, carry)))
    at_static = np.isclose(ev.tau, tau_static)
    picks = {
        P1: ev[ev.model == "Classical"],
        P2: ev[(ev.model == "MA28") & at_static],
        P3: ev[ev.model == "GBM"].merge(best_tau(cal_df, "GBM"), on=KEYS + ["tau"]),
        REF_GBM_QSTAR: ev[(ev.model == "GBM") & at_static],
        REF_MA28_CAL: ev[ev.model == "MA28"].merge(best_tau(cal_df, "MA28"), on=KEYS + ["tau"]),
        REF_ORACLE: ev[ev.model == "GBM"].merge(best_tau(ev, "GBM"), on=KEYS + ["tau"]),
    }
    return pd.concat([d.assign(policy=k) for k, d in picks.items()], ignore_index=True)


def rates(df, by):
    """Tỷ lệ theo từng family rồi lấy trung bình đều giữa các family (đơn vị kg/cái khác nhau nên không cộng gộp)."""
    g = df.groupby(by + ["family"])[QTY + ["cost"]].sum()
    r = pd.DataFrame({"cost_rate": g.cost / g.demand, "fill_rate": g.sold / g.demand,
                      "waste_rate": g.waste / g.recv, "stock_days": g.hold / g.demand})
    return r.groupby(by).mean()


def simulation(pn, frames, preds, sig):
    phys = cached("sim_physical", sig + code_signature(simulate, run_scenario, run_simulations, build_context, classical_levels,
                                                      shock_delta, shelf_days, BUFFER_MODELS, SIM_REPS, CALIB_SKIP, LEAD_TIME_SHOCK_PROB,
                                                      SHELF_LIFE_MULT, LEAD_TIME_MEAN),
                  lambda: run_simulations(pn, frames, preds))
    return phys


def service_frontier(phys, margin, carry):
    """GBM quét mọi tau trong giai đoạn đánh giá: chi phí và mức phục vụ theo từng (L, trễ, hạn sử dụng, đòn bẩy)."""
    ev = economics(phys[(phys.window == "eval") & phys.model.isin(["GBM", *BUFFER_MODELS])], margin, carry)
    return rates(ev, ["model", "Lm", "shock", "mult", "tau"]).reset_index()


def cost_to_reach_fill(fr, targets):
    """Chi phí thấp nhất để đạt mỗi mức fill rate khi chỉ nâng tau (nội suy tuyến tính trên đường biên);
    NaN nếu nâng tau tới 0.95 vẫn chưa đạt."""
    rows = []
    for (md, sh, mu, Lm), g in fr.groupby(["model", "shock", "mult", "Lm"]):
        g = g.sort_values("tau").reset_index(drop=True)
        branch = g.loc[g.cost_rate.idxmin():]                  # nhánh tăng tau từ điểm chi phí thấp nhất
        fill = np.maximum.accumulate(branch.fill_rate.to_numpy())
        for target in targets:
            if fill.max() < target:
                cost = np.nan
            elif fill[0] >= target:
                cost = branch.cost_rate.iloc[0]                 # điểm chi phí thấp nhất đã đạt sẵn
            else:
                cost = float(np.interp(target, fill, branch.cost_rate.to_numpy()))
            rows.append(dict(model=md, shock=sh, mult=mu, Lm=Lm, fill_target=target, cost_rate=cost,
                             min_cost_rate=g.cost_rate.min()))
    return pd.DataFrame(rows)


def lever_comparison(tab_c):
    """Với mỗi (trễ, hạn sử dụng, L >= 2, fill mục tiêu): chi phí giữ nguyên L bằng đòn bẩy rẻ nhất
    (nâng tau hoặc đệm thêm nửa/1 ngày) so với chi phí khi rút ngắn L đi 1 ngày.
    Lưu ý: ở L=1 cú sốc "về sớm" bị chặn (hàng không thể về trước 1 ngày), nên bước L=2 -> L=1 bị phóng đại;
    chạy thử cú sốc chỉ-về-trễ ở L=2 cho thấy chi phí giảm khoảng 2/3 mức chênh với L=1."""
    t = tab_c.pivot_table(index=["shock", "mult", "Lm", "fill_target"], columns="model", values="cost_rate",
                          dropna=False)
    t["cost_buffer"] = t[list(BUFFER_MODELS)].min(axis=1)                 # đòn bẩy đệm thời gian: lấy w rẻ nhất
    t["cost_best_lever"] = t[["GBM", "cost_buffer"]].min(axis=1)          # NaN bị bỏ qua nếu một đòn bẩy còn lại có giá trị
    t = t.rename(columns={"GBM": "cost_raise_tau"}).drop(columns=list(BUFFER_MODELS)).reset_index()
    prev = t[["shock", "mult", "Lm", "fill_target", "cost_best_lever"]].copy()
    prev["Lm"] += 1
    t = t.merge(prev.rename(columns={"cost_best_lever": "cost_shorter_L"}),
                on=["shock", "mult", "Lm", "fill_target"], how="left")
    t = t[t.Lm >= 2]
    t["gain_shorten_1day"] = t.cost_best_lever - t.cost_shorter_L
    t["gain_pct_of_cost"] = t.gain_shorten_1day / t.cost_best_lever
    return t.reset_index(drop=True)


def simulation_report(phys):
    banner("BƯỚC 6b: KẾT QUẢ MÔ PHỎNG")
    m0, c0 = GROSS_MARGIN["base"], CARRY_RATE_ANNUAL["base"]
    pol = policy_results(phys, m0, c0)
    pol.to_csv(OUT_DIR / "sim_policy_family.csv.gz", index=False)
    print(f"Critical ratio cơ sở q* = {critical_ratio(m0, c0):.3f} -> tau tĩnh = "
          f"{min(TAUS, key=lambda x: abs(x - critical_ratio(m0, c0)))}")

    b = BASE_SCEN
    sel = pol[(pol.Lm == b["Lm"]) & (pol.shock == b["shock"]) & (pol.mult == b["mult"])]
    tab_a = rates(sel, ["policy"]).round(3)
    print(f"\nA. So sánh chính sách, kịch bản cơ sở (L={b['Lm']}, trễ={b['shock']}, hạn sử dụng x{b['mult']}, "
          f"biên {m0:.0%}, giữ hàng {c0:.0%}/năm); cost_rate = chi phí / cầu (giá vốn = 1):")
    print(tab_a.to_string())
    tab_a.to_csv(OUT_DIR / "sim_table_base_policies.csv")

    sel = pol[(pol.mult == 1.0) & pol.policy.isin([P1, P2, P3, REF_MA28_CAL, REF_ORACLE])]
    tab_b = rates(sel, ["policy", "shock", "Lm"]).reset_index()
    tab_b.to_csv(OUT_DIR / "sim_table_policies_by_L_shock.csv", index=False)
    for col in ("cost_rate", "fill_rate", "waste_rate"):
        print(f"\nB. {col} theo L và mức trễ (hạn sử dụng x1):")
        print(tab_b.pivot(index=["policy", "shock"], columns="Lm", values=col).round(3).to_string())

    fr = service_frontier(phys, m0, c0)
    fr.to_csv(OUT_DIR / "sim_service_frontier.csv", index=False)
    tab_c = cost_to_reach_fill(fr, FILL_TARGETS)
    tab_c.to_csv(OUT_DIR / "sim_table_cost_to_reach_fill.csv", index=False)
    for md, title in (("GBM", "chỉ nâng tau"), ("GBM+0.5", "phủ thêm nửa ngày cầu rồi chỉnh tau"),
                      ("GBM+1", "phủ thêm 1 ngày cầu rồi chỉnh tau")):
        print(f"\nC. Chi phí để đạt một mức fill rate cho trước bằng cách {title} (nội suy), hạn sử dụng x1; "
              "NaN = nâng tau tới 0.95 vẫn chưa đạt:")
        print(tab_c[(tab_c.mult == 1.0) & (tab_c.model == md)]
              .pivot(index=["shock", "fill_target"], columns="Lm", values="cost_rate").round(3).to_string())

    tab_f = lever_comparison(tab_c)
    tab_f.to_csv(OUT_DIR / "sim_table_shorten_vs_buffer.csv", index=False)
    print("\nF. Rút ngắn 1 ngày lead time so với tăng tồn kho an toàn để đạt cùng fill rate (hạn sử dụng x1). "
          "cost_* = chi phí / cầu; gain = chi phí giữ nguyên L trừ chi phí khi rút ngắn L, tức mức chi tối đa "
          "(tính theo % giá vốn) đáng để bỏ ra cho 1 ngày lead time:")
    print(tab_f[tab_f.mult == 1.0].drop(columns="mult").round(3).to_string(index=False))

    sel = pol[(pol.policy == P3) & (pol.shock == b["shock"]) & (pol.Lm == b["Lm"])]
    w = rates(sel, ["mult"]).round(3)
    wf = sel.groupby(["mult", "family"])[QTY].sum()
    wf = (wf.waste / wf.recv).unstack(0).round(3)
    wf["ERS_retail_loss"] = SHELF_LIFE.ers_retail_loss_pct / 100
    print(f"\nD. Tỷ lệ hao hụt mô phỏng (P3, L={b['Lm']}, trễ={b['shock']}) so với mốc USDA ERS, theo hệ số hạn sử dụng:")
    print(wf.to_string())
    wf.to_csv(OUT_DIR / "sim_table_waste_vs_ers.csv")

    rows = []
    for mg in GROSS_MARGIN["grid"]:
        for cr in CARRY_RATE_ANNUAL["grid"]:
            pl = policy_results(phys, mg, cr)
            pl = pl[(pl.Lm == b["Lm"]) & (pl.shock == b["shock"]) & (pl.mult == 1.0)]
            r = rates(pl, ["policy"]).cost_rate
            rows.append(dict(margin=mg, carry=cr, tau_static=min(TAUS, key=lambda x: abs(x - critical_ratio(mg, cr))),
                             P1=r[P1], P2=r[P2], P3=r[P3], GBM_tai_q=r[REF_GBM_QSTAR], MA28_chon_tau=r[REF_MA28_CAL],
                             ex_post=r[REF_ORACLE]))
    tab_e = pd.DataFrame(rows)
    tab_e.to_csv(OUT_DIR / "sim_table_sensitivity_margin_carry.csv", index=False)
    print(f"\nE. Độ nhạy theo biên lợi nhuận và chi phí giữ hàng (cost_rate, L={b['Lm']}, trễ={b['shock']}):")
    print(tab_e.round(3).to_string(index=False))

    sel = pol[(pol.Lm == b["Lm"]) & (pol.shock == b["shock"])]
    tab_h = rates(sel, ["policy", "mult"]).reset_index()
    tab_h.to_csv(OUT_DIR / "sim_table_sensitivity_shelf_life.csv", index=False)
    for col in ("cost_rate", "fill_rate", "waste_rate"):
        print(f"\nH. {col} theo hệ số hạn sử dụng (L={b['Lm']}, trễ={b['shock']}):")
        print(tab_h.pivot(index="policy", columns="mult", values=col).round(3).to_string())
    return pol, tab_b, fr, tab_c, wf


# ----------------------------------------------------------------------------
# BƯỚC 6c: KHOẢNG TIN CẬY CHO KHÁC BIỆT GIỮA CÁC CHÍNH SÁCH
# ----------------------------------------------------------------------------
BOOT_B = 2000
N_BLOCKS = 13                          # 92 ngày đánh giá chia thành 13 khối tuần (khối cuối có 8 ngày)
DETAIL_POLICIES = {P1: "Classical", P2: "MA28", P3: "GBM", REF_GBM_QSTAR: "GBM", REF_MA28_CAL: "MA28"}
QD = ["demand", "sold", "waste", "recv", "cost"]


def run_detail(Lm, shock, mult, rep, ctx, picks):
    """Mô phỏng lại các chính sách đã chọn (mỗi chuỗi một tau theo bảng picks), gộp theo (khối tuần, chuỗi) trong
    giai đoạn đánh giá. Trả về {chính sách: mảng (khối, chuỗi, QTY)}."""
    pn, D, levels = ctx["pn"], ctx["D"], ctx["levels"]
    T, N = pn["T"], pn["N"]
    j0, e0 = ctx["j0"], ctx["e0"]
    delta = shock_delta(ctx["U"][rep], LEAD_TIME_SHOCK_PROB[shock])
    K = shelf_days(mult)[[FAMILIES.index(f) for f in pn["fam_of"]]]
    starts = np.arange(0, T - e0, 7)[:N_BLOCKS]
    out = {}
    for policy, model in DETAIL_POLICIES.items():
        if model == "Classical":
            S = levels[Lm][model]
        else:
            tau_f = picks[policy].set_index("family").tau
            idx = np.array([TAUS.index(tau_f[f]) for f in pn["fam_of"]])
            S = levels[Lm][model][:, np.arange(N), idx]
        res = simulate(S, D, K, Lm, delta, j0)
        days = np.arange(j0 + 1, T)
        ev = days >= e0
        out[policy] = np.stack([np.add.reduceat(res[k][ev], starts, axis=0) for k in QTY], axis=-1)
    return out


def collect_detail(ctx, pol):
    """Chạy lại cho mọi (Lm, shock) ở hạn sử dụng x1; cộng gộp các lần bốc ngẫu nhiên. Trả về
    {(chính sách, Lm, shock): mảng (cửa hàng, family, khối, QTY)}."""
    pn = ctx["pn"]
    stores_u, store_i = np.unique(pn["store_of"], return_inverse=True)
    fam_i = np.array([FAMILIES.index(f) for f in pn["fam_of"]])
    jobs = [(Lm, sh, r) for sh, p in LEAD_TIME_SHOCK_PROB.items() for r in range(1 if p == 0 else SIM_REPS)
            for Lm in LEAD_TIME_MEAN]

    def one(Lm, sh, r):
        sel = pol[(pol.Lm == Lm) & (pol.shock == sh) & (pol.mult == 1.0) & (pol.rep == r)]
        picks = {k: sel[sel.policy == k][["family", "tau"]] for k in DETAIL_POLICIES}
        return (Lm, sh), run_detail(Lm, sh, 1.0, r, ctx, picks)

    detail = {}
    for (Lm, sh), d in Parallel(n_jobs=N_JOBS, prefer="threads")(delayed(one)(*j) for j in jobs):
        for policy, A in d.items():
            key = (policy, Lm, sh)
            detail[key] = detail.get(key, 0) + A
    margin, carry = GROSS_MARGIN["base"], CARRY_RATE_ANNUAL["base"]
    cu, hold_cost = margin / (1 - margin), carry / 365
    final = {}
    for key, A in detail.items():                       # A: (khối, chuỗi, QTY) -> (cửa hàng, family, khối, QD)
        q = {k: A[..., i] for i, k in enumerate(QTY)}
        q["cost"] = cu * q["lost"] + (1 - SALVAGE_VALUE) * q["waste"] + hold_cost * q["hold"]
        arr = np.zeros((len(stores_u), len(FAMILIES), A.shape[0], len(QD)))
        for n in range(A.shape[1]):
            for qi, k in enumerate(QD):
                arr[store_i[n], fam_i[n], :, qi] += q[k][:, n]
        final[key] = arr
    return final


def cell_metrics(X):
    """X: (..., family, QD) đã cộng theo cửa hàng-khối được chọn. Tỷ lệ theo family rồi lấy trung bình đều."""
    d, s, w, r, c = (X[..., i] for i in range(len(QD)))
    return dict(cost_rate=(c / d).mean(-1), fill_rate=(s / d).mean(-1), waste_rate=(w / r).mean(-1))


def bootstrap_metrics(arrs, rng, B=BOOT_B):
    """Bootstrap hai chiều (chọn lại cửa hàng và khối tuần độc lập), dùng chung một lần chọn cho mọi mảng trong arrs
    để các so sánh là ghép cặp. Trả về {tên: {chỉ số: mảng B giá trị}}."""
    n_s, _, n_b, _ = next(iter(arrs.values())).shape
    ws = rng.multinomial(n_s, np.full(n_s, 1 / n_s), size=B)
    wb = rng.multinomial(n_b, np.full(n_b, 1 / n_b), size=B)
    return {k: cell_metrics(np.einsum("Bs,Bb,sfbq->Bfq", ws, wb, A)) for k, A in arrs.items()}


def ci_row(draws, point):
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return dict(estimate=point, ci_low=lo, ci_high=hi, share_below_zero=(draws < 0).mean())


def simulation_inference(pn, frames, preds, pol):
    banner("BƯỚC 6c: KHOẢNG TIN CẬY CHO KHÁC BIỆT GIỮA CÁC CHÍNH SÁCH")
    ctx = build_context(pn, frames, preds)
    detail = collect_detail(ctx, pol)

    b = BASE_SCEN                                        # kiểm tra: gộp chi tiết phải khớp bảng A (gộp theo family)
    chk = rates(pol[(pol.Lm == b["Lm"]) & (pol.shock == b["shock"]) & (pol.mult == 1.0)
                    & pol.policy.isin(DETAIL_POLICIES)], ["policy"]).cost_rate
    for k in DETAIL_POLICIES:
        got = cell_metrics(detail[(k, b["Lm"], b["shock"])].sum((0, 2)))["cost_rate"]
        assert np.isclose(got, chk[k], rtol=1e-6), f"gộp chi tiết không khớp bảng A: {k}"

    rng = np.random.default_rng(SEED)
    keys = list(detail)
    boot = bootstrap_metrics({k: detail[k] for k in keys}, rng)
    point = {k: cell_metrics(detail[k].sum((0, 2))) for k in keys}
    rows = []

    def add(kind, scen, comp, metric, d_boot, d_point):
        rows.append(dict(kind=kind, scenario=scen, comparison=comp, metric=metric, **ci_row(d_boot, d_point)))

    for sh in LEAD_TIME_SHOCK_PROB:
        for Lm in LEAD_TIME_MEAN:
            scen = f"{sh}|L={Lm}"
            for other in (P1, P2, REF_MA28_CAL, REF_GBM_QSTAR):
                for met in ("cost_rate", "fill_rate"):
                    add("policy", scen, f"P3 - [{other}]", met,
                        boot[(P3, Lm, sh)][met] - boot[(other, Lm, sh)][met],
                        point[(P3, Lm, sh)][met] - point[(other, Lm, sh)][met])
            if Lm < max(LEAD_TIME_MEAN):
                for met in ("cost_rate", "fill_rate"):
                    add("lead_time", f"{sh}|P3", f"L={Lm + 1} - L={Lm}", met,
                        boot[(P3, Lm + 1, sh)][met] - boot[(P3, Lm, sh)][met],
                        point[(P3, Lm + 1, sh)][met] - point[(P3, Lm, sh)][met])
    for k in (P1, P2, P3, REF_MA28_CAL, REF_GBM_QSTAR):
        for met in ("cost_rate", "fill_rate", "waste_rate"):
            rows.append(dict(kind="level", scenario=f"{b['shock']}|L={b['Lm']}", comparison=k, metric=met,
                             **{**ci_row(boot[(k, b["Lm"], b["shock"])][met], point[(k, b["Lm"], b["shock"])][met]),
                                "share_below_zero": np.nan}))
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT_DIR / "sim_inference_bootstrap.csv", index=False)

    fmt = lambda g: g.assign(ci=g.apply(lambda r: f"{r.estimate:+.4f} [{r.ci_low:+.4f}; {r.ci_high:+.4f}]", axis=1))
    lv = tab[tab.kind == "level"]
    print(f"\nG1. Mức tuyệt đối, kịch bản cơ sở (L={b['Lm']}, trễ={b['shock']}); khoảng tin cậy 95% bootstrap "
          f"({BOOT_B} lần, chọn lại cửa hàng và tuần):")
    print(fmt(lv).pivot(index="comparison", columns="metric", values="ci").to_string())
    pc = tab[(tab.kind == "policy") & (tab.metric == "cost_rate")]
    print("\nG2. Khác biệt chi phí (P3 trừ chính sách còn lại), âm = P3 rẻ hơn; [khoảng tin cậy 95%]:")
    print(fmt(pc).pivot(index=["scenario"], columns="comparison", values="ci").to_string())
    ll = tab[(tab.kind == "lead_time") & (tab.metric == "cost_rate")]
    print("\nG3. P3: chi phí tăng thêm khi lead time dài hơn 1 ngày; [khoảng tin cậy 95%]:")
    print(fmt(ll).pivot(index="scenario", columns="comparison", values="ci").to_string())
    return tab


def simulation_figures(pol, tab_b, fr):
    colors = {1: "tab:green", 2: "tab:blue", 3: "tab:orange", 4: "tab:red"}
    b = BASE_SCEN

    g0 = fr[(fr.shock == b["shock"]) & (fr.mult == 1.0)]
    sel = pol[(pol.shock == b["shock"]) & (pol.mult == 1.0)]
    pts = rates(sel, ["policy", "Lm"]).reset_index()
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for Lm, g in g0.groupby("Lm"):
        ax.plot(g.fill_rate * 100, g.cost_rate, "-", color=colors[Lm], lw=1, alpha=0.7, label=f"GBM quét tau, L = {Lm}")
        ax.scatter(g.fill_rate * 100, g.cost_rate, s=10, color=colors[Lm])
        for marker, name in (("*", P1), ("^", P2), ("D", P3)):
            r = pts[(pts.policy == name) & (pts.Lm == Lm)].iloc[0]
            ax.scatter(r.fill_rate * 100, r.cost_rate, marker=marker, s=70, color=colors[Lm], edgecolor="k", zorder=5)
    for marker, lab in (("*", P1), ("^", P2), ("D", P3)):
        ax.scatter([], [], marker=marker, s=60, color="w", edgecolor="k", label=lab)
    ax.set(title=f"Chi phí và mức phục vụ (lead time {SHOCK_VI[b['shock']]}, hạn sử dụng x1)",
           xlabel="Fill rate (%)", ylabel="Chi phí / cầu (giá vốn = 1)")
    ax.legend(fontsize=7)
    save_fig("fig11_cost_service_frontier")

    names = {P1: ("tab:gray", "-"), P2: ("tab:blue", "-"), P3: ("tab:red", "-"), REF_MA28_CAL: ("tab:green", "-."),
             REF_ORACLE: ("tab:red", "--")}
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    for ax, sh in zip(axes, LEAD_TIME_SHOCK_PROB):
        for k, (c, ls) in names.items():
            g = tab_b[(tab_b.policy == k) & (tab_b.shock == sh)]
            ax.plot(g.Lm, g.cost_rate, marker="o", color=c, ls=ls, label=k)
        ax.set(title=f"Lead time {SHOCK_VI[sh]}", xlabel="Lead time trung bình L (ngày)", xticks=LEAD_TIME_MEAN)
    axes[0].set_ylabel("Chi phí / cầu (giá vốn = 1)")
    axes[0].legend(fontsize=7)
    save_fig("fig12_cost_by_lead_time")


def lever_figure(tab_c):
    levers = {"GBM": ("tab:blue", "Chỉ nâng tau"), "GBM+0.5": ("tab:green", "Phủ thêm nửa ngày cầu"),
              "GBM+1": ("tab:orange", "Phủ thêm 1 ngày cầu")}
    targets = [0.90, 0.95]
    fig, axes = plt.subplots(len(targets), 3, figsize=(11, 6.4), sharex=True, sharey="row")
    for i, tg in enumerate(targets):
        for j, sh in enumerate(LEAD_TIME_SHOCK_PROB):
            ax = axes[i, j]
            for md, (c, lab) in levers.items():
                g = tab_c[(tab_c.model == md) & (tab_c.shock == sh) & (tab_c.mult == 1.0) & np.isclose(tab_c.fill_target, tg)]
                ax.plot(g.Lm, g.cost_rate, marker="o", color=c, label=lab)
            ax.set(title=f"Fill {tg:.0%} - lead time {SHOCK_VI[sh]}", xticks=LEAD_TIME_MEAN)
            if i == len(targets) - 1:
                ax.set_xlabel("Lead time trung bình L (ngày)")
        axes[i, 0].set_ylabel("Chi phí / cầu (giá vốn = 1)")
    axes[0, 0].legend(fontsize=7)
    fig.text(0.5, -0.01, "Điểm vắng của đường 'Chỉ nâng tau' nghĩa là nâng tau tới 0,95 vẫn không đạt fill rate mục tiêu.",
             ha="center", fontsize=8)
    save_fig("fig14_shorten_vs_buffer")


def waste_figure(wf):
    fig, ax = plt.subplots(figsize=(9, 3.6))
    (wf.drop(columns="ERS_retail_loss") * 100).plot.bar(ax=ax, width=0.75)
    ax.scatter(range(len(wf)), wf.ERS_retail_loss * 100, marker="_", s=900, color="k", linewidths=2, zorder=5,
               label="Mốc USDA ERS (hao hụt bán lẻ)")
    ax.set(title=f"Tỷ lệ hao hụt mô phỏng theo hệ số hạn sử dụng (P3, L = {BASE_SCEN['Lm']})",
           xlabel="", ylabel="Hao hụt / hàng nhận (%)")
    plt.setp(ax.get_xticklabels(), rotation=0, fontsize=8)
    ax.legend(title="Hệ số hạn sử dụng", fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    save_fig("fig13_waste_vs_shelf_life")


def main():
    train, stores, holidays, oil = load_data()
    scope, picked, store_audit = select_scope(train, stores)
    summarise_scope(scope)
    eda_figures(train, scope, picked)
    assumptions_report()
    pn, frames, preds, sig = forecasting(scope, stores, holidays, picked)
    driver_analysis(pn, oil)
    phys = simulation(pn, frames, preds, sig)
    pol, tab_b, fr, tab_c, wf = simulation_report(phys)
    simulation_inference(pn, frames, preds, pol)
    simulation_figures(pol, tab_b, fr)
    lever_figure(tab_c)
    waste_figure(wf)


if __name__ == "__main__":
    main()
