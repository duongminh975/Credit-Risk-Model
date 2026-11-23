"""
DỰ ÁN 1 — Mô hình xác suất vỡ nợ (PD) và Thẻ điểm rủi ro tín dụng
=====================================================================
Bộ dữ liệu : Statlog (German Credit) — 1.000 hồ sơ vay vốn, 20 thuộc tính

Phương pháp thực hiện theo quy trình xây dựng thẻ điểm tín dụng bán lẻ
chuẩn:
    1. Phân nhóm (coarse classing) từng thuộc tính thành các khoảng
    2. Chuyển đổi từng nhóm bằng trọng số bằng chứng (WOE)
    3. Tính giá trị thông tin (IV) để lựa chọn biến
    4. Hồi quy Logistic trên các biến đã chuyển đổi WOE
    5. Chuyển log-odds thành thẻ điểm dạng điểm số (phương pháp PDO)
    6. Đánh giá mô hình: AUC / Gini / thống kê KS
    7. Xếp hạng rủi ro và tính tổn thất kỳ vọng theo hạng
       (phân tích ngưỡng quyết định của danh mục)

Quy ước biến mục tiêu: default = 1 (vỡ nợ/rủi ro), non-default = 0 (không vỡ nợ/tốt)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

RANDOM_STATE = 42
pd.set_option("display.width", 200)

# ----------------------------------------------------------------------------
# 1. TẢI VÀ CHUẨN BỊ DỮ LIỆU
# ----------------------------------------------------------------------------
df = pd.read_csv("german_credit.csv")

# Bộ dữ liệu quy ước 1 = tốt, 0 = xấu. Đảo giá trị để mô hình hóa xác suất VỠ NỢ.
df["default"] = 1 - df["credit_risk"]
df = df.drop(columns=["credit_risk"])

TARGET = "default"
features = [c for c in df.columns if c != TARGET]

train, test = train_test_split(
    df, test_size=0.30, random_state=RANDOM_STATE, stratify=df[TARGET]
)

print("=" * 78)
print("1. DỮ LIỆU")
print("=" * 78)
print(f"Tổng số hồ sơ vay       : {len(df):,}")
print(f"Tập huấn luyện / kiểm tra: {len(train):,} / {len(test):,}")
print(f"Tỷ lệ vỡ nợ toàn bộ      : {df[TARGET].mean():.2%}")
print(f"Tỷ lệ vỡ nợ tập huấn luyện: {train[TARGET].mean():.2%}")
print(f"Tỷ lệ vỡ nợ tập kiểm tra : {test[TARGET].mean():.2%}")

# ----------------------------------------------------------------------------
# 2. PHÂN NHÓM (biến số -> nhóm theo phân vị; biến phân loại -> giữ nguyên)
# ----------------------------------------------------------------------------
NUMERIC = ["duration", "amount", "age", "installment_rate",
           "present_residence", "number_credits", "people_liable"]

bin_edges = {}


def make_bins(series, n_bins=5):
    """Phân nhóm thô dựa trên phân vị. Trả về các điểm cắt ở bên trong."""
    qs = np.linspace(0, 1, n_bins + 1)[1:-1]
    edges = sorted(set(np.quantile(series, qs)))
    return [-np.inf] + edges + [np.inf]


def apply_bins(frame):
    out = pd.DataFrame(index=frame.index)
    for col in features:
        if col in NUMERIC and frame[col].nunique() > 5:
            out[col] = pd.cut(frame[col], bins=bin_edges[col]).astype(str)
        else:
            out[col] = frame[col].astype(str)
    return out


for col in NUMERIC:
    if train[col].nunique() > 5:
        bin_edges[col] = make_bins(train[col])

train_binned = apply_bins(train)
test_binned = apply_bins(test)

# ----------------------------------------------------------------------------
# 3. WOE và IV
# ----------------------------------------------------------------------------
#   WOE = ln( %Khách_hàng_tốt_trong_nhóm / %Khách_hàng_xấu_trong_nhóm ) -> WOE cao hơn = rủi ro thấp hơn
#   IV  = TỔNG ( %Khách_hàng_tốt - %Khách_hàng_xấu ) * WOE
# ----------------------------------------------------------------------------
y_train = train[TARGET].values
tot_bad = y_train.sum()
tot_good = len(y_train) - tot_bad

woe_maps = {}
iv_rows = []
bin_detail = []

for col in features:
    g = pd.DataFrame({"bin": train_binned[col], "y": y_train})
    agg = g.groupby("bin", observed=True)["y"].agg(["count", "sum"])
    agg.columns = ["n", "bad"]
    agg["good"] = agg["n"] - agg["bad"]

    # Hiệu chỉnh liên tục 0.5 giúp tránh ln(0) khi một số nhóm có ít quan sát
    agg["dist_good"] = (agg["good"] + 0.5) / (tot_good + 0.5 * len(agg))
    agg["dist_bad"] = (agg["bad"] + 0.5) / (tot_bad + 0.5 * len(agg))
    agg["woe"] = np.log(agg["dist_good"] / agg["dist_bad"])
    agg["iv_part"] = (agg["dist_good"] - agg["dist_bad"]) * agg["woe"]
    agg["bad_rate"] = agg["bad"] / agg["n"]

    woe_maps[col] = agg["woe"].to_dict()
    iv_rows.append({"feature": col, "iv": agg["iv_part"].sum(), "n_bins": len(agg)})

    for b, r in agg.iterrows():
        bin_detail.append({
            "feature": col, "bin": b, "n": int(r["n"]), "bad": int(r["bad"]),
            "bad_rate": r["bad_rate"], "woe": r["woe"],
        })

iv_table = pd.DataFrame(iv_rows).sort_values("iv", ascending=False).reset_index(drop=True)


def iv_strength(v):
    if v < 0.02:
        return "Không có khả năng dự báo"
    if v < 0.10:
        return "Yếu"
    if v < 0.30:
        return "Trung bình"
    if v < 0.50:
        return "Mạnh"
    return "Mạnh bất thường"


iv_table["predictive_power"] = iv_table["iv"].apply(iv_strength)

print()
print("=" * 78)
print("2. GIÁ TRỊ THÔNG TIN — xếp hạng biến")
print("=" * 78)
print(iv_table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# Giữ các biến có IV từ mức 'Yếu' trở lên — loại biến có IV < 0.02
SELECTED = iv_table.loc[iv_table["iv"] >= 0.02, "feature"].tolist()
print(f"\nSố biến được giữ lại (IV >= 0.02): {len(SELECTED)} of {len(features)}")

# ----------------------------------------------------------------------------
# 4. CHUYỂN ĐỔI WOE VÀ HỒI QUY LOGISTIC
# ----------------------------------------------------------------------------
def to_woe(binned, cols):
    out = pd.DataFrame(index=binned.index)
    for c in cols:
        # Nhóm chưa xuất hiện trong tập kiểm tra -> WOE = 0 (trung tính), phương án dự phòng tiêu chuẩn
        out[c] = binned[c].map(woe_maps[c]).fillna(0.0)
    return out


X_train = to_woe(train_binned, SELECTED)
X_test = to_woe(test_binned, SELECTED)
y_test = test[TARGET].values

model = LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_STATE)
model.fit(X_train, y_train)

pd_train = model.predict_proba(X_train)[:, 1]
pd_test = model.predict_proba(X_test)[:, 1]

# ----------------------------------------------------------------------------
# 5. ĐÁNH GIÁ MÔ HÌNH — AUC / Gini / KS
# ----------------------------------------------------------------------------
def ks_stat(y, p):
    """Kolmogorov-Smirnov: khoảng cách lớn nhất giữa hàm phân phối tích lũy của nhóm tốt và nhóm xấu."""
    fpr, tpr, _ = roc_curve(y, p)
    return np.max(tpr - fpr)


auc_tr, auc_te = roc_auc_score(y_train, pd_train), roc_auc_score(y_test, pd_test)
ks_tr, ks_te = ks_stat(y_train, pd_train), ks_stat(y_test, pd_test)

print()
print("=" * 78)
print("3. HIỆU SUẤT MÔ HÌNH")
print("=" * 78)
print(f"{'Chỉ tiêu':<12}{'Huấn luyện':>12}{'Kiểm tra':>12}")
print("-" * 36)
print(f"{'AUC':<12}{auc_tr:>12.4f}{auc_te:>12.4f}")
print(f"{'Gini':<12}{2*auc_tr-1:>12.4f}{2*auc_te-1:>12.4f}")
print(f"{'KS':<12}{ks_tr:>12.4f}{ks_te:>12.4f}")
print(f"\nChênh lệch AUC (huấn luyện - kiểm tra): {auc_tr - auc_te:.4f}")

# ----------------------------------------------------------------------------
# 6. POINTS-BASED SCORECARD
# ----------------------------------------------------------------------------
#   Điểm = Offset + Factor * ln(odds_good)
#   Factor = PDO / ln(2)      Offset = Điểm_cơ_sở - Factor * ln(Odds_cơ_sở)
# ----------------------------------------------------------------------------
PDO = 20          # số điểm cần tăng để odds tăng gấp đôi
BASE_SCORE = 600  # điểm số tại tỷ lệ odds cơ sở
BASE_ODDS = 50    # tỷ lệ khách hàng tốt:xấu là 50:1 tại điểm số cơ sở

factor = PDO / np.log(2)
offset = BASE_SCORE - factor * np.log(BASE_ODDS)
n_feat = len(SELECTED)
intercept = model.intercept_[0]

scorecard = []
for i, col in enumerate(SELECTED):
    beta = model.coef_[0][i]
    for b, woe in woe_maps[col].items():
        # Dấu âm: mô hình dự báo vỡ nợ, điểm số tăng khi mức độ tín nhiệm cao hơn
        pts = -(beta * woe + intercept / n_feat) * factor + offset / n_feat
        scorecard.append({"feature": col, "bin": b, "woe": round(woe, 4),
                          "coefficient": round(beta, 4), "points": int(round(pts))})

scorecard_df = pd.DataFrame(scorecard)


def score_rows(binned):
    lookup = {(r.feature, r.bin): r.points for r in scorecard_df.itertuples()}
    default_pts = int(round(offset / n_feat))
    totals = []
    for _, row in binned.iterrows():
        totals.append(sum(lookup.get((c, row[c]), default_pts) for c in SELECTED))
    return np.array(totals)


test_scores = score_rows(test_binned)

print()
print("=" * 78)
print("4. QUY ĐỔI THANG ĐIỂM THẺ ĐIỂM")
print("=" * 78)
print(f"PDO={PDO}, Điểm cơ sở={BASE_SCORE} tại odds {BASE_ODDS}:1")
print(f"Factor={factor:.4f}  Offset={offset:.4f}")
print(f"Khoảng điểm của tập kiểm tra: {test_scores.min()} – {test_scores.max()} "
      f"(mean {test_scores.mean():.0f})")

# ----------------------------------------------------------------------------
# 7. XẾP HẠNG RỦI RO VÀ TỔN THẤT KỲ VỌNG
# ----------------------------------------------------------------------------
res = pd.DataFrame({"score": test_scores, "actual_default": y_test, "pd": pd_test})
res["grade"] = pd.qcut(res["score"], 5, labels=["E (rủi ro cao nhất)", "D", "C", "B", "A (rủi ro thấp nhất)"])

grade_tbl = res.groupby("grade", observed=True).agg(
    accounts=("actual_default", "size"),
    defaults=("actual_default", "sum"),
    actual_dr=("actual_default", "mean"),
    predicted_pd=("pd", "mean"),
    avg_score=("score", "mean"),
).reindex(["A (rủi ro thấp nhất)", "B", "C", "D", "E (rủi ro cao nhất)"])

# Tổn thất kỳ vọng = PD x LGD x EAD (công thức theo Basel)
LGD = 0.45   # Mức tham chiếu Basel Foundation IRB đối với khoản vay bán lẻ không có tài sản bảo đảm
EAD = 250000  # dư nợ giả định bình quân mỗi hồ sơ, đơn vị EUR
grade_tbl["expected_loss_eur"] = (grade_tbl["predicted_pd"] * LGD * EAD
                                  * grade_tbl["accounts"]).round(0)

print()
print("=" * 78)
print("5. XẾP HẠNG RỦI RO (tập kiểm tra) — kiểm tra thứ tự rủi ro")
print("=" * 78)
print(grade_tbl.to_string(float_format=lambda x: f"{x:,.4f}"))

# Phân tích ngưỡng: điều gì xảy ra nếu từ chối các nhóm có điểm thấp nhất?
print()
print("=" * 78)
print("6. PHÂN TÍCH NGƯỠNG — phê duyệt hồ sơ có điểm từ X trở lên")
print("=" * 78)
print(f"{'Ngưỡng':>8}{'Được duyệt':>10}{'Tỷ lệ duyệt':>11}{'Tỷ lệ xấu':>10}{'Số khoản xấu tránh được':>13}")
print("-" * 52)
for cut in np.percentile(test_scores, [0, 10, 20, 30, 40, 50]):
    keep = res[res["score"] >= cut]
    avoided = int(res["actual_default"].sum() - keep["actual_default"].sum())
    print(f"{cut:>8.0f}{len(keep):>10}{len(keep)/len(res)*100:>10.1f}%"
          f"{keep['actual_default'].mean()*100:>9.1f}%{avoided:>13}")

# ----------------------------------------------------------------------------
# 8. XUẤT KẾT QUẢ
# ----------------------------------------------------------------------------
iv_table.to_csv("output_iv_table.csv", index=False)
scorecard_df.sort_values(["feature", "points"]).to_csv("output_scorecard.csv", index=False)
pd.DataFrame(bin_detail).to_csv("output_bin_detail.csv", index=False)
grade_tbl.to_csv("output_risk_grades.csv")

summary = {
    "n_applications": len(df),
    "default_rate": round(float(df[TARGET].mean()), 4),
    "features_screened": len(features),
    "features_retained": len(SELECTED),
    "auc_train": round(float(auc_tr), 4), "auc_test": round(float(auc_te), 4),
    "gini_test": round(float(2 * auc_te - 1), 4), "ks_test": round(float(ks_te), 4),
    "score_min": int(test_scores.min()), "score_max": int(test_scores.max()),
}
pd.DataFrame([summary]).to_csv("output_summary.csv", index=False)

print()
print("=" * 78)
print("CÁC CHỈ SỐ TỔNG QUAN CỦA MÔ HÌNH")
print("=" * 78)
for k, v in summary.items():
    print(f"  {k:<20}: {v}")
