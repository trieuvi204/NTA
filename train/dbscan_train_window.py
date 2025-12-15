#train dbscan
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import json, joblib, os

INPUT_CSV = "dbscan_flows.csv"
ART_DIR = "artifacts/dbscan_window"
os.makedirs(ART_DIR, exist_ok=True)

# FEATURE ENGINEERING
FEATURES = [
    "flows",        #Tổng số flow trong 1 window
    "uniq_src_ip",  # Mức độ hoạt động  
    "uniq_dst_ip",  #
    "uniq_dst_port",#
    "entropy_dst_port", # Mức độ phân tán port đích 
    "avg_in_bytes",      # 
    "avg_out_bytes",     # Kích thước và thời gian flow
    "avg_duration_ms",   #
    "small_flow_ratio",    # Tỷ lệ flow nhỏ (scan/flood có rất nhiều flow nhỏ, bin thường ít )
    "avg_src_to_dst_bps",   #Tốc độ truyền
    "avg_dst_to_src_bps",   #
    "src_iat_avg",           #thời gian giữa các gói tin (Tool tấn công → IAT đều, std thấp, Người dùng thật → IAT ngẫu nhiên)
    "src_iat_std",           #
    "dst_iat_avg",           #
    "dst_iat_std"            #   
]

# Hàm tính entropy, mức độ phân tán
def shannon_entropy(s):
    if len(s) == 0: return 0
    p = s.value_counts(normalize=True) #tính xác suất xuất hiện (đếm số lần xuất hiện của mỗi giá trị, chia cho tổng số phần tử)
    return float(-(p * np.log2(p + 1e-9)).sum()) #1e-9: tránh log(0) gây NaN

# trích xuất các đặc trưng trong 1 window
def extract_features(df):
    total_bytes = df["IN_BYTES"] + df["OUT_BYTES"] # tổng số byte của mỗi flow

    return {
        "flows": len(df),
        "uniq_src_ip": df["IPV4_SRC_ADDR"].nunique(),
        "uniq_dst_ip": df["IPV4_DST_ADDR"].nunique(),
        "uniq_dst_port": df["L4_DST_PORT"].nunique(),
        "entropy_dst_port": shannon_entropy(df["L4_DST_PORT"]), # độ phân tán port đích để phát hiện portscan
        "avg_in_bytes": df["IN_BYTES"].mean(),  # tính trung bình byte vào/ra của các flow trong window
        "avg_out_bytes": df["OUT_BYTES"].mean(),#
        "avg_duration_ms": df["FLOW_DURATION_MILLISECONDS"].mean(),# thời gian trung bình của flow trong window
        "small_flow_ratio": (total_bytes < 200).mean(), # tỷ lệ flow nhỏ < 200 bytes
        "avg_src_to_dst_bps": df["SRC_TO_DST_SECOND_BYTES"].mean(),# 
        "avg_dst_to_src_bps": df["DST_TO_SRC_SECOND_BYTES"].mean(),#
        "src_iat_avg": df["SRC_TO_DST_IAT_AVG"].mean(),
        "src_iat_std": df["SRC_TO_DST_IAT_STDDEV"].mean(),
        "dst_iat_avg": df["DST_TO_SRC_IAT_AVG"].mean(),
        "dst_iat_std": df["DST_TO_SRC_IAT_STDDEV"].mean(),
    }

# LOAD CSV
print("[+] Loading:", INPUT_CSV)
df = pd.read_csv(INPUT_CSV)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["window"] = df["timestamp"].dt.floor("1s")
# trích xuất đặc trưng cho từng window
windows = []
for win, group in df.groupby("window"):
    feats = extract_features(group)
    windows.append(feats)

dfW = pd.DataFrame(windows)
print("[OK] Windows:", dfW.shape)
# xử lý giá trị thiếu
dfW = dfW.fillna(0)

# TRAIN DBSCAN
X = dfW[FEATURES].values
## Chuẩn hóa dữ liệu
scaler = StandardScaler() # sử dụng vì DBSCAN nhạy với khoảng cách, cần chuẩn hóa để các đặc trưng có cùng thang đo
X_scaled = scaler.fit_transform(X)
# Huấn luyện DBSCAN
dbs = DBSCAN(eps=2.5, min_samples=10)
dbs.fit(X_scaled)
# Đánh giá kết quả
labels = dbs.labels_
noise_ratio = np.mean(labels == -1)

print("[RESULT] Noise windows:", np.sum(labels == -1))
print("[RESULT] Noise ratio:", noise_ratio * 100, "%")

# SAVE ARTIFACTS
joblib.dump(dbs, f"{ART_DIR}/dbscan.pkl")
joblib.dump(scaler, f"{ART_DIR}/scaler.pkl")
json.dump(FEATURES, open(f"{ART_DIR}/feature_order.json", "w"))

print("[OK] Saved to:", ART_DIR)
