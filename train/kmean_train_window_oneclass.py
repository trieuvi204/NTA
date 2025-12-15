# train_window_oneclass.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib
import json
import os


BENIGN_CSV_PATH = "dataset/Monday_benign.csv"  
ARTIFACT_DIR = "artifacts_window"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

WINDOW_SEC = 1
THRESHOLD_PERCENTILE = 99.5  # percentile để làm ngưỡng phát hiện anomaly, chỉ các điểm vượt qua ngưỡng này mới bị coi là anomaly 


# tính entropy
def shannon_entropy(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    counts = series.value_counts()
    probs = counts / counts.sum()
    return float(-(probs * np.log2(probs + 1e-12)).sum())


# BUILD WINDOW FEATURES
def build_windows(df: pd.DataFrame, window_sec: int = 1) -> pd.DataFrame:
    """
    Từ raw flow CICFlowMeter → group theo time window → tạo feature window-based.
    """

    df = df.copy()

    # chuẩn hóa tên cột
    df.columns = [
        c.strip().lower().replace(" ", "_").replace("/", "_").replace(".", "_")
        for c in df.columns
    ]

    # bắt buộc phải có timestamp
    if "timestamp" not in df.columns:
        raise ValueError("File không có cột 'timestamp' (CICFlowMeter output mới luôn có).")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])

    # map tên cột port
    if "dst_port" not in df.columns and "destination_port" in df.columns:
        df["dst_port"] = df["destination_port"]

    # đảm bảo các cột numeric tồn tại
    numeric_cols = [
        "pkt_len_mean",
        "syn_flag_cnt",
        "flow_iat_mean",
    ]
    for c in numeric_cols:
        if c not in df.columns:
            df[c] = 0.0

    df["pkt_len_mean"] = pd.to_numeric(df["pkt_len_mean"], errors="coerce").fillna(0)
    df["syn_flag_cnt"] = pd.to_numeric(df["syn_flag_cnt"], errors="coerce").fillna(0)
    df["flow_iat_mean"] = pd.to_numeric(df["flow_iat_mean"], errors="coerce").fillna(0)

    # tạo window theo giây
    df["window"] = df["timestamp"].dt.floor(f"{window_sec}S")

    groups = df.groupby("window")

    rows = []
    for window_ts, g in groups:
        flows = len(g)
        if flows == 0:
            continue

        uniq_dst_ports = g["dst_port"].nunique() if "dst_port" in g.columns else 0
        syn_count = g["syn_flag_cnt"].sum()

        small_pkts = (g["pkt_len_mean"] < 80).sum()
        small_pkt_ratio = small_pkts / flows

        entropy_dst_port = shannon_entropy(g["dst_port"]) if "dst_port" in g.columns else 0.0
        avg_flow_iat_mean = g["flow_iat_mean"].mean()

        rows.append({
            "window_ts": window_ts,
            "flows": flows,
            "uniq_dst_ports": uniq_dst_ports,
            "syn_count": syn_count,
            "small_pkt_ratio": small_pkt_ratio,
            "entropy_dst_port": entropy_dst_port,
            "avg_flow_iat_mean": avg_flow_iat_mean,
        })

    win_df = pd.DataFrame(rows)
    return win_df


# main train

def main():
    print(f"Đang load benign data từ: {BENIGN_CSV_PATH}")
    df = pd.read_csv(BENIGN_CSV_PATH, low_memory=False)
    print(f"Số dòng raw flows: {len(df)}")

    # Trích xuất window features
    win_df = build_windows(df, window_sec=WINDOW_SEC)
    print(f"Số window sinh ra: {len(win_df)}")

    # Chuẩn bị dữ liệu cho KMeans One-Class
    feature_cols = [
        "flows",
        "uniq_dst_ports",
        "syn_count",
        "small_pkt_ratio",
        "entropy_dst_port",
        "avg_flow_iat_mean",
    ]

    X = win_df[feature_cols].values

    print("Chuẩn hoá dữ liệu (StandardScaler)...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("Train One-Class KMeans (1 cluster)...")
    kmeans = KMeans(n_clusters=1, random_state=42, n_init=10)
    kmeans.fit(X_scaled)

    # Tính threshold khoảng cách
    ## Lấy tâm cụm
    centroid = kmeans.cluster_centers_[0]
    ## Tính khoảng cách từ mỗi điểm đến tâm cụm
    dists = np.linalg.norm(X_scaled - centroid, axis=1)
    ## Lấy ngưỡng khoảng cách tại percentile đã định
    threshold = np.percentile(dists, THRESHOLD_PERCENTILE)

    print(f"Train xong. Threshold ({THRESHOLD_PERCENTILE}%) = {threshold:.4f}")

    # Lưu artifact
    joblib.dump(kmeans, os.path.join(ARTIFACT_DIR, "kmeans_window_oneclass.pkl"))
    joblib.dump(scaler, os.path.join(ARTIFACT_DIR, "scaler_window_oneclass.pkl"))
    np.save(os.path.join(ARTIFACT_DIR, "threshold_window.npy"),
            np.array(threshold, dtype=np.float32))

    with open(os.path.join(ARTIFACT_DIR, "feature_order_window.json"), "w") as f:
        json.dump(feature_cols, f, indent=2)

    print("Artifacts đã lưu trong thư mục artifacts_window/")


if __name__ == "__main__":
    main()
