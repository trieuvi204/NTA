import numpy as np
import pandas as pd
import joblib, json
from numpy.linalg import norm

# ==== LOAD ARTIFACTS ====
DBSCAN_MODEL = joblib.load("artifacts/dbscan_window/dbscan.pkl")
SCALER = joblib.load("artifacts/dbscan_window/scaler.pkl")
FEATURE_ORDER = json.load(open("artifacts/dbscan_window/feature_order.json"))

EPS = float(DBSCAN_MODEL.eps)


#tính entropy
def shannon_entropy(series):
    if series.empty:
        return 0.0
    p = series.value_counts(normalize=True)
    return float(-(p * np.log2(p + 1e-12)).sum())


#  FEATURE EXTRACTION
def extract_window_features(df):

    df = df.copy()

    # Chuẩn hóa tên cột → tránh lỗi mismatch
    df.columns = df.columns.str.strip()

    # Nếu thiếu cột, setup giá trị mặc định 0
    required = [
        "IPV4_SRC_ADDR","IPV4_DST_ADDR","L4_DST_PORT",
        "IN_BYTES","OUT_BYTES","FLOW_DURATION_MILLISECONDS",
        "SRC_TO_DST_SECOND_BYTES","DST_TO_SRC_SECOND_BYTES",
        "SRC_TO_DST_IAT_AVG","SRC_TO_DST_IAT_STDDEV",
        "DST_TO_SRC_IAT_AVG","DST_TO_SRC_IAT_STDDEV"
    ]

    for col in required:
        if col not in df.columns:
            df[col] = 0.0

    total_bytes = df["IN_BYTES"] + df["OUT_BYTES"]

    feats = {
        "flows": len(df),

        "uniq_src_ip": df["IPV4_SRC_ADDR"].nunique(),
        "uniq_dst_ip": df["IPV4_DST_ADDR"].nunique(),
        "uniq_dst_port": df["L4_DST_PORT"].nunique(),

        "entropy_dst_port": shannon_entropy(df["L4_DST_PORT"]),

        "avg_in_bytes": df["IN_BYTES"].mean(),
        "avg_out_bytes": df["OUT_BYTES"].mean(),
        "avg_duration_ms": df["FLOW_DURATION_MILLISECONDS"].mean(),

        "small_flow_ratio": (total_bytes < 200).mean(),

        "avg_src_to_dst_bps": df["SRC_TO_DST_SECOND_BYTES"].mean(),
        "avg_dst_to_src_bps": df["DST_TO_SRC_SECOND_BYTES"].mean(),

        "src_iat_avg": df["SRC_TO_DST_IAT_AVG"].mean(),
        "src_iat_std": df["SRC_TO_DST_IAT_STDDEV"].mean(),
        "dst_iat_avg": df["DST_TO_SRC_IAT_AVG"].mean(),
        "dst_iat_std": df["DST_TO_SRC_IAT_STDDEV"].mean(),
    }

    return feats


#  MAIN DETECTOR
def detect_dbscan(df_window):

    if df_window.empty:
        return {
            "label": "Benign",
            "distance": 0.0,
            "flows": 0,
            "src_ips": [],
            "dst_ips": [],
            "threshold": EPS
        }

    # Ensure correct column format
    dfw = df_window.copy()
    dfw.columns = dfw.columns.str.strip()

    # Collect IPs for UI
    src_ips = dfw["IPV4_SRC_ADDR"].unique().tolist() if "IPV4_SRC_ADDR" in dfw.columns else []
    dst_ips = dfw["IPV4_DST_ADDR"].unique().tolist() if "IPV4_DST_ADDR" in dfw.columns else []

    # Extract features
    feats = extract_window_features(dfw)

    # Create feature vector – đảm bảo đúng thứ tự FEATURE_ORDER
    X = np.array([[feats.get(f, 0.0) for f in FEATURE_ORDER]], dtype=float)

    # Scale
    X_scaled = SCALER.transform(X)

    # DBSCAN distance → khoảng cách tới vùng benign
    core = DBSCAN_MODEL.components_
    dists = norm(core - X_scaled, axis=1)
    min_dist = float(dists.min())

    # Compare distance vs EPS
    label = "Attack" if min_dist > EPS else "Benign"

    return {
        "label": label,
        "distance": min_dist,
        "flows": int(feats["flows"]),
        "src_ips": src_ips,
        "dst_ips": dst_ips,
        "threshold": EPS
    }
