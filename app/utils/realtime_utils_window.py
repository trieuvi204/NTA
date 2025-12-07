# realtime_utils_window.py 

import numpy as np
import pandas as pd
import joblib
import json
import os

from configs.configs import KMEANS_ARTIFACTS_DIR


KMEANS = joblib.load(os.path.join(KMEANS_ARTIFACTS_DIR, "kmeans_window_oneclass.pkl"))
SCALER = joblib.load(os.path.join(KMEANS_ARTIFACTS_DIR, "scaler_window_oneclass.pkl"))
THRESHOLD = float(np.load(os.path.join(KMEANS_ARTIFACTS_DIR, "threshold_window.npy")))
FEATURE_ORDER = json.load(open(os.path.join(KMEANS_ARTIFACTS_DIR, "feature_order_window.json")))


def shannon_entropy(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    counts = series.value_counts()
    probs = counts / counts.sum()
    return float(-(probs * np.log2(probs + 1e-12)).sum())


def build_window_feature(df_window: pd.DataFrame) -> pd.DataFrame:
    if df_window.empty:
        return pd.DataFrame(columns=FEATURE_ORDER)

    df = df_window.copy()
    df.columns = [
        c.strip().lower().replace(" ", "_").replace("/", "_").replace(".", "_")
        for c in df.columns
    ]

    if "dst_port" not in df.columns:
        df["dst_port"] = pd.to_numeric(df.iloc[:, 3], errors="coerce").fillna(0)

    for c in ["pkt_len_mean", "flow_iat_mean"]:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    flows = len(df)
    uniq_dst_ports = df["dst_port"].nunique()
    entropy_dst_port = shannon_entropy(df["dst_port"])
    avg_flow_iat_mean = df["flow_iat_mean"].mean()
    small_pkt_ratio = (df["pkt_len_mean"] < 80).sum() / flows

    row = {
        "flows": flows,
        "uniq_dst_ports": uniq_dst_ports,
        "entropy_dst_port": entropy_dst_port,
        "avg_flow_iat_mean": avg_flow_iat_mean,
        "small_pkt_ratio": small_pkt_ratio
    }

    win_df = pd.DataFrame([row])
    win_df = win_df.reindex(columns=FEATURE_ORDER, fill_value=0)

    return win_df


def detect_window(df_window: pd.DataFrame):

    if df_window.empty:
        return {
            "label": "Benign",
            "distance": 0.0,
            "threshold": THRESHOLD,
            "features": None,
            "src_ips": [],
            "dst_ips": []
        }

    df = df_window.copy()
    df.columns = [
        c.strip().lower().replace(" ", "_").replace("/", "_").replace(".", "_")
        for c in df.columns
    ]

    if "src_ip" not in df.columns:
        df["src_ip"] = df.iloc[:, 0]

    if "dst_ip" not in df.columns:
        df["dst_ip"] = df.iloc[:, 1]

    src_ips = df["src_ip"].unique().tolist()
    dst_ips = df["dst_ip"].unique().tolist()

    feat_df = build_window_feature(df)
    X_scaled = SCALER.transform(feat_df.values)

    centroid = KMEANS.cluster_centers_[0]
    dist = float(np.linalg.norm(X_scaled - centroid, axis=1)[0])

    is_attack = dist > THRESHOLD
    label = "Attack" if is_attack else "Benign"

    return {
        "label": label,
        "distance": dist,
        "threshold": THRESHOLD,
        "features": {col: float(feat_df.iloc[0][col]) for col in FEATURE_ORDER},
        "src_ips": src_ips,
        "dst_ips": dst_ips
    }
