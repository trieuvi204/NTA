#kmeans_router
from fastapi import APIRouter, WebSocket
import os, asyncio
import pandas as pd
from utils.realtime_utils_window import detect_window, THRESHOLD


router = APIRouter()
connected_clients = set()

CSV_FILE = "flows.csv"
WINDOW_SEC = 1
LAST_KMEANS_WINDOW = None


@router.websocket("/ws")
async def ws_kmeans(ws: WebSocket):
    await ws.accept()
    connected_clients.add(ws)
    print("[KMEANS] Client connected")

    try:
        while True:
            await asyncio.sleep(1)
    except:
        pass
    finally:
        connected_clients.remove(ws)
        print("[KMEANS] Client disconnected")


async def broadcast_kmeans(message: dict):
    dead = []
    for ws in connected_clients:
        try:
            await ws.send_json(message)
        except:
            dead.append(ws)

    for ws in dead:
        if ws in connected_clients:
            connected_clients.remove(ws)


async def start_kmeans_monitor():
    global LAST_KMEANS_WINDOW

    while True:
        try:
            if not os.path.exists(CSV_FILE):
                await asyncio.sleep(1)
                continue

            df = pd.read_csv(CSV_FILE, low_memory=False)
            if df.empty:
                await asyncio.sleep(1)
                continue

            df.columns = [c.lower().replace(" ", "_") for c in df.columns]
            if "timestamp" not in df.columns:
                await asyncio.sleep(1)
                continue

            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df["window"] = df["timestamp"].dt.floor("1s")

            latest = df["window"].max()

            if LAST_KMEANS_WINDOW and latest <= LAST_KMEANS_WINDOW:
                await asyncio.sleep(1)
                continue

            LAST_KMEANS_WINDOW = latest
            df_window = df[df["window"] == latest]

            result = detect_window(df_window)

            msg = {
                "event": "window_checked",
                "window_ts": latest.isoformat(),
                "label": result["label"],
                "distance": result["distance"],
                "threshold": THRESHOLD,
                "features": result["features"],
                "flows_in_window": len(df_window),
                "src_ips": result["src_ips"],
                "dst_ips": result["dst_ips"]
            }

            await broadcast_kmeans(msg)

        except Exception as e:
            print("[KMEANS ERROR]", e)

        await asyncio.sleep(1)
