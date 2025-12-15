# app/routers/dbscan_router.py
from fastapi import APIRouter, WebSocket
import pandas as pd, asyncio, os
from utils.realtime_utils_dbscan import detect_dbscan

router = APIRouter()
CLIENTS = set()
CSV_FILE = "dbscan_flows.csv"
LAST_WIN = None

@router.websocket("/ws_dbscan")
async def ws_dbscan(ws: WebSocket):
    await ws.accept()
    CLIENTS.add(ws)
    try:
        while True:
            await asyncio.sleep(1)
    except:
        pass
    finally:
        CLIENTS.remove(ws)


async def broadcast(msg):
    bad = []
    for ws in CLIENTS:
        try: await ws.send_json(msg)
        except: bad.append(ws)
    for dead in bad:
        CLIENTS.remove(dead)


async def start_dbscan_monitor():
    global LAST_WIN
    while True:
        try:
            if not os.path.exists(CSV_FILE):
                await asyncio.sleep(1)
                continue

            df = pd.read_csv(CSV_FILE)
            if df.empty:
                await asyncio.sleep(1)
                continue

            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df["window"] = df["timestamp"].dt.floor("1s")
            win = df["window"].max()

            if LAST_WIN == win:
                await asyncio.sleep(1)
                continue

            LAST_WIN = win
            dfw = df[df["window"] == win]

            out = detect_dbscan(dfw)
            out["window_ts"] = win.isoformat()

            await broadcast(out)

        except Exception as e:
            print("[DBSCAN ERR]", e)

        await asyncio.sleep(1)
