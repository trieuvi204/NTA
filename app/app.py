
# app.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import asyncio

from routers.kmeans_router import router as kmeans_router, start_kmeans_monitor
from routers.dbscan_router import router as dbscan_router, start_dbscan_monitor


app = FastAPI(
    title="Unified NTA Realtime App",
    description="KMeans + DBSCAN realtime detection",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


#  Include Routers
app.include_router(kmeans_router)
app.include_router(dbscan_router)


#   Background Monitors
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(start_kmeans_monitor())
    asyncio.create_task(start_dbscan_monitor())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
