from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Query, Path
from fastapi.middleware.cors import CORSMiddleware
from hold_seat import hold_seat, confirm_seat
import redis.asyncio as aioredis
import json
import os

import sentry_sdk
sentry_sdk.init(
    dsn=os.environ.get("SENTRY_DSN"),
    send_default_pii=True,
)

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

app = FastAPI()

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost")

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        dead = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception:
                dead.append(connection)
        for connection in dead:
            self.active_connections.remove(connection)

manager = ConnectionManager()

@app.post("/seats/{seat_id}/hold")
@limiter.limit("10/minute")
async def hold(
    request: Request,
    seat_id: int = Path(..., ge=1, le=8),
    user_id: str = Query(..., min_length=1, max_length=100, pattern=r"^[a-zA-Z0-9_-]+$"),
):
    won = hold_seat(seat_id, user_id)
    if won:
        r = aioredis.from_url(REDIS_URL)
        await r.publish("seat_updates", json.dumps({"seat_id": seat_id, "status": "held", "user_id": user_id}))
        await r.aclose()
    return {"held": won}

@app.post("/seats/{seat_id}/confirm")
async def confirm(seat_id: int, user_id: str, idempotency_key: str):
    result = confirm_seat(seat_id, user_id, idempotency_key)
    if result["confirmed"] and not result["replay"]:
        r = aioredis.from_url(REDIS_URL)
        await r.publish("seat_updates", json.dumps({"seat_id": seat_id, "status": "booked", "user_id": user_id}))
        await r.aclose()
    return result

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    r = aioredis.from_url(REDIS_URL)
    pubsub = r.pubsub()
    await pubsub.subscribe("seat_updates")
    try:
        async for message in pubsub.listen():
            if message["type"] == "message":
                await manager.broadcast(message["data"].decode())
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    finally:
        manager.disconnect(websocket)
        await r.aclose()
