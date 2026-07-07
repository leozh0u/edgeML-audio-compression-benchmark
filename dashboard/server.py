"""
Dashboard backend.

Two services:
  1. HTTP (port 8000): serves the static SPA (index.html) and /api/benchmark,
     which returns results/benchmark.json (the merged Pareto table).
  2. WebSocket (port 8765): live inference stream. On-device the ESP32-S3 will
     POST/stream predictions here; until the board is flashed, --simulate emits
     synthetic predictions so the whole UI loop is demonstrable right now.

The ESP32 bridge hooks in at broadcast(): whatever feeds real predictions (a
serial reader, an MQTT subscriber, or an HTTP ingest endpoint) just calls
broadcast(prediction_dict). Nothing else changes.

Run:  python dashboard/server.py --simulate
Then open http://localhost:8000
"""

import argparse
import asyncio
import json
import random
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import websockets

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
RESULTS = REPO_ROOT / "results" / "benchmark.json"

# 50 ESC-50 classes (index order matches meta/esc50.csv targets).
ESC50_CLASSES = [
    "dog", "rooster", "pig", "cow", "frog", "cat", "hen", "insects", "sheep", "crow",
    "rain", "sea_waves", "crackling_fire", "crickets", "chirping_birds", "water_drops",
    "wind", "pouring_water", "toilet_flush", "thunderstorm", "crying_baby", "sneezing",
    "clapping", "breathing", "coughing", "footsteps", "laughing", "brushing_teeth",
    "snoring", "drinking_sipping", "door_wood_knock", "mouse_click", "keyboard_typing",
    "door_wood_creaks", "can_opening", "washing_machine", "vacuum_cleaner", "clock_alarm",
    "clock_tick", "glass_breaking", "helicopter", "chainsaw", "siren", "car_horn",
    "engine", "train", "church_bells", "airplane", "fireworks", "hand_saw",
]

CLIENTS = set()
_loop = None  # asyncio loop reference so sync code can schedule broadcasts


class APIHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(HERE), **kwargs)

    def do_GET(self):
        if self.path == "/api/benchmark":
            payload = RESULTS.read_bytes() if RESULTS.exists() else b"[]"
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(payload)
        else:
            super().do_GET()

    def log_message(self, *args):
        pass  # quiet


def start_http(port):
    srv = ThreadingHTTPServer(("", port), APIHandler)
    print(f"HTTP  : http://localhost:{port}")
    threading.Thread(target=srv.serve_forever, daemon=True).start()


async def ws_handler(ws):
    CLIENTS.add(ws)
    try:
        await ws.send(json.dumps({"type": "hello", "classes": ESC50_CLASSES}))
        await ws.wait_closed()
    finally:
        CLIENTS.discard(ws)


def broadcast(prediction: dict):
    """Thread-safe entry point for pushing a prediction to all dashboard clients.
    The ESP32 bridge calls this."""
    if _loop is None:
        return
    msg = json.dumps({"type": "prediction", **prediction})
    for ws in list(CLIENTS):
        asyncio.run_coroutine_threadsafe(ws.send(msg), _loop)


async def simulator():
    """Stand-in for the ESP32 feed: emit a plausible prediction every ~1.5s."""
    while True:
        await asyncio.sleep(1.5)
        idx = random.randrange(len(ESC50_CLASSES))
        conf = round(random.uniform(0.55, 0.98), 3)
        broadcast({
            "label": ESC50_CLASSES[idx],
            "class_id": idx,
            "confidence": conf,
            "latency_ms": round(random.uniform(8, 22), 1),
            "source": "simulated",
        })


async def main_async(args):
    global _loop
    _loop = asyncio.get_running_loop()
    start_http(args.http_port)
    print(f"WS    : ws://localhost:{args.ws_port}")
    if args.simulate:
        print("mode  : SIMULATE (synthetic predictions; swap for ESP32 feed later)")
        asyncio.create_task(simulator())
    else:
        print("mode  : LIVE (waiting for ESP32 to call broadcast())")
    async with websockets.serve(ws_handler, "", args.ws_port):
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--http-port", type=int, default=8000)
    ap.add_argument("--ws-port", type=int, default=8765)
    ap.add_argument("--simulate", action="store_true",
                    help="emit synthetic predictions until the ESP32 is flashed")
    args = ap.parse_args()
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("\nbye")
