# Dashboard

Real-time benchmark + inference dashboard. Zero-build React (loaded from CDN via
`React.createElement`, no npm/bundler) plus a Python backend using only stdlib
HTTP + the `websockets` package.

## Run

```bash
../esc50env/bin/python3 server.py --simulate
# then open http://localhost:8000
```

- `--simulate` emits synthetic predictions every ~1.5s so the live panel works
  before the ESP32 is flashed. Drop the flag for a real device feed.
- Requires `results/benchmark.json` (produced by `scripts/benchmark.py`) for the
  Pareto chart + table; the page still loads without it.

## What it shows
- **Accuracy vs size Pareto frontier** (SVG scatter, log-x), colored by stage
  (FP32 / PTQ / QAT / prune), with the Pareto-optimal set connected.
- **All variants** table (accuracy, size, host-CPU latency).
- **Live inference** panel over WebSocket (`ws://<host>:8765`): top-1 class,
  confidence bar, per-inference latency.

## Wiring the ESP32 later
The device just needs to deliver predictions to `server.broadcast({...})`. Two
easy options: (a) the ESP32 opens a WebSocket to a small ingest endpoint that
calls `broadcast`, or (b) a serial reader thread parses the board's USB output
and calls `broadcast`. No frontend changes needed — it already renders whatever
arrives on the WebSocket.

## Ports
HTTP `8000`, WebSocket `8765` (override with `--http-port` / `--ws-port`).
