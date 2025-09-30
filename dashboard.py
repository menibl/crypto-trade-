# =============================================
# dashboard.py  — FastAPI control panel for the bot (with .env editor + live reload)
# =============================================
# Features:
# - Start / Stop the bot loop (via stop.flag file signaling)
# - Panic liquidate: sell every non‑QUOTE asset to QUOTE (best‑effort, skips dust)
# - Status: quote balance, open positions (positions.json), recent trades (trades.json)
# - **NEW**: Edit .env keys from the UI + apply immediately (touches reload.flag)
# - Serves a small web UI at http://localhost:8000
#
# Usage:
#   pip install fastapi uvicorn python-dotenv binance-connector
#   python dashboard.py
#   → open http://localhost:8000
#
# Notes:
# - The bot must support hot-reload by watching reload.flag or .env mtime. See patch at bottom.
# - Keep dashboard.py in the same folder as: .env, bot_5pct.py, positions.json, trades.json

import os, json, math, time, logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv, dotenv_values, set_key
from binance.spot import Spot as SpotClient
from binance.error import ClientError

APP_DIR = Path(__file__).parent
STATE_POS = APP_DIR / "positions.json"
STATE_TRD = APP_DIR / "trades.json"
STOP_FLAG = APP_DIR / "stop.flag"
RELOAD_FLAG = APP_DIR / "reload.flag"
ENV_FILE  = APP_DIR / ".env"

TESTNET_URL = "https://testnet.binance.vision"
PROD_URL    = "https://api.binance.com"

# ------------------- helpers -------------------

def load_env() -> Dict[str, Any]:
    load_dotenv(dotenv_path=ENV_FILE)
    env = (os.getenv("BINANCE_ENV") or "prod").lower()
    return {
        "api_key": os.getenv("BINANCE_API_KEY"),
        "api_secret": os.getenv("BINANCE_API_SECRET"),
        "base_url": PROD_URL if env == "prod" else TESTNET_URL,
        "env": env,
        "quote": (os.getenv("QUOTE_ASSET") or "USDT").upper(),
    }

def get_client():
    cfg = load_env()
    if not cfg["api_key"] or not cfg["api_secret"]:
        raise RuntimeError("Missing API keys in .env")
    return SpotClient(api_key=cfg["api_key"], api_secret=cfg["api_secret"], base_url=cfg["base_url"]), cfg

def read_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except Exception:
        return default

def write_json(path: Path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2))

# ------------------- liquidation -------------------

def round_down(x, step):
    if step <= 0: return x
    return math.floor(x / step) * step

def symbol_filters(client: SpotClient, symbol: str):
    s = client.exchange_info(symbol=symbol)["symbols"][0]
    lot  = next(f for f in s["filters"] if f["filterType"] == "LOT_SIZE")
    price = next(f for f in s["filters"] if f["filterType"] == "PRICE_FILTER")
    notional = next((f for f in s["filters"] if f["filterType"] in ("NOTIONAL","MIN_NOTIONAL")), None)
    return {
        "stepSize": float(lot["stepSize"]),
        "minQty": float(lot.get("minQty", 0.0)),
        "tickSize": float(price["tickSize"]),
        "minNotional": float(notional["minNotional"]) if notional else None,
    }

def get_balance(client: SpotClient, asset: str):
    acct = client.account(recvWindow=5000)
    for b in acct["balances"]:
        if b["asset"] == asset:
            return float(b["free"]), float(b["locked"])
    return 0.0, 0.0

def sell_all_to_quote(client: SpotClient) -> Dict[str, Any]:
    cfg = load_env()
    acct = client.account(recvWindow=5000)
    assets = [(b["asset"], float(b["free"])) for b in acct["balances"] if float(b["free"]) > 0]
    to_sell = [(a, q) for a, q in assets if a != cfg["quote"]]
    results = []
    for base, qty_free in to_sell:
        symbol = f"{base}{cfg['quote']}"
        try:
            info = client.exchange_info(symbol=symbol)["symbols"][0]
            if info.get("status") != "TRADING":
                results.append({"symbol": symbol, "status": "skip", "reason": "not TRADING"})
                continue
        except Exception:
            results.append({"symbol": symbol, "status": "skip", "reason": "pair not found"})
            continue

        f = symbol_filters(client, symbol)
        step, minq = f["stepSize"], f["minQty"]
        target = qty_free * 0.998
        q = round_down(target, step)
        if q <= 0 or (minq and q < minq):
            results.append({"symbol": symbol, "status": "skip", "reason": "dust"})
            continue

        attempt = 0
        while attempt < 3:
            try:
                ord = client.new_order(symbol=symbol, side="SELL", type="MARKET", quantity=str(q), recvWindow=5000)
                recv = float(ord.get("cummulativeQuoteQty", 0.0))
                exe  = float(ord.get("executedQty", 0.0))
                results.append({"symbol": symbol, "status": "sold", "qty": exe, "quote_recv": recv})
                break
            except ClientError as e:
                if getattr(e, "error_code", None) == -2010 or "-2010" in str(e):
                    attempt += 1
                    q = round_down(q * 0.98, step)
                    if q <= 0 or (minq and q < minq):
                        results.append({"symbol": symbol, "status": "skip", "reason": "fell under minQty"})
                        break
                    time.sleep(0.2)
                    continue
                raise
        time.sleep(0.2)
    return {"summary": results}

# ------------------- API -------------------

from logging.handlers import RotatingFileHandler
def init_dashboard_logging():
    log_name = os.getenv("BOT_LOG_FILE", "bot.log")
    log_level = os.getenv("BOT_LOG_LEVEL", "INFO").upper()
    log_path = APP_DIR / log_name
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if not logging.getLogger().handlers:
        fh = RotatingFileHandler(str(log_path), maxBytes=5_000_000, backupCount=3, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        fh.setLevel(getattr(logging, log_level, logging.INFO))
        logging.getLogger().addHandler(fh)
        logging.getLogger().setLevel(getattr(logging, log_level, logging.INFO))
    logging.info("dashboard started; writing logs to %s", log_path.resolve())

init_dashboard_logging()

app = FastAPI(title="Trading Bot Dashboard")

# ---- NEW: .env editor ----
# Lock down to these keys; add/remove as you need.
ALLOWED_ENV_KEYS = [
    "LOG_LEVEL","QUOTE_ASSET","BASE_BLACKLIST",
    "TARGET_TP_PCT","OPTIONAL_SL_PCT","TRAILING_STOP_PCT","TRAILING_ARM_PNL_PCT",
    "MAX_OPEN_POSITIONS","ALLOCATION_USD",
    "USE_ALL_BALANCE","BALANCE_RESERVE_PCT","CONSOLIDATE_AT_START",
    "AUTO_SPLIT","SPLIT_SLICES","MIN_SLICE_USD",
    "MIN_24H_QUOTE_VOL","MAX_SPREAD_BPS","SCAN_INTERVAL_SEC","EOD_HHMM",
    "PORTFOLIO_TARGET_PCT","PORTFOLIO_STOP_PCT",
    "ROTATION_ENABLED","ROTATION_EDGE_PCT","FEE_BPS","SLIPPAGE_BPS",
    "REGIME_CHECK_ENABLED","REGIME_TF","REGIME_EMA_SHORT","REGIME_EMA_LONG","REGIME_MIN_AGREE","REGIME_COOLDOWN_MIN",
    "BUY_TF","BUY_EMA_SHORT","BUY_EMA_LONG","BUY_MIN_SLOPE_PCT_PER_BAR",
    "HTF_ALIGN_ENABLED","HTF_TF","HTF_EMA_SHORT","HTF_EMA_LONG",
    "USE_MAKER","MAKER_WAIT_SEC","LOSS_COOLDOWN_MIN"
]

@app.get("/env")
def get_env():
    env = dotenv_values(ENV_FILE)
    return {k: env.get(k, "") for k in ALLOWED_ENV_KEYS}

class EnvPayload(BaseModel):
    apply: bool = True
    updates: Dict[str, str] = {}

@app.post("/env")
def set_env(payload: EnvPayload):
    env = dotenv_values(ENV_FILE)
    changed = 0
    for k, v in (payload.updates or {}).items():
        if k in ALLOWED_ENV_KEYS:
            set_key(ENV_FILE.as_posix(), k, "" if v is None else str(v), quote_mode="never")
            changed += 1
    if payload.apply:
        RELOAD_FLAG.write_text(str(datetime.utcnow()))
    return {"ok": True, "changed": changed, "applied": payload.apply}

@app.post("/reload")
def reload_now():
    RELOAD_FLAG.write_text(str(datetime.utcnow()))
    return {"ok": True, "message": "reload.flag touched"}

# ------------------- legacy endpoints -------------------

@app.get("/status")
def status():
    c, cfg = get_client()
    quote, _ = get_balance(c, cfg["quote"])
    running = not STOP_FLAG.exists()
    positions = read_json(STATE_POS, {"positions": {}})
    trades = read_json(STATE_TRD, {"trades": []})
    return {
        "env": cfg["env"],
        "quote": cfg["quote"],
        "running": running,
        "quote_free": quote,
        "positions": positions.get("positions", {}),
        "trades": trades.get("trades", [])[-100:],
    }

class Command(BaseModel):
    confirm: bool = True

@app.post("/start")
def start(cmd: Command):
    if STOP_FLAG.exists():
        STOP_FLAG.unlink()
    logging.info("DASHBOARD: start requested → clearing stop.flag")
    return {"ok": True}

@app.post("/stop")
def stop(cmd: Command):
    STOP_FLAG.write_text("stop")
    logging.info("DASHBOARD: stop requested → creating stop.flag")
    return {"ok": True}

@app.post("/panic")
def panic(cmd: Command):
    logging.warning("DASHBOARD: PANIC requested → sell all to QUOTE")
    c, _ = get_client()
    res = sell_all_to_quote(c)
    logging.warning("DASHBOARD: PANIC result → %s", res)
    return {"ok": True, "result": res}

# ------------- UI -------------
STATIC_DIR = APP_DIR / "static"
STATIC_DIR.mkdir(exist_ok=True)
INDEX_HTML = STATIC_DIR / "index.html"

UI_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Trading Bot Dashboard</title>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <style>
    body{font-family:system-ui,Arial,sans-serif; margin:20px;}
    h1{margin:0 0 12px}
    button{padding:8px 12px; margin-right:8px; border-radius:8px; border:1px solid #ccc; cursor:pointer}
    .row{display:flex; gap:16px; flex-wrap:wrap}
    .card{border:1px solid #e5e7eb; border-radius:12px; padding:12px; box-shadow:0 1px 3px rgba(0,0,0,.06)}
    table{border-collapse:collapse; width:100%}
    th,td{border-bottom:1px solid #eee; padding:6px 8px; text-align:left}
    .pill{display:inline-block; padding:2px 8px; border-radius:999px; background:#f3f4f6}
    input[type=text]{width:240px;padding:6px;margin:4px 0;}
    .grid{display:grid;grid-template-columns:repeat(2, minmax(260px,1fr));gap:12px}
  </style>
</head>
<body>
  <h1>Trading Bot Dashboard</h1>
  <div class="row">
    <div class="card">
      <div>Running: <span id="running" class="pill">-</span> · Env: <span id="env" class="pill">-</span> · Quote: <span id="quote" class="pill">-</span></div>
      <div>Quote free: <b id="quote_free">-</b></div>
      <div style="margin-top:8px">
        <button onclick="doStart()">Start</button>
        <button onclick="doStop()">Stop</button>
        <button onclick="doPanic()" style="background:#fee2e2">Panic → QUOTE</button>
        <button onclick="refresh()">Refresh</button>
      </div>
    </div>
  </div>

  <div class="row" style="margin-top:16px">
    <div class="card" style="flex:1; min-width:320px">
      <h3>Open Positions</h3>
      <table id="pos"><thead><tr><th>Symbol</th><th>Qty</th><th>Entry</th><th>Opened</th><th>Source</th></tr></thead><tbody></tbody></table>
    </div>
    <div class="card" style="flex:1; min-width:320px">
      <h3>Recent Trades</h3>
      <table id="trades"><thead><tr><th>Time</th><th>Action</th><th>Symbol</th><th>Qty</th><th>Avg Price</th><th>Quote</th><th>Reason</th></tr></thead><tbody></tbody></table>
    </div>
  </div>

  <div class="row" style="margin-top:16px">
    <div class="card" style="flex:1; min-width:320px">
      <h3>.env Editor</h3>
      <div id="envGrid" class="grid"></div>
      <div style="margin-top:8px">
        <button onclick="saveEnv()">Save .env</button>
        <button onclick="applyReload()">Save + Apply</button>
        <button onclick="reloadOnly()">Reload from .env</button>
      </div>
    </div>
  </div>

<script>
async function refresh(){
  const r = await fetch('/status'); const s = await r.json();
  document.getElementById('env').textContent = s.env;
  document.getElementById('quote').textContent = s.quote;
  document.getElementById('running').textContent = s.running ? 'YES' : 'NO';
  document.getElementById('quote_free').textContent = (+s.quote_free).toFixed(6);

  const tbody = document.querySelector('#pos tbody'); tbody.innerHTML = '';
  for (const [sym, p] of Object.entries(s.positions||{})){
    const tr = document.createElement('tr');
    tr.innerHTML = `<td>${sym}</td><td>${(+p.qty).toFixed(8)}</td><td>${(+p.entry).toFixed(8)}</td><td>${p.opened_at||''}</td><td>${p.source||''}</td>`;
    tbody.appendChild(tr);
  }

  const trades = (s.trades||[]).slice().reverse();
  const ttb = document.querySelector('#trades tbody'); ttb.innerHTML = '';
  for (const t of trades){
    const tr = document.createElement('tr');
    tr.innerHTML = `<td>${t.ts||''}</td><td>${t.side||''}</td><td>${t.symbol||''}</td><td>${(+t.qty||0).toFixed(8)}</td><td>${t.avg?(+t.avg).toFixed(8):''}</td><td>${t.quote?(+t.quote).toFixed(8):''}</td><td>${t.reason||''}</td>`;
    ttb.appendChild(tr);
  }
}

async function loadEnv(){
  const r = await fetch('/env'); const env = await r.json();
  const g = document.getElementById('envGrid'); g.innerHTML='';
  Object.entries(env).forEach(([k,v])=>{
    const label = document.createElement('label'); label.textContent = k;
    const input = document.createElement('input'); input.type='text'; input.id='env_'+k; input.value = v ?? '';
    const wrap = document.createElement('div'); wrap.appendChild(label); wrap.appendChild(input);
    g.appendChild(wrap);
  });
}

async function saveEnv(){
  const updates = {};
  document.querySelectorAll('#envGrid input').forEach(inp=>{
    const k = inp.id.replace('env_',''); updates[k] = inp.value;
  });
  const r = await fetch('/env', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({apply:false, updates})});
  const j = await r.json(); alert('Saved: '+j.changed+' keys');
}

async function applyReload(){
  const updates = {};
  document.querySelectorAll('#envGrid input').forEach(inp=>{
    const k = inp.id.replace('env_',''); updates[k] = inp.value;
  });
  const r = await fetch('/env', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({apply:true, updates})});
  const j = await r.json(); alert('Saved & applied: '+j.changed+' keys');
}

async function reloadOnly(){
  const r = await fetch('/reload', {method:'POST'});
  const j = await r.json(); alert(j.message||'Reload signal sent');
}

refresh(); loadEnv();
setInterval(refresh, 5000);
</script>
</body>
</html>
"""

# Write index.html
STATIC_DIR = APP_DIR / "static"
STATIC_DIR.mkdir(exist_ok=True)
INDEX_HTML = STATIC_DIR / "index.html"
INDEX_HTML.write_text(UI_HTML, encoding="utf-8")

# Mount static (serves index.html at /)
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")

# =========================
# PATCH for bot_5pct.py
# =========================
# 1) At top-level (near other path constants):
#    from pathlib import Path
#    ENV_FILE   = Path(__file__).parent / ".env"
#    RELOAD_FLAG= Path(__file__).parent / "reload.flag"
#    _LAST_ENV_MTIME = 0.0
#
# 2) Add a function to reload when .env changed (or when reload.flag exists):
#    from dotenv import load_dotenv
#    def reload_cfg_if_changed(cfg, client):
#        global _LAST_ENV_MTIME
#        need = RELOAD_FLAG.exists()
#        try:
#            mtime = ENV_FILE.stat().st_mtime
#        except FileNotFoundError:
#            mtime = _LAST_ENV_MTIME
#        if mtime > _LAST_ENV_MTIME:
#            need = True
#        if not need:
#            return cfg, client, False
#        # re-load .env and rebuild cfg
#        load_dotenv(ENV_FILE, override=True)
#        new_cfg = load_env()   # ← your existing loader that returns dict
#        # if base_url or keys changed, rebuild client
#        if (new_cfg.get("base_url") != cfg.get("base_url")
#            or new_cfg.get("api_key") != cfg.get("api_key")
#            or new_cfg.get("api_secret") != cfg.get("api_secret")):
#            client = client_from_cfg(new_cfg)
#        _LAST_ENV_MTIME = mtime
#        if RELOAD_FLAG.exists():
#            try: RELOAD_FLAG.unlink()
#            except Exception: pass
#        return new_cfg, client, True
#
# 3) In your main loop (each tick), call:
#        cfg, client, reloaded = reload_cfg_if_changed(cfg, client)
#        if reloaded:
#            logging.info("Hot-reload: applied new .env settings")
#
# Done. Now the dashboard “Save + Apply” touches reload.flag; the bot detects and reloads within seconds.