# =============================================
# dashboard.py — FastAPI control panel for the bot
# =============================================
# Features:
# - Start / Stop the bot loop (via stop.flag signaling)
# - Panic liquidate: sell every non-QUOTE asset to QUOTE (best-effort, skips dust)
# - Status: quote balance, open positions (positions.json), recent trades (trades.json)
# - Equity total (cash + positions) and daily % change (baseline resets each day)
# - Edit .env keys from the UI + apply immediately (touches reload.flag)
# - Serves a small web UI at http://localhost:8000
#
# Usage:
#   pip install fastapi uvicorn python-dotenv binance-connector
#   python dashboard.py
#   → open http://localhost:8000
#
# Notes:
# - Keep dashboard.py in the same folder as: .env, bot_5pct_v3_info.py, positions.json, trades.json

import os, json, math, time, logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv, dotenv_values, set_key

from binance.spot import Spot as SpotClient
from binance.error import ClientError, ServerError
from logging.handlers import RotatingFileHandler

# ------------------- paths & constants -------------------

APP_DIR     = Path(__file__).parent
STATE_POS   = APP_DIR / "positions.json"
STATE_TRD   = APP_DIR / "trades.json"
STOP_FLAG   = APP_DIR / "stop.flag"
RELOAD_FLAG = APP_DIR / "reload.flag"
ENV_FILE    = APP_DIR / ".env"
EQUITY_DAY  = APP_DIR / "equity_day.json"

TESTNET_URL = "https://testnet.binance.vision"
PROD_URL    = "https://api.binance.com"
STABLES     = {"USDT","USDC","FDUSD","TUSD","BUSD","DAI","USDP","EURT","USTC"}

# ------------------- logging -------------------

def init_dashboard_logging():
    log_name = os.getenv("BOT_LOG_FILE", "bot.log")
    log_level = (os.getenv("BOT_LOG_LEVEL", "INFO") or "INFO").upper()
    log_path = APP_DIR / log_name
    log_path.parent.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(getattr(logging, log_level, logging.INFO))

    # avoid duplicate handlers on reload
    for h in list(root.handlers):
        root.removeHandler(h)

    fh = RotatingFileHandler(str(log_path), maxBytes=5_000_000, backupCount=3, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    fh.setLevel(getattr(logging, log_level, logging.INFO))
    root.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    ch.setLevel(getattr(logging, log_level, logging.INFO))
    root.addHandler(ch)

    logging.info("dashboard started; writing logs to %s", log_path.resolve())

init_dashboard_logging()

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

def get_client() -> Tuple[SpotClient, Dict[str, Any]]:
    cfg = load_env()
    if not cfg["api_key"] or not cfg["api_secret"]:
        raise RuntimeError("Missing API keys in .env")
    client = SpotClient(api_key=cfg["api_key"], api_secret=cfg["api_secret"], base_url=cfg["base_url"], timeout=10)
    return client, cfg

def read_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default

def write_json(path: Path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def get_balance(client: SpotClient, asset: str) -> Tuple[float, float]:
    acct = client.account(recvWindow=5000)
    for b in acct["balances"]:
        if b["asset"] == asset:
            return float(b["free"]), float(b["locked"])
    return 0.0, 0.0

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
        "base": s["baseAsset"],
        "quote": s["quoteAsset"],
    }

def round_down(x, step):
    if step <= 0: return x
    return math.floor(x / step) * step

def mid_price_safe(client: SpotClient, symbol: str) -> float:
    """Try (bid+ask)/2; fallback to ticker_price."""
    try:
        bk = client.book_ticker(symbol=symbol)
        bid = float(bk.get("bidPrice") or 0)
        ask = float(bk.get("askPrice") or 0)
        if bid > 0 and ask > 0:
            return (bid + ask) / 2.0
    except Exception:
        pass
    try:
        tp = client.ticker_price(symbol=symbol)
        p = float(tp.get("price") or 0)
        return p if p > 0 else 0.0
    except Exception:
        return 0.0

def portfolio_values(client: SpotClient, cfg: Dict[str, Any], positions_map: Dict[str, Any]) -> Dict[str, Any]:
    q_free, q_locked = get_balance(client, cfg["quote"])
    pos_val = 0.0
    snapshots: List[Dict[str, Any]] = []

    for sym, ps in (positions_map or {}).items():
        qty = float(ps.get("qty", 0.0) or 0.0)
        entry = float(ps.get("entry", 0.0) or 0.0)
        if qty <= 0:
            continue
        price = mid_price_safe(client, sym)
        if price <= 0:
            continue

        unrl = qty * (price - entry) if entry > 0 else 0.0
        pnl_pct = ((price/entry) - 1.0)*100.0 if entry > 0 else 0.0

        # Optional TP/SL distances if exist on pstate (MOB) — otherwise '-'
        tp1_pct = float(ps.get("tp1_pct", 0.0) or 0.0)
        tp2_pct = float(ps.get("tp2_pct", 0.0) or 0.0)
        sl_pct  = float(ps.get("sl_pct",  0.0) or 0.0)

        def dist_pct(target, ref):
            return ((target/ref)-1.0)*100.0 if ref>0 else 0.0

        toTP1 = dist_pct(entry*(1.0+tp1_pct), price) if tp1_pct>0 else None
        toTP2 = dist_pct(entry*(1.0+tp2_pct), price) if tp2_pct>0 else None
        toSL  = dist_pct(entry*(1.0-sl_pct),  price) if sl_pct>0  else None

        pos_val += qty * price
        snapshots.append({
            "symbol": sym,
            "qty": qty,
            "price": price,
            "entry": entry,
            "pnl_pct": pnl_pct,
            "unrealized": unrl,
            "toTP1": toTP1,
            "toTP2": toTP2,
            "toSL": toSL,
            "opened_at": ps.get("opened_at",""),
            "source": ps.get("source",""),
        })

    equity = q_free + q_locked + pos_val
    return {
        "quote_free": q_free,
        "quote_locked": q_locked,
        "positions_value": pos_val,
        "equity_total": equity,
        "pos_snapshots": snapshots
    }

def load_day_baseline() -> Dict[str, Any]:
    return read_json(EQUITY_DAY, {})

def save_day_baseline(date_str: str, equity: float):
    write_json(EQUITY_DAY, {"date": date_str, "equity": equity})

# ------------------- liquidation (panic) -------------------

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
                # insufficient balance / min notional? reduce and retry
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

# ------------------- API app -------------------

app = FastAPI(title="Trading Bot Dashboard")

# ---- NEW: .env editor ----
ALLOWED_ENV_KEYS = [
    "BINANCE_ENV","QUOTE_ASSET","LOG_LEVEL","LOG_FILE","LOG_MAX_BYTES","LOG_BACKUP_COUNT",
    "STRATEGY",
    "SCAN_INTERVAL_SEC","MIN_24H_QUOTE_VOL","MIN_DEPTH_10BPS_USD","USE_RS_FILTER",
    "MAX_SPREAD_BPS",
    "ENTRY_BREAKOUT_MULT","ENTRY_VOL_MULT","MOMENTUM_RET_15M","OB_IMBALANCE_MIN","EVENT_VOL_Z",
    "USE_ALL_BALANCE","ALLOCATION_USD","MAX_OPEN_POSITIONS","AUTO_SPLIT","SPLIT_SLICES",
    "MIN_SLICE_USD","BALANCE_RESERVE_PCT",
    "TP1_PCT","TP1_FRACTION","TP2_PCT","SL_MIN_PCT","ATR_MULT_SL",
    "TRAILING_ARM_PNL_PCT","TRAILING_STOP_PCT","MAX_TIME_HOURS","REGIME_CHECK_ENABLED",
    "REQUIRE_VOL_CONFIRMATION","REQUIRE_ORDERBOOK_IMBALANCE","SL_POLICY",
    "TRAIL_STEP_UP_PCT","TRAIL_LOCK_PCT",
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
        RELOAD_FLAG.write_text(datetime.utcnow().isoformat())
    return {"ok": True, "changed": changed, "applied": payload.apply}

@app.post("/reload")
def reload_now():
    RELOAD_FLAG.write_text(datetime.utcnow().isoformat())
    return {"ok": True, "message": "reload.flag touched"}

# ------------------- legacy endpoints -------------------

@app.get("/status")
def status():
    # positions/trades
    positions_all = read_json(STATE_POS, {"positions": {}})
    positions_map = positions_all.get("positions", {})
    trades = read_json(STATE_TRD, {"trades": []}).get("trades", [])[-100:]

    # running flag
    running = not STOP_FLAG.exists()

    # env / client
    try:
        c, cfg = get_client()
    except Exception as e:
        # allow UI to load even without keys
        cfg = load_env()
        return {
            "env": cfg["env"],
            "quote": cfg["quote"],
            "running": running,
            "quote_free": 0.0,
            "quote_locked": 0.0,
            "positions": positions_map,
            "trades": trades,
            "positions_value": 0.0,
            "equity_total": 0.0,
            "day_change_pct": 0.0,
            "pos_snapshots": [],
            "warn": f"status degraded (no API): {e}"
        }

    # equity & snapshots
    vals = portfolio_values(c, cfg, positions_map)

    # daily baseline
    today = datetime.now().strftime("%Y-%m-%d")
    base = load_day_baseline()
    if base.get("date") != today or "equity" not in base:
        save_day_baseline(today, vals["equity_total"])
        day_change_pct = 0.0
    else:
        b = float(base.get("equity", 0.0) or 0.0)
        day_change_pct = ((vals["equity_total"]/b) - 1.0) if b > 0 else 0.0

    return {
        "env": cfg["env"],
        "quote": cfg["quote"],
        "running": running,
        "quote_free": vals["quote_free"],
        "quote_locked": vals["quote_locked"],
        "positions": positions_map,
        "trades": trades,
        "positions_value": vals["positions_value"],
        "equity_total": vals["equity_total"],
        "day_change_pct": day_change_pct,
        "pos_snapshots": vals["pos_snapshots"],
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
    try:
        c, _ = get_client()
    except Exception as e:
        raise HTTPException(400, f"Cannot run PANIC without API keys: {e}")
    try:
        res = sell_all_to_quote(c)
        logging.warning("DASHBOARD: PANIC result → %s", res)
        return {"ok": True, "result": res}
    except (ClientError, ServerError) as e:
        raise HTTPException(502, f"Binance error: {e}")

# ------------------- UI -------------------

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
    :root{--ok:#16a34a;--bad:#dc2626;--muted:#6b7280;}
    body{font-family:system-ui,Arial,sans-serif; margin:20px;}
    h1{margin:0 0 12px}
    button{padding:8px 12px; margin-right:8px; border-radius:8px; border:1px solid #ccc; cursor:pointer}
    .row{display:flex; gap:16px; flex-wrap:wrap}
    .card{border:1px solid #e5e7eb; border-radius:12px; padding:12px; box-shadow:0 1px 3px rgba(0,0,0,.06)}
    table{border-collapse:collapse; width:100%}
    th,td{border-bottom:1px solid #eee; padding:6px 8px; text-align:left; font-size:14px}
    .pill{display:inline-block; padding:2px 8px; border-radius:999px; background:#f3f4f6}
    input[type=text]{width:240px;padding:6px;margin:4px 0;}
    .grid{display:grid;grid-template-columns:repeat(2, minmax(260px,1fr));gap:12px}
    .muted{color:var(--muted)}
    .posok{color:var(--ok)} .posbad{color:var(--bad)}
  </style>
</head>
<body>
  <h1>Trading Bot Dashboard</h1>
  <div class="row">
    <div class="card" style="flex:1">
      <div>Running: <span id="running" class="pill">-</span> · Env: <span id="env" class="pill">-</span> · Quote: <span id="quote" class="pill">-</span></div>
      <div>Quote free: <b id="quote_free">-</b> · Quote locked: <span id="quote_locked">-</span></div>
      <div>Equity total: <b id="equity_total">-</b> · Positions value: <span id="pos_val">-</span> · Day Δ: <span id="day_change_pct">-</span></div>
      <div style="margin-top:8px">
        <button onclick="doStart()">Start</button>
        <button onclick="doStop()">Stop</button>
        <button onclick="doPanic()" style="background:#fee2e2">Panic → QUOTE</button>
        <button onclick="refresh()">Refresh</button>
      </div>
      <div id="warn" class="muted" style="margin-top:6px"></div>
    </div>
  </div>

  <div class="row" style="margin-top:16px">
    <div class="card" style="flex:1; min-width:360px">
      <h3>Open Positions</h3>
      <table id="pos">
        <thead>
          <tr>
            <th>Symbol</th><th>Qty</th><th>Price</th><th>Entry</th><th>PnL %</th><th>Unrealized</th><th>toTP1</th><th>toTP2</th><th>toSL</th><th>Opened</th><th>Source</th>
          </tr>
        </thead>
        <tbody></tbody>
      </table>
    </div>
    <div class="card" style="flex:1; min-width:360px">
      <h3>Recent Trades</h3>
      <table id="trades">
        <thead><tr><th>Time</th><th>Action</th><th>Symbol</th><th>Qty</th><th>Avg Price</th><th>Quote</th><th>Reason</th></tr></thead>
        <tbody></tbody>
      </table>
    </div>
  </div>

  <div class="row" style="margin-top:16px">
    <div class="card" style="flex:1; min-width:360px">
      <h3>.env Editor</h3>
      <div id="envGrid" class="grid"></div>
      <div style="margin-top:8px">
        <button onclick="saveEnv()">Save .env</button>
        <button onclick="applyReload()">Save + Apply</button>
        <button onclick="reloadOnly()">Reload</button>
      </div>
    </div>
  </div>

<script>
async function api(path, opts){
  const r = await fetch(path, opts||{});
  if(!r.ok){ throw new Error(await r.text()); }
  return await r.json();
}

async function refresh(){
  try{
    const s = await api('/status');
    document.getElementById('env').textContent = s.env;
    document.getElementById('quote').textContent = s.quote;
    document.getElementById('running').textContent = s.running ? 'YES' : 'NO';

    // amounts
    document.getElementById('quote_free').textContent = (+s.quote_free).toFixed(6) + ' ' + s.quote;
    document.getElementById('quote_locked').textContent = (+s.quote_locked).toFixed(6) + ' ' + s.quote;
    document.getElementById('equity_total').textContent = (+s.equity_total).toFixed(2) + ' ' + s.quote;
    document.getElementById('pos_val').textContent = (+s.positions_value).toFixed(2) + ' ' + s.quote;

    // day Δ
    const dp = (+s.day_change_pct || 0) * 100;
    const el = document.getElementById('day_change_pct');
    el.textContent = (dp>=0? '+' : '') + dp.toFixed(2) + '%';
    el.style.color = dp>0 ? '#16a34a' : (dp<0 ? '#dc2626' : '');

    document.getElementById('warn').textContent = s.warn || '';

    // positions (snapshots)
    const tbody = document.querySelector('#pos tbody'); tbody.innerHTML = '';
    const snaps = s.pos_snapshots || [];
    for (const p of snaps){
      const tr = document.createElement('tr');
      const cls = (p.pnl_pct||0) >= 0 ? 'posok' : 'posbad';
      function fmt(x, d=6){ return (x===null || x===undefined) ? '-' : (+x).toFixed(d); }
      tr.innerHTML = `
        <td>${p.symbol}</td>
        <td>${fmt(p.qty, 6)}</td>
        <td>${fmt(p.price, 6)}</td>
        <td>${fmt(p.entry, 6)}</td>
        <td class="${cls}">${fmt(p.pnl_pct, 2)}%</td>
        <td>${fmt(p.unrealized, 2)} ${s.quote}</td>
        <td>${p.toTP1==null?'-':fmt(p.toTP1,2)+'%'}</td>
        <td>${p.toTP2==null?'-':fmt(p.toTP2,2)+'%'}</td>
        <td>${p.toSL==null?'-':fmt(p.toSL,2)+'%'}</td>
        <td>${p.opened_at||''}</td>
        <td>${p.source||''}</td>`;
      tbody.appendChild(tr);
    }

    // trades
    const trades = (s.trades||[]).slice().reverse();
    const ttb = document.querySelector('#trades tbody'); ttb.innerHTML = '';
    for (const t of trades){
      const tr = document.createElement('tr');
      tr.innerHTML = `<td>${t.ts||''}</td><td>${t.side||''}</td><td>${t.symbol||''}</td>
        <td>${(+t.qty||0).toFixed(8)}</td><td>${t.avg?(+t.avg).toFixed(8):''}</td>
        <td>${t.quote?(+t.quote).toFixed(8):''}</td><td>${t.reason||''}</td>`;
      ttb.appendChild(tr);
    }
  }catch(err){
    alert('Status error: '+err.message);
  }
}

async function loadEnv(){
  const env = await api('/env');
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
  const j = await api('/env', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({apply:false, updates})});
  alert('Saved: '+j.changed+' keys');
}

async function applyReload(){
  const updates = {};
  document.querySelectorAll('#envGrid input').forEach(inp=>{
    const k = inp.id.replace('env_',''); updates[k] = inp.value;
  });
  const j = await api('/env', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({apply:true, updates})});
  alert('Saved & applied: '+j.changed+' keys');
}

async function reloadOnly(){
  const j = await api('/reload', {method:'POST'});
  alert(j.message||'Reload signal sent');
}

async function doStart(){ await api('/start', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({confirm:true})}); refresh(); }
async function doStop(){ await api('/stop',  {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({confirm:true})}); refresh(); }
async function doPanic(){ if(confirm('Sell ALL assets to QUOTE?')){ const r = await api('/panic', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({confirm:true})}); alert('Done. Check logs.'); refresh(); } }

refresh(); loadEnv();
setInterval(refresh, 5000);
</script>
</body>
</html>
"""

# Write index.html
INDEX_HTML.write_text(UI_HTML, encoding="utf-8")

# Mount static (serves index.html at /)
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
