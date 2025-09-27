# README (TL;DR)
# ----------------
# בוט ספוט לבינאנס שמנסה להשיג רווח של 5% לכל פוזיציה, מוכר אוטומטית בהגעה ל-5% או בסוף יום (23:59 Asia/Jerusalem) את כל ההחזקות ל-QUOTE (ברירת מחדל USDT).
# כולל סריקת שוק (נזילות/ספרד/מומנטום), קנייה ב-MARKET לפי סכום בדולרים, וניטור מחירים למכירה.
#
# ▶️ הוראות הרצה קצרות:
# 1) pip install -r requirements.txt
# 2) העתק את .env.example ל-.env ועדכן מפתחות ופרמטרים (כולל QUOTE_ASSET=USDT או BTC),
# 3) python bot_5pct.py
#
# הערות:
# - הבוט משתמש ב-quoteOrderQty לקנייה (נוח, לא צריך לעגל כמות), ובמכירה מעגל כמות ל-stepSize.
# - הבוט בוחר מועמדים מתוך צמדי QUOTE (למשל כל מה שנגמר ב-USDT), מסנן נזילות/ספרד, ובודק מגמת 20 נרות 1ד׳.
# - בסוף יום (23:59 Asia/Jerusalem) מוכר את כל ההחזקות ל-QUOTE.
# - אין כאן ייעוץ השקעות; קוד לדוגמה בלבד. בדוק תחילה על TESTNET.

# --------------
# requirements.txt
# --------------
# fastapi==0.115.0  # (לא חובה כאן, נשאר מהסקלטון קודם)
# python-dotenv==1.0.1
# binance-connector==3.6.0
# pyyaml==6.0.2

# --------------
# .env.example
# --------------
# Binance
# BINANCE_API_KEY=__PUT_YOUR_KEY__
# BINANCE_API_SECRET=__PUT_YOUR_SECRET__
# BINANCE_ENV=prod            # prod | testnet

# Quote & scanning
# QUOTE_ASSET=USDT            # או BTC אם מסחר מול BTC
# BASE_BLACKLIST=UP,DOWN,BEAR,BULL,VENFT,BRL
# MIN_24H_QUOTE_VOL=10000000  # 10M
# MAX_SPREAD_BPS=15           # 0.15%
# SCAN_INTERVAL_SEC=30
# EOD_HHMM=23:59
# LOG_LEVEL=INFO

# Positioning / per-trade targets
# TARGET_TP_PCT=0.05          # 5% לכל פוזיציה
# OPTIONAL_SL_PCT=0.00        # 0 -> ללא SL אוטומטי
# TRAILING_STOP_PCT=0.02      # חדש: טריילינג סטופ (לדוגמה 2% מהשיא)
# TRAILING_ARM_PNL_PCT=0.02   # מפעיל טריילינג כאשר הפוזיציה ברווח >= 2%
# MAX_OPEN_POSITIONS=3        # בשגרה; יותאם אוטומטית לפי Auto-Split/All-Balance
# ALLOCATION_USD=20           # בשגרה: כמה לקנות לעסקה

# Use full balance mode (אופציונלי) (אופציונלי)
# USE_ALL_BALANCE=false       # אם true: עסקה אחת בכל היתרה
# BALANCE_RESERVE_PCT=0.02    # שמירת 2% רזרבה
# CONSOLIDATE_AT_START=false  # למכור הכל ל-QUOTE בתחילת היום

# NEW: Auto-Split (חלוקה אוטומטית של היתרה לכמה עסקאות במקביל)
# AUTO_SPLIT=true
# SPLIT_SLICES=3              # חלוקה ל-3 סלייסים (נכסים שונים)
# MIN_SLICE_USD=20            # מינימום לסלייס (כדי לעבור minNotional)

# Portfolio targets (יעד/סטופ ברמת התיק כולו)
# PORTFOLIO_TARGET_PCT=0.05   # כשהתיק כולו מגיע +5% → מכירת הכל ל-QUOTE
# PORTFOLIO_STOP_PCT=0.02     # אופציונלי: אם יורד -2% → מכירת הכל ועצירה

# Manage existing balances (אופציונלי)
# MANAGE_EXISTING_ASSETS=BTTC
# USE_COST_BASIS_FROM_TRADES=true
# COST_BASIS_LOOKBACK_DAYS=60
# USE_COST_BASIS_FROM_TRADES=true
# COST_BASIS_LOOKBACK_DAYS=60

# --------------
# bot_5pct.py
# --------------
# -------------- bot_5pct.py --------------
import os, sys, json, math, time, logging
from pathlib import Path
from typing import Dict, List, Tuple, Any
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from binance.spot import Spot as SpotClient
from binance.error import ClientError
from decimal import Decimal, ROUND_DOWN, getcontext
getcontext().prec = 28  # די והותר לקריפטו

def _dec(x) -> Decimal:
    return x if isinstance(x, Decimal) else Decimal(str(x))

def _step_to_quant(step_str: str) -> Decimal:
    # קולט מחרוזת stepSize כמו "0.0001" ומחזיר Decimal("0.0001")
    return Decimal(step_str)

def _quantize_to_step(amount, step_str: str) -> Decimal:
    step = _step_to_quant(step_str)
    # כימות כלפי מטה בדיוק של גודל ה־step
    return (_dec(amount) // step) * step

def _to_str(d: Decimal) -> str:
    # מחרוזת בלי scientific, בלי גרר מיותר
    s = format(d, 'f')
    # הסרת אפסים מיותרים מימין, אבל להשאיר "0" אם הכל נעלם
    if '.' in s:
        s = s.rstrip('0').rstrip('.')
    return s if s else '0'

# ---------- Paths / Const ----------
APP_DIR     = Path(__file__).parent
STATE_FILE  = APP_DIR / "positions.json"
TRADES_FILE = APP_DIR / "trades.json"
STOP_FLAG   = APP_DIR / "stop.flag"

TESTNET_URL = "https://testnet.binance.vision"
PROD_URL    = "https://api.binance.com"

IL_TZ = ZoneInfo("Asia/Jerusalem")
STABLES = {"USDT","USDC","FDUSD","TUSD","BUSD","DAI","USDP","EURT","USTC"}

# ---------- Small utils ----------
def now_local() -> datetime:
    return datetime.now(IL_TZ)

def iso_utc() -> str:
    return datetime.now(timezone.utc).isoformat()

def read_json(path: Path, default: Any):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except Exception:
        return default

def write_json(path: Path, obj: Any):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2))

def log_trade(side: str, symbol: str, qty: float, avg: float | None = None, quote: float | None = None, reason: str = ""):
    data = read_json(TRADES_FILE, {"trades": []})
    data["trades"].append({
        "ts": iso_utc(),
        "side": side, "symbol": symbol,
        "qty": qty, "avg": avg, "quote": quote, "reason": reason
    })
    write_json(TRADES_FILE, data)

# ---------- Env / Config ----------
def load_env() -> Dict[str, Any]:
    load_dotenv(APP_DIR / ".env")

    api_key    = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")
    env        = (os.getenv("BINANCE_ENV") or "prod").lower()
    base_url   = PROD_URL if env == "prod" else TESTNET_URL
    
    if not api_key or not api_secret:
        raise RuntimeError("Missing API keys in .env")

    # logging
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=getattr(logging, log_level, logging.INFO),
                        format="%(asctime)s %(levelname)s %(message)s")

    # core
    quote_asset        = os.getenv("QUOTE_ASSET", "USDT").upper()
    blacklist_keywords = [s.strip().upper() for s in os.getenv("BASE_BLACKLIST", "UP,DOWN,BEAR,BULL,VENFT").split(",") if s.strip()]

    # trade sizing / exits
    target_tp_pct     = float(os.getenv("TARGET_TP_PCT", "0.05"))         # 5%
    optional_sl_pct   = float(os.getenv("OPTIONAL_SL_PCT", "0.0"))        # off by default
    trailing_stop_pct = float(os.getenv("TRAILING_STOP_PCT", "0.0"))      # e.g. 0.02
    trailing_arm_pct  = float(os.getenv("TRAILING_ARM_PNL_PCT", "0.02"))  # e.g. arm at +2%

    # portfolio guards
    portfolio_target_pct = float(os.getenv("PORTFOLIO_TARGET_PCT", "0.0"))
    portfolio_stop_pct   = float(os.getenv("PORTFOLIO_STOP_PCT", "0.0"))

    # positions / allocation
    max_positions_cfg = int(os.getenv("MAX_OPEN_POSITIONS", "3"))
    alloc_quote       = float(os.getenv("ALLOCATION_USD", "20"))

    # scanning
    min_quote_vol   = float(os.getenv("MIN_24H_QUOTE_VOL", "10000000"))
    max_spread_bps  = float(os.getenv("MAX_SPREAD_BPS", "15"))
    scan_interval   = int(os.getenv("SCAN_INTERVAL_SEC", "30"))
    eod_hhmm        = os.getenv("EOD_HHMM", "23:59")

    # all-balance / split
    use_all_balance     = os.getenv("USE_ALL_BALANCE", "false").lower() == "true"
    balance_reserve_pct = float(os.getenv("BALANCE_RESERVE_PCT", "0.02"))
    consolidate_at_start= os.getenv("CONSOLIDATE_AT_START", "false").lower() == "true"

    auto_split     = os.getenv("AUTO_SPLIT", "false").lower() == "true"
    split_slices   = int(os.getenv("SPLIT_SLICES", "3"))
    min_slice_usd  = float(os.getenv("MIN_SLICE_USD", "20"))

    # manage existing balances (e.g., BTTC)
    manage_assets_csv = (os.getenv("MANAGE_EXISTING_ASSETS", "") or "").upper().strip()
    manage_assets     = [a for a in (s.strip() for s in manage_assets_csv.split(",")) if a]
    use_cost_basis    = os.getenv("USE_COST_BASIS_FROM_TRADES", "true").lower() == "true"
    cost_basis_lookback_days = int(os.getenv("COST_BASIS_LOOKBACK_DAYS", "30"))

    # enforce
    if use_all_balance:
        auto_split = False
        max_positions = 1
    else:
        max_positions = max_positions_cfg

        return {
        "api_key": api_key, "api_secret": api_secret, "env": env, "base_url": base_url,
        "quote_asset": quote_asset, "blacklist_keywords": blacklist_keywords,
        "alloc_quote": alloc_quote, "max_positions": max_positions,
        "target_tp_pct": target_tp_pct, "optional_sl_pct": optional_sl_pct,
        "trailing_stop_pct": trailing_stop_pct, "trailing_arm_pct": trailing_arm_pct,
        "min_24h_quote_vol": min_quote_vol, "max_spread_bps": max_spread_bps,
        "scan_interval": scan_interval, "eod_hhmm": eod_hhmm,
        "use_all_balance": use_all_balance, "balance_reserve_pct": balance_reserve_pct,
        "consolidate_at_start": consolidate_at_start,
        "auto_split": auto_split, "split_slices": split_slices, "min_slice_usd": min_slice_usd,
        "portfolio_target_pct": portfolio_target_pct, "portfolio_stop_pct": portfolio_stop_pct,
        "manage_assets": manage_assets, "use_cost_basis": use_cost_basis,
        "cost_basis_lookback_days": cost_basis_lookback_days,

        # --- חדש לרוטציה ---
        "rotation_enabled": os.getenv("ROTATION_ENABLED", "false").lower() == "true",
        "rotation_edge_pct": float(os.getenv("ROTATION_EDGE_PCT", "0.20")),
        "fee_bps": float(os.getenv("FEE_BPS", "10")),            # עמלה בסיסית 0.1% (10bps)
        "slippage_bps": float(os.getenv("SLIPPAGE_BPS", "5")),   # סליפג’ מוערך 0.05%
                "rotation_enabled": os.getenv("ROTATION_ENABLED", "false").lower() == "true",
        "rotation_edge_pct": float(os.getenv("ROTATION_EDGE_PCT", "0.20")),
        "fee_bps": float(os.getenv("FEE_BPS", "10")),
        "slippage_bps": float(os.getenv("SLIPPAGE_BPS", "5")),

        # --- Regime / only-up filters ---
        "regime_enabled": os.getenv("REGIME_CHECK_ENABLED", "false").lower() == "true",
        "regime_symbols": [s.strip().upper() for s in os.getenv("REGIME_SYMBOLS","BTCUSDT,ETHUSDT").split(",") if s.strip()],
        "regime_tf": os.getenv("REGIME_TF","15m"),
        "regime_ema_short": int(os.getenv("REGIME_EMA_SHORT","20")),
        "regime_ema_long":  int(os.getenv("REGIME_EMA_LONG","50")),
        "regime_min_agree": int(os.getenv("REGIME_MIN_AGREE","1")),
        "regime_cooldown_min": int(os.getenv("REGIME_COOLDOWN_MIN","90")),

        "buy_tf": os.getenv("BUY_TF","5m"),
        "buy_ema_short": int(os.getenv("BUY_EMA_SHORT","20")),
        "buy_ema_long":  int(os.getenv("BUY_EMA_LONG","50")),
        "buy_min_slope_pct_per_bar": float(os.getenv("BUY_MIN_SLOPE_PCT_PER_BAR","0.0002")),

    }
def reconcile_positions(client: SpotClient, state: Dict, cfg):
    """מיישרת את positions.json למציאות:
    - אם אין יתרה ב-BASE → מוחק פוזיציה.
    - מצמצם qty ליתרה החופשית המעוגלת ל-stepSize.
    - אם לאחר עיגול השווי < minNotional → מוחק ומסמן דאסט בלוג.
    """
    changed = False
    for sym, pos in list(state["positions"].items()):
        f = symbol_filters(client, sym)
        base = f["raw"]["baseAsset"]
        step = f["stepSize"]
        min_qty = f["minQty"]
        min_notional = f["minNotional"]
        free_base, _ = get_account_bal(client, base)

        q = round_down(free_base, step)
        if q <= 0 or (min_qty and q < min_qty):
            log_trade("SELL", sym, 0.0, avg=None, quote=0.0, reason="RECONCILE_DUST_DROP")
            state["positions"].pop(sym, None)
            changed = True
            continue

        bid, ask = book(client, sym)
        mid = (bid + ask)/2.0 if bid and ask else ask or bid or 0.0
        if min_notional and mid*q < min_notional:
            log_trade("SELL", sym, 0.0, avg=None, quote=0.0, reason="RECONCILE_MIN_NOTIONAL_DROP")
            state["positions"].pop(sym, None)
            changed = True
            continue

        if abs(q - float(pos["qty"])) > 0:  # עדכן לכמות האמיתית
            pos["qty"] = q
            changed = True

    if changed:
        save_positions(state)


def client_from_cfg(cfg) -> SpotClient:
    return SpotClient(api_key=cfg["api_key"], api_secret=cfg["api_secret"], base_url=cfg["base_url"])

# ---------- State ----------
def load_positions() -> Dict:
    return read_json(STATE_FILE, {"positions": {}})

def save_positions(state: Dict):
    write_json(STATE_FILE, state)

# ---------- Time helpers ----------
def is_eod(cfg) -> bool:
    hh, mm = map(int, cfg["eod_hhmm"].split(":"))
    n = now_local()
    return (n.hour, n.minute) >= (hh, mm)

# ---------- Exchange helpers ----------
def get_account_bal(client: SpotClient, asset: str) -> Tuple[float, float]:
    acct = client.account(recvWindow=5000)
    for b in acct["balances"]:
        if b["asset"] == asset:
            return float(b["free"]), float(b["locked"])
    return 0.0, 0.0

def spot_symbols_for_quote(client: SpotClient, quote: str, blacklist_keywords: List[str]) -> List[str]:
    info = client.exchange_info()
    syms = []
    for s in info["symbols"]:
        if s.get("status") != "TRADING":
            continue
        if s.get("isSpotTradingAllowed") is False:
            continue
        sym = s["symbol"]
        if not sym.endswith(quote):
            continue
        base = s["baseAsset"].upper()
        if any(k in base for k in blacklist_keywords):
            continue
        if base in STABLES:  # לא סוחרים סטייבל מול USDT
            continue

        syms.append(sym)
    return syms

def symbol_filters(client: SpotClient, symbol: str):
    s = client.exchange_info(symbol=symbol)["symbols"][0]
    lot   = next(f for f in s["filters"] if f["filterType"] == "LOT_SIZE")
    price = next(f for f in s["filters"] if f["filterType"] == "PRICE_FILTER")
    notional = next((f for f in s["filters"] if f["filterType"] in ("NOTIONAL","MIN_NOTIONAL")), None)
    return {
        "stepSize": float(lot["stepSize"]),
        "minQty":   float(lot.get("minQty", 0.0)),
        "tickSize": float(price["tickSize"]),
        "minNotional": float(notional["minNotional"]) if notional else None,
        "raw": s,
    }

def round_down(qty: float, step: float) -> float:
    if step <= 0: return qty
    return math.floor(qty / step) * step

def book(client: SpotClient, symbol: str) -> Tuple[float, float]:
    bk = client.book_ticker(symbol=symbol)
    return float(bk["bidPrice"]), float(bk["askPrice"])

def get_klines_closes(client: SpotClient, symbol: str, interval: str, limit: int) -> List[float]:
    kl = client.klines(symbol=symbol, interval=interval, limit=limit)
    return [float(k[4]) for k in kl] if kl else []

def ema(series: List[float], period: int) -> List[float]:
    if not series or period <= 1 or len(series) < period:
        return []
    k = 2 / (period + 1)
    out = [sum(series[:period]) / period]  # seed with SMA
    for x in series[period:]:
        out.append(out[-1] + k * (x - out[-1]))
    return out

def pct_slope_per_bar(series: List[float]) -> float:
    """שיפוע לוגריתמי בקירוב: (last/first)-1 חלקי #ברים; אם חסר – 0"""
    if not series or len(series) < 2:
        return 0.0
    first, last = series[0], series[-1]
    if first <= 0: 
        return 0.0
    return (last / first - 1.0) / (len(series)-1)

def kline_momentum(client: SpotClient, symbol: str, limit: int = 20) -> float:
    kl = client.klines(symbol=symbol, interval="1m", limit=limit)
    if not kl or len(kl) < 2:
        return 0.0
    first_open = float(kl[0][1])
    last_close = float(kl[-1][4])
    return (last_close / first_open) - 1.0

# ---------- Strategy ----------
def scan_candidates(client: SpotClient, cfg) -> List[str]:
    all_syms = spot_symbols_for_quote(client, cfg["quote_asset"], cfg["blacklist_keywords"])
    tickers = client.ticker_24hr()
    pool: list[tuple[str, float, float]] = []
    for t in tickers:
        sym = t.get("symbol")
        if not sym or not sym.endswith(cfg["quote_asset"]):
            continue
        if sym not in all_syms:
            continue
        qv = float(t.get("quoteVolume", 0.0))
        if qv < cfg["min_24h_quote_vol"]:
            continue
        bid, ask = book(client, sym)
        if ask <= 0:
            continue
        spread_bps = (ask - bid) / ask * 10_000
        if spread_bps > cfg["max_spread_bps"]:
            continue
        pool.append((sym, qv, spread_bps))
    pool = sorted(pool, key=lambda x: -x[1])[:25]
    ranked: list[tuple[str, float, float, float]] = []
    for sym, qv, sp in pool:
        mom = kline_momentum(client, sym, limit=20)
        ranked.append((sym, mom, qv, sp))
    ranked.sort(key=lambda x: (-x[1], -x[2]))
    cands = [sym for sym, mom, _, _ in ranked if mom > 0][:10]
    logging.info(f"scan: chosen candidates {cands}")
    return cands

def desired_new_positions(state: Dict, candidates: List[str], cfg) -> List[str]:
    open_syms = set(state["positions"].keys())
    room = max(cfg["max_positions"] - len(open_syms), 0)
    return [s for s in candidates if s not in open_syms][:room]

# ---------- Trade ops ----------
def _round_down_dec(amount: float, decimals: int) -> float:
    if decimals < 0:
        return amount
    factor = 10 ** decimals
    return math.floor(amount * factor) / factor
def _get_symbol_filters_dict(client: SpotClient, symbol: str) -> dict:
    """מחזיר מיפוי נוח של הפילטרים לפי filterType"""
    info = client.exchange_info(symbol=symbol)
    s = info["symbols"][0]
    filt_map = {f["filterType"]: f for f in s.get("filters", [])}
    return {
        "base": s["baseAsset"],
        "quote": s["quoteAsset"],
        "raw": s,
        "lot": filt_map.get("LOT_SIZE", {}),
        "price": filt_map.get("PRICE_FILTER", {}),
        "notional": filt_map.get("NOTIONAL") or filt_map.get("MIN_NOTIONAL") or {},
        # חלק מהשמות שונים בזמנים שונים; מכוסה בשורה למעלה
    }

def _lot_step_str(client: SpotClient, symbol: str) -> str:
    """מחזיר stepSize כמחרוזת מדויקת דרך LOT_SIZE; זורק שגיאה אם חסר"""
    f = _get_symbol_filters_dict(client, symbol)
    step_str = f["lot"].get("stepSize")
    if not step_str:
        raise KeyError(f"{symbol} LOT_SIZE.stepSize missing")
    return step_str

def _min_qty_dec(client: SpotClient, symbol: str) -> Decimal:
    f = _get_symbol_filters_dict(client, symbol)
    v = f["lot"].get("minQty", "0")
    return _dec(v)

def _min_notional_dec(client: SpotClient, symbol: str) -> Decimal:
    f = _get_symbol_filters_dict(client, symbol)
    v = f["notional"].get("minNotional", "0")
    return _dec(v)

def buy_market_quote(client: SpotClient, symbol: str, quote_amount: float):
    meta = _get_symbol_filters_dict(client, symbol)
    sraw = meta["raw"]
    quote_asset = meta["quote"]
    quote_prec = int(sraw.get("quotePrecision", sraw.get("quoteAssetPrecision", 8)))
    min_notional = _min_notional_dec(client, symbol)

    free_q, _ = get_account_bal(client, quote_asset)
    amt = min(quote_amount, free_q)

    if min_notional > 0 and _dec(amt) < min_notional:
        raise ValueError(f"amount {amt} below minNotional {min_notional} for {symbol}")

    qd = _dec(amt).quantize(Decimal(1).scaleb(-quote_prec), rounding=ROUND_DOWN)
    if qd <= 0:
        raise ValueError("quote amount after precision rounding is 0")

    attempts, last_exc = 0, None
    while attempts < 3:
        try:
            ord = client.new_order(
                symbol=symbol, side="BUY", type="MARKET",
                quoteOrderQty=_to_str(qd), recvWindow=5000
            )
            executed_qty = float(ord.get("executedQty", 0.0))
            cumm_quote   = float(ord.get("cummulativeQuoteQty", 0.0))
            avg_price    = (cumm_quote / executed_qty) if executed_qty > 0 else None
            return executed_qty, avg_price
        except Exception as e:
            last_exc = e
            qd = (qd * Decimal("0.98")).quantize(Decimal(1).scaleb(-quote_prec), rounding=ROUND_DOWN)
            attempts += 1
            time.sleep(0.2)
    if last_exc:
        raise last_exc
    raise RuntimeError("buy_market_quote failed without captured exception")


def sell_market_all(client: SpotClient, symbol: str, desired_qty: float):
    # --- שליפת מטה־דאטה בטוחה ---
    meta = _get_symbol_filters_dict(client, symbol)
    base = meta["base"]
    step_str = meta["lot"].get("stepSize")
    if not step_str:
        raise KeyError(f"{symbol} LOT_SIZE.stepSize missing")
    min_qty = _dec(meta["lot"].get("minQty", "0"))
    min_notional = _dec(meta["notional"].get("minNotional", "0"))

    # --- יתרה בבסיס וכימות לכמות חוקית ---
    free_base, _ = get_account_bal(client, base)
    q_dec = _quantize_to_step(min(desired_qty, free_base), step_str)
    if q_dec <= 0 or (min_qty > 0 and q_dec < min_qty):
        raise ValueError("dust")

    # --- בדיקת NOTIONAL (אם רלוונטי) ---
    bid, ask = book(client, symbol)
    mid = (bid + ask)/2.0 if bid and ask else ask or bid or 0.0
    if min_notional > 0 and mid > 0:
        if q_dec * _dec(mid) < min_notional:
            need = (min_notional / _dec(mid))
            q_dec = _quantize_to_step(min(_dec(free_base), q_dec.max(need)), step_str)
            if q_dec <= 0 or (min_qty > 0 and q_dec < min_qty) or (q_dec*_dec(mid) < min_notional):
                raise ValueError("dust")

    # --- שליחה עם רידוּיים חכמים ---
    attempts, last_exc = 0, None
    while attempts < 5:
        try:
            qty_str = _to_str(q_dec)
            ord = client.new_order(symbol=symbol, side="SELL", type="MARKET",
                                   quantity=qty_str, recvWindow=5000)
            cumm_quote   = float(ord.get("cummulativeQuoteQty", 0.0))
            executed_qty = float(ord.get("executedQty", 0.0))
            avg_price    = (cumm_quote / executed_qty) if executed_qty > 0 else None
            return executed_qty, avg_price, cumm_quote

        except Exception as e:
            last_exc = e
            msg = str(e)
            free_base, _ = get_account_bal(client, base)

            if "-2010" in msg:  # insufficient balance
                q_dec = _quantize_to_step(free_base, step_str)
            elif "-1111" in msg:  # precision
                q_dec = _quantize_to_step(q_dec * Decimal("0.98"), step_str)
            elif "-1013" in msg:  # NOTIONAL
                if mid > 0 and min_notional > 0:
                    need = (min_notional / _dec(mid))
                    q_dec = _quantize_to_step(min(_dec(free_base), (q_dec*Decimal("1.01")).max(need)), step_str)
                else:
                    q_dec = _quantize_to_step(q_dec * Decimal("0.98"), step_str)
            else:
                q_dec = _quantize_to_step(min(_dec(free_base), q_dec * Decimal("0.98")), step_str)

            attempts += 1
            if q_dec <= 0 or (min_qty > 0 and q_dec < min_qty) or (mid > 0 and min_notional > 0 and q_dec*_dec(mid) < min_notional):
                return 0.0, None, 0.0
            time.sleep(0.15)

    if last_exc:
        raise RuntimeError(f"sell_market_all exhausted retries for {symbol}: {last_exc}") from last_exc
    raise RuntimeError(f"sell_market_all exhausted retries for {symbol} (no exception captured)")



# ---------- Open positions ----------
def maybe_open_positions(client: SpotClient, state: Dict, cfg):
    """
    פותחת פוזיציות חדשות בהתאם למצב (.env) ואם אין מקום – מפעילה רוטציה חכמה:
    מחליפה את הפוזיציה ה"חלשה" במועמד "חזק" רק אם היתרון ≥ ROTATION_EDGE_PCT
    וגם מכסה את עלות round-trip (עמלות+סליפג').
    """
    # --- סריקה ומועמדים ---
    cands = scan_candidates(client, cfg)
    to_open = desired_new_positions(state, cands, cfg)
    if not to_open and not cfg.get("rotation_enabled", False):
        return

    # ============================
    # Mode 1: ALL-BALANCE
    # ============================
    if to_open and cfg["use_all_balance"]:
        best = to_open[0]
        free_q, _ = get_account_bal(client, cfg["quote_asset"])
        alloc = max(free_q * (1.0 - cfg["balance_reserve_pct"]), 0.0)
        logging.info(f"ALL-BALANCE MODE | free={free_q} alloc={alloc}")
        if alloc > 0:
            try:
                executed_qty, entry = buy_market_quote(client, best, alloc)
                if executed_qty > 0 and entry:
                    state["positions"][best] = {
                        "qty": executed_qty, "entry": entry, "opened_at": iso_utc(), "source": "all_balance"
                    }
                    save_positions(state)
                    log_trade("BUY", best, executed_qty, avg=entry, quote=None, reason="all_balance")
                    logging.info(f"OPEN {best} qty={executed_qty} entry≈{entry}")
            except Exception as e:
                logging.exception(f"buy failed for {best}: {e}")
        return  # במצב זה תמיד עסקה אחת

    # ============================
    # Mode 2: AUTO_SPLIT
    # ============================
    if to_open and cfg["auto_split"]:
        free_q, _ = get_account_bal(client, cfg["quote_asset"])
        budget = max(free_q * (1.0 - cfg["balance_reserve_pct"]), 0.0)
        slots_free = max(cfg["max_positions"] - len(state["positions"]), 0)
        slices = min(cfg["split_slices"], slots_free, len(to_open))
        if budget <= 0 or slices <= 0:
            # אין תקציב/מקום – נפל לרוטציה בהמשך (אם מופעלת)
            pass
        else:
            per_slice = budget / slices if slices > 0 else 0.0
            # ודא מינימום לסלייס
            while slices > 0 and per_slice < cfg["min_slice_usd"]:
                slices -= 1
                per_slice = budget / slices if slices > 0 else 0.0
            if slices > 0 and per_slice >= cfg["min_slice_usd"]:
                chosen = to_open[:slices]
                logging.info(f"AUTO-SPLIT MODE | budget={budget} slices={slices} per_slice={per_slice}")
                for sym in chosen:
                    try:
                        executed_qty, entry = buy_market_quote(client, sym, per_slice)
                        if executed_qty > 0 and entry:
                            state["positions"][sym] = {
                                "qty": executed_qty, "entry": entry, "opened_at": iso_utc(), "source": "auto_split"
                            }
                            save_positions(state)
                            log_trade("BUY", sym, executed_qty, avg=entry, quote=None, reason="auto_split")
                            logging.info(f"OPEN {sym} qty={executed_qty} entry≈{entry}")
                            time.sleep(0.25)
                    except Exception as e:
                        logging.exception(f"buy failed for {sym}: {e}")
                return
            # אם לא עמדנו במינימום – ננסה המשך/רוטציה
        # אין return כאן כדי לתת הזדמנות לרוטציה אם אין מקום/תקציב

    # ============================
    # Mode 3: הקצאה קבועה לכל עסקה
    # ============================
    if to_open and not cfg["auto_split"] and not cfg["use_all_balance"]:
        free_q, _ = get_account_bal(client, cfg["quote_asset"])
        if free_q >= cfg["alloc_quote"]:
            slots_free = max(cfg["max_positions"] - len(state["positions"]), 0)
            # נפתח עד slots_free עסקאות, כל אחת בסכום קבוע
            for sym in to_open[:slots_free]:
                try:
                    executed_qty, entry = buy_market_quote(client, sym, cfg["alloc_quote"])
                    if executed_qty > 0 and entry:
                        state["positions"][sym] = {
                            "qty": executed_qty, "entry": entry, "opened_at": iso_utc(), "source": "fixed_alloc"
                        }
                        save_positions(state)
                        log_trade("BUY", sym, executed_qty, avg=entry, quote=None, reason="fixed_alloc")
                        logging.info(f"OPEN {sym} qty={executed_qty} entry≈{entry}")
                except Exception as e:
                    logging.exception(f"buy failed for {sym}: {e}")
                time.sleep(0.3)
            # גם אם פתחנו חלק – חוזרים. אם לא נפתח כלום בגלל מגבלות minNotional, ייתכן שרוטציה עדיין רלוונטית
        # אם אין מספיק יתרה להקצאה – נמשיך לרוטציה (אם מופעלת)

    # ============================
    # Rotation (אם אין מקום לפתוח חדש)
    # ============================
    if cfg.get("rotation_enabled", False):
        open_syms = list(state["positions"].keys())
        room = max(cfg["max_positions"] - len(open_syms), 0)
        if room <= 0 and cands:
            # מועמדים שאינם פתוחים כרגע
            fresh = [s for s in cands if s not in open_syms]
            if not fresh:
                return
            # בחר הטוב ביותר לפי ציון
            try:
                best_new = max(fresh, key=lambda s: score_symbol(client, s))
                score_new = score_symbol(client, best_new)
            except Exception:
                return

            # מצא הפוזיציה ה"חלשה" ביותר
            try:
                worst_open = min(open_syms, key=lambda s: score_symbol(client, s))
                score_old  = score_symbol(client, worst_open)
            except Exception:
                return

            # יתרון נדרש + עלות round-trip
            edge_req  = cfg.get("rotation_edge_pct", 0.20)
            costs_pct = est_roundtrip_cost_pct(cfg)  # מבוסס FEE_BPS ו-SLIPPAGE_BPS
            cond_edge = (score_new >= score_old * (1.0 + edge_req))
            cond_cost = (score_new - score_old) >= (abs(score_old) * costs_pct * 1.5)

            if score_new > 0 and cond_edge and cond_cost:
                logging.info(
                    f"ROTATION: replacing {worst_open} (score={score_old:.5f}) → "
                    f"{best_new} (score={score_new:.5f}); edge_req={edge_req*100:.1f}%, costs≈{costs_pct*100:.2f}%"
                )
                try:
                    # מכור את ה"חלשה"
                    qty_old = float(state["positions"][worst_open]["qty"])
                    ex_qty, avg, quote_recv = sell_market_all(client, worst_open, qty_old)
                    log_trade("SELL", worst_open, ex_qty, avg=avg, quote=quote_recv, reason="ROTATE_OUT")
                    state["positions"].pop(worst_open, None)
                    save_positions(state)

                    # קנה את החדשה – העדפה להשתמש ב-quote שהתקבל
                    free_q, _ = get_account_bal(client, cfg["quote_asset"])
                    buy_amt = quote_recv if quote_recv and quote_recv > 0 else free_q
                    if buy_amt <= 0:
                        logging.info("ROTATION: no quote available to re-enter; skipping buy")
                        return
                    ex_qty_new, entry_new = buy_market_quote(client, best_new, buy_amt)
                    if ex_qty_new > 0 and entry_new:
                        state["positions"][best_new] = {
                            "qty": ex_qty_new, "entry": entry_new, "opened_at": iso_utc(), "source": "rotation"
                        }
                        save_positions(state)
                        log_trade("BUY", best_new, ex_qty_new, avg=entry_new, quote=None, reason="rotation")
                        logging.info(f"ROTATION OPEN {best_new} qty={ex_qty_new} entry≈{entry_new}")
                except Exception as e:
                    logging.exception(f"rotation failed: {e}")


# ---------- Monitor & exits ----------
def monitor_and_close(client: SpotClient, state: Dict, cfg):
    to_delete = []
    for sym, pos in list(state["positions"].items()):
        try:
            # מחירים – השתמש ב-BID כדי למדוד PnL שמרני (זה המחיר שבו נוכל למכור)
            bid, ask = book(client, sym)
            last_sellable = bid if bid else ((bid or 0) + (ask or 0)) / 2.0
            entry = float(pos["entry"])
            if entry <= 0 or not last_sellable:
                continue

            pnl_pct = (last_sellable / entry) - 1.0

            # ---------- 1) STOP-LOSS קודם לכל ----------
            sl_pct = float(cfg.get("optional_sl_pct", 0.0) or 0.0)
            if sl_pct > 0 and pnl_pct <= -sl_pct:
                qty = float(pos["qty"])
                logging.info(f"SL CHECK {sym}: pnl={pnl_pct*100:.2f}% thresh={-sl_pct*100:.2f}% → SELL (SL)")
                _, avg, quote_recv = sell_market_all(client, sym, qty)
                log_trade("SELL", sym, qty, avg=avg, quote=quote_recv, reason="SL")
                to_delete.append(sym)
                continue

            # ---------- 2) Trailing Stop (אם מופעל) ----------
            trail_pct = float(cfg.get("trailing_stop_pct", 0.0) or 0.0)
            arm_pct   = float(cfg.get("trailing_arm_pct", 0.0) or 0.0)
            if trail_pct > 0 and arm_pct > 0 and pnl_pct >= arm_pct:
                if not pos.get("trail_active"):
                    pos["trail_active"] = True
                    pos["trail_peak"] = last_sellable
                    save_positions(state)
                    logging.info(f"TRAIL ARMED {sym} at ~{last_sellable}")
                else:
                    if last_sellable > float(pos.get("trail_peak", last_sellable)):
                        pos["trail_peak"] = last_sellable
                        save_positions(state)
                    peak = float(pos.get("trail_peak", last_sellable))
                    if peak > 0 and (peak - last_sellable) / peak >= trail_pct:
                        qty = float(pos["qty"])
                        logging.info(f"TRAIL CHECK {sym}: drop={(peak-last_sellable)/peak*100:.2f}% ≥ {trail_pct*100:.2f}% → SELL (TRAIL)")
                        _, avg, quote_recv = sell_market_all(client, sym, qty)
                        log_trade("SELL", sym, qty, avg=avg, quote=quote_recv, reason="TRAIL")
                        to_delete.append(sym)
                        continue

            # ---------- 3) Take Profit ----------
            tp_pct = float(cfg.get("target_tp_pct", 0.0) or 0.0)
            if tp_pct > 0 and pnl_pct >= tp_pct:
                qty = float(pos["qty"])
                logging.info(f"TP CHECK {sym}: pnl={pnl_pct*100:.2f}% ≥ {tp_pct*100:.2f}% → SELL (TP)")
                _, avg, quote_recv = sell_market_all(client, sym, qty)
                log_trade("SELL", sym, qty, avg=avg, quote=quote_recv, reason="TP")
                to_delete.append(sym)
                continue

            # לוג דיאגנוסטי קליל כדי לראות מצב
            logging.debug(f"MON {sym}: pnl={pnl_pct*100:.2f}% entry={entry} bid={last_sellable}")

        except Exception as e:
            logging.exception(f"monitor failed for {sym}: {e}")
        time.sleep(0.2)

    for sym in to_delete:
        state["positions"].pop(sym, None)
    if to_delete:
        save_positions(state)


def eod_liquidate_all(client: SpotClient, state: Dict, cfg):
    if not state["positions"]:
        return
    logging.info("EOD liquidation: closing all positions to QUOTE")

    # פיוס מהיר לפני מכירה – מצמצם כמות ליתרה אמיתית/סטפסייז:
    try:
        reconcile_positions(client, state, cfg)
    except Exception as e:
        logging.warning(f"reconcile before EOD failed: {e}")

    for sym, pos in list(state["positions"].items()):
        try:
            qty = float(pos["qty"])
            ex, avg, recv = sell_market_all(client, sym, qty)

            if ex <= 0 or recv <= 0:
                logging.info(f"EOD {sym}: treated as dust/zero → skip & drop")
                log_trade("SELL", sym, 0.0, avg=None, quote=0.0, reason="EOD_DUST_SKIP")
            else:
                logging.info(f"EOD EXIT {sym} recv≈{recv}")
                log_trade("SELL", sym, ex, avg=avg, quote=recv, reason="EOD")

            state["positions"].pop(sym, None)
            save_positions(state)

        except ValueError as ve:
            if str(ve) == "dust":
                logging.info(f"EOD {sym}: dust/too small → skip & drop")
                log_trade("SELL", sym, 0.0, avg=None, quote=0.0, reason="EOD_DUST_SKIP")
                state["positions"].pop(sym, None)
                save_positions(state)
            else:
                logging.exception(f"EOD sell failed for {sym}: {ve}")
        except Exception as e:
            # לא נתקעים – מתעדים וממשיכים לסימבול הבא
            logging.exception(f"EOD sell failed for {sym}: {e}")
        time.sleep(0.25)



# ---------- Existing balances bootstrap ----------
def vwap_cost_basis_from_trades(client: SpotClient, symbol: str, base_free: float, lookback_days: int) -> float | None:
    try:
        end = int(time.time() * 1000)
        start = end - lookback_days * 24 * 3600 * 1000
        trades = client.my_trades(symbol=symbol, startTime=start, endTime=end)
        if not trades:
            return None
        remain = base_free
        cost = 0.0
        qty_acc = 0.0
        for tr in reversed(trades):
            if not tr.get("isBuyer"):
                continue
            qty = float(tr["qty"])
            price = float(tr["price"])
            take = min(remain, qty)
            cost += take * price
            qty_acc += take
            remain -= take
            if remain <= 0:
                break
        return (cost / qty_acc) if qty_acc > 0 else None
    except Exception as e:
        logging.warning(f"cost-basis fetch failed for {symbol}: {e}")
        return None

def init_existing_positions(client: SpotClient, state: Dict, cfg):
    if not cfg["manage_assets"]:
        return
    for base in cfg["manage_assets"]:
        symbol = f"{base}{cfg['quote_asset']}"
        try:
            s = client.exchange_info(symbol=symbol)["symbols"][0]
            if s.get("status") != "TRADING":
                logging.info(f"{symbol} not TRADING; skip")
                continue
        except Exception:
            logging.info(f"symbol {symbol} not found; skip")
            continue
        free, _ = get_account_bal(client, base)
        if free <= 0:
            logging.info(f"no free balance for {base}; skip")
            continue
        if symbol in state["positions"]:
            logging.info(f"position for {symbol} already tracked; skip")
            continue
        entry = vwap_cost_basis_from_trades(client, symbol, free, cfg["cost_basis_lookback_days"]) if cfg["use_cost_basis"] else None
        bid, ask = book(client, symbol)
        mid = (bid + ask) / 2.0 if bid and ask else ask or bid
        entry = entry or mid
        state["positions"][symbol] = {"qty": free, "entry": entry, "opened_at": iso_utc(), "source": "existing_balance"}
        save_positions(state)
        logging.info(f"INIT FROM BALANCE: {symbol} qty={free} entry≈{entry}")

# ---------- Portfolio ----------
def position_mark(client: SpotClient, symbol: str, qty: float) -> float:
    bid, ask = book(client, symbol)
    mid = (bid + ask) / 2.0 if bid and ask else ask or bid
    return qty * mid

def portfolio_value_quote(client: SpotClient, state: Dict, cfg) -> float:
    free_q, _ = get_account_bal(client, cfg["quote_asset"])
    val = free_q
    for sym, pos in state["positions"].items():
        val += position_mark(client, sym, float(pos["qty"]))
    return val
def spread_bps_from_book(bid: float, ask: float) -> float:
    if ask <= 0: return 1e9
    return (ask - bid) / ask * 10_000

def score_symbol(client: SpotClient, sym: str) -> float:
    """
    ציון פשטני: מומנטום 20ד' (יחסי) מנורמל ע"י העונש על ספרד.
    ציון גבוה=עדיף. אם אין נתונים, מחזיר 0.
    """
    try:
        mom = kline_momentum(client, sym, limit=20)           # ~ “כוחה” של התנועה
        bid, ask = book(client, sym)
        spr_bps = spread_bps_from_book(bid, ask)
        if spr_bps <= 0: spr_bps = 1.0
        # ננרמל: ככל שהספרד גדול, מורידים ציון.  (1 + spr_bps/1000) הוא עונש עדין.
        return max(0.0, mom) / (1.0 + spr_bps / 1000.0)
    except Exception:
        return 0.0

def est_roundtrip_cost_pct(cfg) -> float:
    """
    עלות משוערת לעסקת החלפה: מכירה (SELL) לפוזיציה הישנה + קנייה (BUY) לחדשה.
    2 * עמלה + 2 * סליפג' קטן (bps → pct).
    """
    rt_bps = 2*(cfg["fee_bps"] + cfg["slippage_bps"])
    return rt_bps / 10_000.0
def regime_signal(client: SpotClient, cfg) -> int:
    """
    מחזיר +1 אם 'שוק עולה', -1 אם 'שוק יורד', 0 אם ניטרלי/אין נתונים.
    קריטריונים לכל סמל ברשימת REGIME_SYMBOLS על TF נתון:
      - EMA_SHORT > EMA_LONG
      - סגירה אחרונה > EMA_SHORT אחרונה
      - שיפוע EMA_SHORT חיובי (ל-N ברים אחרונים)
    אם לפחות regime_min_agree סמלים עומדים בקריטריונים → +1
    אם אף אחד לא עומד והסימנים הפוכים → -1
    אחרת 0.
    """
    if not cfg.get("regime_enabled", False):
        return +1  # אם לא מופעל — לא חוסמים קניות

    agree = 0
    disagree = 0
    tf = cfg["regime_tf"]
    pS, pL = cfg["regime_ema_short"], cfg["regime_ema_long"]
    need = cfg["regime_min_agree"]

    for sym in cfg["regime_symbols"]:
        closes = get_klines_closes(client, sym, tf, limit=max(pL*3, 60))
        if len(closes) < pL+2:
            continue
        eS = ema(closes, pS)
        eL = ema(closes, pL)
        if not eS or not eL:
            continue
        eS_last = eS[-1]
        eL_last = eL[-1]
        # שיפוע EMA-קצר על 10 נק׳ אחרונות (או כל אורך שיש)
        slope = pct_slope_per_bar(eS[-10:] if len(eS) >= 10 else eS)
        last_close = closes[-1]

        bull = (eS_last > eL_last) and (last_close > eS_last) and (slope > 0)
        bear = (eS_last < eL_last) and (last_close < eS_last) and (slope < 0)
        if bull:
            agree += 1
        elif bear:
            disagree += 1

    if agree >= need:
        return +1
    if disagree == len(cfg["regime_symbols"]):
        return -1
    return 0
def symbol_uptrend_ok(client: SpotClient, symbol: str, cfg) -> bool:
    """
    True רק אם:
      - Momentum 20×1m חיובי
      - BUY_TF: EMA_SHORT > EMA_LONG
      - BUY_TF: close אחרון > EMA_SHORT
      - BUY_TF: שיפוע EMA_SHORT ≥ buy_min_slope_pct_per_bar
    """
    try:
        if kline_momentum(client, symbol, limit=20) <= 0.0:
            return False
        closes = get_klines_closes(client, symbol, cfg["buy_tf"], limit=max(cfg["buy_ema_long"]*3, 60))
        if len(closes) < cfg["buy_ema_long"]+2:
            return False
        eS = ema(closes, cfg["buy_ema_short"])
        eL = ema(closes, cfg["buy_ema_long"])
        if not eS or not eL:
            return False
        last_close = closes[-1]
        eS_last, eL_last = eS[-1], eL[-1]
        if not (eS_last > eL_last and last_close > eS_last):
            return False
        slope = pct_slope_per_bar(eS[-10:] if len(eS) >= 10 else eS)
        return slope >= cfg["buy_min_slope_pct_per_bar"]
    except Exception:
        return False

# ---------- Main ----------
def main():
    # נטרול stop.flag אם נשאר מריצה קודמת
    if STOP_FLAG.exists():
        STOP_FLAG.unlink()

    cfg = load_env()
    client = client_from_cfg(cfg)
    state = load_positions()
    reconcile_positions(client, state, cfg)


    # אופציונלי: בתחילת היום לרכז הכל ל-QUOTE כדי להשתמש בכל היתרה
    if cfg.get("consolidate_at_start", False) and state["positions"]:
        logging.info("Consolidate-at-start enabled: liquidating all current positions to QUOTE")
        eod_liquidate_all(client, state, cfg)

    # אתחול פוזיציות קיימות (למשל BTTC) אם מוגדר
    init_existing_positions(client, state, cfg)

    logging.info(
        f"Starting 5% bot | QUOTE={cfg['quote_asset']} "
        f"target={cfg['target_tp_pct']*100:.1f}% max_pos={cfg['max_positions']} "
        f"all_balance={cfg['use_all_balance']} auto_split={cfg['auto_split']} "
        f"reserve={cfg['balance_reserve_pct']*100:.1f}%"
    )

    # נקודת בסיס לתיק (ליעד/סטופ פורטפוליו)
    start_value = portfolio_value_quote(client, state, cfg)
    logging.info(f"portfolio start value ≈ {start_value} {cfg['quote_asset']}")

    last_scan = 0.0
    wait_until_ts = 0.0  # קול־דאון אחרי שוק יורד / סטופ פורטפוליו

    while True:
        if STOP_FLAG.exists():
            logging.info('stop.flag detected → exiting main loop')
            break

        try:
            now_ts = time.time()

            # --- בדיקת משטר שוק (regime) ---
            sig = regime_signal(client, cfg)   # +1 שורי, 0 ניטרלי, -1 דובי
            if sig < 0:
                # שוק יורד → מוכרים הכל ונכנסים לקול־דאון
                if state["positions"]:
                    logging.info("REGIME BEAR: liquidate all and enter cooldown")
                    eod_liquidate_all(client, state, cfg)
                    # אתחול נקודת בסיס אחרי מכירה
                    start_value = portfolio_value_quote(client, state, cfg)
                # הארכת קול־דאון
                wait_until_ts = max(wait_until_ts, now_ts + cfg.get("regime_cooldown_min", 90)*60)

            # --- פתיחת פוזיציות חדשות (לפי סריקה) ---
            if now_ts - last_scan > cfg["scan_interval"]:
                can_buy = (sig > 0) and (now_ts >= wait_until_ts)
                if can_buy:
                    maybe_open_positions(client, state, cfg)
                else:
                    logging.info("BUY BLOCKED: regime not bullish or cooldown active")
                last_scan = now_ts

            # --- ניטור פוזיציות קיימות ויציאות TP/SL/Trailing ---
            if state["positions"]:
                monitor_and_close(client, state, cfg)

            # --- יעד/סטופ ברמת התיק ---
            cur_val = portfolio_value_quote(client, state, cfg)
            pnl_port = (cur_val / start_value) - 1.0 if start_value > 0 else 0.0

            if cfg.get("portfolio_target_pct", 0.0) > 0 and pnl_port >= cfg["portfolio_target_pct"]:
                logging.info(f"PORTFOLIO TARGET HIT: {pnl_port*100:.2f}% → liquidate all to QUOTE")
                eod_liquidate_all(client, state, cfg)
                start_value = portfolio_value_quote(client, state, cfg)
                # קול־דאון קצר אחרי מימוש יעד
                wait_until_ts = max(wait_until_ts, time.time() + 15*60)

            elif cfg.get("portfolio_stop_pct", 0.0) > 0 and pnl_port <= -cfg["portfolio_stop_pct"]:
                logging.info(f"PORTFOLIO STOP HIT: {pnl_port*100:.2f}% → liquidate all to QUOTE")
                eod_liquidate_all(client, state, cfg)
                start_value = portfolio_value_quote(client, state, cfg)
                # קול־דאון לפי ההגדרה
                wait_until_ts = max(wait_until_ts, time.time() + cfg.get("regime_cooldown_min", 90)*60)

            # --- סוף יום: מכירה של הכל ל-QUOTE ---
            if is_eod(cfg):
                eod_liquidate_all(client, state, cfg)
                start_value = portfolio_value_quote(client, state, cfg)
                time.sleep(65)   # להתרחק מהשעה של EOD
                last_scan = 0.0  # התחלה חדשה ללולאת סריקה

        except Exception as e:
            logging.exception(f"main loop error: {e}")
            time.sleep(2)

        time.sleep(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("bye")
