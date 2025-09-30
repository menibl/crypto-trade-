# -------------- bot_5pct_v3_info.py (defensive + explain logs at INFO+DEBUG) --------------
# Changes vs v3:
# - For each candidate symbol, we now log a one-line INFO summary: ALLOW/BLOCK + key reason.
# - Detailed step-by-step reasons remain at DEBUG level.
# - Rest is identical to v3 (maker-first buys, dust-safe sells, cooldowns, EOD skip, etc.).

import os, json, math, time, logging, sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from decimal import Decimal, ROUND_DOWN, getcontext

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from dotenv import load_dotenv
from binance.spot import Spot as SpotClient
from pathlib import Path
ENV_FILE    = Path(__file__).parent / ".env"
RELOAD_FLAG = Path(__file__).parent / "reload.flag"
_LAST_ENV_MTIME = 0.0

# --- Optional link to crypto_bot.py ---
CRYPTO_BOT_AVAILABLE = False
RegimeRouter = TrendBreakoutStrategy = None
BotConfig = default_config = None
try:
    import importlib.util
    here = Path(__file__).parent
    cb_path = here / "crypto_bot.py"
    if cb_path.exists():
        spec = importlib.util.spec_from_file_location("crypto_bot", cb_path.as_posix())
        cb = importlib.util.module_from_spec(spec)  # type: ignore
        sys.modules["crypto_bot"] = cb  # type: ignore
        spec.loader.exec_module(cb)     # type: ignore
        RegimeRouter = cb.RegimeRouter
        TrendBreakoutStrategy = cb.TrendBreakoutStrategy
        BotConfig = cb.BotConfig
        default_config = cb.default_config
        CRYPTO_BOT_AVAILABLE = True
except Exception:
    CRYPTO_BOT_AVAILABLE = False

getcontext().prec = 28

# ---------- Paths / Const ----------
APP_DIR     = Path(__file__).parent
STATE_FILE  = APP_DIR / "positions.json"
TRADES_FILE = APP_DIR / "trades.json"
STOP_FLAG   = APP_DIR / "stop.flag"
TESTNET_URL = "https://testnet.binance.vision"
PROD_URL    = "https://api.binance.com"
IL_TZ = ZoneInfo("Asia/Jerusalem")
STABLES = {"USDT","USDC","FDUSD","TUSD","BUSD","DAI","USDP","EURT","USTC"}

# ---------- Utils ----------
def _dec(x) -> Decimal:
    return x if isinstance(x, Decimal) else Decimal(str(x))

def _to_str(d: Decimal) -> str:
    s = format(d, 'f')
    if '.' in s:
        s = s.rstrip('0').rstrip('.')
    return s if s else '0'

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
def reload_cfg_if_changed(cfg, client):
    """
    Reload .env into cfg when either reload.flag exists or .env mtime changed.
    If API keys / base_url changed → rebuild Spot client.
    Returns (cfg, client, reloaded: bool).
    """
    global _LAST_ENV_MTIME
    need = RELOAD_FLAG.exists()
    try:
        mtime = ENV_FILE.stat().st_mtime
    except FileNotFoundError:
        mtime = _LAST_ENV_MTIME
    if mtime > _LAST_ENV_MTIME:
        need = True
    if not need:
        return cfg, client, False

    load_dotenv(ENV_FILE, override=True)
    new_cfg = load_env()   # assumes you already have a load_env() that returns dict

    if (new_cfg.get("base_url") != cfg.get("base_url")
        or new_cfg.get("api_key") != cfg.get("api_key")
        or new_cfg.get("api_secret") != cfg.get("api_secret")):
        client = client_from_cfg(new_cfg)

    _LAST_ENV_MTIME = mtime
    if RELOAD_FLAG.exists():
        try: RELOAD_FLAG.unlink()
        except Exception: pass

    return new_cfg, client, True
# ---------- ENV / CONFIG ----------
def load_env() -> Dict[str, Any]:
    load_dotenv(ENV_FILE)

    # --- helpers: קריאת משתני סביבה חסינה לערכים ריקים ---
    def env_str(name: str, default: Optional[str] = None) -> Optional[str]:
        v = os.getenv(name)
        if v is None: 
            return default
        v = v.strip()
        return default if v == "" else v

    def env_int(name: str, default: int) -> int:
        v = env_str(name, None)
        try: return int(v) if v is not None else default
        except: return default

    def env_float(name: str, default: float) -> float:
        v = env_str(name, None)
        try: return float(v) if v is not None else default
        except: return default

    def env_bool(name: str, default: bool) -> bool:
        v = (env_str(name, None) or "").lower()
        if v in ("1","true","yes","y","on"):  return True
        if v in ("0","false","no","n","off"): return False
        return default

    api_key    = env_str("BINANCE_API_KEY")
    api_secret = env_str("BINANCE_API_SECRET")
    env_name   = (env_str("BINANCE_ENV","prod") or "prod").lower()
    base_url   = PROD_URL if env_name == "prod" else TESTNET_URL
    if not api_key or not api_secret:
        raise RuntimeError("Missing API keys in .env")

    # logging
    log_level = (env_str("LOG_LEVEL", "INFO") or "INFO").upper()
    logging.basicConfig(level=getattr(logging, log_level, logging.INFO),
                        format="%(asctime)s %(levelname)s %(message)s")
    log_file = env_str("LOG_FILE", None)
    if log_file:
        from logging.handlers import RotatingFileHandler
        handler = RotatingFileHandler(
            log_file,
            maxBytes=env_int("LOG_MAX_BYTES", 10485760),
            backupCount=env_int("LOG_BACKUP_COUNT", 3),
            encoding="utf-8",
        )
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logging.getLogger().addHandler(handler)

    # core
    quote_asset        = (env_str("QUOTE_ASSET", "USDT") or "USDT").upper()
    blacklist_keywords = [s.strip().upper() for s in (env_str("BASE_BLACKLIST","UP,DOWN,BEAR,BULL,VENFT,BRL") or "").split(",") if s.strip()]

    # trade sizing / exits
    target_tp_pct     = env_float("TARGET_TP_PCT", 0.05)
    optional_sl_pct   = env_float("OPTIONAL_SL_PCT", 0.02)
    trailing_stop_pct = env_float("TRAILING_STOP_PCT", 0.02)
    trailing_arm_pct  = env_float("TRAILING_ARM_PNL_PCT", 0.02)

    # portfolio guards
    portfolio_target_pct = env_float("PORTFOLIO_TARGET_PCT", 0.05)
    portfolio_stop_pct   = env_float("PORTFOLIO_STOP_PCT", 0.02)

    # positions / allocation
    max_positions_cfg = env_int("MAX_OPEN_POSITIONS", 2)
    alloc_quote       = env_float("ALLOCATION_USD", 20.0)

    # scanning
    min_quote_vol   = env_float("MIN_24H_QUOTE_VOL", 15000000.0)
    max_spread_bps  = env_float("MAX_SPREAD_BPS", 10.0)
    scan_interval   = env_int("SCAN_INTERVAL_SEC", 60)
    eod_hhmm        = env_str("EOD_HHMM", "23:59")
    eod_skip_min    = env_int("EOD_SKIP_MIN", 90)

    # all-balance / split
    use_all_balance      = env_bool("USE_ALL_BALANCE", False)
    balance_reserve_pct  = env_float("BALANCE_RESERVE_PCT", 0.02)
    consolidate_at_start = env_bool("CONSOLIDATE_AT_START", False)

    auto_split    = env_bool("AUTO_SPLIT", True)
    split_slices  = env_int("SPLIT_SLICES", 2)
    min_slice_usd = env_float("MIN_SLICE_USD", 20.0)

    # manage existing balances (optional)
    manage_assets_csv = (env_str("MANAGE_EXISTING_ASSETS", "") or "").upper().strip()
    manage_assets     = [a for a in (s.strip() for s in manage_assets_csv.split(",")) if a]
    use_cost_basis    = env_bool("USE_COST_BASIS_FROM_TRADES", True)
    cost_basis_lookback_days = env_int("COST_BASIS_LOOKBACK_DAYS", 60)

    # rotation & costs
    rotation_enabled  = env_bool("ROTATION_ENABLED", True)
    rotation_edge_pct = env_float("ROTATION_EDGE_PCT", 0.20)
    fee_bps           = env_float("FEE_BPS", 10.0)
    slippage_bps      = env_float("SLIPPAGE_BPS", 5.0)

    # regime checks
    regime_enabled    = env_bool("REGIME_CHECK_ENABLED", True)
    regime_symbols    = [s.strip().upper() for s in (env_str("REGIME_SYMBOLS","BTCUSDT,ETHUSDT") or "").split(",") if s.strip()]
    regime_tf         = env_str("REGIME_TF","15m")
    regime_ema_short  = env_int("REGIME_EMA_SHORT", 20)
    regime_ema_long   = env_int("REGIME_EMA_LONG", 50)
    regime_min_agree  = env_int("REGIME_MIN_AGREE", 1)
    regime_cooldown_min = env_int("REGIME_COOLDOWN_MIN", 90)

    # buy filter
    buy_tf            = env_str("BUY_TF","5m")
    buy_ema_short     = env_int("BUY_EMA_SHORT", 20)
    buy_ema_long      = env_int("BUY_EMA_LONG", 50)
    buy_min_slope_pct_per_bar = env_float("BUY_MIN_SLOPE_PCT_PER_BAR", 0.0002)

    # HTF alignment
    htf_align_enabled = env_bool("HTF_ALIGN_ENABLED", True)
    htf_tf            = env_str("HTF_TF","1h")
    htf_ema_short     = env_int("HTF_EMA_SHORT", 20)
    htf_ema_long      = env_int("HTF_EMA_LONG", 50)

    # maker preference
    use_maker         = env_bool("USE_MAKER", True)
    maker_wait_sec    = env_int("MAKER_WAIT_SEC", 10)

    # loss cooldown
    loss_cooldown_min = env_int("LOSS_COOLDOWN_MIN", 20)

    if use_all_balance:
        auto_split = False
        max_positions = 1
    else:
        max_positions = max_positions_cfg

    cfg = {
        "api_key": api_key, "api_secret": api_secret, "env": env_name, "base_url": base_url,
        "quote_asset": quote_asset, "blacklist_keywords": blacklist_keywords,
        "alloc_quote": alloc_quote, "max_positions": max_positions,
        "target_tp_pct": target_tp_pct, "optional_sl_pct": optional_sl_pct,
        "trailing_stop_pct": trailing_stop_pct, "trailing_arm_pct": trailing_arm_pct,
        "min_24h_quote_vol": min_quote_vol, "max_spread_bps": max_spread_bps,
        "scan_interval": scan_interval, "eod_hhmm": eod_hhmm, "eod_skip_min": eod_skip_min,
        "use_all_balance": use_all_balance, "balance_reserve_pct": balance_reserve_pct,
        "consolidate_at_start": consolidate_at_start,
        "auto_split": auto_split, "split_slices": split_slices, "min_slice_usd": min_slice_usd,
        "portfolio_target_pct": portfolio_target_pct, "portfolio_stop_pct": portfolio_stop_pct,
        "manage_assets": manage_assets, "use_cost_basis": use_cost_basis,
        "cost_basis_lookback_days": cost_basis_lookback_days,
        "rotation_enabled": rotation_enabled, "rotation_edge_pct": rotation_edge_pct,
        "fee_bps": fee_bps, "slippage_bps": slippage_bps,
        "regime_enabled": regime_enabled, "regime_symbols": regime_symbols,
        "regime_tf": regime_tf, "regime_ema_short": regime_ema_short, "regime_ema_long": regime_ema_long,
        "regime_min_agree": regime_min_agree, "regime_cooldown_min": regime_cooldown_min,
        "buy_tf": buy_tf, "buy_ema_short": buy_ema_short, "buy_ema_long": buy_ema_long,
        "buy_min_slope_pct_per_bar": buy_min_slope_pct_per_bar,
        "htf_align_enabled": htf_align_enabled, "htf_tf": htf_tf, "htf_ema_short": htf_ema_short, "htf_ema_long": htf_ema_long,
        "use_maker": use_maker, "maker_wait_sec": maker_wait_sec,
        "loss_cooldown_min": loss_cooldown_min,
    }

    # <<< חשוב: טען גם את הרחבות MOB >>>
    cfg.update(load_env_mobo_extras(os.environ))

    return cfg



# ---------- Client ----------
def client_from_cfg(cfg) -> SpotClient:
    return SpotClient(api_key=cfg["api_key"], api_secret=cfg["api_secret"], base_url=cfg["base_url"])

# ---------- Exchange helpers ----------
def book(client: SpotClient, symbol: str) -> Tuple[float, float]:
    bk = client.book_ticker(symbol=symbol)
    return float(bk["bidPrice"]), float(bk["askPrice"])

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
        if base in STABLES:
            continue
        syms.append(sym)
    return syms

def _get_symbol_filters_dict(client: SpotClient, symbol: str) -> dict:
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
    }

def _quantize_to_step(amount: Decimal, step_str: str) -> Decimal:
    step = Decimal(step_str)
    if step <= 0:
        return amount
    return (amount // step) * step

def _price_tick(meta) -> Decimal:
    return Decimal(str(meta["price"].get("tickSize", "0.00000001")))

def spread_bps_from_book(bid: float, ask: float) -> float:
    if ask <= 0: return 1e9
    return (ask - bid) / ask * 10_000

def kline_momentum(client: SpotClient, symbol: str, limit: int = 20) -> float:
    kl = client.klines(symbol=symbol, interval="1m", limit=limit)
    if not kl or len(kl) < 2:
        return 0.0
    first_open = float(kl[0][1])
    last_close = float(kl[-1][4])
    return (last_close / first_open) - 1.0

def get_klines_closes(client: SpotClient, symbol: str, interval: str, limit: int) -> List[float]:
    kl = client.klines(symbol=symbol, interval=interval, limit=limit)
    return [float(k[4]) for k in kl] if kl else []

# ---------- EMA & ATR helpers ----------
def ema_series(series: List[float], period: int) -> List[float]:
    if not series or period <= 1 or len(series) < period:
        return []
    k = 2 / (period + 1)
    out = [sum(series[:period]) / period]
    for x in series[period:]:
        out.append(out[-1] + k * (x - out[-1]))
    return out

def pct_slope_per_bar(series: List[float]) -> float:
    if not series or len(series) < 2:
        return 0.0
    first, last = series[0], series[-1]
    if first <= 0:
        return 0.0
    return (last / first - 1.0) / (len(series)-1)

def atr_like(closes: List[float], n: int = 14) -> float:
    if not closes or len(closes) < n+1:
        return 0.0
    diffs = [abs(closes[i]-closes[i-1]) for i in range(1,len(closes))]
    if len(diffs) < n:
        return 0.0
    return sum(diffs[-n:]) / n

# ---------- Scoring & candidates ----------
def score_symbol(client: SpotClient, sym: str) -> float:
    try:
        mom = kline_momentum(client, sym, limit=20)
        bid, ask = book(client, sym)
        spr_bps = spread_bps_from_book(bid, ask)
        if spr_bps <= 0: spr_bps = 1.0
        return max(0.0, mom) / (1.0 + spr_bps / 1000.0)
    except Exception:
        return 0.0

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
    logging.info(f"scan: candidates {cands}")
    return cands

# ---------- Maker/Market buy ----------
def buy_market_quote(client: SpotClient, symbol: str, quote_amount: float):
    meta = _get_symbol_filters_dict(client, symbol)
    sraw = meta["raw"]
    quote_asset = meta["quote"]
    quote_prec = int(sraw.get("quotePrecision", sraw.get("quoteAssetPrecision", 8)))
    min_notional = _dec(meta["notional"].get("minNotional", "0"))

    free_q, _ = get_account_bal(client, quote_asset)
    amt = min(quote_amount, free_q)

    if min_notional > 0 and _dec(amt) < min_notional:
        logging.info(f"{symbol}: amount {amt} below minNotional {min_notional} -> skip buy")
        return 0.0, None

    qd = _dec(amt).quantize(Decimal(1).scaleb(-quote_prec), rounding=ROUND_DOWN)
    if qd <= 0:
        logging.info(f"{symbol}: rounded quote amount is 0 -> skip buy")
        return 0.0, None

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
        logging.exception(f"buy_market_quote failed for {symbol}: {e}")
        return 0.0, None

def buy_limit_maker_try(client: SpotClient, symbol: str, target_quote: float, wait_sec: int = 10):
    meta = _get_symbol_filters_dict(client, symbol)
    lot = meta["lot"]; pricef = meta["price"]; notional = meta["notional"]
    step_str = lot.get("stepSize", "0.00000001")
    tick = _price_tick(meta)
    min_notional = _dec(notional.get("minNotional","0"))
    bid, ask = book(client, symbol)
    if bid <= 0:
        return 0.0, None, 0.0
    limit_price = max(_dec(str(bid)) - tick*2, tick)
    qty_est = _dec(str(target_quote)) / limit_price
    qty_q = _quantize_to_step(qty_est, step_str)
    if qty_q <= 0:
        return 0.0, None, 0.0
    if min_notional > 0 and (qty_q * limit_price) < min_notional:
        logging.info(f"{symbol}: maker notional too small -> skip maker")
        return 0.0, None, 0.0
    try:
        ord = client.new_order(symbol=symbol, side="BUY", type="LIMIT_MAKER",
                               price=_to_str(limit_price), quantity=_to_str(qty_q), recvWindow=5000)
        order_id = ord.get("orderId")
        t0 = time.time()
        while time.time() - t0 < wait_sec:
            q = client.get_order(symbol=symbol, orderId=order_id)
            status = q.get("status","")
            executed_qty = _dec(q.get("executedQty","0"))
            cqq = _dec(q.get("cummulativeQuoteQty","0"))
            if status in ("FILLED","PARTIALLY_FILLED") and executed_qty > 0:
                avg = (cqq / executed_qty) if executed_qty > 0 else None
                if status == "FILLED":
                    return float(executed_qty), float(avg) if avg else None, float(cqq)
            time.sleep(0.5)
        try:
            client.cancel_order(symbol=symbol, orderId=order_id)
        except Exception as ce:
            logging.warning(f"cancel maker order failed: {ce}")
        q = client.get_order(symbol=symbol, orderId=order_id)
        executed_qty = float(q.get("executedQty","0"))
        cqq = float(q.get("cummulativeQuoteQty","0"))
        avg = (cqq / executed_qty) if executed_qty > 0 else None
        return executed_qty, avg, cqq
    except Exception as e:
        logging.info(f"{symbol}: LIMIT_MAKER rejected or failed ({e})")
        return 0.0, None, 0.0

# ---------- Sells (dust-safe) ----------
def sell_market_all(client: SpotClient, symbol: str, desired_qty: float):
    meta = _get_symbol_filters_dict(client, symbol)
    base = meta["base"]
    lot = meta["lot"]
    step_str = lot.get("stepSize", "0.00000001")
    min_qty = _dec(lot.get("minQty", "0"))
    min_notional = _dec(meta["notional"].get("minNotional", "0"))

    free_base, _ = get_account_bal(client, base)
    q_dec = _quantize_to_step(_dec(str(min(desired_qty, free_base))), step_str)
    if q_dec <= 0 or (min_qty > 0 and q_dec < min_qty):
        logging.info(f"{symbol}: dust/too small to sell (qty) -> skipping")
        return 0.0, None, 0.0

    bid, ask = book(client, symbol)
    mid = _dec(str((bid + ask)/2.0 if bid and ask else ask or bid or 0.0))
    if min_notional > 0 and mid > 0 and (q_dec * mid) < min_notional:
        logging.info(f"{symbol}: dust/too small to sell (notional) -> skipping")
        return 0.0, None, 0.0

    try:
        qty_str = _to_str(q_dec)
        ord = client.new_order(symbol=symbol, side="SELL", type="MARKET",
                               quantity=qty_str, recvWindow=5000)
        cumm_quote   = float(ord.get("cummulativeQuoteQty", 0.0))
        executed_qty = float(ord.get("executedQty", 0.0))
        avg_price    = (cumm_quote / executed_qty) if executed_qty > 0 else None
        return executed_qty, avg_price, cumm_quote
    except Exception as e:
        logging.exception(f"sell_market_all failed for {symbol}: {e}")
        return 0.0, None, 0.0
    

def try_sell_with_retries(client, symbol: str, qty: float, reason: str, cfg):
    """
    עוטף את sell_market_all עם רידו, לוג ברור, וטיפול בשגיאות נפוצות.
    מחזיר (executed_qty, avg, quote_recv) או זורק שגיאה בסוף.
    """
    import time
    max_try = int(cfg.get("sell_retry_max", 5))
    backoff = max(int(cfg.get("sell_retry_backoff_ms", 250)), 50) / 1000.0

    last_exc = None
    for i in range(1, max_try+1):
        try:
            ex, avg, recv = sell_market_all(client, symbol, qty)
            logging.info(f"SELL OK {symbol} qty={ex:.8f} avg≈{avg} quote≈{recv} reason={reason}")
            return ex, avg, recv
        except ValueError as ve:
            # dust כבר מטופל אצלך; נעביר הלאה כדי שהמדיניות (drop/keep) תפעל
            if str(ve) == "dust":
                logging.warning(f"SELL {symbol} blocked: dust (minNotional); reason={reason}")
                raise
            last_exc = ve
            logging.warning(f"SELL retry {i}/{max_try} for {symbol} failed: {ve} | reason={reason}")
        except Exception as e:
            last_exc = e
            logging.warning(f"SELL retry {i}/{max_try} for {symbol} failed: {e} | reason={reason}")
        time.sleep(backoff)
    raise RuntimeError(f"SELL {symbol} exhausted retries; last={last_exc}")

def is_sellable(client: SpotClient, symbol: str, qty: float) -> bool:
    meta = _get_symbol_filters_dict(client, symbol)
    lot = meta["lot"]
    step_str = lot.get("stepSize", "0.00000001")
    min_qty = _dec(lot.get("minQty","0"))
    notional = _dec(meta["notional"].get("minNotional","0"))
    bid, ask = book(client, symbol)
    mid = _dec(str((bid + ask)/2.0 if bid and ask else ask or bid or 0.0))
    free_base, _ = get_account_bal(client, meta["base"])
    q_dec = _quantize_to_step(_dec(str(min(qty, free_base))), step_str)
    if q_dec <= 0 or (min_qty > 0 and q_dec < min_qty):
        return False
    if notional > 0 and mid > 0 and (q_dec * mid) < notional:
        return False
    return True

# ---------- Indicators and decisions ----------
def ema_series_safe(closes: List[float], p: int) -> List[float]:
    try: return ema_series(closes, p)
    except Exception: return []

def simple_regime_signal(client: SpotClient, cfg) -> int:
    try:
        closes = get_klines_closes(client, "BTCUSDT", cfg["regime_tf"], limit=200)
        if len(closes) < max(cfg["regime_ema_long"]+2, 60):
            return +1
        eS = ema_series_safe(closes, cfg["regime_ema_short"])
        eL = ema_series_safe(closes, cfg["regime_ema_long"])
        slope = pct_slope_per_bar(eS[-10:] if len(eS) >= 10 else eS)
        if eS and eL and eS[-1] > eL[-1] and closes[-1] > eS[-1] and slope > 0:
            return +1
        if eS and eL and eS[-1] < eL[-1] and closes[-1] < eS[-1] and slope < 0:
            return -1
        return 0
    except Exception:
        return +1

def regime_signal(client: SpotClient, cfg) -> int:
    if not cfg.get("regime_enabled", False):
        return +1
    if CRYPTO_BOT_AVAILABLE and RegimeRouter and default_config:
        try:
            import pandas as pd
            closes = get_klines_closes(client, "BTCUSDT", cfg["regime_tf"], limit=300)
            if len(closes) < 60:
                logging.debug("[regime] source=crypto_bot insufficient data -> default +1")
                return +1
            c = pd.Series(closes)
            o = c.shift(1).fillna(c.iloc[0])
            h = pd.concat([o, c], axis=1).max(axis=1)
            l = pd.concat([o, c], axis=1).min(axis=1)
            v = pd.Series([1.0]*len(c))
            df = pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v})
            router = RegimeRouter(default_config().regime)
            r = router.classify(df)
            logging.debug(f"[regime] source=crypto_bot router_class={r}")
            return +1 if r in ("trend","mixed") else 0
        except Exception as e:
            logging.debug(f"[regime] crypto_bot failed ({e}) -> fallback")
            s = simple_regime_signal(client, cfg)
            logging.debug(f"[regime] source=fallback signal={s}")
            return s
    else:
        s = simple_regime_signal(client, cfg)
        logging.debug(f"[regime] source=fallback signal={s}")
        return s

def explain_buy_decision(client: SpotClient, symbol: str, cfg) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    mom = kline_momentum(client, symbol, limit=20)
    if mom <= 0.0:
        reasons.append(f"FAIL: momentum_20x1m <= 0 ({mom:.5f})")
        return False, reasons
    reasons.append(f"PASS: momentum_20x1m {mom:.5f}")

    if cfg.get("htf_align_enabled", True):
        closes_h = get_klines_closes(client, symbol, cfg["htf_tf"], limit=max(cfg["htf_ema_long"]*3, 120))
        if len(closes_h) >= cfg["htf_ema_long"]+2:
            eS_h = ema_series_safe(closes_h, cfg["htf_ema_short"])
            eL_h = ema_series_safe(closes_h, cfg["htf_ema_long"])
            if not (eS_h and eL_h and eS_h[-1] > eL_h[-1]):
                reasons.append("FAIL: HTF EMA short not above long")
                return False, reasons
            reasons.append("PASS: HTF EMA short > long")
        else:
            reasons.append("WARN: HTF data insufficient -> allow")

    closes = get_klines_closes(client, symbol, cfg["buy_tf"], limit=max(cfg["buy_ema_long"]*3, 120))
    if len(closes) < cfg["buy_ema_long"]+2:
        reasons.append("FAIL: insufficient data on BUY_TF")
        return False, reasons
    eS = ema_series_safe(closes, cfg["buy_ema_short"])
    eL = ema_series_safe(closes, cfg["buy_ema_long"])
    if not eS or not eL:
        reasons.append("FAIL: EMA calc failed")
        return False, reasons
    last_close = closes[-1]
    if not (eS[-1] > eL[-1] and last_close > eS[-1]):
        reasons.append("FAIL: not in uptrend (EMA short>long and price>EMA short required)")
        return False, reasons
    reasons.append("PASS: uptrend on BUY_TF (EMA short>long and price>EMA short)")

    slope = pct_slope_per_bar(eS[-10:] if len(eS) >= 10 else eS)
    if slope < cfg["buy_min_slope_pct_per_bar"]:
        reasons.append(f"FAIL: EMA short slope {slope:.6f} < min {cfg['buy_min_slope_pct_per_bar']}")
        return False, reasons
    reasons.append(f"PASS: EMA short slope {slope:.6f}")

    extended = (last_close / eS[-1] - 1.0) >= 0.005
    if extended:
        reasons.append("FAIL: price too extended above EMA short (>0.5%)")
        return False, reasons
    reasons.append("PASS: not extended above EMA short")

    N = 6
    near = any(abs(c - e) / e <= 0.002 for c, e in zip(closes[-N:], eS[-N:]))
    resume = closes[-1] > max(closes[-2], eS[-1])
    if not (near and resume):
        reasons.append("FAIL: no recent pullback near EMA or no resume")
        return False, reasons
    reasons.append("PASS: pullback and resume confirmed")

    atr = atr_like(closes)
    if atr > 0 and (closes[-1] - closes[-2]) > 1.5 * atr:
        reasons.append("FAIL: last candle too large vs ATR (chase)")
        return False, reasons
    reasons.append("PASS: last candle size acceptable")

    if CRYPTO_BOT_AVAILABLE and TrendBreakoutStrategy and default_config:
        try:
            import pandas as pd
            c = pd.Series(closes[-120:])
            o = c.shift(1).fillna(c.iloc[0])
            h = pd.concat([o, c], axis=1).max(axis=1)
            l = pd.concat([o, c], axis=1).min(axis=1)
            v = pd.Series([1.0]*len(c))
            df = pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v})
            tcfg = default_config().trend_breakout
            from types import SimpleNamespace
            risk = getattr(default_config(), "risk", SimpleNamespace(atr_multiple_stop=2.0))
            tstrat = TrendBreakoutStrategy(tcfg, risk)
            tsig = tstrat.signal(df)
            ok = (getattr(tsig, "side", None) in (None, "long"))
            if not ok:
                reasons.append("FAIL: TrendBreakout gate blocked long")
                return False, reasons
            reasons.append("PASS: TrendBreakout gate allows long")
        except Exception as e:
            reasons.append(f"WARN: TrendBreakout check failed ({e}); allow")

    return True, reasons

# ---------- Position logic & rotation ----------
def desired_new_positions(state: Dict, candidates: List[str], cfg) -> List[str]:
    open_syms = set(state["positions"].keys())
    room = max(cfg["max_positions"] - len(open_syms), 0)
    return [s for s in candidates if s not in open_syms][:room]

def estimate_buy_budget(client: SpotClient, cfg, n_slots: int) -> float:
    free_q, _ = get_account_bal(client, cfg["quote_asset"])
    if cfg["use_all_balance"]:
        return max(free_q * (1.0 - cfg["balance_reserve_pct"]), 0.0)
    if cfg["auto_split"] and n_slots > 0:
        budget = max(free_q * (1.0 - cfg["balance_reserve_pct"]), 0.0)
        return max(0.0, budget / n_slots)
    return min(free_q, cfg["alloc_quote"])

def buy_with_preference(client: SpotClient, symbol: str, quote_amount: float, cfg) -> Tuple[float, Optional[float]]:
    if quote_amount <= 0:
        return 0.0, None
    if cfg.get("use_maker", True):
        filled_qty, avg, filled_quote = buy_limit_maker_try(client, symbol, quote_amount, cfg.get("maker_wait_sec", 10))
        if filled_qty > 0:
            return filled_qty, avg
    return buy_market_quote(client, symbol, quote_amount)

def maybe_open_positions(client: SpotClient, state: Dict, cfg):
    cands = scan_candidates(client, cfg)

    allowed = []
    for s in cands:
        ok, reasons = explain_buy_decision(client, s, cfg)
        # INFO-level summary
        if ok:
            # find a notable PASS or just say OK
            pass_msgs = [r for r in reasons if r.startswith("PASS")]
            key = pass_msgs[-1] if pass_msgs else "OK"
            logging.info(f"[buy_check] {s}: ALLOW | {key}")
        else:
            # first FAIL if any
            fail = next((r for r in reasons if r.startswith("FAIL")), reasons[0] if reasons else "blocked")
            logging.info(f"[buy_check] {s}: BLOCK | {fail}")
        # DEBUG-level full list
        for r in reasons:
            logging.debug(f"[buy_check] {s}: {r}")
        if ok:
            allowed.append(s)

    to_open = desired_new_positions(state, allowed, cfg)
    if not to_open and not cfg.get("rotation_enabled", False):
        return

    if to_open and cfg["use_all_balance"]:
        best = to_open[0]
        budget = estimate_buy_budget(client, cfg, 1)
        if budget > 0:
            executed_qty, entry = buy_with_preference(client, best, budget, cfg)
            if executed_qty > 0 and entry:
                state["positions"][best] = {"qty": executed_qty, "entry": entry, "opened_at": iso_utc(), "source": "all_balance"}
                save_positions(state); log_trade("BUY", best, executed_qty, avg=entry, quote=None, reason="all_balance")
                logging.info(f"OPEN {best} qty={executed_qty} entry~={entry}")
        return

    if to_open and cfg["auto_split"]:
        slots_free = max(cfg["max_positions"] - len(state["positions"]), 0)
        slices = min(cfg["split_slices"], slots_free, len(to_open))
        if slices > 0:
            per_slice = estimate_buy_budget(client, cfg, slices)
            if per_slice >= cfg["min_slice_usd"]:
                for sym in to_open[:slices]:
                    executed_qty, entry = buy_with_preference(client, sym, per_slice, cfg)
                    if executed_qty > 0 and entry:
                        state["positions"][sym] = {"qty": executed_qty, "entry": entry, "opened_at": iso_utc(), "source": "auto_split"}
                        save_positions(state); log_trade("BUY", sym, executed_qty, avg=entry, quote=None, reason="auto_split")
                        logging.info(f"OPEN {sym} qty={executed_qty} entry~={entry}")
                        time.sleep(0.25)
                return

    if to_open and not cfg["auto_split"] and not cfg["use_all_balance"]:
        slots_free = max(cfg["max_positions"] - len(state["positions"]), 0)
        for sym in to_open[:slots_free]:
            executed_qty, entry = buy_with_preference(client, sym, cfg["alloc_quote"], cfg)
            if executed_qty > 0 and entry:
                state["positions"][sym] = {"qty": executed_qty, "entry": entry, "opened_at": iso_utc(), "source": "fixed_alloc"}
                save_positions(state); log_trade("BUY", sym, executed_qty, avg=entry, quote=None, reason="fixed_alloc")
                logging.info(f"OPEN {sym} qty={executed_qty} entry~={entry}")
            time.sleep(0.3)

    if cfg.get("rotation_enabled", False):
        open_syms = list(state["positions"].keys())
        room = max(cfg["max_positions"] - len(open_syms), 0)
        if room <= 0 and allowed:
            fresh = [s for s in allowed if s not in open_syms]
            if not fresh:
                return
            best_new = max(fresh, key=lambda s: score_symbol(client, s))
            score_new = score_symbol(client, best_new)
            worst_open = min(open_syms, key=lambda s: score_symbol(client, s))
            score_old  = score_symbol(client, worst_open)
            edge_req  = cfg.get("rotation_edge_pct", 0.20)
            costs_pct = (2*(cfg["fee_bps"] + cfg["slippage_bps"])) / 10_000.0
            cond_edge = (score_new >= score_old * (1.0 + edge_req))
            cond_cost = (score_new - score_old) >= (abs(score_old) * costs_pct * 1.5)
            if score_new > 0 and cond_edge and cond_cost:
                logging.info(f"ROTATION: {worst_open} -> {best_new} | edge={edge_req*100:.1f}% costs~{costs_pct*100:.2f}%")
                qty_old = float(state["positions"][worst_open]["qty"])
                if not is_sellable(client, worst_open, qty_old):
                    logging.info(f"ROTATION: {worst_open} dust/not sellable -> drop from state")
                    state["positions"].pop(worst_open, None); save_positions(state); return
                ex_qty, avg, quote_recv = sell_market_all(client, worst_open, qty_old)
                log_trade("SELL", worst_open, ex_qty, avg=avg, quote=quote_recv, reason="ROTATE_OUT")
                state["positions"].pop(worst_open, None); save_positions(state)
                buy_amt = quote_recv or get_account_bal(client, cfg["quote_asset"])[0]
                if buy_amt > 0:
                    ex_qty_new, entry_new = buy_with_preference(client, best_new, buy_amt, cfg)
                    if ex_qty_new > 0 and entry_new:
                        state["positions"][best_new] = {"qty": ex_qty_new, "entry": entry_new, "opened_at": iso_utc(), "source": "rotation"}
                        save_positions(state); log_trade("BUY", best_new, ex_qty_new, avg=entry_new, quote=None, reason="rotation")
                        logging.info(f"ROTATION OPEN {best_new} qty={ex_qty_new} entry~={entry_new}")

# ---------- Portfolio & monitoring ----------
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

def monitor_and_close(client: SpotClient, state: Dict, cfg) -> bool:
    loss_triggered = False
    to_delete = []
    for sym, pos in list(state["positions"].items()):
        try:
            bid, ask = book(client, sym)
            last_sellable = bid if bid else ((bid or 0) + (ask or 0)) / 2.0
            entry = float(pos["entry"])
            if entry <= 0 or not last_sellable:
                continue
            pnl_pct = (last_sellable / entry) - 1.0

                        # ---------- 1) STOP-LOSS קודם לכל ----------
            sl_pct = float(cfg.get("optional_sl_pct", 0.0) or 0.0)
            if entry <= 0 or not last_sellable:
                logging.debug(f"{sym} skip: entry={entry} last={last_sellable} (no data)")
            else:
                logging.debug(
                    f"SL-CHECK {sym}: qty={pos['qty']} entry={entry:.8f} bid={last_sellable:.8f} "
                    f"pnl={pnl_pct*100:.3f}% sl={-sl_pct*100:.2f}%"
                )

            if sl_pct > 0 and pnl_pct <= -sl_pct:
                qty = float(pos["qty"])
                logging.info(f"SL TRIGGER {sym}: pnl={pnl_pct*100:.2f}% ≤ -{sl_pct*100:.2f}% → SELL (SL)")
                try:
                    _, avg, quote_recv = try_sell_with_retries(client, sym, qty, reason="SL", cfg=cfg)
                    log_trade("SELL", sym, qty, avg=avg, quote=quote_recv, reason="SL")
                    to_delete.append(sym)
                    continue
                except ValueError as ve:
                    if str(ve) == "dust":
                        dp = (cfg.get("dust_policy") or "keep").lower()
                        logging.warning(f"{sym} SL blocked by dust; policy={dp}")
                        if dp == "drop":
                            log_trade("SELL", sym, 0.0, avg=None, quote=0.0, reason="DUST_DROP_SL")
                            to_delete.append(sym)
                            continue
                    else:
                        logging.exception(f"SL sell failed for {sym}: {ve}")

            # ---------- 1b) FAILSAFE (קשיח) ----------
            fail_pct = float(cfg.get("failsafe_sl_pct", 0.0) or 0.0)
            if fail_pct > 0 and pnl_pct <= -fail_pct:
                qty = float(pos["qty"])
                logging.warning(f"FAILSAFE SL {sym}: pnl={pnl_pct*100:.2f}% ≤ -{fail_pct*100:.2f}% → SELL (FAILSAFE)")
                try:
                    _, avg, quote_recv = try_sell_with_retries(client, sym, qty, reason="FAILSAFE", cfg=cfg)
                    log_trade("SELL", sym, qty, avg=avg, quote=quote_recv, reason="FAILSAFE")
                    to_delete.append(sym)
                    continue
                except ValueError as ve:
                    if str(ve) == "dust":
                        dp = (cfg.get("dust_policy") or "keep").lower()
                        logging.warning(f"{sym} FAILSAFE blocked by dust; policy={dp}")
                        if dp == "drop":
                            log_trade("SELL", sym, 0.0, avg=None, quote=0.0, reason="DUST_DROP_FAILSAFE")
                            to_delete.append(sym)
                            continue
                    else:
                        logging.exception(f"FAILSAFE sell failed for {sym}: {ve}")
    

            trail_pct = float(cfg.get("trailing_stop_pct", 0.0) or 0.0)
            arm_pct   = float(cfg.get("trailing_arm_pct", 0.0) or 0.0)
            if trail_pct > 0 and arm_pct > 0 and pnl_pct >= arm_pct:
                if not pos.get("trail_active"):
                    pos["trail_active"] = True; pos["trail_peak"] = last_sellable; save_positions(state)
                else:
                    if last_sellable > float(pos.get("trail_peak", last_sellable)):
                        pos["trail_peak"] = last_sellable; save_positions(state)
                    peak = float(pos.get("trail_peak", last_sellable))
                    if peak > 0 and (peak - last_sellable) / peak >= trail_pct:
                        qty = float(pos["qty"])
                        ex, avg, quote_recv = sell_market_all(client, sym, qty)
                        if ex <= 0 and quote_recv <= 0:
                            logging.info(f"{sym}: TRAIL dust skip -> drop from state")
                        log_trade("SELL", sym, ex, avg=avg, quote=quote_recv, reason="TRAIL")
                        to_delete.append(sym); continue

            tp_pct = float(cfg.get("target_tp_pct", 0.0) or 0.0)
            if tp_pct > 0 and pnl_pct >= tp_pct:
                qty = float(pos["qty"])
                ex, avg, quote_recv = sell_market_all(client, sym, qty)
                if ex <= 0 and quote_recv <= 0:
                    logging.info(f"{sym}: TP dust skip -> drop from state")
                log_trade("SELL", sym, ex, avg=avg, quote=quote_recv, reason="TP")
                to_delete.append(sym); continue

            logging.debug(f"MON {sym}: pnl={pnl_pct*100:.2f}% entry={entry} bid={last_sellable}")

        except Exception as e:
            logging.exception(f"monitor failed for {sym}: {e}")
        time.sleep(0.2)

    for sym in to_delete:
        state["positions"].pop(sym, None)
    if to_delete:
        save_positions(state)
    return loss_triggered

# ---------- Time helpers ----------
def is_eod(cfg) -> bool:
    hh, mm = map(int, cfg["eod_hhmm"].split(":"))
    n = now_local()
    return (n.hour, n.minute) >= (hh, mm)

def parse_iso(ts: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(ts.replace("Z","+00:00"))
    except Exception:
        return None

def eod_liquidate_all(client: SpotClient, state: Dict, cfg):
    if not state["positions"]:
        return
    logging.info("EOD: closing positions to QUOTE (skip very fresh positions)")
    for sym, pos in list(state["positions"].items()):
        try:
            opened_at = parse_iso(pos.get("opened_at",""))
            if opened_at:
                age_min = max(0, (now_local() - opened_at.astimezone(IL_TZ)).total_seconds()/60.0)
                if age_min < cfg.get("eod_skip_min", 90):
                    logging.info(f"EOD skip {sym}: age {age_min:.1f}m < {cfg.get('eod_skip_min',90)}m")
                    continue

            qty = float(pos["qty"])
            if not is_sellable(client, sym, qty):
                logging.info(f"EOD {sym}: dust/not sellable -> drop from state")
                state["positions"].pop(sym, None); save_positions(state); continue

            ex, avg, recv = sell_market_all(client, sym, qty)
            if ex <= 0 or recv <= 0:
                log_trade("SELL", sym, 0.0, avg=None, quote=0.0, reason="EOD_DUST_SKIP")
            else:
                log_trade("SELL", sym, ex, avg=avg, quote=recv, reason="EOD")
            state["positions"].pop(sym, None); save_positions(state)
        except Exception as e:
            logging.exception(f"EOD sell failed for {sym}: {e}")
        time.sleep(0.25)
# ===================== MOB (Momentum+Breakout) Patch =====================
# Drop-in helpers + strategy logic (Spot). No pandas, only stdlib.

import math, time, statistics
from datetime import datetime, timezone

def _as_float(x, default=0.0):
    try:
        if x is None: return float(default)
        s = str(x).strip()
        if s == "": return float(default)
        return float(s)
    except Exception:
        return float(default)

def _as_int(x, default=0):
    try:
        if x is None: return int(default)
        s = str(x).strip()
        if s == "": return int(default)
        return int(s)
    except Exception:
        return int(default)

def _as_bool(x, default=False):
    if x is None: return default
    s = str(x).strip().lower()
    if s in ("1","true","yes","y","on"): return True
    if s in ("0","false","no","n","off"): return False
    return default

# ---- Safe-extend your existing load_env() to include new keys ----
# Call this function *inside* your load_env() before returning, e.g.:
# cfg.update(load_env_mobo_extras(os.environ))
def load_env_mobo_extras(env: dict):
    ema_stack = (env.get("EMA_STACK") or "20,50,100").split(",")
    ema_stack = [ _as_int(v, 20) for v in ema_stack ]
    rs_looks = (env.get("RS_LOOKBACKS") or "1h,4h").split(",")
    return {
        "strategy": (env.get("STRATEGY") or "LEGACY_5PCT").upper(),
        "min_vol_24h": _as_float(env.get("MIN_24H_QUOTE_VOL"), 50_000_000),
        "max_spread_bps": _as_float(env.get("MAX_SPREAD_BPS"), 10),
        "min_depth10bps": _as_float(env.get("MIN_DEPTH_10BPS_USD"), 200_000),
        "use_rs_filter": _as_bool(env.get("USE_RS_FILTER"), True),
        "rs_lookbacks": rs_looks,

        "breakout_mult": _as_float(env.get("ENTRY_BREAKOUT_MULT"), 1.002),
        "entry_vol_mult": _as_float(env.get("ENTRY_VOL_MULT"), 2.0),
        "ema_stack": ema_stack,
        "momentum_ret_15m": _as_float(env.get("MOMENTUM_RET_15M"), 0.03),
        "ob_imb_min": _as_float(env.get("OB_IMBALANCE_MIN"), 0.60),
        "event_vol_z": _as_float(env.get("EVENT_VOL_Z"), 3.0),

        "risk_per_trade_pct": _as_float(env.get("RISK_PER_TRADE_PCT"), 0.75),
        "cap_per_asset_pct": _as_float(env.get("CAP_PER_ASSET_PCT"), 10.0),
        "max_positions": _as_int(env.get("MAX_OPEN_POSITIONS"), 3),
        "max_time_hours": _as_float(env.get("MAX_TIME_HOURS"), 12),
        "tp1_pct": _as_float(env.get("TP1_PCT"), 0.05),
        "tp1_frac": _as_float(env.get("TP1_FRACTION"), 0.50),
        "tp2_pct": _as_float(env.get("TP2_PCT"), 0.12),
        "sl_min_pct": _as_float(env.get("SL_MIN_PCT"), 0.04),
        "atr_mult_sl": _as_float(env.get("ATR_MULT_SL"), 2.0),

        "daily_stop_R": _as_float(env.get("DAILY_STOP_R"), -2.0),
        "daily_stop_eq_pct": _as_float(env.get("DAILY_STOP_EQUITY_PCT"), -0.03),
        "kill_slip_bps": _as_float(env.get("KILL_SLIPPAGE_BPS"), 40.0),
        "kill_latency_ms": _as_float(env.get("KILL_LATENCY_MS"), 500.0),
    }

# ---- Data helpers ----
def get_klines(client, symbol, interval="5m", limit=150):
    r = client.klines(symbol, interval=interval, limit=limit)
    out = []
    for k in r:
        out.append({
            "open_time": int(k[0]),
            "open": _as_float(k[1]),
            "high": _as_float(k[2]),
            "low": _as_float(k[3]),
            "close": _as_float(k[4]),
            "volume": _as_float(k[5]),
        })
    return out

def ema_last(values, period):
    # simple EMA, return last; if not enough data → SMA fallback
    k = 2/(period+1)
    ema = None
    for v in values:
        ema = (v if ema is None else (v*k + ema*(1-k)))
    return float(ema if ema is not None else (sum(values)/max(1,len(values))))

def vol_zscore_20(klines):
    vols = [c["volume"] for c in klines[-20:]]
    if len(vols) < 2: return 0.0
    m = statistics.mean(vols[:-1])
    sd = statistics.pstdev(vols[:-1]) or 1e-9
    z = (vols[-1]-m)/sd
    return float(z)

def atr_pct(klines, window=14):
    trs = []
    for i in range(1, len(klines)):
        h = klines[i]["high"]; l = klines[i]["low"]; pc = klines[i-1]["close"]
        tr = max(h-l, abs(h-pc), abs(l-pc))
        trs.append(tr)
    if len(trs) < window+1: window = max(2, min(window, len(trs)))
    atr = statistics.mean(trs[-window:]) if trs else 0.0
    close = klines[-1]["close"]
    return (atr/close) if close>0 else 0.0

def ret_pct_nbars(klines, bars):
    if len(klines) <= bars: return 0.0
    c0 = klines[-bars-1]["close"]; c1 = klines[-1]["close"]
    return (c1/c0 - 1.0) if c0>0 else 0.0

def highest_high(klines, lookback=20):
    if len(klines) < lookback: lookback = len(klines)
    return max(c["high"] for c in klines[-lookback:]) if lookback>0 else 0.0

def orderbook_metrics(client, symbol, bps=10):
    ob = client.depth(symbol=symbol, limit=500)
    best_bid = _as_float(ob["bids"][0][0]); best_ask = _as_float(ob["asks"][0][0])
    mid = 0.5*(best_bid+best_ask)
    band = mid * (bps/10_000)
    def side_sum(levels, is_bid):
        s = 0.0
        for px, qty in (( _as_float(p), _as_float(q) ) for p,q in levels):
            if is_bid and px >= mid - band: s += px*qty
            if not is_bid and px <= mid + band: s += px*qty
        return s
    bid_usd = side_sum(ob["bids"], True)
    ask_usd = side_sum(ob["asks"], False)
    imb = bid_usd / max(1e-9, (bid_usd+ask_usd))
    spread_bps = ( (best_ask - best_bid) / mid ) * 10_000 if mid>0 else 99999
    return {"mid": mid, "spread_bps": spread_bps, "depth10_bid": bid_usd, "depth10_ask": ask_usd, "imb": imb}

def ticker_24h_quote_volume(client, symbol):
    t = client.ticker_24hr(symbol)
    # quoteVolume key exists on Binance; testnet too
    return _as_float(t.get("quoteVolume"), 0.0)

def rs_ok_vs_btc_eth(client, symbol, lookbacks=("1h","4h")):
    def last_ret(sym, interval):
        k = get_klines(client, sym, interval=interval, limit=5)
        return ret_pct_nbars(k, 1)  # שינוי ברבעון האחרון של הטיים-פריים
    try:
        r_asset = [ last_ret(symbol, tf) for tf in lookbacks ]
        r_btc   = [ last_ret("BTCUSDT", tf) for tf in lookbacks ]
        r_eth   = [ last_ret("ETHUSDT", tf) for tf in lookbacks ]
        for a,b,e in zip(r_asset, r_btc, r_eth):
            if not (a > max(b,e)):  # חייב להיות חזק משניהם
                return False
        return True
    except Exception:
        return False

# ---- Filters & signals ----
def mobo_filters_ok(client, symbol, cfg):
    vol_ok = ticker_24h_quote_volume(client, symbol) >= cfg["min_vol_24h"]
    obm = orderbook_metrics(client, symbol, bps=10)
    spread_ok = obm["spread_bps"] <= cfg["max_spread_bps"]
    depth_ok = (obm["depth10_bid"] >= cfg["min_depth10bps"]) and (obm["depth10_ask"] >= cfg["min_depth10bps"])
    rs_ok = (not cfg["use_rs_filter"]) or rs_ok_vs_btc_eth(client, symbol, tuple(cfg["rs_lookbacks"]))
    if not vol_ok: logging.debug(f"[MOB] {symbol}: FAIL vol24")
    if not spread_ok: logging.debug(f"[MOB] {symbol}: FAIL spread {obm['spread_bps']:.2f}bps")
    if not depth_ok: logging.debug(f"[MOB] {symbol}: FAIL depth10bps bid={obm['depth10_bid']:.0f} ask={obm['depth10_ask']:.0f}")
    if cfg["use_rs_filter"] and not rs_ok: logging.debug(f"[MOB] {symbol}: FAIL RS vs BTC/ETH")
    return vol_ok and spread_ok and depth_ok and rs_ok, obm

def mobo_entry_signal(client, symbol, cfg):
    k5 = get_klines(client, symbol, interval="5m", limit=120)
    if len(k5) < 40: return False, {}
    ema20 = ema_last([c["close"] for c in k5], cfg["ema_stack"][0])
    ema50 = ema_last([c["close"] for c in k5], cfg["ema_stack"][1])
    ema100= ema_last([c["close"] for c in k5], cfg["ema_stack"][2])
    ret15 = ret_pct_nbars(k5, 3)  # 3×5m = 15m
    zvol  = vol_zscore_20(k5)
    hh20  = highest_high(k5, 20)
    close = k5[-1]["close"]
    breakout = (close > hh20 * cfg["breakout_mult"]) and (k5[-1]["volume"] >= cfg["entry_vol_mult"] * (sum(c["volume"] for c in k5[-20:])/20.0))
    momentum = (ema20>ema50>ema100) and (ret15 >= cfg["momentum_ret_15m"])
    # orderbook imbalance
    obm = orderbook_metrics(client, symbol, bps=10)
    momentum = momentum and (obm["imb"] >= cfg["ob_imb_min"])
    event_sig = (zvol > cfg["event_vol_z"])
    signal = breakout or momentum or event_sig
    if not signal:
        logging.debug(f"[MOB] {symbol}: no-entry | bo={breakout} mo={momentum} ev={event_sig} z={zvol:.2f} ret15={ret15:.4f} imb={obm['imb']:.2f}")
    return signal, {"ema20":ema20,"ema50":ema50,"ema100":ema100,"ret15":ret15,"zvol":zvol,"hh20":hh20,"close":close,"obm":obm}

def mobo_atr_sl_tp(cfg, klines5m):
    apct = atr_pct(klines5m, 14)
    sl_pct = min(cfg["sl_min_pct"], cfg["atr_mult_sl"]*apct)
    tp1_pct = cfg["tp1_pct"]; tp2_pct = cfg["tp2_pct"]
    return apct, sl_pct, tp1_pct, tp2_pct

def mobo_position_size_quote(equity_quote, atr_percent, cfg):
    # risk_per_trade_pct supports "0.75" (percent) or "0.0075" (fraction)
    r = cfg["risk_per_trade_pct"]
    risk_frac = (r/100.0) if r>1 else r
    eff = max(0.015, 1.5*atr_percent)  # safety floor
    dollars = (equity_quote * risk_frac) / eff if eff>0 else 0.0
    cap = (cfg["cap_per_asset_pct"]/100.0) * equity_quote
    return max(0.0, min(dollars, cap))

# ---- Execution (simple & robust: market only + bot-managed TP/SL) ----
def _symbol_filters(client, symbol):
    s = client.exchange_info(symbol=symbol)["symbols"][0]
    lot  = next(f for f in s["filters"] if f["filterType"] == "LOT_SIZE")
    price= next(f for f in s["filters"] if f["filterType"] == "PRICE_FILTER")
    notional = next((f for f in s["filters"] if f["filterType"] in ("NOTIONAL","MIN_NOTIONAL")), None)
    return { "stepSize": _as_float(lot["stepSize"]), "minQty": _as_float(lot.get("minQty",0.0)),
             "tickSize": _as_float(price["tickSize"]), "minNotional": _as_float(notional["minNotional"]) if notional else None }

def _round_down(x, step):
    if step<=0: return x
    return math.floor(x/step)*step

def mobo_open_position(client, symbol, quote_amt, cfg):
    # market buy all (פשוט ועמיד; אם תרצה 60/40 השתמש ב-limit maker לחלק)
    ob = orderbook_metrics(client, symbol)
    price = ob["mid"] if ob["mid"]>0 else _as_float(client.ticker_price(symbol).get("price"), 0.0)
    flt = _symbol_filters(client, symbol)
    qty = _round_down(quote_amt / max(price,1e-9), flt["stepSize"])
    if flt["minQty"] and qty < flt["minQty"]:
        raise ValueError("qty below minQty")
    ord = client.new_order(symbol=symbol, side="BUY", type="MARKET", quantity=str(qty))
    exe_qty = _as_float(ord.get("executedQty"), 0.0)
    cqq = _as_float(ord.get("cummulativeQuoteQty"), 0.0)
    avg = (cqq/exe_qty) if exe_qty>0 else price
    return exe_qty, avg, cqq

def mobo_close_market(client, symbol, qty):
    if qty<=0: return 0.0, 0.0
    ord = client.new_order(symbol=symbol, side="SELL", type="MARKET", quantity=str(qty))
    exe_qty = _as_float(ord.get("executedQty"), 0.0)
    cqq = _as_float(ord.get("cummulativeQuoteQty"), 0.0)
    avg = (cqq/exe_qty) if exe_qty>0 else 0.0
    return exe_qty, avg, cqq

# ---- State helpers (positions.json compatible) ----
def mobo_make_state(sym, qty, entry, atrp, sl_pct, tp1, tp2, now_ts):
    return {
        "strategy": "MOB",
        "qty": qty,
        "entry": entry,
        "opened_at": now_ts,
        "atr_pct": atrp,
        "sl_pct": sl_pct,
        "tp1_pct": tp1,
        "tp2_pct": tp2,
        "tp1_done": False,
        "trail_active": False,
        "trail_anchor": entry,   # for BE & trailing
        "last_trail_up": 0.0,
    }

def mobo_manage_one(client, symbol, pstate, cfg):
    # returns (updated_state, maybe_closed, sold_quote_pnl)
    # prices
    ob = orderbook_metrics(client, symbol)
    price = ob["mid"]
    entry = pstate["entry"]
    qty = pstate["qty"]
    # SL logic
    be_price = pstate["trail_anchor"]
    sl_pct = pstate["sl_pct"]
    stop_price = (be_price*(1.0 - sl_pct))
    # TP/TP1/Trail
    tp1_price = entry*(1.0 + pstate["tp1_pct"])
    tp2_price = entry*(1.0 + pstate["tp2_pct"])

    quote_pnl = 0.0
    # TP1: take half & move to BE
    if (not pstate["tp1_done"]) and price >= tp1_price:
        sell_qty = qty * cfg["tp1_frac"]
        exq, avg, cqq = mobo_close_market(client, symbol, sell_qty)
        if exq>0:
            pstate["qty"] = qty - exq
            pstate["tp1_done"] = True
            pstate["trail_active"] = True
            pstate["trail_anchor"] = max(entry, price)  # move to BE+
            quote_pnl += (cqq - exq*entry)

    # Trailing: for every +3% since last_trail_up, raise BE anchor by 1.5%
    if pstate["trail_active"]:
        move_from_anchor = (price / pstate["trail_anchor"] - 1.0) if pstate["trail_anchor"]>0 else 0.0
        if move_from_anchor >= 0.03 - 1e-6:
            pstate["trail_anchor"] = price * (1.0 - 0.015)  # lock 1.5% below
            pstate["last_trail_up"] = price

    # Hard SL check
    # we enforce SL on remaining qty
    if pstate["qty"] > 0 and price <= stop_price:
        exq, avg, cqq = mobo_close_market(client, symbol, pstate["qty"])
        if exq>0:
            quote_pnl += (cqq - exq*entry)
            pstate["qty"] = 0.0
            return pstate, True, quote_pnl

    # TP2: close remainder if reached
    if pstate["qty"] > 0 and price >= tp2_price:
        exq, avg, cqq = mobo_close_market(client, symbol, pstate["qty"])
        if exq>0:
            quote_pnl += (cqq - exq*entry)
            pstate["qty"] = 0.0
            return pstate, True, quote_pnl

    # TTL
    try:
        opened = datetime.fromisoformat(pstate["opened_at"].replace("Z","").replace("+00:00","")).replace(tzinfo=timezone.utc)
    except Exception:
        opened = datetime.now(timezone.utc)
    hours_open = (datetime.now(timezone.utc) - opened).total_seconds()/3600.0
    if pstate["qty"]>0 and hours_open >= cfg["max_time_hours"]:
        exq, avg, cqq = mobo_close_market(client, symbol, pstate["qty"])
        if exq>0:
            quote_pnl += (cqq - exq*entry)
            pstate["qty"] = 0.0
            return pstate, True, quote_pnl

    return pstate, False, quote_pnl

# ---- One-iteration orchestrator ----
def mobo_try_enter_for_symbol(client, cfg, symbol, equity_quote, positions_map, log_trade_fn):
    ok, obm = mobo_filters_ok(client, symbol, cfg)
    if not ok: return False
    sig, info = mobo_entry_signal(client, symbol, cfg)
    if not sig: return False

    k5 = get_klines(client, symbol, interval="5m", limit=60)
    apct, sl_pct, tp1, tp2 = mobo_atr_sl_tp(cfg, k5)
    quote_amt = mobo_position_size_quote(equity_quote, apct, cfg)
    if quote_amt <= 0: 
        logging.debug(f"[MOB] {symbol}: sizing=0 (equity={equity_quote:.2f}, atr%={apct:.4f})")
        return False

    try:
        qty, avg, cq = mobo_open_position(client, symbol, quote_amt, cfg)
        if qty <= 0: 
            logging.info(f"[MOB] {symbol}: market BUY executed 0")
            return False
        now_ts = datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()
        state = mobo_make_state(symbol, qty, avg, apct, sl_pct, tp1, tp2, now_ts)
        positions_map[symbol] = state
        # trade log
        if log_trade_fn:
            log_trade_fn("BUY", symbol, qty, avg=avg, quote=cq, reason="MOB-entry")
        logging.info(f"[MOB] BUY {symbol} qty={qty:.8f} avg={avg:.8f} tp1={tp1*100:.1f}% tp2={tp2*100:.1f}% sl={sl_pct*100:.1f}%")
        return True
    except Exception as e:
        logging.error(f"[MOB] entry failed for {symbol}: {e}")
        return False

def mobo_manage_all(client, cfg, positions_map, log_trade_fn):
    realized = 0.0
    to_delete = []
    for sym, ps in positions_map.items():
        if ps.get("strategy") != "MOB": 
            continue
        new_ps, closed, q_pnl = mobo_manage_one(client, sym, ps, cfg)
        positions_map[sym] = new_ps
        if q_pnl != 0 and log_trade_fn:
            # we log SELL chunks inside manage_one
            pass
        realized += q_pnl
        if closed:
            if log_trade_fn and q_pnl != 0:
                side = "SELL"
                qty_sold = ps["qty"]  # after close became 0; for log we can’t recover exact
                log_trade_fn(side, sym, qty=None, avg=None, quote=None, reason="MOB-exit")
            to_delete.append(sym)
    for s in to_delete:
        positions_map.pop(s, None)
    return realized
# ===================== End MOB Patch =====================

# ---------- State ----------
def load_positions() -> Dict:
    return read_json(STATE_FILE, {"positions": {}})

def save_positions(state: Dict):
    write_json(STATE_FILE, state)

# ---------- Main ----------
def main():
    if STOP_FLAG.exists():
        try: STOP_FLAG.unlink()
        except: pass

    cfg = load_env()
    client = client_from_cfg(cfg)
    state = load_positions()

    if cfg.get("consolidate_at_start", False) and state["positions"]:
        logging.info("Consolidate-at-start: liquidating all current positions to QUOTE")
        eod_liquidate_all(client, state, cfg)

    logging.info(
        f"Starting bot | STRATEGY={cfg.get('strategy','LEGACY_5PCT')} "
        f"QUOTE={cfg['quote_asset']} target={cfg['target_tp_pct']*100:.1f}% "
        f"max_pos={cfg['max_positions']} maker={cfg.get('use_maker',True)} auto_split={cfg.get('auto_split',True)} "
        f"reserve={cfg.get('balance_reserve_pct',0.0)*100:.1f}%"
    )

    start_value = portfolio_value_quote(client, state, cfg)
    logging.info(f"portfolio start value ~= {start_value} {cfg['quote_asset']}")

    last_scan = 0.0
    wait_until_ts = 0.0

    while True:
        if STOP_FLAG.exists():
            logging.info('stop.flag detected -> exiting main loop')
            break
        try:
            # hot reload
            cfg, client, reloaded = reload_cfg_if_changed(cfg, client)
            if reloaded:
                logging.info("Hot-reload: applied new .env settings")

            now_ts = time.time()
            sig = regime_signal(client, cfg)   # +1 bull, 0 neutral, -1 bear

            if sig < 0:
                if state["positions"]:
                    logging.info("REGIME BEAR: liquidate all and enter cooldown")
                    eod_liquidate_all(client, state, cfg)
                    start_value = portfolio_value_quote(client, state, cfg)
                wait_until_ts = max(wait_until_ts, now_ts + cfg.get("regime_cooldown_min", 90)*60)

            # --- open logic (by strategy) ---
            if now_ts - last_scan > cfg["scan_interval"]:
                can_buy = (sig > 0) and (now_ts >= wait_until_ts)
                strat = cfg.get("strategy","LEGACY_5PCT").upper()

                if can_buy:
                    if strat == "MOMENTUM_BREAKOUT":
                        # סריקה מהירה: טופ-25 לפי נפח 24h לאותו QUOTE, סדר לפי נפח
                        syms = spot_symbols_for_quote(client, cfg["quote_asset"], cfg["blacklist_keywords"])
                        t24 = { t["symbol"]: float(t.get("quoteVolume",0.0)) for t in client.ticker_24hr() }
                        syms = [s for s in syms if t24.get(s,0.0) >= cfg.get("min_24h_quote_vol",50_000_000)]
                        syms.sort(key=lambda s: -t24.get(s,0.0))
                        # פתח עד שמגיעים לתקרה
                        for s in syms[:25]:
                            if s in state["positions"]: 
                                continue
                            if len(state["positions"]) >= cfg["max_positions"]:
                                break
                            equity_quote = portfolio_value_quote(client, state, cfg)
                            free_q, _ = get_account_bal(client, cfg["quote_asset"])
                            # נפתח רק אם יש גם מזומן זמין
                            if free_q <= 10: 
                                break
                            opened = mobo_try_enter_for_symbol(
                                client, cfg, s, equity_quote=min(equity_quote, free_q*10), 
                                positions_map=state["positions"], log_trade_fn=log_trade
                            )
                            if opened:
                                save_positions(state)
                                time.sleep(0.25)
                    else:
                        # LEGACY
                        maybe_open_positions(client, state, cfg)
                else:
                    logging.info("BUY BLOCKED: regime not bullish or cooldown active")

                last_scan = now_ts

            # --- manage logic (by strategy) ---
            strat = cfg.get("strategy","LEGACY_5PCT").upper()
            if state["positions"]:
                if strat == "MOMENTUM_BREAKOUT":
                    realized = mobo_manage_all(client, cfg, state["positions"], log_trade)
                    if realized != 0:
                        save_positions(state)
                else:
                    loss = monitor_and_close(client, state, cfg)
                    if loss:
                        wait_until_ts = max(wait_until_ts, time.time() + cfg.get("loss_cooldown_min", 20)*60)

            # --- portfolio guards ---
            cur_val = portfolio_value_quote(client, state, cfg)
            pnl_port = (cur_val / start_value) - 1.0 if start_value > 0 else 0.0

            if cfg.get("portfolio_target_pct", 0.0) > 0 and pnl_port >= cfg["portfolio_target_pct"]:
                logging.info(f"PORTFOLIO TARGET HIT: {pnl_port*100:.2f}% -> liquidate all to QUOTE")
                eod_liquidate_all(client, state, cfg)
                start_value = portfolio_value_quote(client, state, cfg)
                wait_until_ts = max(wait_until_ts, time.time() + 15*60)
            elif cfg.get("portfolio_stop_pct", 0.0) > 0 and pnl_port <= -cfg["portfolio_stop_pct"]:
                logging.info(f"PORTFOLIO STOP HIT: {pnl_port*100:.2f}% -> liquidate all to QUOTE")
                eod_liquidate_all(client, state, cfg)
                start_value = portfolio_value_quote(client, state, cfg)
                wait_until_ts = max(wait_until_ts, time.time() + cfg.get("regime_cooldown_min", 90)*60)

            # --- EOD ---
            if is_eod(cfg):
                eod_liquidate_all(client, state, cfg)
                start_value = portfolio_value_quote(client, state, cfg)
                time.sleep(65)
                last_scan = 0.0

        except Exception as e:
            logging.exception(f"main loop error: {e}")
            time.sleep(2)

        time.sleep(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("bye")