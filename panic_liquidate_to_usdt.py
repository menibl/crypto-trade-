# ------------- panic_liquidate_to_usdt.py -------------
import os, math, time, json, logging
from pathlib import Path
from typing import Any, Dict, Tuple, List
from datetime import datetime, timezone

from dotenv import load_dotenv
from binance.spot import Spot as SpotClient
from binance.error import ClientError

APP_DIR     = Path(__file__).parent
TRADES_FILE = APP_DIR / "trades.json"

TESTNET_URL = "https://testnet.binance.vision"
PROD_URL    = "https://api.binance.com"

# ---------- Logging ----------
def setup_logging():
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=getattr(logging, level, logging.INFO),
                        format="%(asctime)s %(levelname)s %(message)s")

# ---------- JSON helpers ----------
def read_json(path: Path, default: Any):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except Exception:
        return default

def write_json(path: Path, obj: Any):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2))

# ---------- Trade logger (same schema as bot_5pct.py) ----------
def log_trade(side: str, symbol: str, qty: float,
              avg: float | None = None, quote: float | None = None, reason: str = ""):
    data = read_json(TRADES_FILE, {"trades": []})
    data["trades"].append({
        "ts": datetime.now(timezone.utc).isoformat(),
        "side": side, "symbol": symbol,
        "qty": qty, "avg": avg, "quote": quote, "reason": reason
    })
    write_json(TRADES_FILE, data)

# ---------- Env / client ----------
def load_env() -> Dict[str, Any]:
    load_dotenv(APP_DIR / ".env")
    api_key    = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")
    env        = (os.getenv("BINANCE_ENV") or "prod").lower()
    base_url   = PROD_URL if env == "prod" else TESTNET_URL
    quote      = (os.getenv("QUOTE_ASSET") or "USDT").upper()
    if not api_key or not api_secret:
        raise RuntimeError("Missing API keys in .env")
    return {"api_key": api_key, "api_secret": api_secret, "base_url": base_url, "quote": quote, "env": env}

def client_from_cfg(cfg) -> SpotClient:
    return SpotClient(api_key=cfg["api_key"], api_secret=cfg["api_secret"], base_url=cfg["base_url"])

# ---------- Exchange helpers ----------
def exchange_info_for(client: SpotClient, symbol: str):
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

def book_mid(client: SpotClient, symbol: str) -> float:
    bk = client.book_ticker(symbol=symbol)
    bid, ask = float(bk["bidPrice"]), float(bk["askPrice"])
    return (bid + ask) / 2.0 if bid and ask else ask or bid

# ---------- Core: sell everything to QUOTE ----------
def list_balances(client: SpotClient) -> Dict[str, Tuple[float, float]]:
    """returns asset -> (free, locked)"""
    out: Dict[str, Tuple[float, float]] = {}
    acct = client.account(recvWindow=5000)
    for b in acct["balances"]:
        free, locked = float(b["free"]), float(b["locked"])
        if free > 0 or locked > 0:
            out[b["asset"]] = (free, locked)
    return out

def sell_all_non_quote_to_quote(client: SpotClient, quote_asset: str, dry_run: bool=False) -> Dict[str, Any]:
    result = {"sold": [], "skipped": [], "errors": []}
    bals = list_balances(client)
    for base, (free, locked) in bals.items():
        if base == quote_asset:
            continue
        qty = free  # locked לא נטפל כאן
        if qty <= 0:
            continue
        symbol = f"{base}{quote_asset}"
        try:
            # ודא שהסימבול קיים/נסחר
            try:
                f = exchange_info_for(client, symbol)
            except Exception:
                result["skipped"].append({"symbol": symbol, "reason": "symbol_not_found"})
                continue

            q = round_down(qty, f["stepSize"])
            if q <= 0 or q < f["minQty"]:
                result["skipped"].append({"symbol": symbol, "qty": qty, "reason": "dust_qty"})
                log_trade("SELL", symbol, 0.0, avg=None, quote=0.0, reason="PANIC_DUST_SKIP")
                continue

            mid = book_mid(client, symbol)
            notional_est = q * mid
            if f["minNotional"] and notional_est < f["minNotional"]:
                result["skipped"].append({"symbol": symbol, "qty": q, "reason": "below_minNotional"})
                log_trade("SELL", symbol, 0.0, avg=None, quote=0.0, reason="PANIC_MIN_NOTIONAL_SKIP")
                continue

            if dry_run:
                result["sold"].append({"symbol": symbol, "qty": q, "quote_est": notional_est, "dry_run": True})
                continue

            # שלח מכירה
            ord = client.new_order(symbol=symbol, side="SELL", type="MARKET",
                                   quantity=str(q), recvWindow=5000)
            recv = float(ord.get("cummulativeQuoteQty", 0.0))
            exe  = float(ord.get("executedQty", 0.0))
            avg  = (recv / exe) if exe > 0 else None

            log_trade("SELL", symbol, exe, avg=avg, quote=recv, reason="PANIC")
            result["sold"].append({"symbol": symbol, "executed_qty": exe, "quote_recv": recv})

        except ClientError as e:
            result["errors"].append({"symbol": symbol, "code": e.error_code, "msg": e.error_message})
            logging.exception(f"panic sell failed for {symbol}: {e}")
        except Exception as e:
            result["errors"].append({"symbol": symbol, "msg": str(e)})
            logging.exception(f"panic sell failed for {symbol}: {e}")
        time.sleep(0.15)
    return result

# ---------- CLI ----------
def main():
    cfg = load_env()
    setup_logging()
    client = client_from_cfg(cfg)
    logging.info(f"PANIC: liquidate everything to {cfg['quote']} (env={cfg['env']})")
    res = sell_all_non_quote_to_quote(client, cfg["quote"], dry_run=False)
    logging.info(f"Done. Summary: sold={len(res['sold'])} skipped={len(res['skipped'])} errors={len(res['errors'])}")
    # הדפסה נעימה בקונסול
    print(json.dumps(res, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
