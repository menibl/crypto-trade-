# -*- coding: utf-8 -*-
"""
Panic liquidation: convert all non-USDT assets to USDT with safety guards.
"""

import os, time, logging, uuid
from decimal import Decimal, ROUND_DOWN
from pathlib import Path
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv
from binance.spot import Spot as SpotClient
from binance.error import ClientError

LOG_FILE = os.getenv("PANIC_LOG_FILE", str(Path(__file__).parent / "panic.log"))

def init_logger() -> logging.Logger:
    logger = logging.getLogger("panic")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = RotatingFileHandler(LOG_FILE, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
    ch = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh.setFormatter(fmt); ch.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(ch)
    return logger

log = init_logger()

def _to_dec(x) -> Decimal:
    return Decimal(str(x))

def build_client() -> SpotClient:
    load_dotenv()
    api_key = os.getenv("BINANCE_API_KEY"); api_sec = os.getenv("BINANCE_API_SECRET")
    if not api_key or not api_sec:
        raise RuntimeError("BINANCE_API_KEY/SECRET missing in environment")
    return SpotClient(key=api_key, secret=api_sec)

def exchange_filters(client: SpotClient, symbol: str):
    info = client.exchange_info(symbol=symbol)["symbols"][0]
    tick = step = None; min_notional = Decimal("0")
    for f in info["filters"]:
        if f["filterType"] == "PRICE_FILTER":
            tick = _to_dec(f["tickSize"])
        elif f["filterType"] == "LOT_SIZE":
            step = _to_dec(f["stepSize"])
        elif f["filterType"] in ("MIN_NOTIONAL", "NOTIONAL"):
            mn = f.get("minNotional") or f.get("notional")
            if mn: min_notional = _to_dec(mn)
    if not tick or not step:
        raise RuntimeError(f"missing filters for {symbol}")
    return tick, step, min_notional

def normalize_qty(qty: Decimal, step: Decimal) -> Decimal:
    return (qty / step).to_integral_value(rounding=ROUND_DOWN) * step

def normalize_price(price: Decimal, tick: Decimal) -> Decimal:
    return (price / tick).to_integral_value(rounding=ROUND_DOWN) * tick

def last_price(client: SpotClient, symbol: str) -> Decimal:
    return _to_dec(client.ticker_price(symbol)["price"])

def sell_market(client: SpotClient, symbol: str, qty: Decimal):
    try:
        cid = f"panic-{uuid.uuid4().hex[:18]}"
        r = client.new_order(symbol=symbol, side="SELL", type="MARKET", quantity=str(qty), newClientOrderId=cid, recvWindow=5000)
        log.info(f"SELL {symbol} MARKET {qty} -> OK")
        return True, r
    except ClientError as e:
        log.error(f"SELL {symbol} error {e.error_code} {e.error_message}")
        return False, {"code": e.error_code, "err": e.error_message}
    except Exception as e:
        log.error(f"SELL {symbol} exception {e}")
        return False, {"err": str(e)}

def main():
    client = build_client()
    acc = client.account()
    balances = acc.get("balances", [])

    # Optional: chunk size to avoid slamming the book
    chunk_usd = _to_dec(os.getenv("PANIC_CHUNK_USD", "1000"))

    for b in balances:
        asset = b["asset"]
        free = _to_dec(b["free"])
        locked = _to_dec(b["locked"])
        total = free + locked

        if asset in ("USDT", "BUSD") or total <= 0:
            continue

        symbol = f"{asset}USDT"
        # skip assets without USDT market
        try:
            tick, step, min_notional = exchange_filters(client, symbol)
        except Exception:
            log.warning(f"skip {asset} (no {symbol} market)")
            continue

        price = last_price(client, symbol)
        if price <= 0:
            log.warning(f"skip {symbol} (price<=0)")
            continue

        qty_left = normalize_qty(total, step)
        while qty_left > 0:
            # chunk by USD notional
            target_qty = qty_left
            if chunk_usd > 0:
                chunk_qty = (chunk_usd / price).quantize(step, rounding=ROUND_DOWN)
                if chunk_qty > 0:
                    target_qty = min(qty_left, chunk_qty)

            # ensure min notional
            notional = target_qty * price
            if notional < min_notional:
                log.info(f"stop {symbol}: below minNotional ({notional}<{min_notional})")
                break

            ok, r = sell_market(client, symbol, target_qty)
            if not ok:
                log.error(f"panic sell failed for {symbol}: {r}")
                break

            qty_left -= target_qty
            time.sleep(0.3)  # gentle pacing

    log.info("Panic liquidation completed")

if __name__ == "__main__":
    main()
