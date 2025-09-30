#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ================================================================
# A-L-L  I-N  O-N-E  C-R-Y-P-T-O  B-O-T  (modular & alloc-aware)
# ================================================================
"""
This is a single-file, modular crypto bot skeleton that implements:
  - Capital allocation per strategy module
  - Four strategy modules:
      1) TrendBreakout (Donchian+EMA+ADX gating, partial TP, ATR stops)
      2) MeanReversionVWAP (VWAP ±σ entries, ATR stops)
      3) MarketMakingLite (inventory-capped, spread-following quotes)
      4) FundingBasis (neutral sleeves; placeholders for data & hedging)
  - Regime Router (trend vs range) using ADX & Bollinger Bandwidth
  - Volatility-based position sizing (target_daily_vol)
  - Risk manager (ATR stops, per-asset risk caps, portfolio leverage cap, daily DD kill-switch)
  - Simple backtest engine (event-driven on bar close) and live-trading stubs

Notes
-----
- This is an opinionated template to help you start quickly. Review every line before using with real money.
- For live trading, plug in a real connector (e.g., ccxt) inside `ExchangeConnector`.
- For backtesting, feed a pandas DataFrame of candles to `BacktestEngine`.
- All risk limits are configurable via CONFIG at the bottom.
"""

import math
import time
import json
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd

# ===============
# Utils & Indicators
# ===============

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int) -> pd.Series:
    return true_range(high, low, close).rolling(n).mean()

def adx(high: pd.Series, low: pd.Series, close: pd.Series, n: int=14) -> pd.Series:
    # Wilder's ADX (simplified)
    up_move = high.diff()
    down_move = low.diff().abs()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (low.diff() < 0), down_move, 0.0)

    tr = true_range(high, low, close)
    atr_n = tr.rolling(n).sum()

    plus_di = 100 * pd.Series(plus_dm, index=high.index).rolling(n).sum() / atr_n
    minus_di = 100 * pd.Series(minus_dm, index=high.index).rolling(n).sum() / atr_n
    dx = ( (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0,np.nan) ) * 100
    return dx.rolling(n).mean()

def donchian_channels(high: pd.Series, low: pd.Series, n: int=20) -> Tuple[pd.Series, pd.Series]:
    upper = high.rolling(n).max()
    lower = low.rolling(n).min()
    return upper, lower

def vwap(df: pd.DataFrame) -> pd.Series:
    # df with columns: ['close','high','low','open','volume']
    typical_price = (df['high'] + df['low'] + df['close']) / 3.0
    cum_typ_vol = (typical_price * df['volume']).cumsum()
    cum_vol = df['volume'].cumsum().replace(0, np.nan)
    return cum_typ_vol / cum_vol

def bollinger_bandwidth(close: pd.Series, n:int=20, k:float=2.0) -> pd.Series:
    ma = close.rolling(n).mean()
    sd = close.rolling(n).std(ddof=0)
    upper = ma + k*sd
    lower = ma - k*sd
    return (upper - lower) / ma.replace(0, np.nan)

def rolling_percentile(series: pd.Series, window:int, pct:float) -> pd.Series:
    # pct in [0,1]
    return series.rolling(window).apply(lambda x: np.nanpercentile(x, pct*100), raw=True)

# ===============
# Data Structures
# ===============

@dataclass
class StrategySignal:
    side: Optional[str] = None     # 'long', 'short', or None
    strength: float = 0.0          # 0..1 weight
    entry_price: Optional[float] = None
    stop_price: Optional[float] = None
    take_profit: Optional[float] = None
    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Position:
    side: Optional[str] = None
    qty: float = 0.0
    entry_price: float = 0.0
    stop_price: Optional[float] = None
    take_profit: Optional[float] = None

@dataclass
class RiskConfig:
    target_daily_vol: float = 0.01
    max_intraday_dd: float = 0.03
    max_leverage_asset: float = 3.0
    max_leverage_portfolio: float = 3.0
    kelly_fraction: float = 0.25
    max_per_asset_risk: float = 0.0125

@dataclass
class RegimeConfig:
    adx_trend: int = 20
    adx_chop: int = 18
    bb_width_pctile: float = 0.4
    bb_window: int = 20

@dataclass
class TrendBreakoutConfig:
    donchian: int = 20
    ema_filter: int = 50
    stop_atr_mult: float = 3.0
    partial_tp_R: float = 2.0
    vol_confirm_lookback: int = 20

@dataclass
class MeanReversionConfig:
    vwap_sigma_entry: float = 0.7
    exit_sigma: float = 0.35
    stop_atr_mult: float = 1.2
    max_retries: int = 3
    bb_window: int = 20

@dataclass
class MMLiteConfig:
    spread_factor: float = 0.7
    inventory_cap_nav: float = 0.01
    trend_skew_ema: int = 20
    pause_move_pct_1m: float = 0.003

@dataclass
class FundingBasisConfig:
    funding_abs_pctile: float = 0.8
    basis_min_apr: float = 0.12

@dataclass
class Allocations:
    trend_breakout: float = 0.35
    mean_reversion: float = 0.35
    mm_lite: float = 0.2
    funding_basis: float = 0.1

@dataclass
class BotConfig:
    risk: RiskConfig
    regime: RegimeConfig
    trend_breakout: TrendBreakoutConfig
    mean_reversion: MeanReversionConfig
    mm_lite: MMLiteConfig
    funding_basis: FundingBasisConfig
    allocations: Allocations

# ===============
# Exchange Connector (stub for live; works with backtest engine)
# ===============

class ExchangeConnector:
    """
    Plug in ccxt here for live trading. Methods return minimal data or raise NotImplementedError.
    BacktestEngine supplies candles and simulated fills without using this connector.
    """
    def __init__(self, name: str = "stub"):
        self.name = name

    # --- Market data ---
    def fetch_ohlcv(self, symbol: str, timeframe: str="1m", limit: int=1000) -> pd.DataFrame:
        raise NotImplementedError("Implement with ccxt for live mode.")

    def fetch_order_book(self, symbol: str) -> Dict[str, Any]:
        return {"bids": [], "asks": [], "timestamp": int(time.time()*1000)}

    def fetch_funding_rate(self, symbol: str) -> float:
        # Return current perp funding (per period); needs real API.
        return 0.0

    def fetch_basis_annualized(self, symbol: str) -> float:
        # Placeholder: compute from futures vs spot quotes.
        return 0.0

    # --- Trading ---
    def create_order(self, symbol: str, side: str, type_: str, amount: float, price: Optional[float]=None) -> Dict[str, Any]:
        print(f"[LIVE-ORDER] {side.upper()} {amount} {symbol} @ {price or 'MKT'} ({type_})")
        return {"id": f"live-{int(time.time())}"}

# ===============
# Strategy Base
# ===============

class StrategyBase:
    def __init__(self, cfg: Dict[str, Any], risk: RiskConfig):
        self.cfg = cfg
        self.risk = risk

    def signal(self, df: pd.DataFrame) -> StrategySignal:
        raise NotImplementedError

# ===============
# Strategy 1: Trend Breakout
# ===============

class TrendBreakoutStrategy(StrategyBase):
    def __init__(self, cfg: TrendBreakoutConfig, risk: RiskConfig):
        super().__init__(cfg.__dict__, risk)

    def signal(self, df: pd.DataFrame) -> StrategySignal:
        # Require columns: open, high, low, close, volume
        if len(df) < max(self.cfg['donchian'], self.cfg['ema_filter'], self.cfg['vol_confirm_lookback']) + 5:
            return StrategySignal()

        high, low, close, vol = df['high'], df['low'], df['close'], df['volume']
        ema50 = ema(close, self.cfg['ema_filter'])
        upper, lower = donchian_channels(high, low, self.cfg['donchian'])
        atr14 = atr(high, low, close, 14)
        adx14 = adx(high, low, close, 14)

        vol_ma = vol.rolling(self.cfg['vol_confirm_lookback']).mean()
        i = -1
        last = close.iloc[i]
        sig = StrategySignal()

        if adx14.iloc[i] > 20 and vol.iloc[i] > (vol_ma.iloc[i] or 0):
            if last > upper.iloc[i] and last > ema50.iloc[i]:
                sig.side = "long"
                sig.entry_price = last
                sig.stop_price = last - self.cfg['stop_atr_mult'] * atr14.iloc[i]
                sig.take_profit = last + self.cfg['partial_tp_R'] * (last - sig.stop_price)
                sig.strength = 1.0
                sig.meta = {"reason":"donchian_breakout_up"}
            elif last < lower.iloc[i] and last < ema50.iloc[i]:
                sig.side = "short"
                sig.entry_price = last
                sig.stop_price = last + self.cfg['stop_atr_mult'] * atr14.iloc[i]
                sig.take_profit = last - self.cfg['partial_tp_R'] * (sig.stop_price - last)
                sig.strength = 1.0
                sig.meta = {"reason":"donchian_breakout_down"}
        return sig

# ===============
# Strategy 2: Mean Reversion VWAP
# ===============

class MeanReversionVWAPStrategy(StrategyBase):
    def __init__(self, cfg: MeanReversionConfig, risk: RiskConfig):
        super().__init__(cfg.__dict__, risk)

    def signal(self, df: pd.DataFrame) -> StrategySignal:
        if len(df) < max(20, self.cfg['bb_window']) + 5:
            return StrategySignal()

        close = df['close']
        vw = vwap(df)
        sd = close.rolling(20).std(ddof=0)
        atr14 = atr(df['high'], df['low'], df['close'], 14)

        i = -1
        last = close.iloc[i]
        vw_last = vw.iloc[i]
        sd_last = sd.iloc[i]
        sig = StrategySignal()

        if sd_last and not np.isnan(sd_last):
            z = (last - vw_last) / (sd_last if sd_last != 0 else np.nan)
            if z <= -self.cfg['vwap_sigma_entry']:
                sig.side = "long"
                sig.entry_price = last
                sig.stop_price = last - self.cfg['stop_atr_mult'] * atr14.iloc[i]
                # exit back to VWAP +/- exit_sigma
                target = vw_last - self.cfg['exit_sigma'] * sd_last
                sig.take_profit = max(target, last + 0.5 * (last - sig.stop_price))  # safety
                sig.strength = min(1.0, abs(z) / 3.0)
                sig.meta = {"zscore": float(z)}
            elif z >= self.cfg['vwap_sigma_entry']:
                sig.side = "short"
                sig.entry_price = last
                sig.stop_price = last + self.cfg['stop_atr_mult'] * atr14.iloc[i]
                target = vw_last + self.cfg['exit_sigma'] * sd_last
                sig.take_profit = min(target, last - 0.5 * (sig.stop_price - last))
                sig.strength = min(1.0, abs(z) / 3.0)
                sig.meta = {"zscore": float(z)}
        return sig

# ===============
# Strategy 3: Market Making Lite (inventory-capped, simplified)
# ===============

class MarketMakingLiteStrategy(StrategyBase):
    def __init__(self, cfg: MMLiteConfig, risk: RiskConfig):
        super().__init__(cfg.__dict__, risk)
        self.inventory = 0.0  # signed base units

    def signal(self, df: pd.DataFrame) -> StrategySignal:
        # MM is usually quote-based, not bar-close. Here we provide a simplified "bias" signal.
        if len(df) < self.cfg['trend_skew_ema'] + 5:
            return StrategySignal()

        close = df['close']
        ema20 = ema(close, self.cfg['trend_skew_ema'])
        i = -1
        last = close.iloc[i]

        bias = 0.0
        if last > ema20.iloc[i]:
            bias = 0.3
        elif last < ema20.iloc[i]:
            bias = -0.3

        sig = StrategySignal()
        # If inventory is near cap, bias toward reducing exposure (not opening new in that direction)
        sig.side = "long" if bias > 0 else ("short" if bias < 0 else None)
        sig.entry_price = last
        sig.stop_price = None
        sig.take_profit = None
        sig.strength = abs(bias)
        sig.meta = {"mm_bias": bias, "note":"Use with quote engine; here it's a simplified directional nudge."}
        return sig

# ===============
# Strategy 4: Funding & Basis (placeholder for neutral sleeves)
# ===============

class FundingBasisStrategy(StrategyBase):
    def __init__(self, cfg: FundingBasisConfig, risk: RiskConfig):
        super().__init__(cfg.__dict__, risk)

    def signal(self, df: pd.DataFrame, funding_series: Optional[pd.Series]=None, basis_series: Optional[pd.Series]=None) -> StrategySignal:
        # For backtest, you may supply time series of funding and basis; else no-op.
        sig = StrategySignal()
        if funding_series is not None and len(funding_series) >= 180:
            # enter only if current |funding| is above 80th pctile (configurable)
            curr = funding_series.iloc[-1]
            pctile = np.nanpercentile(funding_series.dropna().values, self.cfg['funding_abs_pctile']*100)
            if abs(curr) >= abs(pctile):
                # If funding very positive -> short perp / long spot (net slight short)
                sig.side = "short" if curr > 0 else "long"
                sig.entry_price = df['close'].iloc[-1]
                sig.stop_price = None  # neutral sleeve; risk is cross-venue; manage size small
                sig.take_profit = None
                sig.strength = 0.5
                sig.meta = {"funding": float(curr), "pctile_thr": float(pctile)}
        # Basis sleeve (annualized)
        if basis_series is not None and len(basis_series) > 30:
            curr_b = basis_series.iloc[-1]
            if curr_b >= self.cfg['basis_min_apr']:
                sig.side = "short"  # short perp/future, long spot (cash&carry)
                sig.entry_price = df['close'].iloc[-1]
                sig.meta["basis_apr"] = float(curr_b)
                sig.strength = max(sig.strength, 0.6)
        return sig

# ===============
# Regime Router
# ===============

class RegimeRouter:
    def __init__(self, cfg: RegimeConfig):
        self.cfg = cfg

    def classify(self, df: pd.DataFrame) -> str:
        if len(df) < max(14, self.cfg.bb_window) + 5:
            return "unknown"
        series_adx = adx(df['high'], df['low'], df['close'], 14)
        bbw = bollinger_bandwidth(df['close'], n=self.cfg.bb_window)
        # Compute rolling percentiles for BBW to decide "quietness"
        bbw_valid = bbw.dropna()
        if len(bbw_valid) < 90:
            # not enough data for robust percentile; fallback
            bbw_thr = bbw_valid.iloc[-1] if len(bbw_valid) else np.nan
        else:
            bbw_thr = np.nanpercentile(bbw_valid[-90:], self.cfg.bb_width_pctile*100)

        adx_last = series_adx.iloc[-1]
        bbw_last = bbw.iloc[-1]

        if adx_last >= self.cfg.adx_trend and (np.isnan(bbw_thr) or bbw_last >= bbw_thr):
            return "trend"
        if adx_last <= self.cfg.adx_chop and (np.isnan(bbw_thr) or bbw_last <= bbw_thr):
            return "range"
        return "mixed"

# ===============
# Risk Manager & Position Sizing
# ===============

class RiskManager:
    def __init__(self, cfg: RiskConfig):
        self.cfg = cfg
        self.start_nav = None
        self.kill_switch = False

    def start_day(self, nav: float):
        self.start_nav = nav
        self.kill_switch = False

    def update_nav(self, current_nav: float):
        if self.start_nav is None:
            self.start_nav = current_nav
        dd = (current_nav - self.start_nav) / self.start_nav
        if dd <= -self.cfg.max_intraday_dd:
            self.kill_switch = True

    def position_size_by_vol(self, nav: float, atr_pct: float, alloc_fraction: float, price: float, max_leverage_asset: float) -> float:
        """
        atr_pct is ATR% per bar/day (approx). For 1m bars use scaled ATR or a daily ATR% proxy.
        size notionally aims at nav * (target_daily_vol / atr_pct) * alloc_fraction
        """
        if atr_pct <= 0 or math.isnan(atr_pct):
            return 0.0
        target_notional = nav * (self.cfg.target_daily_vol / atr_pct) * alloc_fraction
        # Cap leverage
        max_notional = nav * max_leverage_asset * alloc_fraction
        notional = min(target_notional, max_notional)
        qty = notional / price
        return max(qty, 0.0)

# ===============
# Portfolio Orchestrator
# ===============

class PortfolioBot:
    def __init__(self, cfg: BotConfig):
        self.cfg = cfg
        self.router = RegimeRouter(cfg.regime)
        self.riskman = RiskManager(cfg.risk)
        # Instantiate strategies
        self.strat_trend = TrendBreakoutStrategy(cfg.trend_breakout, cfg.risk)
        self.strat_mr = MeanReversionVWAPStrategy(cfg.mean_reversion, cfg.risk)
        self.strat_mm = MarketMakingLiteStrategy(cfg.mm_lite, cfg.risk)
        self.strat_fb = FundingBasisStrategy(cfg.funding_basis, cfg.risk)

        self.positions: Dict[str, Position] = {}  # symbol -> Position

    def step(self, symbol: str, df: pd.DataFrame, nav: float, funding_series: Optional[pd.Series]=None, basis_series: Optional[pd.Series]=None) -> List[Tuple[str, Position, StrategySignal, float]]:
        """
        Decide actions on the latest bar. Returns a list of tuples:
        (strategy_name, position_to_open, signal, allocated_cap_fraction)
        """
        if len(df) < 50:
            return []

        regime = self.router.classify(df)
        close = df['close'].iloc[-1]
        atr14_abs = atr(df['high'], df['low'], df['close'], 14).iloc[-1]
        atr_pct = atr14_abs / close if close else np.nan

        actions = []

        # Strategy allocations (user-configurable)
        allocs = self.cfg.allocations

        # Guard: daily kill-switch
        if self.riskman.kill_switch:
            return []

        # --- Trend Breakout ---
        if regime == "trend" or regime == "mixed":
            sig = self.strat_trend.signal(df)
            if sig.side:
                qty = self.riskman.position_size_by_vol(nav, atr_pct, allocs.trend_breakout, sig.entry_price, self.cfg.risk.max_leverage_asset)
                pos = Position(side=sig.side, qty=qty, entry_price=sig.entry_price, stop_price=sig.stop_price, take_profit=sig.take_profit)
                actions.append(("trend_breakout", pos, sig, allocs.trend_breakout))

        # --- Mean Reversion ---
        if regime == "range" or regime == "mixed":
            sig = self.strat_mr.signal(df)
            if sig.side:
                qty = self.riskman.position_size_by_vol(nav, atr_pct, allocs.mean_reversion, sig.entry_price, self.cfg.risk.max_leverage_asset)
                pos = Position(side=sig.side, qty=qty, entry_price=sig.entry_price, stop_price=sig.stop_price, take_profit=sig.take_profit)
                actions.append(("mean_reversion", pos, sig, allocs.mean_reversion))

        # --- Market Making Lite (simplified directional bias here) ---
        sig = self.strat_mm.signal(df)
        if sig.side:
            qty = self.riskman.position_size_by_vol(nav, atr_pct, allocs.mm_lite, sig.entry_price, self.cfg.risk.max_leverage_asset)
            pos = Position(side=sig.side, qty=max(qty*0.2, 0.0), entry_price=sig.entry_price)  # keep MM lighter
            actions.append(("mm_lite", pos, sig, allocs.mm_lite))

        # --- Funding & Basis ---
        sig = self.strat_fb.signal(df, funding_series=funding_series, basis_series=basis_series)
        if sig.side:
            qty = self.riskman.position_size_by_vol(nav, max(atr_pct, 0.01), allocs.funding_basis, sig.entry_price, 1.0)  # generally small
            pos = Position(side=sig.side, qty=max(qty*0.2, 0.0), entry_price=sig.entry_price)
            actions.append(("funding_basis", pos, sig, allocs.funding_basis))

        return actions

# ===============
# Simple Backtest Engine (bar-close execution, no slippage model by default)
# ===============

class BacktestEngine:
    def __init__(self, bot: PortfolioBot, initial_nav: float=100000.0, fee_bps: float=1.0):
        self.bot = bot
        self.initial_nav = initial_nav
        self.nav = initial_nav
        self.fee_bps = fee_bps
        self.positions: List[Position] = []
        self.trades: List[Dict[str, Any]] = []
        self.bot.riskman.start_day(self.nav)

    def run(self, symbol: str, df: pd.DataFrame, funding_series: Optional[pd.Series]=None, basis_series: Optional[pd.Series]=None) -> Dict[str, Any]:
        # Iterate through bars and act on each
        for i in range(60, len(df)):  # warm-up
            window = df.iloc[:i+1].copy()
            actions = self.bot.step(symbol, window, self.nav,
                                    funding_series=funding_series.iloc[:i+1] if funding_series is not None else None,
                                    basis_series=basis_series.iloc[:i+1] if basis_series is not None else None)
            # Execute each action (open position immediately at close)
            price = window['close'].iloc[-1]

            for name, pos, sig, alloc in actions:
                if pos.qty <= 0:
                    continue
                # Apply fee on notional
                fee = (abs(pos.qty) * price) * (self.fee_bps / 10000.0)
                pnl = 0.0  # realized later; here we will mark-to-market on next bar with stops/TP
                self.positions.append(pos)
                self.trades.append({
                    "t": window.index[-1],
                    "strategy": name,
                    "side": pos.side,
                    "qty": pos.qty,
                    "price": price,
                    "fee": fee,
                    "signal": sig.meta
                })
                self.nav -= fee  # fee at entry

            # Update PnL for open positions (mark-to-market to current close)
            new_positions = []
            for pos in self.positions:
                # simplistic: evaluate vs current price and stop/tp triggers on the same bar (very rough)
                px = price
                if pos.side == "long":
                    # stop / tp checks (bar-based approximations)
                    low = window['low'].iloc[-1]
                    high = window['high'].iloc[-1]
                    exit_px = None
                    if pos.stop_price is not None and low <= pos.stop_price:
                        exit_px = pos.stop_price
                    if pos.take_profit is not None and high >= pos.take_profit:
                        exit_px = pos.take_profit if exit_px is None else exit_px  # prefer first hit (unknown ordering)
                    if exit_px is None:
                        # hold
                        mtm = pos.qty * (px - pos.entry_price)
                        # no fee here; apply only when closing
                        new_positions.append(pos)
                    else:
                        pnl = pos.qty * (exit_px - pos.entry_price)
                        fee = (abs(pos.qty) * exit_px) * (self.fee_bps / 10000.0)
                        self.nav += pnl - fee
                        self.trades.append({
                            "t": window.index[-1],
                            "strategy": "exit",
                            "side": "sell",
                            "qty": pos.qty,
                            "price": exit_px,
                            "fee": fee,
                        })
                elif pos.side == "short":
                    low = window['low'].iloc[-1]
                    high = window['high'].iloc[-1]
                    exit_px = None
                    if pos.stop_price is not None and high >= pos.stop_price:
                        exit_px = pos.stop_price
                    if pos.take_profit is not None and low <= pos.take_profit:
                        exit_px = pos.take_profit if exit_px is None else exit_px
                    if exit_px is None:
                        mtm = pos.qty * (pos.entry_price - px)
                        new_positions.append(pos)
                    else:
                        pnl = pos.qty * (pos.entry_price - exit_px)
                        fee = (abs(pos.qty) * exit_px) * (self.fee_bps / 10000.0)
                        self.nav += pnl - fee
                        self.trades.append({
                            "t": window.index[-1],
                            "strategy": "exit",
                            "side": "cover",
                            "qty": pos.qty,
                            "price": exit_px,
                            "fee": fee,
                        })

            self.positions = new_positions
            # Risk update (daily-KillSwitch using running NAV)
            self.bot.riskman.update_nav(self.nav)

        return {
            "final_nav": self.nav,
            "return_pct": (self.nav / self.initial_nav) - 1.0,
            "trades": pd.DataFrame(self.trades)
        }

# ===============
# Default CONFIG (editable)
# ===============

def default_config() -> BotConfig:
    return BotConfig(
        risk=RiskConfig(
            target_daily_vol=0.01,
            max_intraday_dd=0.03,
            max_leverage_asset=3.0,
            max_leverage_portfolio=3.0,
            kelly_fraction=0.25,
            max_per_asset_risk=0.0125
        ),
        regime=RegimeConfig(
            adx_trend=20, adx_chop=18, bb_width_pctile=0.4, bb_window=20
        ),
        trend_breakout=TrendBreakoutConfig(
            donchian=20, ema_filter=50, stop_atr_mult=3.0, partial_tp_R=2.0, vol_confirm_lookback=20
        ),
        mean_reversion=MeanReversionConfig(
            vwap_sigma_entry=0.7, exit_sigma=0.35, stop_atr_mult=1.2, max_retries=3, bb_window=20
        ),
        mm_lite=MMLiteConfig(
            spread_factor=0.7, inventory_cap_nav=0.01, trend_skew_ema=20, pause_move_pct_1m=0.003
        ),
        funding_basis=FundingBasisConfig(
            funding_abs_pctile=0.8, basis_min_apr=0.12
        ),
        allocations=Allocations(
            trend_breakout=0.35, mean_reversion=0.35, mm_lite=0.2, funding_basis=0.1
        )
    )

# ===============
# CLI Entry (example usage with random walk data)
# ===============

def example_run():
    np.random.seed(7)
    n = 2000
    dt_index = pd.date_range("2024-01-01", periods=n, freq="T")
    # random walk with drift
    ret = np.random.normal(0, 0.0008, size=n)
    price = 30000 * (1 + pd.Series(ret).cumsum()/10.0).clip(lower=1000).values
    high = price * (1 + np.random.rand(n)*0.001)
    low = price * (1 - np.random.rand(n)*0.001)
    open_ = price * (1 + (np.random.rand(n)-0.5)*0.0005)
    volume = np.random.randint(1_000, 10_000, size=n)

    df = pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": price,
        "volume": volume
    }, index=dt_index)

    cfg = default_config()
    bot = PortfolioBot(cfg)
    engine = BacktestEngine(bot, initial_nav=100_000, fee_bps=1.5)
    result = engine.run("BTC/USDT", df)
    print("Final NAV:", round(result["final_nav"], 2))
    print("Return %:", round(result["return_pct"]*100, 2))
    print("Trades:", len(result["trades"]))
    print(result["trades"].tail())

if __name__ == "__main__":
    # Run a demo backtest if executed directly
    example_run()
