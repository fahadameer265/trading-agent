"""
indicators.py — Deterministic technical-indicator engine for the Trading Agent.

Purpose
-------
* Download daily OHLCV market data from Yahoo Finance via ``yfinance``.
* Compute RSI, MACD (dual-verified), EMA trend (double-confirmed) and ATR with
  ``pandas-ta``.
* Expose a single façade, :func:`run_all_indicators`, which is the ONLY
  function called by ``agent.py`` (Node 1 — technical analysis).

Design rules
------------
* No public function in this module raises. Every failure is logged and
  converted into ``None`` / ``"NEUTRAL"`` plus a human-readable error string.
* Errors that make a trade decision impossible (no data, too little data, no
  price, pipeline crash) are prefixed with :data:`CRITICAL_PREFIX` so the agent
  can force a HOLD.
* All returned floats are rounded to 4 decimal places.
* No LLM is involved anywhere in this module — it is pure, reproducible math.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd

# Compatibility shim: pandas-ta 0.3.14b0 imports ``numpy.NaN``, which was removed
# in NumPy 2.0. Restoring the alias keeps older pandas-ta builds importable and
# is a harmless no-op on NumPy 1.x / newer pandas-ta releases.
if not hasattr(np, "NaN"):
    np.NaN = np.nan  # type: ignore[attr-defined]

import pandas_ta as ta  # noqa: E402  (must follow the NumPy shim above)
import yfinance as yf  # noqa: E402

import config  # noqa: E402

logger = logging.getLogger(__name__)

CRITICAL_PREFIX = "CRITICAL:"
REQUIRED_COLUMNS: tuple[str, ...] = ("Open", "High", "Low", "Close")
NUMERIC_COLUMNS: tuple[str, ...] = ("Open", "High", "Low", "Close", "Volume")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _safe_float(value: Any) -> float | None:
    """Convert ``value`` to a finite float, or return ``None``."""
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result) or math.isinf(result):
        return None
    return result


def _round4(value: Any) -> float | None:
    """Round ``value`` to 4 decimal places, or return ``None`` if not numeric."""
    result = _safe_float(value)
    return None if result is None else round(result, 4)


def _last_valid(series: pd.Series | None) -> float | None:
    """Return the last non-NaN value of a Series as a float, or ``None``."""
    if series is None or not isinstance(series, pd.Series):
        return None
    cleaned = pd.to_numeric(series, errors="coerce").dropna()
    if cleaned.empty:
        return None
    return _safe_float(cleaned.iloc[-1])


def _find_column(frame: pd.DataFrame, prefix: str) -> str | None:
    """Return the first column whose name starts with ``prefix`` (case-sensitive)."""
    for column in frame.columns:
        if str(column).startswith(prefix):
            return str(column)
    return None


def _has_columns(df: pd.DataFrame | None, columns: tuple[str, ...]) -> bool:
    """Return True if ``df`` is a non-empty DataFrame containing every column."""
    return (
        isinstance(df, pd.DataFrame)
        and not df.empty
        and all(column in df.columns for column in columns)
    )


# ---------------------------------------------------------------------------
# 1. Data acquisition
# ---------------------------------------------------------------------------
def fetch_ohlcv(symbol: str, days: int = config.DATA_LOOKBACK_DAYS) -> pd.DataFrame | None:
    """
    Download daily OHLCV bars for ``symbol`` covering the last ``days`` calendar days.

    Column names are standardised to Title Case (``Open``, ``High``, ``Low``,
    ``Close``, ``Volume``), rows with a NaN ``Close`` are dropped, duplicate
    timestamps are removed and the index is sorted ascending.

    Returns
    -------
    pd.DataFrame | None
        The cleaned DataFrame, or ``None`` if the download failed or is empty.
    """
    try:
        if not isinstance(symbol, str) or not symbol.strip():
            logger.error("fetch_ohlcv: symbol must be a non-empty string (got %r).", symbol)
            return None
        if not isinstance(days, int) or isinstance(days, bool) or days <= 0:
            logger.error("fetch_ohlcv: days must be a positive integer (got %r).", days)
            return None

        clean_symbol = symbol.strip().upper()
        today = datetime.now(timezone.utc).date()
        start = today - timedelta(days=days)
        end = today + timedelta(days=1)  # yfinance treats 'end' as exclusive

        logger.info("Fetching %s daily OHLCV from %s to %s.", clean_symbol, start, end)
        ticker = yf.Ticker(clean_symbol)
        df = ticker.history(
            start=start.isoformat(),
            end=end.isoformat(),
            interval="1d",
            auto_adjust=True,
            actions=False,
        )

        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            logger.warning("fetch_ohlcv: no data returned for %s.", clean_symbol)
            return None

        df = df.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [str(column).strip().replace("_", " ").title() for column in df.columns]
        df = df.loc[:, ~pd.Index(df.columns).duplicated(keep="first")]

        missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
        if missing:
            logger.error("fetch_ohlcv: %s data is missing columns %s.", clean_symbol, missing)
            return None

        present_numeric = [column for column in NUMERIC_COLUMNS if column in df.columns]
        df[present_numeric] = df[present_numeric].apply(pd.to_numeric, errors="coerce")

        df = df.dropna(subset=["Close"])
        df = df[~df.index.duplicated(keep="last")].sort_index()

        if df.empty:
            logger.warning("fetch_ohlcv: all rows for %s had NaN Close values.", clean_symbol)
            return None

        logger.info("fetch_ohlcv: received %d rows for %s.", len(df), clean_symbol)
        return df
    except Exception as exc:  # network errors, yfinance parsing errors, etc.
        logger.exception("fetch_ohlcv: failed to download %r: %s", symbol, exc)
        return None


# ---------------------------------------------------------------------------
# 2. Latest price
# ---------------------------------------------------------------------------
def get_latest_price(df: pd.DataFrame | None) -> float | None:
    """Return the most recent valid closing price (rounded to 4 dp), or ``None``."""
    try:
        if not _has_columns(df, ("Close",)):
            logger.warning("get_latest_price: DataFrame is empty or has no 'Close' column.")
            return None
        price = _last_valid(df["Close"])  # type: ignore[index]
        if price is None or price <= 0:
            logger.warning("get_latest_price: no positive closing price found.")
            return None
        return _round4(price)
    except Exception as exc:
        logger.exception("get_latest_price: failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# 3. RSI
# ---------------------------------------------------------------------------
def calculate_rsi(df: pd.DataFrame | None, period: int = config.RSI_PERIOD) -> float | None:
    """
    Calculate the latest RSI value using ``pandas_ta.rsi``.

    Returns
    -------
    float | None
        The most recent RSI (0–100, rounded to 4 dp), or ``None`` if the series
        is empty or the calculation failed.
    """
    try:
        if not _has_columns(df, ("Close",)):
            logger.warning("calculate_rsi: DataFrame is empty or has no 'Close' column.")
            return None
        if len(df) < period + 1:  # type: ignore[arg-type]
            logger.warning("calculate_rsi: need at least %d rows, got %d.", period + 1, len(df))  # type: ignore[arg-type]
            return None

        rsi_series = ta.rsi(df["Close"], length=period)  # type: ignore[index]
        if rsi_series is None:
            logger.warning("calculate_rsi: pandas_ta.rsi returned None.")
            return None
        rsi_series = rsi_series.dropna()
        if rsi_series.empty:
            logger.warning("calculate_rsi: RSI series is empty after dropping NaN.")
            return None

        value = _round4(rsi_series.iloc[-1])
        if value is None or not (0.0 <= value <= 100.0):
            logger.warning("calculate_rsi: RSI value %r is out of range.", value)
            return None
        return value
    except Exception as exc:
        logger.exception("calculate_rsi: failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# 4. MACD (dual-verified)
# ---------------------------------------------------------------------------
def calculate_macd(
    df: pd.DataFrame | None,
    fast: int = config.MACD_FAST,
    slow: int = config.MACD_SLOW,
    signal: int = config.MACD_SIGNAL,
) -> tuple[str | None, float | None, float | None]:
    """
    Calculate MACD with ``pandas_ta.macd`` and classify it with dual verification.

    * BULLISH  — histogram > 0 AND macd_line > signal_line
    * BEARISH  — histogram < 0 AND macd_line < signal_line
    * NEUTRAL  — the two checks disagree or the lines are flat

    Returns
    -------
    tuple[str | None, float | None, float | None]
        ``(verdict, macd_value, signal_value)``; ``(None, None, None)`` on failure.
    """
    failure: tuple[str | None, float | None, float | None] = (None, None, None)
    try:
        if not _has_columns(df, ("Close",)):
            logger.warning("calculate_macd: DataFrame is empty or has no 'Close' column.")
            return failure
        if len(df) < slow + signal:  # type: ignore[arg-type]
            logger.warning(
                "calculate_macd: need at least %d rows, got %d.", slow + signal, len(df)  # type: ignore[arg-type]
            )
            return failure

        macd_df = ta.macd(df["Close"], fast=fast, slow=slow, signal=signal)  # type: ignore[index]
        if macd_df is None or not isinstance(macd_df, pd.DataFrame) or macd_df.empty:
            logger.warning("calculate_macd: pandas_ta.macd returned no data.")
            return failure

        macd_col = _find_column(macd_df, "MACD_")
        hist_col = _find_column(macd_df, "MACDh_")
        signal_col = _find_column(macd_df, "MACDs_")
        if macd_col is None or hist_col is None or signal_col is None:
            logger.error("calculate_macd: unexpected MACD columns %s.", list(macd_df.columns))
            return failure

        latest = macd_df[[macd_col, hist_col, signal_col]].dropna()
        if latest.empty:
            logger.warning("calculate_macd: MACD frame is empty after dropping NaN.")
            return failure

        row = latest.iloc[-1]
        macd_line = _safe_float(row[macd_col])
        histogram = _safe_float(row[hist_col])
        signal_line = _safe_float(row[signal_col])
        if macd_line is None or histogram is None or signal_line is None:
            logger.warning("calculate_macd: latest MACD values are not numeric.")
            return failure

        # Dual verification: BOTH the histogram sign AND the line ordering must agree.
        histogram_bullish = histogram > 0
        lines_bullish = macd_line > signal_line
        histogram_bearish = histogram < 0
        lines_bearish = macd_line < signal_line

        if histogram_bullish and lines_bullish:
            verdict = "BULLISH"
        elif histogram_bearish and lines_bearish:
            verdict = "BEARISH"
        else:
            verdict = "NEUTRAL"

        logger.info(
            "calculate_macd: macd=%.4f signal=%.4f hist=%.4f -> %s",
            macd_line,
            signal_line,
            histogram,
            verdict,
        )
        return verdict, _round4(macd_line), _round4(signal_line)
    except Exception as exc:
        logger.exception("calculate_macd: failed: %s", exc)
        return failure


# ---------------------------------------------------------------------------
# 5. EMA trend (double confirmation)
# ---------------------------------------------------------------------------
def calculate_ema_trend(
    df: pd.DataFrame | None,
    short: int = config.EMA_SHORT,
    long: int = config.EMA_LONG,
) -> tuple[str, float | None, float | None]:
    """
    Classify the trend using two EMA confirmations (``pandas_ta.ema``).

    * BULLISH — price > EMA_long AND EMA_short > EMA_long
    * BEARISH — price < EMA_long AND EMA_short < EMA_long
    * NEUTRAL — otherwise (including any missing value)

    Returns
    -------
    tuple[str, float | None, float | None]
        ``(trend, ema_short_value, ema_long_value)``.
    """
    try:
        if not _has_columns(df, ("Close",)):
            logger.warning("calculate_ema_trend: DataFrame is empty or has no 'Close' column.")
            return "NEUTRAL", None, None
        if len(df) < long:  # type: ignore[arg-type]
            logger.warning("calculate_ema_trend: need at least %d rows, got %d.", long, len(df))  # type: ignore[arg-type]
            return "NEUTRAL", None, None

        close = df["Close"]  # type: ignore[index]
        ema_short_value = _last_valid(ta.ema(close, length=short))
        ema_long_value = _last_valid(ta.ema(close, length=long))
        price = get_latest_price(df)

        if price is None or ema_short_value is None or ema_long_value is None:
            logger.warning("calculate_ema_trend: missing price or EMA values; trend is NEUTRAL.")
            return "NEUTRAL", _round4(ema_short_value), _round4(ema_long_value)

        if price > ema_long_value and ema_short_value > ema_long_value:
            trend = "BULLISH"
        elif price < ema_long_value and ema_short_value < ema_long_value:
            trend = "BEARISH"
        else:
            trend = "NEUTRAL"

        logger.info(
            "calculate_ema_trend: price=%.4f ema%d=%.4f ema%d=%.4f -> %s",
            price,
            short,
            ema_short_value,
            long,
            ema_long_value,
            trend,
        )
        return trend, _round4(ema_short_value), _round4(ema_long_value)
    except Exception as exc:
        logger.exception("calculate_ema_trend: failed: %s", exc)
        return "NEUTRAL", None, None


# ---------------------------------------------------------------------------
# 6. ATR
# ---------------------------------------------------------------------------
def calculate_atr(df: pd.DataFrame | None, period: int = config.ATR_PERIOD) -> float | None:
    """
    Calculate the latest Average True Range using ``pandas_ta.atr``.

    Returns
    -------
    float | None
        The most recent positive ATR value (rounded to 4 dp), or ``None``.
    """
    try:
        if not _has_columns(df, ("High", "Low", "Close")):
            logger.warning("calculate_atr: DataFrame is empty or missing High/Low/Close.")
            return None
        if len(df) < period + 1:  # type: ignore[arg-type]
            logger.warning("calculate_atr: need at least %d rows, got %d.", period + 1, len(df))  # type: ignore[arg-type]
            return None

        atr_series = ta.atr(
            high=df["High"],  # type: ignore[index]
            low=df["Low"],  # type: ignore[index]
            close=df["Close"],  # type: ignore[index]
            length=period,
        )
        value = _last_valid(atr_series)
        if value is None or value <= 0:
            logger.warning("calculate_atr: ATR is unavailable or non-positive (%r).", value)
            return None
        return _round4(value)
    except Exception as exc:
        logger.exception("calculate_atr: failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# 7. Façade — the ONLY entry point used by agent.py
# ---------------------------------------------------------------------------
def run_all_indicators(symbol: str, days: int = config.DATA_LOOKBACK_DAYS) -> dict[str, Any]:
    """
    Fetch data for ``symbol`` and compute every indicator the agent needs.

    Returns
    -------
    dict
        Keys: ``price``, ``rsi``, ``macd_signal``, ``ema_trend``, ``ema_short``,
        ``ema_long``, ``atr``, ``errors``. Numeric values are floats rounded to
        4 dp or ``None``. ``errors`` is a list of strings; entries starting with
        :data:`CRITICAL_PREFIX` mean no trade decision should be made.
    """
    result: dict[str, Any] = {
        "price": None,
        "rsi": None,
        "macd_signal": "NEUTRAL",
        "ema_trend": "NEUTRAL",
        "ema_short": None,
        "ema_long": None,
        "atr": None,
        "errors": [],
    }
    errors: list[str] = result["errors"]

    try:
        clean_symbol = symbol.strip().upper() if isinstance(symbol, str) else ""
        if not clean_symbol:
            errors.append(f"{CRITICAL_PREFIX} Invalid ticker symbol {symbol!r}.")
            return result

        df = fetch_ohlcv(clean_symbol, days)
        if df is None:
            errors.append(
                f"{CRITICAL_PREFIX} No market data returned for {clean_symbol}. "
                "Check the ticker symbol and your network connection."
            )
            return result

        row_count = len(df)
        if row_count < config.MIN_DATA_ROWS:
            errors.append(
                f"{CRITICAL_PREFIX} Insufficient data for {clean_symbol}: {row_count} daily rows "
                f"(minimum {config.MIN_DATA_ROWS}). Increase DATA_LOOKBACK_DAYS."
            )
            return result

        price = get_latest_price(df)
        if price is None or price <= 0:
            errors.append(f"{CRITICAL_PREFIX} Could not determine a valid latest price for {clean_symbol}.")
            return result
        result["price"] = _round4(price)

        rsi = calculate_rsi(df, config.RSI_PERIOD)
        if rsi is None:
            errors.append(f"RSI({config.RSI_PERIOD}) could not be calculated.")
        else:
            result["rsi"] = rsi

        macd_verdict, macd_value, signal_value = calculate_macd(
            df, config.MACD_FAST, config.MACD_SLOW, config.MACD_SIGNAL
        )
        if macd_verdict is None:
            errors.append("MACD could not be calculated; treated as NEUTRAL.")
        else:
            result["macd_signal"] = macd_verdict

        ema_trend, ema_short_value, ema_long_value = calculate_ema_trend(
            df, config.EMA_SHORT, config.EMA_LONG
        )
        result["ema_trend"] = ema_trend
        result["ema_short"] = ema_short_value
        result["ema_long"] = ema_long_value
        if ema_short_value is None or ema_long_value is None:
            errors.append(
                f"EMA{config.EMA_SHORT}/EMA{config.EMA_LONG} could not be fully calculated; "
                "trend treated as NEUTRAL."
            )

        atr = calculate_atr(df, config.ATR_PERIOD)
        if atr is None:
            errors.append(
                f"ATR({config.ATR_PERIOD}) could not be calculated; the risk engine will fall back "
                f"to {config.ATR_FALLBACK_PCT:.0%} of price."
            )
        else:
            result["atr"] = atr

        logger.info(
            "run_all_indicators(%s): price=%s rsi=%s macd=%s (macd=%s, signal=%s) "
            "ema_trend=%s ema_short=%s ema_long=%s atr=%s rows=%d",
            clean_symbol,
            result["price"],
            result["rsi"],
            result["macd_signal"],
            macd_value,
            signal_value,
            result["ema_trend"],
            result["ema_short"],
            result["ema_long"],
            result["atr"],
            row_count,
        )
        return result
    except Exception as exc:
        logger.exception("run_all_indicators: unexpected failure for %r: %s", symbol, exc)
        errors.append(f"{CRITICAL_PREFIX} Indicator pipeline failed unexpectedly: {exc}")
        result["price"] = None
        return result
