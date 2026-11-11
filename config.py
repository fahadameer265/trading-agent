"""
config.py — Central configuration for the LangGraph Trading Agent.

Purpose
-------
This module is the single source of truth for every tunable parameter used by
the agent. It:

* Loads secrets (``OPENAI_API_KEY``) from the environment / a local ``.env``
  file via ``python-dotenv``. Secrets are never hard-coded.
* Defines all indicator, risk-management and runtime constants used by
  ``indicators.py`` and ``agent.py``.
* Exposes :func:`validate_config`, which returns a list of human-readable error
  strings (an empty list means the configuration is valid).

Conventions
-----------
* Error strings that concern the OpenAI API key always start with the literal
  prefix ``"OPENAI_API_KEY"``. The agent treats these as *non-fatal* (the
  sentiment node degrades to NEUTRAL/0.5), whereas every other configuration
  error is *fatal* and forces a HOLD without running the graph.
* ``DATA_LOOKBACK_DAYS`` is expressed in CALENDAR days. 90 calendar days yield
  roughly 60–63 daily trading bars, which satisfies ``MIN_DATA_ROWS``.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Environment loading
# ---------------------------------------------------------------------------
# 1) A .env file that sits next to this module (project root).
# 2) A .env file in the current working directory (or its parents).
# Existing environment variables are never overridden.
_PROJECT_ENV_PATH = Path(__file__).resolve().parent / ".env"
if _PROJECT_ENV_PATH.exists():
    load_dotenv(dotenv_path=_PROJECT_ENV_PATH, override=False)
load_dotenv(override=False)

# ---------------------------------------------------------------------------
# Secrets
# ---------------------------------------------------------------------------
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "").strip()

# ---------------------------------------------------------------------------
# LLM settings (sentiment classification ONLY — never trade decisions)
# ---------------------------------------------------------------------------
OPENAI_MODEL: str = "gpt-4o-mini"
LLM_TEMPERATURE: float = 0.0
LLM_MAX_TOKENS: int = 150
LLM_TIMEOUT_SECONDS: float = 30.0
LLM_MAX_RETRIES: int = 2

# ---------------------------------------------------------------------------
# Market data
# ---------------------------------------------------------------------------
DATA_LOOKBACK_DAYS: int = 90  # calendar days of daily bars to download
MIN_DATA_ROWS: int = 60       # minimum daily bars required to run indicators

# ---------------------------------------------------------------------------
# Technical indicator parameters
# ---------------------------------------------------------------------------
RSI_PERIOD: int = 14

MACD_FAST: int = 12
MACD_SLOW: int = 26
MACD_SIGNAL: int = 9

EMA_SHORT: int = 20
EMA_LONG: int = 50

ATR_PERIOD: int = 14

# ---------------------------------------------------------------------------
# Risk management parameters
# ---------------------------------------------------------------------------
ATR_MULTIPLIER: float = 1.5     # stop distance = ATR_MULTIPLIER * ATR
MIN_RR_RATIO: float = 2.0       # minimum reward:risk (1:2)
ATR_FALLBACK_PCT: float = 0.02  # used ONLY if ATR is unavailable (2% of price)

# ---------------------------------------------------------------------------
# Signal thresholds
# ---------------------------------------------------------------------------
RSI_OVERSOLD_MAX: float = 45.0    # BUY requires RSI strictly below this
RSI_OVERBOUGHT_MIN: float = 70.0  # SELL triggers when RSI strictly above this

# ---------------------------------------------------------------------------
# LangGraph runtime
# ---------------------------------------------------------------------------
CHECKPOINT_THREAD_ID: str = "trading-agent-thread-01"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "WARNING").strip().upper() or "WARNING"

_PLACEHOLDER_KEY_MARKERS = (
    "your-openai-api-key",
    "your_openai_api_key",
    "sk-xxxx",
    "changeme",
)


def _is_positive_int(value: object) -> bool:
    """Return True if ``value`` is a strictly positive ``int`` (bools excluded)."""
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _is_positive_number(value: object) -> bool:
    """Return True if ``value`` is a strictly positive int/float (bools excluded)."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and value == value  # excludes NaN
        and value > 0
    )


def validate_config() -> list[str]:
    """
    Validate the active configuration.

    Returns
    -------
    list[str]
        A list of human-readable error strings. An empty list means the
        configuration is valid. Errors relating to the OpenAI key start with
        the prefix ``"OPENAI_API_KEY"`` and are considered non-fatal by the
        agent; all others are fatal.
    """
    errors: list[str] = []

    # --- Secrets -----------------------------------------------------------
    if not OPENAI_API_KEY:
        errors.append(
            "OPENAI_API_KEY is not set. Add it to your environment or .env file; "
            "sentiment will default to NEUTRAL/0.5 until it is configured."
        )
    elif any(marker in OPENAI_API_KEY.lower() for marker in _PLACEHOLDER_KEY_MARKERS):
        errors.append(
            "OPENAI_API_KEY still contains the placeholder value from .env.example; "
            "replace it with a real key."
        )
    elif not OPENAI_API_KEY.startswith("sk-"):
        errors.append(
            "OPENAI_API_KEY does not look like a valid OpenAI key (expected prefix 'sk-')."
        )

    # --- LLM ---------------------------------------------------------------
    if not isinstance(OPENAI_MODEL, str) or not OPENAI_MODEL.strip():
        errors.append("OPENAI_MODEL must be a non-empty string.")
    if not isinstance(LLM_TEMPERATURE, (int, float)) or LLM_TEMPERATURE != 0.0:
        errors.append("LLM_TEMPERATURE must be 0.0 for deterministic sentiment output.")
    if not _is_positive_int(LLM_MAX_TOKENS):
        errors.append("LLM_MAX_TOKENS must be a positive integer.")
    if not _is_positive_number(LLM_TIMEOUT_SECONDS):
        errors.append("LLM_TIMEOUT_SECONDS must be a positive number.")
    if not isinstance(LLM_MAX_RETRIES, int) or isinstance(LLM_MAX_RETRIES, bool) or LLM_MAX_RETRIES < 0:
        errors.append("LLM_MAX_RETRIES must be a non-negative integer.")

    # --- Market data -------------------------------------------------------
    if not _is_positive_int(DATA_LOOKBACK_DAYS):
        errors.append("DATA_LOOKBACK_DAYS must be a positive integer.")
    if not _is_positive_int(MIN_DATA_ROWS):
        errors.append("MIN_DATA_ROWS must be a positive integer.")
    if (
        _is_positive_int(DATA_LOOKBACK_DAYS)
        and _is_positive_int(MIN_DATA_ROWS)
        and DATA_LOOKBACK_DAYS < MIN_DATA_ROWS
    ):
        errors.append(
            f"DATA_LOOKBACK_DAYS ({DATA_LOOKBACK_DAYS}) can never yield "
            f"MIN_DATA_ROWS ({MIN_DATA_ROWS}) daily bars."
        )

    # --- Indicator periods -------------------------------------------------
    for name, value in (
        ("RSI_PERIOD", RSI_PERIOD),
        ("MACD_FAST", MACD_FAST),
        ("MACD_SLOW", MACD_SLOW),
        ("MACD_SIGNAL", MACD_SIGNAL),
        ("EMA_SHORT", EMA_SHORT),
        ("EMA_LONG", EMA_LONG),
        ("ATR_PERIOD", ATR_PERIOD),
    ):
        if not _is_positive_int(value):
            errors.append(f"{name} must be a positive integer (got {value!r}).")

    if _is_positive_int(MACD_FAST) and _is_positive_int(MACD_SLOW) and MACD_FAST >= MACD_SLOW:
        errors.append("MACD_FAST must be smaller than MACD_SLOW.")
    if _is_positive_int(EMA_SHORT) and _is_positive_int(EMA_LONG) and EMA_SHORT >= EMA_LONG:
        errors.append("EMA_SHORT must be smaller than EMA_LONG.")
    if _is_positive_int(MIN_DATA_ROWS):
        if _is_positive_int(EMA_LONG) and EMA_LONG > MIN_DATA_ROWS:
            errors.append("EMA_LONG cannot exceed MIN_DATA_ROWS.")
        if (
            _is_positive_int(MACD_SLOW)
            and _is_positive_int(MACD_SIGNAL)
            and MACD_SLOW + MACD_SIGNAL > MIN_DATA_ROWS
        ):
            errors.append("MACD_SLOW + MACD_SIGNAL cannot exceed MIN_DATA_ROWS.")

    # --- Risk management ---------------------------------------------------
    if not _is_positive_number(ATR_MULTIPLIER):
        errors.append("ATR_MULTIPLIER must be a positive number.")
    if not isinstance(MIN_RR_RATIO, (int, float)) or isinstance(MIN_RR_RATIO, bool) or MIN_RR_RATIO < 2.0:
        errors.append("MIN_RR_RATIO must be at least 2.0 (minimum 1:2 reward:risk).")
    if not _is_positive_number(ATR_FALLBACK_PCT) or ATR_FALLBACK_PCT >= 1.0:
        errors.append("ATR_FALLBACK_PCT must be between 0 and 1 (exclusive).")

    # --- Thresholds --------------------------------------------------------
    if not (0.0 < RSI_OVERSOLD_MAX < RSI_OVERBOUGHT_MIN < 100.0):
        errors.append(
            "RSI thresholds must satisfy 0 < RSI_OVERSOLD_MAX < RSI_OVERBOUGHT_MIN < 100."
        )

    # --- Runtime -----------------------------------------------------------
    if not isinstance(CHECKPOINT_THREAD_ID, str) or not CHECKPOINT_THREAD_ID.strip():
        errors.append("CHECKPOINT_THREAD_ID must be a non-empty string.")
    if LOG_LEVEL not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
        errors.append(
            f"LOG_LEVEL '{LOG_LEVEL}' is invalid; use DEBUG, INFO, WARNING, ERROR or CRITICAL."
        )

    return errors
