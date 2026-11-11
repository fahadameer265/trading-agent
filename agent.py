"""
agent.py — LangGraph orchestration for the Trading Agent.

Purpose
-------
Wires three nodes into a linear LangGraph pipeline with an in-memory
checkpointer and prints a colour-coded terminal report:

    START -> technical_analysis -> sentiment_analysis -> risk_signal -> END

* ``technical_analysis`` — calls ``indicators.run_all_indicators`` (pure math).
* ``sentiment_analysis`` — asks an OpenAI model for a JSON sentiment label and
  confidence score ONLY. The LLM never decides the trade action.
* ``risk_signal`` — deterministic Python rules decide BUY / SELL / HOLD and
  compute ATR-based stop-loss / take-profit with an enforced minimum 1:2 R:R.

Safety guarantees
-----------------
* Every node wraps its logic in try/except and falls back to HOLD.
* All floats written to state are rounded to 4 decimal places.
* A BUY is only accepted if TP > price > SL and R:R >= MIN_RR_RATIO.

Usage
-----
    python agent.py            # analyses AAPL
    python agent.py MSFT       # analyses MSFT

This tool is for research and education only; it is not financial advice.
"""

import json
import logging
import math
import os
import re
import sys
import textwrap
from typing import Any, List, Literal, TypedDict, cast

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

import config
import indicators

logger = logging.getLogger("trading_agent")

VALID_SENTIMENTS: frozenset[str] = frozenset({"BULLISH", "BEARISH", "NEUTRAL"})
VALID_SIGNALS: frozenset[str] = frozenset({"BULLISH", "BEARISH", "NEUTRAL"})
DEFAULT_SENTIMENT = "NEUTRAL"
DEFAULT_CONFIDENCE = 0.5
NEUTRAL_RSI_PLACEHOLDER = 50.0
RR_TOLERANCE = 1e-9
SYMBOL_PATTERN = re.compile(r"^[A-Z0-9.\-^=]{1,20}$")


# ===========================================================================
# PART A — Agent state
# ===========================================================================
class AgentState(TypedDict):
    """Shared state passed between LangGraph nodes."""

    symbol: str
    price: float
    rsi: float
    macd_signal: str
    ema_trend: str
    ema_short: float | None
    ema_long: float | None
    atr: float | None
    sentiment: str
    confidence_score: float
    stop_loss: float
    take_profit: float
    risk_reward_ratio: float
    action: Literal["BUY", "SELL", "HOLD"]
    reasoning: str
    errors: List[str]


# ===========================================================================
# Shared helpers
# ===========================================================================
def _round4(value: Any, default: float = 0.0) -> float:
    """Return ``value`` as a finite float rounded to 4 dp, or ``default``."""
    if value is None or isinstance(value, bool):
        return round(default, 4)
    try:
        result = float(value)
    except (TypeError, ValueError):
        return round(default, 4)
    if math.isnan(result) or math.isinf(result):
        return round(default, 4)
    return round(result, 4)


def _round4_optional(value: Any) -> float | None:
    """Return ``value`` rounded to 4 dp, or ``None`` if it is not a finite number."""
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result) or math.isinf(result):
        return None
    return round(result, 4)


def _has_critical_errors(errors: List[str]) -> bool:
    """Return True if any error string is flagged as critical."""
    return any(str(error).startswith(indicators.CRITICAL_PREFIX) for error in errors)


def _initial_state(symbol: str) -> AgentState:
    """Return a fully populated, safe-by-default initial state."""
    return AgentState(
        symbol=symbol,
        price=0.0,
        rsi=NEUTRAL_RSI_PLACEHOLDER,
        macd_signal="NEUTRAL",
        ema_trend="NEUTRAL",
        ema_short=None,
        ema_long=None,
        atr=None,
        sentiment=DEFAULT_SENTIMENT,
        confidence_score=DEFAULT_CONFIDENCE,
        stop_loss=0.0,
        take_profit=0.0,
        risk_reward_ratio=0.0,
        action="HOLD",
        reasoning="",
        errors=[],
    )


# ===========================================================================
# PART B — Node 1: Technical analysis
# ===========================================================================
def technical_analysis_node(state: AgentState) -> AgentState:
    """
    Run every technical indicator for ``state["symbol"]`` and merge into state.

    Forces ``action = "HOLD"`` if the price is 0.0 or a critical data error
    occurred. Never raises.
    """
    errors: List[str] = list(state.get("errors", []))
    symbol = str(state.get("symbol", "")).strip().upper()
    try:
        logger.info("[technical_analysis] Running indicators for %s.", symbol)
        result = indicators.run_all_indicators(symbol)
        errors.extend(str(error) for error in result.get("errors", []))

        price = _round4(result.get("price"), 0.0)

        rsi_value = _round4_optional(result.get("rsi"))
        if rsi_value is None:
            rsi_value = NEUTRAL_RSI_PLACEHOLDER
            if price > 0.0:
                errors.append(
                    f"RSI unavailable; using neutral placeholder {NEUTRAL_RSI_PLACEHOLDER} "
                    "(cannot satisfy BUY or SELL RSI conditions)."
                )

        macd_signal = str(result.get("macd_signal") or "NEUTRAL").upper()
        if macd_signal not in VALID_SIGNALS:
            errors.append(f"Unexpected MACD signal {macd_signal!r}; treated as NEUTRAL.")
            macd_signal = "NEUTRAL"

        ema_trend = str(result.get("ema_trend") or "NEUTRAL").upper()
        if ema_trend not in VALID_SIGNALS:
            errors.append(f"Unexpected EMA trend {ema_trend!r}; treated as NEUTRAL.")
            ema_trend = "NEUTRAL"

        new_state: AgentState = {
            **state,
            "symbol": symbol,
            "price": price,
            "rsi": _round4(rsi_value, NEUTRAL_RSI_PLACEHOLDER),
            "macd_signal": macd_signal,
            "ema_trend": ema_trend,
            "ema_short": _round4_optional(result.get("ema_short")),
            "ema_long": _round4_optional(result.get("ema_long")),
            "atr": _round4_optional(result.get("atr")),
            "errors": errors,
        }

        if price == 0.0 or _has_critical_errors(errors):
            logger.warning("[technical_analysis] Forcing HOLD for %s: data unavailable.", symbol)
            new_state["action"] = "HOLD"
            new_state["reasoning"] = (
                "Forced HOLD: market data was unavailable or incomplete, so no trade "
                "decision can be made safely."
            )

        return new_state
    except Exception as exc:
        logger.exception("[technical_analysis] Unexpected failure: %s", exc)
        errors.append(f"{indicators.CRITICAL_PREFIX} technical_analysis_node failed: {exc}")
        return {
            **state,
            "symbol": symbol,
            "price": 0.0,
            "action": "HOLD",
            "reasoning": "Forced HOLD: the technical analysis step failed unexpectedly.",
            "errors": errors,
        }


# ===========================================================================
# PART C — Node 2: Sentiment analysis (LLM — sentiment ONLY)
# ===========================================================================
SENTIMENT_SYSTEM_PROMPT = """You are a disciplined market-sentiment classifier inside an automated trading system.

Your ONLY job is to classify short-term market sentiment for one stock using ONLY the data supplied in the user message. You do NOT decide whether to buy, sell or hold; a separate deterministic risk engine makes that decision.

OUTPUT RULES (mandatory):
1. Respond with a single raw JSON object and NOTHING else. No markdown, no code fences, no prose, no explanations, no extra keys.
2. Schema: {"sentiment": "BULLISH" | "BEARISH" | "NEUTRAL", "confidence_score": <number from 0.0 to 1.0>}
3. If you are uncertain, or the data is ambiguous or insufficient, respond exactly: {"sentiment": "NEUTRAL", "confidence_score": 0.5}
4. Do NOT invent, assume or reference any news, earnings, events, analyst ratings or facts that are not present in the user message. Fabricated news events are strictly forbidden. Base your classification strictly on the figures provided.
5. Never output trade instructions such as BUY, SELL or HOLD."""

_FENCE_PATTERN = re.compile(r"^\s*```[a-zA-Z0-9_-]*\s*(.*?)\s*```\s*$", re.DOTALL)


def _message_text(response: Any) -> str:
    """Extract plain text from a LangChain message (handles str or content blocks)."""
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        return "".join(parts)
    return str(content)


def _strip_markdown_fences(raw: str) -> str:
    """Remove surrounding ``` / ```json fences and any stray backticks."""
    text = raw.strip()
    match = _FENCE_PATTERN.match(text)
    if match:
        return match.group(1).strip()
    if text.startswith("```"):
        text = text[3:]
        if text.lower().startswith("json"):
            text = text[4:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip().strip("`").strip()


def parse_sentiment_response(raw: str) -> tuple[str, float]:
    """
    Parse and validate the LLM's JSON sentiment output.

    Strips markdown fences, falls back to extracting the first ``{...}`` block,
    validates the sentiment label and clamps the confidence to [0.0, 1.0].

    Raises
    ------
    ValueError
        If the output cannot be parsed or fails validation.
    """
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("LLM returned an empty response.")

    text = _strip_markdown_fences(raw)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise ValueError(f"No JSON object found in LLM output: {raw[:120]!r}") from None
        data = json.loads(text[start : end + 1])

    if not isinstance(data, dict):
        raise ValueError("LLM output JSON is not an object.")

    sentiment = str(data.get("sentiment", "")).strip().upper()
    if sentiment not in VALID_SENTIMENTS:
        raise ValueError(f"Invalid sentiment value {data.get('sentiment')!r}.")

    raw_confidence = data.get("confidence_score")
    if raw_confidence is None or isinstance(raw_confidence, bool):
        raise ValueError("confidence_score is missing or not numeric.")
    confidence = float(raw_confidence)
    if math.isnan(confidence) or math.isinf(confidence):
        raise ValueError("confidence_score is not a finite number.")
    confidence = max(0.0, min(1.0, confidence))

    return sentiment, round(confidence, 4)


def sentiment_node(state: AgentState) -> AgentState:
    """
    Classify market sentiment with the LLM. Never decides the trade action.

    Skips the LLM when Node 1 forced HOLD with price 0.0. On ANY failure the
    sentiment defaults to NEUTRAL/0.5, the error is logged and the pipeline
    continues with a provisional HOLD (the risk node makes the final call).
    """
    errors: List[str] = list(state.get("errors", []))
    try:
        price = _round4(state.get("price"), 0.0)
        if state.get("action") == "HOLD" and price == 0.0:
            logger.info("[sentiment_analysis] Skipping LLM call: HOLD already forced (price 0.0).")
            return {
                **state,
                "sentiment": DEFAULT_SENTIMENT,
                "confidence_score": DEFAULT_CONFIDENCE,
                "errors": errors,
            }

        if not config.OPENAI_API_KEY:
            errors.append("Sentiment skipped: OPENAI_API_KEY is not configured; defaulted to NEUTRAL/0.5.")
            return {
                **state,
                "sentiment": DEFAULT_SENTIMENT,
                "confidence_score": DEFAULT_CONFIDENCE,
                "errors": errors,
            }

        llm = ChatOpenAI(
            model=config.OPENAI_MODEL,
            temperature=0.0,
            max_tokens=150,
            api_key=config.OPENAI_API_KEY,
            timeout=config.LLM_TIMEOUT_SECONDS,
            max_retries=config.LLM_MAX_RETRIES,
        )

        rsi = _round4(state.get("rsi"), NEUTRAL_RSI_PLACEHOLDER)
        user_message = (
            f"Ticker symbol: {state.get('symbol', '')}\n"
            f"Current price: {price:.4f}\n"
            f"EMA trend (price vs EMA{config.EMA_LONG} and EMA{config.EMA_SHORT} vs "
            f"EMA{config.EMA_LONG}): {state.get('ema_trend', 'NEUTRAL')}\n"
            f"RSI({config.RSI_PERIOD}): {rsi:.2f}\n\n"
            "Return only the JSON object."
        )

        logger.info("[sentiment_analysis] Calling %s for %s.", config.OPENAI_MODEL, state.get("symbol"))
        response = llm.invoke(
            [SystemMessage(content=SENTIMENT_SYSTEM_PROMPT), HumanMessage(content=user_message)]
        )
        raw_text = _message_text(response)
        logger.debug("[sentiment_analysis] Raw LLM output: %r", raw_text)

        sentiment, confidence = parse_sentiment_response(raw_text)
        logger.info("[sentiment_analysis] sentiment=%s confidence=%.4f", sentiment, confidence)
        return {
            **state,
            "sentiment": sentiment,
            "confidence_score": _round4(confidence, DEFAULT_CONFIDENCE),
            "errors": errors,
        }
    except Exception as exc:
        logger.error("[sentiment_analysis] Failed (%s): %s", type(exc).__name__, exc)
        errors.append(
            f"Sentiment analysis failed ({type(exc).__name__}: {exc}); defaulted to NEUTRAL/0.5."
        )
        return {
            **state,
            "sentiment": DEFAULT_SENTIMENT,
            "confidence_score": DEFAULT_CONFIDENCE,
            "action": "HOLD",
            "errors": errors,
        }


# ===========================================================================
# PART D — Node 3: Risk & signal engine (PURE PYTHON — NO LLM)
# ===========================================================================
def _check_line(passed: bool, description: str, detail: str) -> str:
    """Format one condition result for the reasoning string."""
    return f"  [{'PASS' if passed else 'FAIL'}] {description} ({detail})"


def risk_signal_node(state: AgentState) -> AgentState:
    """
    Decide BUY / SELL / HOLD with deterministic rules and compute risk levels.

    BUY  — ALL of: EMA trend BULLISH, RSI < RSI_OVERSOLD_MAX, MACD BULLISH,
           sentiment != BEARISH.
    SELL — ANY of: price < EMA_long, RSI > RSI_OVERBOUGHT_MIN, MACD BEARISH.
    HOLD — otherwise.

    For BUY: stop = price - ATR_MULTIPLIER * ATR, target = price + stop distance
    * MIN_RR_RATIO; ATR falls back to ATR_FALLBACK_PCT of price if missing. A
    BUY is reverted to HOLD unless TP > price > SL and R:R >= MIN_RR_RATIO.
    """
    errors: List[str] = list(state.get("errors", []))
    base: AgentState = {
        **state,
        "stop_loss": 0.0,
        "take_profit": 0.0,
        "risk_reward_ratio": 0.0,
        "errors": errors,
    }
    try:
        price = _round4(state.get("price"), 0.0)

        # ---- Data integrity guard -------------------------------------
        if price <= 0.0 or _has_critical_errors(errors):
            return {
                **base,
                "action": "HOLD",
                "reasoning": (
                    "HOLD (data integrity guard): no valid price or a critical data error "
                    "occurred, so trade rules were not evaluated."
                ),
            }

        rsi = _round4(state.get("rsi"), NEUTRAL_RSI_PLACEHOLDER)
        macd_signal = str(state.get("macd_signal", "NEUTRAL")).upper()
        ema_trend = str(state.get("ema_trend", "NEUTRAL")).upper()
        ema_long = _round4_optional(state.get("ema_long"))
        atr = _round4_optional(state.get("atr"))
        sentiment = str(state.get("sentiment", DEFAULT_SENTIMENT)).upper()
        confidence = _round4(state.get("confidence_score"), DEFAULT_CONFIDENCE)

        # ---- BUY conditions (ALL required) ----------------------------
        buy_conditions = [
            (
                ema_trend == "BULLISH",
                "EMA trend is BULLISH",
                f"ema_trend={ema_trend}",
            ),
            (
                rsi < config.RSI_OVERSOLD_MAX,
                f"RSI < {config.RSI_OVERSOLD_MAX:g}",
                f"RSI={rsi:.2f}",
            ),
            (
                macd_signal == "BULLISH",
                "MACD is BULLISH (histogram > 0 and MACD > signal)",
                f"macd_signal={macd_signal}",
            ),
            (
                sentiment != "BEARISH",
                "Sentiment is not BEARISH",
                f"sentiment={sentiment}, confidence={confidence:.0%}",
            ),
        ]

        # ---- SELL conditions (ANY triggers) ---------------------------
        if ema_long is None:
            price_below_ema = False
            ema_detail = f"EMA{config.EMA_LONG} unavailable"
        else:
            price_below_ema = price < ema_long
            ema_detail = f"price={price:.4f}, EMA{config.EMA_LONG}={ema_long:.4f}"

        sell_conditions = [
            (price_below_ema, f"Price < EMA{config.EMA_LONG}", ema_detail),
            (
                rsi > config.RSI_OVERBOUGHT_MIN,
                f"RSI > {config.RSI_OVERBOUGHT_MIN:g}",
                f"RSI={rsi:.2f}",
            ),
            (
                macd_signal == "BEARISH",
                "MACD is BEARISH",
                f"macd_signal={macd_signal}",
            ),
        ]

        buy_signal = all(passed for passed, _, _ in buy_conditions)
        sell_signal = any(passed for passed, _, _ in sell_conditions)

        lines: List[str] = [
            f"BUY conditions ({sum(p for p, _, _ in buy_conditions)}/{len(buy_conditions)} met, all required):"
        ]
        lines += [_check_line(p, d, x) for p, d, x in buy_conditions]
        lines.append(
            f"SELL conditions ({sum(p for p, _, _ in sell_conditions)}/{len(sell_conditions)} met, any one triggers):"
        )
        lines += [_check_line(p, d, x) for p, d, x in sell_conditions]

        # ---- SELL takes precedence (capital protection) ---------------
        if sell_signal:
            triggered = ", ".join(d for p, d, _ in sell_conditions if p)
            lines.append(f"DECISION: SELL. Triggered by: {triggered}.")
            return {**base, "action": "SELL", "reasoning": "\n".join(lines)}

        if not buy_signal:
            failed = ", ".join(d for p, d, _ in buy_conditions if not p)
            lines.append(f"DECISION: HOLD. No SELL trigger, and BUY rules not met: {failed}.")
            return {**base, "action": "HOLD", "reasoning": "\n".join(lines)}

        # ---- BUY: ATR-based risk parameters ----------------------------
        if atr is not None and atr > 0.0:
            atr_used = atr
            atr_note = f"ATR({config.ATR_PERIOD})={atr:.4f}"
        else:
            atr_used = price * config.ATR_FALLBACK_PCT
            atr_note = (
                f"ATR unavailable; fallback {config.ATR_FALLBACK_PCT:.0%} of price = {atr_used:.4f}"
            )
            errors.append(f"ATR unavailable; used {config.ATR_FALLBACK_PCT:.0%} of price as volatility proxy.")

        risk_per_unit = config.ATR_MULTIPLIER * atr_used
        stop_loss = price - risk_per_unit
        take_profit = price + (risk_per_unit * config.MIN_RR_RATIO)
        risk_amount = price - stop_loss
        if risk_amount <= 0.0:
            lines.append("DECISION: HOLD. Risk per unit is zero or negative.")
            return {**base, "action": "HOLD", "reasoning": "\n".join(lines)}
        risk_reward = (take_profit - price) / risk_amount

        stop_loss_r = round(stop_loss, 4)
        take_profit_r = round(take_profit, 4)
        risk_reward_r = round(risk_reward, 4)

        lines.append(
            f"Risk model: {atr_note}; stop distance = {config.ATR_MULTIPLIER:g} x {atr_used:.4f} "
            f"= {risk_per_unit:.4f}; SL={stop_loss_r:.4f}, TP={take_profit_r:.4f}, "
            f"R:R=1:{risk_reward_r:.2f}."
        )

        # ---- Sanity checks --------------------------------------------
        if not (take_profit > price > stop_loss) or not (take_profit_r > price > stop_loss_r):
            lines.append("DECISION: HOLD. Sanity check failed: TP > price > SL is not true.")
            return {**base, "action": "HOLD", "reasoning": "\n".join(lines)}
        if stop_loss <= 0.0:
            lines.append("DECISION: HOLD. Computed stop-loss is at or below zero.")
            return {**base, "action": "HOLD", "reasoning": "\n".join(lines)}
        if risk_reward + RR_TOLERANCE < config.MIN_RR_RATIO:
            lines.append(
                f"DECISION: HOLD. R:R 1:{risk_reward:.2f} is below the minimum 1:{config.MIN_RR_RATIO:g}."
            )
            return {**base, "action": "HOLD", "reasoning": "\n".join(lines)}

        lines.append("DECISION: BUY. All 4 BUY conditions met, no SELL trigger, risk checks passed.")
        return {
            **base,
            "action": "BUY",
            "stop_loss": stop_loss_r,
            "take_profit": take_profit_r,
            "risk_reward_ratio": risk_reward_r,
            "reasoning": "\n".join(lines),
            "errors": errors,
        }
    except Exception as exc:
        logger.exception("[risk_signal] Unexpected failure: %s", exc)
        errors.append(f"risk_signal_node failed: {exc}")
        return {
            **base,
            "action": "HOLD",
            "reasoning": "HOLD: the risk engine failed unexpectedly, so no trade is taken.",
            "errors": errors,
        }


# ===========================================================================
# PART E — Graph assembly
# ===========================================================================
def build_trading_graph():
    """Build and compile the linear LangGraph pipeline with a MemorySaver checkpointer."""
    graph = StateGraph(AgentState)
    graph.add_node("technical_analysis", technical_analysis_node)
    graph.add_node("sentiment_analysis", sentiment_node)
    graph.add_node("risk_signal", risk_signal_node)

    graph.add_edge(START, "technical_analysis")
    graph.add_edge("technical_analysis", "sentiment_analysis")
    graph.add_edge("sentiment_analysis", "risk_signal")
    graph.add_edge("risk_signal", END)

    return graph.compile(checkpointer=MemorySaver())


# ===========================================================================
# PART F — Terminal report
# ===========================================================================
_ANSI = {
    "GREEN": "\033[92m",
    "RED": "\033[91m",
    "YELLOW": "\033[93m",
    "CYAN": "\033[96m",
    "BOLD": "\033[1m",
    "DIM": "\033[2m",
    "RESET": "\033[0m",
}
_ACTION_COLOURS = {"BUY": "GREEN", "SELL": "RED", "HOLD": "YELLOW"}
_REPORT_WIDTH = 72


def _style(text: str, *styles: str) -> str:
    """Wrap ``text`` in ANSI codes (disabled when the NO_COLOR env var is set)."""
    if os.getenv("NO_COLOR"):
        return text
    return "".join(_ANSI[s] for s in styles) + text + _ANSI["RESET"]


def _fmt_money(value: Any) -> str:
    """Format a price-like value, or 'N/A'."""
    number = _round4_optional(value)
    return "N/A" if number is None or number == 0.0 else f"${number:,.4f}"


def _fmt_number(value: Any, decimals: int = 4) -> str:
    """Format a plain number, or 'N/A'."""
    number = _round4_optional(value)
    return "N/A" if number is None else f"{number:,.{decimals}f}"


def _section(title: str) -> None:
    """Print a section header."""
    print()
    print(_style(title, "BOLD", "CYAN"))
    print("-" * _REPORT_WIDTH)


def _row(label: str, value: str) -> None:
    """Print one aligned label/value row."""
    print(f"  {label:<24}{value}")


def print_analysis_report(state: AgentState) -> None:
    """Print a formatted, colour-coded summary of the final agent state."""
    symbol = str(state.get("symbol", "?"))
    action = str(state.get("action", "HOLD")).upper()
    if action not in _ACTION_COLOURS:
        action = "HOLD"

    print("=" * _REPORT_WIDTH)
    print(_style(f"  TRADING AGENT REPORT  |  {symbol}", "BOLD"))
    print("=" * _REPORT_WIDTH)

    _section("TECHNICAL INDICATORS")
    _row("Price", _fmt_money(state.get("price")))
    _row(f"RSI({config.RSI_PERIOD})", _fmt_number(state.get("rsi"), 2))
    _row(
        f"MACD({config.MACD_FAST},{config.MACD_SLOW},{config.MACD_SIGNAL})",
        str(state.get("macd_signal", "N/A")),
    )
    _row("EMA trend", str(state.get("ema_trend", "N/A")))
    _row(f"EMA{config.EMA_SHORT}", _fmt_money(state.get("ema_short")))
    _row(f"EMA{config.EMA_LONG}", _fmt_money(state.get("ema_long")))
    _row(f"ATR({config.ATR_PERIOD})", _fmt_number(state.get("atr")))

    _section("SENTIMENT ANALYSIS")
    _row("Sentiment", str(state.get("sentiment", "N/A")))
    confidence = _round4(state.get("confidence_score"), DEFAULT_CONFIDENCE)
    _row("Confidence", f"{confidence * 100:.1f}%")

    _section("TRADE DECISION")
    _row("Action", _style(f" {action} ", "BOLD", _ACTION_COLOURS[action]))
    if action == "BUY":
        _row("Stop-loss", _fmt_money(state.get("stop_loss")))
        _row("Take-profit", _fmt_money(state.get("take_profit")))
        rr = _round4(state.get("risk_reward_ratio"), 0.0)
        _row("Risk:Reward", f"1:{rr:.2f}")

    _section("REASONING")
    reasoning = str(state.get("reasoning") or "No reasoning recorded.")
    for paragraph in reasoning.splitlines() or [reasoning]:
        indent = len(paragraph) - len(paragraph.lstrip(" "))
        prefix = "  " + " " * indent
        wrapped = textwrap.fill(
            paragraph.strip(),
            width=_REPORT_WIDTH,
            initial_indent=prefix,
            subsequent_indent=prefix + "    ",
        )
        print(wrapped if wrapped else "")

    errors = [str(e) for e in state.get("errors", []) if str(e).strip()]
    if errors:
        _section("WARNINGS")
        for error in errors:
            print(
                _style(
                    textwrap.fill(
                        error,
                        width=_REPORT_WIDTH,
                        initial_indent="  ! ",
                        subsequent_indent="    ",
                    ),
                    "YELLOW",
                )
            )

    print()
    print("=" * _REPORT_WIDTH)
    print(_style("  Research tool only. Not financial advice.", "DIM"))
    print("=" * _REPORT_WIDTH)


# ===========================================================================
# PART G — Run analysis
# ===========================================================================
def run_analysis(symbol: str) -> AgentState:
    """
    Validate configuration, build the graph and analyse ``symbol``.

    Missing/invalid OpenAI keys are non-fatal (sentiment degrades to
    NEUTRAL/0.5). Any other configuration error returns a HOLD state without
    running the graph. Never raises.
    """
    clean_symbol = (symbol or "").strip().upper()
    state = _initial_state(clean_symbol)

    config_errors = config.validate_config()
    for error in config_errors:
        logger.warning("Config: %s", error)
    state["errors"].extend(f"Config: {error}" for error in config_errors)

    fatal_errors = [e for e in config_errors if not e.startswith("OPENAI_API_KEY")]
    if fatal_errors:
        state["reasoning"] = "HOLD: configuration is invalid; fix the warnings below and re-run."
        return state

    if not SYMBOL_PATTERN.match(clean_symbol):
        state["errors"].append(f"{indicators.CRITICAL_PREFIX} Invalid ticker symbol {symbol!r}.")
        state["reasoning"] = "HOLD: the ticker symbol is invalid."
        return state

    try:
        graph = build_trading_graph()
        run_config = {"configurable": {"thread_id": config.CHECKPOINT_THREAD_ID}}
        final_state = graph.invoke(state, config=run_config)
        return cast(AgentState, final_state)
    except Exception as exc:
        logger.exception("run_analysis: graph execution failed: %s", exc)
        state["errors"].append(f"Graph execution failed: {exc}")
        state["action"] = "HOLD"
        state["reasoning"] = "HOLD: the analysis pipeline failed unexpectedly."
        return state


# ===========================================================================
# PART H — CLI entry point
# ===========================================================================
if __name__ == "__main__":
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL, logging.WARNING),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        stream=sys.stderr,
    )
    if os.name == "nt":
        os.system("")  # enables ANSI escape processing on Windows terminals

    ticker_symbol = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    final = run_analysis(ticker_symbol)
    print_analysis_report(final)
