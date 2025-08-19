#!/usr/bin/env python3
"""
Crypto Long Scanner — SoftGuard + Faster Trades
- Strict 1h trend logic (EMA ribbon slope ↑, HH/HL, ADX)
- SOFT BTC guard with diagnostics (env: BTC_GUARD_MODE=soft|strict)
- Env-configurable score gate, Top-K, retest tolerance, etc.
- READY_TO_LONG enabled behind score gate for faster-but-curated trades
- Structural stops (EMA200 / swing-low with ATR buffer)
"""

import os, time, math, logging
import ccxt, numpy as np, pandas as pd
from pathlib import Path

# ===================== Config (ENV overrides supported) =====================
EXCHANGE_ID = os.getenv("EXCHANGE_ID", "binance")
TIMEFRAME   = os.getenv("TIMEFRAME", "1h")
LOOKBACK    = int(os.getenv("LOOKBACK", "400"))
TOP_N_BY_VOLUME = int(os.getenv("TOP_N_BY_VOLUME", "60"))
SLEEP_BETWEEN_CALLS_SEC = float(os.getenv("SLEEP_BETWEEN_CALLS_SEC", "0.8"))

# Risk & targets
RISK_PER_TRADE      = float(os.getenv("RISK_PER_TRADE", "0.0075"))   # 0.75% of equity
ACCOUNT_EQUITY_USDT = float(os.getenv("ACCOUNT_EQUITY_USDT", "1000"))
ATR_MULT_STOP_BUF   = float(os.getenv("ATR_MULT_STOP_BUF", "0.75"))  # buffer below structural stop
TP1_ATR             = float(os.getenv("TP1_ATR", "2.0"))
TP2_ATR             = float(os.getenv("TP2_ATR", "3.5"))

# Momentum/volume thresholds
VOL_MULT_FOR_BREAKOUT = float(os.getenv("VOL_MULT_FOR_BREAKOUT", "1.20"))
ADX_MIN               = float(os.getenv("ADX_MIN", "25"))

# High-Conviction gating (relaxed for faster trades)
MIN_SCORE  = int(os.getenv("MIN_SCORE", "70"))
TOP_K      = int(os.getenv("TOP_K", "8"))
ALLOW_READY_IF_SCORE = os.getenv("ALLOW_READY_IF_SCORE", "true").lower() in ("1","true","yes","y")

# BTC guardrail
REQUIRE_BTC_TREND = os.getenv("REQUIRE_BTC_TREND", "true").lower() in ("1","true","yes","y")
BTC_SYMBOL        = os.getenv("BTC_SYMBOL", "BTC/USDT:USDT")
BTC_GUARD_MODE    = os.getenv("BTC_GUARD_MODE", "soft").lower()  # soft | strict

# Breakout retest tolerance (fraction, e.g. 0.005 = 0.5% throwback)
RETEST_TOL = float(os.getenv("RETEST_TOL", "0.005"))

# ---------- Logging ----------
_DEFAULT_LOG = Path(os.getenv("SCANNER_LOG_PATH", Path.home() / "Downloads" / "scanner.log"))
_DEFAULT_LOG.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=str(_DEFAULT_LOG),
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    force=True,
)
logging.getLogger().addHandler(logging.StreamHandler())  # mirror to stdout

# ---------- Telegram (optional) ----------
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")

def telegram_send(text: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    import requests
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML", "disable_web_page_preview": True},
            timeout=10,
        )
        logging.info("Telegram alert sent")
    except Exception as e:
        logging.exception(f"Telegram error: {e}")

# ===================== Indicators =====================
def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    up = np.where(d > 0, d, 0.0)
    dn = np.where(d < 0, -d, 0.0)
    ru = pd.Series(up, index=s.index).ewm(span=n, adjust=False).mean()
    rd = pd.Series(dn, index=s.index).ewm(span=n, adjust=False).mean()
    rs = ru / (rd + 1e-9)
    return 100 - (100 / (1 + rs))

def macd_line(s: pd.Series, fast=12, slow=26, signal=9):
    ef, es = ema(s, fast), ema(s, slow)
    m = ef - es
    sig = ema(m, signal)
    return m, sig, m - sig

def atr(h: pd.Series, l: pd.Series, c: pd.Series, n: int = 14) -> pd.Series:
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(span=n, adjust=False).mean()

def adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    trv = atr(h, l, c, n)
    plus_dm  = (h - h.shift(1)).clip(lower=0)
    minus_dm = (l.shift(1) - l).clip(lower=0)
    plus_di  = 100 * (plus_dm.ewm(span=n, adjust=False).mean() / trv)
    minus_di = 100 * (minus_dm.ewm(span=n, adjust=False).mean() / trv)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
    return dx.ewm(span=n, adjust=False).mean()

def slope(s: pd.Series, w: int = 10) -> pd.Series:
    return (s - s.shift(w)) / (w + 1e-9)

# HH/HL pivots + ribbon slope
def find_swings(series: pd.Series, window: int = 3):
    highs, lows = [], []
    for i in range(window, len(series) - window):
        if series.iloc[i] == series.iloc[i - window : i + window + 1].max():
            highs.append((i, series.iloc[i]))
        if series.iloc[i] == series.iloc[i - window : i + window + 1].min():
            lows.append((i, series.iloc[i]))
    return highs, lows

def bullish_structure(df: pd.DataFrame, window: int = 3):
    highs, lows = find_swings(df["high"], window)
    if len(highs) < 2 or len(lows) < 2:
        return False
    _, h0 = highs[-2]
    _, h1 = highs[-1]
    _, l0 = lows[-2]
    _, l1 = lows[-1]
    return (h1 > h0 * 1.001) and (l1 > l0 * 1.001)

def ribbon_slope_up(df: pd.DataFrame, lookback: int = 10):
    em20, em50, em200 = df["ema20"], df["ema50"], df["ema200"]
    breadth = (em20 - em50) + (em50 - em200)
    return (em200.iloc[-1] > em200.iloc[-lookback]) and (breadth.iloc[-1] > breadth.iloc[-lookback])

# ===================== Exchange helpers =====================
def build_exchange():
    klass = getattr(ccxt, EXCHANGE_ID)
    return klass({"enableRateLimit": True, "timeout": 20000, "options": {"defaultType": "future"}})

def fetch_ohlcv_df(ex, symbol, timeframe=TIMEFRAME, limit=LOOKBACK) -> pd.DataFrame:
    raw = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df

def get_usdt_perp_symbols(ex, top_n=TOP_N_BY_VOLUME):
    markets = ex.load_markets()
    cands = [s for s, m in markets.items() if m.get("swap") and m.get("linear") and m.get("quote") == "USDT" and m.get("active", True)]
    try:
        tix = ex.fetch_tickers(cands)
        def vol(s):
            t = tix.get(s, {})
            qv = t.get("quoteVolume") or (t.get("info", {}) or {}).get("quoteVolume") or (t.get("info", {}) or {}).get("quoteVolume24h")
            try:
                return float(qv)
            except:
                return 0.0
        cands = sorted(cands, key=vol, reverse=True)
    except Exception as e:
        logging.warning(f"fetch_tickers failed; unranked symbols. {e}")
    return cands[:top_n]

# ===================== BTC guard (soft/strict) =====================
def btc_trend_guard(ex, mode: str = "soft"):
    """
    strict:  price > ema200 AND ema200 slope up
    soft:    NOT (price < ema200 AND ema200 slope down)  # only block when clearly bearish
    Returns (ok: bool, diag: dict)
    """
    try:
        df = fetch_ohlcv_df(ex, BTC_SYMBOL, TIMEFRAME, LOOKBACK)
        if len(df) < 100:
            return True, {"reason": "thin_data"}
        df["ema200"] = ema(df["close"], 200)
        close = float(df["close"].iloc[-1])
        ema2  = float(df["ema200"].iloc[-1])
        slope_up = (df["ema200"].iloc[-1] > df["ema200"].iloc[-11])  # ~last 10 bars
        if mode == "strict":
            ok = (close > ema2) and slope_up
        else:  # soft
            slope_down = not slope_up
            ok = not (close < ema2 and slope_down)
        diag = {"close": round(close, 2), "ema200": round(ema2, 2), "slope_up": bool(slope_up), "mode": mode}
        return bool(ok), diag
    except Exception as e:
        logging.warning(f"BTC guard failed; allowing trades. {e}")
        return True, {"reason": "exception"}

# ===================== Core logic =====================
def classify_symbol(df: pd.DataFrame):
    # Indicators
    df["ema20"]  = ema(df["close"], 20)
    df["ema50"]  = ema(df["close"], 50)
    df["ema200"] = ema(df["close"], 200)
    df["rsi"]    = rsi(df["close"], 14)
    m, ms, mh    = macd_line(df["close"])
    df["macd"], df["macd_sig"], df["macd_hist"] = m, ms, mh
    df["atr"]    = atr(df["high"], df["low"], df["close"])
    df["adx"]    = adx(df, 14)
    df["vol_sma20"] = df["volume"].rolling(20).mean()

    last, prev = df.iloc[-1], df.iloc[-2]

    # Strict 1h trend
    trend_up = (
        last["close"] > last["ema200"]
        and last["ema50"] > last["ema200"]
        and ribbon_slope_up(df, 10)
        and bullish_structure(df, 3)
        and last["adx"] > ADX_MIN
    )

    # Pullback & momentum improvement (READY_TO_LONG)
    pullback_zone      = (last["close"] <= last["ema20"]) and (last["close"] >= last["ema20"] - last["atr"])
    rsi_neutral_rising = (45 <= last["rsi"] <= 60) and (last["rsi"] > prev["rsi"])
    macd_improving     = last["macd_hist"] > prev["macd_hist"]
    ready_to_long      = trend_up and pullback_zone and rsi_neutral_rising and macd_improving

    # Breakout + retest
    prior_high20 = df["high"].rolling(20).max().iloc[-2]
    broke_out    = (last["close"] > prior_high20) and (last["rsi"] > 55) and (last["macd"] > last["macd_sig"])
    vol_ok       = last["volume"] > (VOL_MULT_FOR_BREAKOUT * (last["vol_sma20"] if not math.isnan(last["vol_sma20"]) else 0))
    retest_ok    = (prev["low"] <= prior_high20 * (1.0 + RETEST_TOL))  # env-tunable
    long_signal  = trend_up and broke_out and vol_ok and retest_ok

    # Structural stop
    entry = float(last["close"])
    swing_low = float(df["low"].rolling(10).min().iloc[-2])
    structural_ref = min(float(last["ema200"]), swing_low)
    stop = max(0.0, structural_ref - ATR_MULT_STOP_BUF * float(last["atr"]))
    a = float(last["atr"])
    tp1, tp2 = entry + TP1_ATR * a, entry + TP2_ATR * a

    # Volatility-aware leverage (6x..10x)
    atr_pct = a / entry
    lev = 10.0 - min(4.0, 400 * atr_pct)   # maps typical 0–1% ATR% to ~10x→6x
    lev = max(6.0, min(10.0, lev))

    risk_amount = ACCOUNT_EQUITY_USDT * RISK_PER_TRADE
    stop_dist   = max(1e-8, entry - stop)
    qty         = (risk_amount / stop_dist) * lev

    # Score
    score = 0
    score += 34 if trend_up else 0
    score += 17 if last["ema20"] > last["ema50"] > last["ema200"] else 0
    score += 16 if last["rsi"]  > 55 else (8 if 50 <= last["rsi"] <= 55 else 0)
    score += 14 if last["macd"] > last["macd_sig"] else 0
    score += 10 if vol_ok else 0
    score += 9  if retest_ok else 0
    score = int(min(100, score))

    # State
    state = "LONG" if long_signal else ("READY_TO_LONG" if (ALLOW_READY_IF_SCORE and ready_to_long and score >= MIN_SCORE) else "NEUTRAL")

    return {
        "state": state,
        "entry": round(entry, 6),
        "stop":  round(stop, 6),
        "tp1":   round(tp1, 6),
        "tp2":   round(tp2, 6),
        "atr":   round(a, 6),
        "suggested_qty": round(qty, 3),
        "qty_usdt_equiv": round(qty * entry, 2),
        "score": score,
        "context": {
            "trend_up": bool(trend_up),
            "retest_ok": bool(retest_ok),
            "vol_ok": bool(vol_ok),
            "prior_high20": round(float(prior_high20), 6),
        },
    }

def scan():
    ex = build_exchange()
    symbols = get_usdt_perp_symbols(ex, TOP_N_BY_VOLUME)
    logging.info(f"Scan start: {len(symbols)} symbols | tf={TIMEFRAME} | guard_mode={BTC_GUARD_MODE} | min_score={MIN_SCORE} | top_k={TOP_K}")

    btc_ok, btc_diag = (True, {"reason": "disabled"})
    if REQUIRE_BTC_TREND:
        btc_ok, btc_diag = btc_trend_guard(ex, BTC_GUARD_MODE)
    logging.info(f"BTC guard: {'OK' if btc_ok else 'BLOCKED'} | {btc_diag}")

    rows, alerts = [], []
    considered, passed_guard = 0, 0

    for sym in symbols:
        try:
            df = fetch_ohlcv_df(ex, sym, TIMEFRAME, LOOKBACK)
            if len(df) < 250:
                logging.info(f"Skip {sym}: insufficient candles ({len(df)})")
                time.sleep(SLEEP_BETWEEN_CALLS_SEC)
                continue

            considered += 1
            r = {"symbol": sym, **classify_symbol(df)}

            # Gate by BTC regime and score
            if btc_ok:
                allow = (r["score"] >= MIN_SCORE) and (r["state"] in ("LONG", "READY_TO_LONG"))
            else:
                # if blocked, allow only strong LONG with a higher bar
                allow = (r["state"] == "LONG") and (r["score"] >= max(MIN_SCORE, 82))

            if allow:
                passed_guard += 1
                rows.append(r)
                alerts.append(f"<b>{r['state']}</b> {sym} | score {r['score']} | entry {r['entry']} | stop {r['stop']} | TP1 {r['tp1']}")

            time.sleep(SLEEP_BETWEEN_CALLS_SEC)
        except ccxt.RateLimitExceeded as e:
            logging.warning(f"Rate limit {sym}: sleep 5s | {e}")
            time.sleep(5)
        except Exception as e:
            logging.exception(f"Error {sym}: {e}")

    # Prioritize & curate Top-K
    rows.sort(key=lambda x: (x["state"] != "LONG", -x["score"]))
    rows = rows[:TOP_K]

    # Save & alert
    if rows:
        pd.DataFrame(rows).to_csv("scan_results.csv", index=False)
    logging.info(f"Scan done: considered={considered}, allowed={len(rows)} (guard_passed_before_topk={passed_guard}), min_score={MIN_SCORE}, top_k={TOP_K}")
    if alerts:
        telegram_send("<b>Crypto Scanner — SoftGuard</b>\n\n" + "\n".join(alerts[:20]))
    return rows

if __name__ == "__main__":
    logging.info(f"Process start | log={_DEFAULT_LOG}")
    try:
        scan()
    except Exception as e:
        logging.exception(f"Fatal: {e}")
    finally:
        logging.info("Process end")
