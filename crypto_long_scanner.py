#!/usr/bin/env python3
"""
Crypto Long Scanner — v2 with Buy-Zone / Support Bounce Logic
"""
import os, time, math
import ccxt, numpy as np, pandas as pd
from datetime import datetime, timezone

# ---------- Config ----------
EXCHANGE_ID = 'binance'
TIMEFRAME = '1h'
LOOKBACK = 400
TOP_N_BY_VOLUME = 60
SLEEP_BETWEEN_CALLS_SEC = 0.8
RISK_PER_TRADE = 0.0075
ACCOUNT_EQUITY_USDT = 1000
VOL_MULT_FOR_BREAKOUT = 1.20
ATR_MULT_STOP = 1.5
TP1_ATR = 2.0
TP2_ATR = 3.5

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")


# ---------- 1h Trend Filter Utilities ----------
def find_swings(series: pd.Series, window:int=3):
    """
    Return lists of pivot highs and lows as tuples (index, price).
    """
    highs, lows = [], []
    for i in range(window, len(series)-window):
        if series.iloc[i] == series.iloc[i-window:i+window+1].max():
            highs.append((i, series.iloc[i]))
        if series.iloc[i] == series.iloc[i-window:i+window+1].min():
            lows.append((i, series.iloc[i]))
    return highs, lows

def bullish_structure(df: pd.DataFrame, window:int=3):
    """
    Check HH/HL structure on the 1h series using recent pivots.
    Returns True if last two swing highs form HH and last two swing lows form HL.
    """
    highs, lows = find_swings(df['high'], window=window)
    if len(highs) < 2 or len(lows) < 2:
        return False
    h1_idx, h1 = highs[-1]
    h0_idx, h0 = highs[-2]
    l1_idx, l1 = lows[-1]
    l0_idx, l0 = lows[-2]
    hh = h1 > h0 * 1.001
    hl = l1 > l0 * 1.001
    return bool(hh and hl)

def ribbon_slope_up(df: pd.DataFrame, lookback:int=10):
    """
    EMA ribbon breadth increasing and EMA200 sloping up.
    """
    em20, em50, em200 = df['ema20'], df['ema50'], df['ema200']
    breadth = (em20 - em50) + (em50 - em200)
    return (em200.iloc[-1] > em200.iloc[-lookback]) and (breadth.iloc[-1] > breadth.iloc[-lookback])

# ---------- Helpers ----------
def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()
def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=series.index).ewm(span=length, adjust=False).mean()
    roll_down = pd.Series(down, index=series.index).ewm(span=length, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-9)
    return 100 - (100 / (1 + rs))
def macd_line(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd = ema_fast - ema_slow
    macd_signal = ema(macd, signal)
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist
def atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(span=length, adjust=False).mean()
def rolling_high(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(length).max()
def rolling_low(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(length).min()
def slope(series: pd.Series, window: int = 5) -> pd.Series:
    return (series - series.shift(window)) / (window + 1e-9)
def safe_float(v, default=np.nan):
    try: return float(v)
    except Exception: return default
def telegram_send(text: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    import requests
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        requests.post(url, json={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True
        }, timeout=10)
    except Exception as e:
        print("Telegram error:", e)

# ---------- Price Action Utilities ----------
def is_bullish_engulfing(o, h, l, c):
    prev_o, prev_c = o.shift(1), c.shift(1)
    return (c > o) & (prev_c < prev_o) & (c >= prev_o) & (o <= prev_c)
def is_hammer(o, h, l, c, body_ratio_max=0.35, lower_wick_min=2.0):
    body = (c - o).abs()
    total = h - l
    upper_wick = h - c.where(c>o, o)
    lower_wick = (o.where(c>o, c) - l)
    cond_small_body = (body / (total + 1e-9)) <= body_ratio_max
    cond_lower_dominant = (lower_wick / (body + 1e-9)) >= lower_wick_min
    return cond_small_body & cond_lower_dominant
def find_support_zones(df: pd.DataFrame, window=3, lookback=200, tolerance_atr=0.6, min_touches=3):
    lows = df['low'].copy()
    atrv = df['atr'].copy()
    pivots = []
    for i in range(window, min(lookback, len(df)-window)):
        if lows.iloc[i] == lows.iloc[i-window:i+window+1].min():
            pivots.append((i, lows.iloc[i], atrv.iloc[i]))
    zones = []
    for idx, price, a in pivots:
        placed = False
        for z in zones:
            low, high, touches = z
            tol = (a if not math.isnan(a) else 0) * tolerance_atr
            if (low - tol) <= price <= (high + tol):
                new_low = min(low, price)
                new_high = max(high, price + tol)
                z[0], z[1], z[2] = new_low, new_high, touches+1
                placed = True
                break
        if not placed:
            zones.append([price, price, 1])
    zones = [z for z in zones if z[2] >= min_touches]
    zones.sort(key=lambda z: (-z[2], -z[1]))
    return zones[:5]
def bounce_from_zone(df: pd.DataFrame, zones, bars_confirm=2):
    if not zones: return None
    last = df.iloc[-1]; prev = df.iloc[-2]
    o, h, l, c = df['open'], df['high'], df['low'], df['close']
    engulf = bool(is_bullish_engulfing(o,h,l,c).iloc[-1])
    hammer = bool(is_hammer(o,h,l,c).iloc[-1])
    rsi_rising = last['rsi'] > prev['rsi']
    for low, high, touches in zones:
        if (last['low'] <= high) and (last['close'] > last['open']) and (engulf or hammer) and rsi_rising:
            return {"zone_low": low, "zone_high": high, "touches": touches, "cue": "engulf" if engulf else "hammer"}
    return None

# ---------- Strategy Logic ----------
def classify_symbol(df: pd.DataFrame):
    df['ema20'] = ema(df['close'], 20)
    df['ema50'] = ema(df['close'], 50)
    df['ema200'] = ema(df['close'], 200)
    df['rsi'] = rsi(df['close'], 14)
    macd, macd_sig, macd_hist = macd_line(df['close'], 12, 26, 9)
    df['macd'] = macd; df['macd_sig'] = macd_sig; df['macd_hist'] = macd_hist
    df['atr'] = atr(df['high'], df['low'], df['close'], 14)
    df['vol_sma20'] = df['volume'].rolling(20).mean()
    df['high20'] = rolling_high(df['high'].shift(1), 20)
    df['low20']  = rolling_low(df['low'].shift(1), 20)

    zones = find_support_zones(df, window=3, lookback=220, tolerance_atr=0.7, min_touches=3)
    bounce = bounce_from_zone(df, zones)

    last = df.iloc[-1]; prev = df.iloc[-2]

    trend_up_basic = (last['close'] > last['ema200']) and (last['ema50'] > last['ema200'])
    trend_up = trend_up_basic and ribbon_slope_up(df, lookback=10) and bullish_structure(df, window=3)
    pullback_zone = (last['close'] <= last['ema20']) and (last['close'] >= last['ema20'] - last['atr'])
    rsi_neutral_rising = (45 <= last['rsi'] <= 60) and (last['rsi'] > prev['rsi'])
    macd_improving = last['macd_hist'] > prev['macd_hist']

    ready_to_long = trend_up and pullback_zone and rsi_neutral_rising and macd_improving

    breakout = (last['close'] > last['high20']) and (last['rsi'] > 55) and (last['macd'] > last['macd_sig'])
    vol_ok = last['volume'] > (VOL_MULT_FOR_BREAKOUT * (last['vol_sma20'] if not math.isnan(last['vol_sma20']) else 0))
    long_signal = trend_up and breakout and vol_ok

    buy_zone_bounce = trend_up and (bounce is not None)

    entry = float(last['close'])
    recent_swing_low = float(df['low'].rolling(10).min().iloc[-2])
    stop_raw = min(float(last['ema200']), recent_swing_low)
    if buy_zone_bounce:
        stop_raw = min(stop_raw, float(bounce['zone_low']))
    stop = max(0.0, stop_raw - ATR_MULT_STOP * float(last['atr']))
    atr_val = float(last['atr'])
    tp1 = entry + TP1_ATR * atr_val
    tp2 = entry + TP2_ATR * atr_val

    risk_amount = ACCOUNT_EQUITY_USDT * RISK_PER_TRADE
    stop_dist = max(1e-8, entry - stop)
    qty = max(0.0, risk_amount / stop_dist)

    score = 0
    score += 34 if trend_up else 0
    score += 17 if last['ema20'] > last['ema50'] > last['ema200'] else 0
    score += 16 if last['rsi'] > 55 else (8 if 50 <= last['rsi'] <= 55 else 0)
    score += 14 if last['macd'] > last['macd_sig'] else 0
    score += 10 if vol_ok else 0
    score += 15 if buy_zone_bounce else 0
    score = int(min(100, score))

    state = "NEUTRAL"
    if buy_zone_bounce:
        state = "BUY_ZONE_BOUNCE"
    if ready_to_long and not long_signal and not buy_zone_bounce:
        state = "READY_TO_LONG"
    if long_signal:
        state = "LONG"

    context = {
        "trend_up": bool(trend_up),
        "pullback_zone": bool(pullback_zone),
        "rsi": round(float(last['rsi']), 2),
        "macd_hist": round(float(last['macd_hist']), 6),
        "vol_ok": bool(vol_ok),
        "ema20": round(float(last['ema20']), 6),
        "ema50": round(float(last['ema50']), 6),
        "ema200": round(float(last['ema200']), 6),
        "high20": round(float(last['high20']), 6) if not math.isnan(last['high20']) else None,
        "low20": round(float(last['low20']), 6) if not math.isnan(last['low20']) else None,
        "zones": [{"low": round(z[0],6), "high": round(z[1],6), "touches": int(z[2])} for z in zones[:3]],
        "bounce": bounce
    }

    return {
        "state": state,
        "entry": round(entry, 6),
        "stop": round(stop, 6),
        "tp1": round(tp1, 6),
        "tp2": round(tp2, 6),
        "atr": round(atr_val, 6),
        "qty_usdt_equiv": round(qty * entry, 2),
        "suggested_qty": round(qty, 3),
        "score": score,
        "context": context
    }

def build_exchange():
    exchange_class = getattr(ccxt, EXCHANGE_ID)
    exchange = exchange_class({
        "enableRateLimit": True,
        "timeout": 20000,
        "options": {"defaultType": "future"}
    })
    return exchange
def get_usdt_perp_symbols(exchange, top_n=TOP_N_BY_VOLUME):
    markets = exchange.load_markets()
    candidates = []
    for sym, m in markets.items():
        if m.get('swap') and m.get('linear') and m.get('quote') == 'USDT':
            if m.get('active', True):
                candidates.append(sym)
    symbols_ranked = candidates
    try:
        tickers = exchange.fetch_tickers(candidates)
        def vol_key(s):
            t = tickers.get(s, {})
            qv = t.get('quoteVolume')
            if qv is None and isinstance(t.get('info'), dict):
                qv = t['info'].get('quoteVolume') or t['info'].get('quoteVolume24h')
            return safe_float(qv, 0.0)
        symbols_ranked = sorted(candidates, key=vol_key, reverse=True)
    except Exception as e:
        print("fetch_tickers failed, using unranked symbols:", e)
    return symbols_ranked[:top_n]
def fetch_ohlcv_df(exchange, symbol, timeframe=TIMEFRAME, limit=LOOKBACK) -> pd.DataFrame:
    raw = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(raw, columns=["timestamp","open","high","low","close","volume"])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    return df
def scan():
    ex = build_exchange()
    symbols = get_usdt_perp_symbols(ex, TOP_N_BY_VOLUME)
    print(f"Scanning {len(symbols)} symbols on {EXCHANGE_ID} futures ({TIMEFRAME}) ...")
    rows, alerts = [], []
    for sym in symbols:
        try:
            df = fetch_ohlcv_df(ex, sym, TIMEFRAME, LOOKBACK)
            if len(df) < 250:
                time.sleep(SLEEP_BETWEEN_CALLS_SEC); continue
            result = classify_symbol(df)
            rows.append({"symbol": sym, **result})
            if result["state"] in ("LONG","READY_TO_LONG","BUY_ZONE_BOUNCE"):
                context = result["context"]
                zone_txt = ""
                if result["state"] == "BUY_ZONE_BOUNCE" and context.get("bounce"):
                    b = context["bounce"]
                    zone_txt = f"\nzone: {b['zone_low']}–{b['zone_high']} (touches={b['touches']}, cue={b['cue']})"
                alerts.append(
                    f"<b>{result['state']}</b> {sym}\n"
                    f"score: {result['score']}\n"
                    f"entry: {result['entry']}\n"
                    f"stop:  {result['stop']}\n"
                    f"TP1:   {result['tp1']}  TP2: {result['tp2']}\n"
                    f"ATR:   {result['atr']}\n"
                    f"qty≈   {result['suggested_qty']} (notional≈{result['qty_usdt_equiv']} USDT)"
                    f"{zone_txt}\n"
                )
            time.sleep(SLEEP_BETWEEN_CALLS_SEC)
        except ccxt.RateLimitExceeded as e:
            print("Rate limit exceeded, sleeping 5s...", e); time.sleep(5)
        except Exception as e:
            print(f"Error {sym}:", e)
    priority = {"LONG": 0, "BUY_ZONE_BOUNCE": 1, "READY_TO_LONG": 2, "NEUTRAL": 3}
    rows.sort(key=lambda r: (priority.get(r["state"], 9), -r["score"]))
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    header = f"{'Symbol':<18} {'State':<16} {'Score':<5} {'Entry':>12} {'Stop':>12} {'TP1':>12} {'TP2':>12}"
    print(f"\n=== Results @{now} ==="); print(header); print("-"*len(header))
    for r in rows[:60]:
        print(f"{r['symbol']:<18} {r['state']:<16} {r['score']:<5} {r['entry']:>12.6f} {r['stop']:>12.6f} {r['tp1']:>12.6f} {r['tp2']:>12.6f}")
    out_df = pd.DataFrame(rows)
    out_df.to_csv("scan_results.csv", index=False)
    out_df.to_json("scan_results.json", orient="records", indent=2, force_ascii=False)
    if alerts:
        text = "<b>Crypto Scanner</b> — Top signals\n\n" + "\n".join(alerts[:20])
        telegram_send(text)
    print("\nSaved: scan_results.csv, scan_results.json")
    return out_df
if __name__ == "__main__":
    scan()