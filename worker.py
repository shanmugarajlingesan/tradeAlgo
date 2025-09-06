import requests
import json
from tabulate import tabulate

TELEGRAM_TOKEN = "8137258652:AAGgbKbx7lDEoLBSaaFSQah7Gupgm5fL9QU"
CHAT_ID = "8425367361"
SCAN_URL = "https://shanntry-tradingalgo.hf.space/scan_all?signalTf=1h&exchangeId=okx&topN=60&minCandles=220&useBinanceUniverse=false&restrictToBinance=true"


def send_telegram(msg: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    requests.post(url, json={"chat_id": CHAT_ID, "text": msg, "parse_mode": "Markdown"})


def scan_and_notify():
    try:
        resp = requests.get(SCAN_URL, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        signals = []
        if isinstance(data, list):
            signals = data
        elif isinstance(data, dict):
            signals = data.get("signals") or data.get("data") or []

        if not signals:
            send_telegram("⚠️ No valid signals found.")
            return

        table_data = []
        for sig in signals:
            symbol = sig.get("symbol") or sig.get("coin") or "?"
            signal = sig.get("signal") or sig.get("status") or "?"
            score = sig.get("score") or "?"
            entry = sig.get("entry") or "-"
            t1 = sig.get("target1") or "-"
            t2 = sig.get("target2") or "-"
            table_data.append([symbol, signal, score, entry, t1, t2])

        table = tabulate(table_data, headers=["Coin", "Signal", "Score", "Entry", "T1", "T2"], tablefmt="pretty")
        send_telegram(f"📊 *Scan Results*\n```\n{table}\n```")

    except Exception as e:
        send_telegram(f"❌ Error: {e}")


if __name__ == "__main__":
    scan_and_notify()
