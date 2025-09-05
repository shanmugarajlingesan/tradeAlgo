import os
import requests
import time
import json
import traceback

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

SCAN_URL = "https://shanntry-tradingalgo.hf.space/scan_all?signalTf=1h&exchangeId=okx&topN=60&minCandles=220&useBinanceUniverse=false&restrictToBinance=true"

def scan_and_notify():
    try:
        resp = requests.get(SCAN_URL)
        resp.raise_for_status()
        data = resp.json()

        message = json.dumps(data, indent=2)[:4000]  # pretty-format & trim
        send_message(message)
    except Exception as e:
        send_message(f"❌ Error in scan:\n{traceback.format_exc()}")

def send_message(text):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": text}
    requests.post(url, json=payload)

if __name__ == "__main__":
    scan_and_notify()
