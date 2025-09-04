import requests
import time
import json
import traceback

# Telegram config
TELEGRAM_TOKEN = "8137258652:AAGgbKbx7lDEoLBSaaFSQah7Gupgm5fL9QU"
CHAT_ID = "8425367361"

# API endpoint
SCAN_URL = "https://shanntry-tradingalgo.hf.space/scan_all?signalTf=1h&exchangeId=okx&topN=60&minCandles=220&useBinanceUniverse=false&restrictToBinance=true"

def send_telegram_message(message: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        res = requests.post(url, json={"chat_id": CHAT_ID, "text": message, "parse_mode": "Markdown"})
        if res.status_code != 200:
            print("Failed to send message:", res.text)
    except Exception as e:
        print("Error sending Telegram:", e)

def format_pretty(response_json):
    """Format the response nicely for Telegram."""
    pretty = ["📊 *Crypto Scan Results* 📊"]
    for item in response_json.get("results", []):
        coin = item.get("symbol", "Unknown")
        signal = item.get("signal", "N/A")
        score = item.get("score", "N/A")
        entry = item.get("entry", "N/A")
        t1 = item.get("target1", "N/A")
        t2 = item.get("target2", "N/A")
        pretty.append(f"▫️ *{coin}*\n Signal: {signal}\n Score: {score}\n Entry: {entry}\n T1: {t1}\n T2: {t2}")
    return "\n\n".join(pretty)

def job():
    try:
        print("Fetching scan data...")
        res = requests.get(SCAN_URL, timeout=60)
        if res.status_code == 200:
            data = res.json()
            message = format_pretty(data)
            send_telegram_message(message)
        else:
            print("Failed scan:", res.text)
    except Exception as e:
        print("Error in job:", e)
        traceback.print_exc()

if __name__ == "__main__":
    while True:
        job()
        print("Sleeping 30 min...")
        time.sleep(1800)  # 30 minutes
