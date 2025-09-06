import requests
import traceback
import time

# Telegram credentials
TELEGRAM_TOKEN = "8137258652:AAGgbKbx7lDEoLBSaaFSQah7Gupgm5fL9QU"
CHAT_ID = "8425367361"

# Scan URL
SCAN_URL = "https://shanntry-tradingalgo.hf.space/scan_all?signalTf=1h&exchangeId=okx&topN=60&minCandles=220&useBinanceUniverse=false&restrictToBinance=true"

def send_message(text: str):
    """Send message to Telegram bot."""
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        resp = requests.post(url, data={"chat_id": CHAT_ID, "text": text, "parse_mode": "Markdown"})
        resp.raise_for_status()
    except Exception as e:
        print(f"Error sending Telegram message: {e}")

def scan_and_notify():
    """Fetch scan results and send to Telegram as a table."""
    try:
        resp = requests.get(SCAN_URL, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        if not isinstance(data, list) or len(data) == 0:
            send_message("⚠️ No valid data from scan API")
            return

        # Pick important columns if available
        important_cols = [c for c in ["symbol", "signal", "score"] if c in data[0]]
        if not important_cols:
            important_cols = list(data[0].keys())[:4]  # fallback first 4 cols

        rows = data[:15]  # avoid Telegram length limit

        # Build table
        table_lines = []
        header_line = " | ".join(h[:10].ljust(10) for h in important_cols)
        sep_line = "-+-".join("-" * 10 for _ in important_cols)
        table_lines.append(header_line)
        table_lines.append(sep_line)

        for row in rows:
            line = " | ".join(str(row.get(h, ""))[:10].ljust(10) for h in important_cols)
            table_lines.append(line)

        table = "```\n" + "\n".join(table_lines) + "\n```"

        send_message("📊 *Scan Results*\n" + table)

    except Exception:
        send_message(f"❌ Error in scan:\n{traceback.format_exc()}")

if __name__ == "__main__":
    print("Starting scan loop...")
    while True:
        scan_and_notify()
        time.sleep(1800)  # 30 minutes
