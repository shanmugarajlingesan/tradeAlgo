# Crypto Scan Worker

Runs every 30 minutes:
- Calls Hugging Face `/scan_all` endpoint
- Sends formatted results to Telegram

### Deploy on Render

1. Fork this repo to GitHub
2. Go to [Render Dashboard](https://dashboard.render.com)
3. Create **New Web Service → Background Worker**
4. Connect GitHub repo
5. Add environment:
   - Python 3.11
6. Render automatically runs `worker: python worker.py` from `Procfile`

Done ✅ It runs 24/7.
