def scan_and_notify():
    try:
        resp = requests.get(SCAN_URL)
        resp.raise_for_status()
        data = resp.json()

        if not isinstance(data, list):
            send_message("⚠️ API did not return a list")
            return

        # Limit number of rows to avoid huge messages
        rows = data[:15]

        # Extract headers from keys of first item
        headers = list(rows[0].keys())

        # Build table
        table_lines = []
        header_line = " | ".join(h[:10].ljust(10) for h in headers)  # trim long headers
        sep_line = "-+-".join("-" * 10 for _ in headers)
        table_lines.append(header_line)
        table_lines.append(sep_line)

        for row in rows:
            line = " | ".join(str(row.get(h, ""))[:10].ljust(10) for h in headers)
            table_lines.append(line)

        table = "```\n" + "\n".join(table_lines) + "\n```"

        send_message(table)

    except Exception:
        send_message(f"❌ Error in scan:\n{traceback.format_exc()}")
