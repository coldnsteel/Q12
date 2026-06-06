import os, sys, json, time, random, threading, http.server, socketserver
from datetime import datetime
from q12_agent import Q12SentinelAgent

TICKERS_ENV = os.getenv("Q12_TICKERS", "AAPL,MSFT,GOOGL,TSLA")
ALERT_THRESHOLD_PCT = float(os.getenv("Q12_ALERT_THRESHOLD_PCT", "2.0"))
POLL_INTERVAL_SECS = int(os.getenv("Q12_POLL_INTERVAL", "15"))
PORT = 8081

class Q12QuietHTTPHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format, *args): pass

def start_ui_server():
    socketserver.TCPServer.allow_reuse_address = True
    try:
        with socketserver.TCPServer(("127.0.0.1", PORT), Q12QuietHTTPHandler) as httpd:
            print(f"[*] Secure Local UI Server active at http://127.0.0.1:{PORT}")
            httpd.serve_forever()
    except Exception as e:
        print(f"[!] Warning: UI server failed to bind to port {PORT}: {e}")

class Q12Engine:
    def __init__(self):
        self.tickers = [t.strip() for t in TICKERS_ENV.split(",") if t.strip()]
        self.agent = Q12SentinelAgent(threshold=ALERT_THRESHOLD_PCT)
        self.cache_file = "q12_market_state.json"
        print(f"[*] Q12 Sovereign Stock Watcher Initialized.")

    def load_cached_prices(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r") as f: return json.load(f)
            except: pass
        return {}

    def save_cached_prices(self, state):
        try:
            with open(self.cache_file, "w") as f: json.dump(state, f, indent=2)
        except Exception as e: pass

    def fetch_mock_or_real_metrics(self):
        metrics = {}
        cached = self.load_cached_prices()
        for ticker in self.tickers:
            prev_price = cached.get(ticker, {}).get("price", random.uniform(100, 300))
            change_percent = random.uniform(-3.5, 3.5)
            new_price = prev_price * (1 + (change_percent / 100))
            metrics[ticker] = {
                "price": round(new_price, 2),
                "change_pct": round(change_percent, 2),
                "timestamp": datetime.now().isoformat()
            }
        self.save_cached_prices(metrics)
        return metrics

    def run_loop(self):
        print("[*] Generating initial telemetry sync frame...")
        market_data = self.fetch_mock_or_real_metrics()
        alerts = self.agent.evaluate_market(market_data)
        report = {
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": "SECURE • MONITORING ACTIVE",
            "metrics": market_data,
            "alerts": alerts
        }
        with open("q12_report.json", "w") as f:
            json.dump(report, f, indent=2)

        server_thread = threading.Thread(target=start_ui_server, daemon=True)
        server_thread.start()
        print("[+] Sovereign monitoring cycle active. Press Ctrl+C to terminate.")
        try:
            while True:
                market_data = self.fetch_mock_or_real_metrics()
                alerts = self.agent.evaluate_market(market_data)
                report = {
                    "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "status": "SECURE • MONITORING ACTIVE",
                    "metrics": market_data,
                    "alerts": alerts
                }
                with open("q12_report.json", "w") as f:
                    json.dump(report, f, indent=2)
                print(f"[{datetime.now().strftime("%T")}]: Sync written.")
                time.sleep(POLL_INTERVAL_SECS)
        except KeyboardInterrupt:
            print("\n[-] Q12 Sentinel offline.")

if __name__ == "__main__":
    engine = Q12Engine()
    engine.run_loop()
