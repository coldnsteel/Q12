import json
from datetime import datetime

class Q12SentinelAgent:
    def __init__(self, threshold):
        self.threshold = threshold
        self.alert_history_file = "q12_alerts.log"

    def log_alert(self, ticker, price, change):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] ⚠️ THRESHOLD BREACH: {ticker} moved {change}% to ${price}\n"
        with open(self.alert_history_file, "a") as f:
            f.write(log_entry)
        return log_entry.strip()

    def evaluate_market(self, market_data):
        triggered_alerts = []
        for ticker, data in market_data.items():
            change = data.get("change_pct", 0)
            price = data.get("price", 0)
            if abs(change) >= self.threshold:
                alert_msg = self.log_alert(ticker, price, change)
                triggered_alerts.append({
                    "ticker": ticker,
                    "price": price,
                    "change_pct": change,
                    "message": alert_msg
                })
        return triggered_alerts
