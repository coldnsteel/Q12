import os
import asyncio
import aiohttp
import datetime
import json
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from sec_api import QueryApi, ExtractorApi
from dotenv import load_dotenv

load_dotenv()  # Load .env for keys

# API Keys from .env
POLYGON_KEY = os.getenv('POLYGON_KEY')
ALPHA_KEY = os.getenv('ALPHA_KEY')
FINNHUB_KEY = os.getenv('FINNHUB_KEY')
SEC_KEY = os.getenv('SEC_KEY')

# Download NLTK data quietly if needed
try:
    nltk.download('vader_lexicon', quiet=True)
except:
    pass

class Q12Agent:
    def __init__(self):
        self.session = None
        self.sia = SentimentIntensityAnalyzer()
        self.sec_query_api = QueryApi(api_key=SEC_KEY) if SEC_KEY else None
        self.sec_extractor_api = ExtractorApi(api_key=SEC_KEY) if SEC_KEY else None
        self.universe = []

    async def get_session(self):
        if self.session is None:
            self.session = aiohttp.ClientSession()
        return self.session

    async def build_universe(self) -> List[str]:
        # Build focused universe: AI/tech/defense, under $25, liquid, pre-earnings 0-12 weeks
        url = f"https://api.polygon.io/v3/reference/tickers?market=stocks&active=true&limit=1000&apiKey={POLYGON_KEY}&sort=ticker"
        async with (await self.get_session()).get(url) as resp:
            if resp.status != 200:
                raise Exception("Polygon API error")
            data = await resp.json()
            tickers = [t['ticker'] for t in data['results'] if t.get('market_cap', 0) > 1e8 and t.get('share_price', 100) < 25]  # Example filter
            # Add pre-earnings filter logic here (use Finnhub for earnings dates)
            self.universe = tickers[:50]  # Limit for demo
            return self.universe

    async def fetch_data(self, ticker: str, endpoint: str, params: Dict = None) -> Dict:
        # Async fetch with fallback: Finnhub > Alpha > Polygon
        try:
            base_urls = [
                f"https://finnhub.io/api/v1/{endpoint}?token={FINNHUB_KEY}",
                f"https://www.alphavantage.co/query?function={endpoint}&apikey={ALPHA_KEY}",
                f"https://api.polygon.io/v2/{endpoint}?apiKey={POLYGON_KEY}"
            ]
            async with (await self.get_session()).get(base_urls[0], params=params) as resp:  # Try Finnhub first
                if resp.status == 200:
                    return await resp.json()
            # Fallbacks if needed
        except Exception as e:
            print(f"Data fetch error for {ticker}: {e}")
            return {}

    async def detect_short_drop(self, short_data: List[Dict]) -> float:
        if len(short_data) < 2:
            return 0
        prev = short_data[-2]
        current = short_data[-1]
        if prev['days_to_cover'] > 0 and current['days_to_cover'] < prev['days_to_cover']:
            drop_ratio = (prev['days_to_cover'] - current['days_to_cover']) / prev['days_to_cover']
            return drop_ratio * 20  # Score contribution
        return 0

    async def ml_anomaly(self, df: pd.DataFrame) -> float:
        if df.empty:
            return 0
        features = df[['volume', 'obv']].fillna(0)
        model = IsolationForest(contamination=0.1)
        anomalies = model.fit_predict(features)
        score = np.mean(anomalies == -1) * 20  # Anomaly ratio contribution
        return score

    async def get_signals(self, ticker: str) -> Optional[Dict[str, Any]]:
        data = await self.fetch_data(ticker, "stock/candle", {"symbol": ticker, "timeframe": "day", "from": "12 weeks ago"})
        if not data:
            return None
        df = pd.DataFrame(data)
        # Calculate signals: volume, insiders, 13F, news sentiment, etc.
        score = 0  # Accumulate 0-100
        # Example calculations...
        short_score = await self.detect_short_drop(data.get('shorts', []))
        ml_score = await self.ml_anomaly(df)
        score += short_score + ml_score  # Add others
        signals = {"ticker": ticker, "score": min(score, 100), "classification": "Accumulating" if score > 50 else "Neutral"}
        # Generate PNG chart
        plt.plot(df['date'], df['volume'])
        plt.savefig(os.path.join('q12_reports', f"{ticker}_chart.png"))
        return signals

    async def scan(self) -> List[Dict[str, Any]]:
        if not self.universe:
            await self.build_universe()
        tasks = [self.get_signals(t) for t in self.universe]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        data = [r for r in results if isinstance(r, dict)]
        with open(os.path.join('q12_reports', 'scan_report.json'), 'w') as f:
            json.dump(data, f, indent=4)
        return data
