import pandas as pd
import numpy as np
import aiohttp
import asyncio
import datetime
import os
import scipy.stats
from sklearn.ensemble import IsolationForest
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import matplotlib.pyplot as plt
from sec_api import InsiderTradingApi, ThirteenFApi  # Still sync; wrapped in to_thread

# --- CONFIGURATION (MUST BE REPLACED WITH ACTUAL KEYS) ---
POLYGON_KEY = 'your_polygon_key'
ALPHA_KEY = 'your_alpha_key'
FINNHUB_KEY = 'your_finnhub_key'
SEC_KEY = 'your_sec_key'
OUTPUT_DIR = 'q12_reports'

# Initialize Sync Clients (SEC is sync)
try:
    insider_api = InsiderTradingApi(SEC_KEY)
    thirteenf_api = ThirteenFApi(SEC_KEY)
    # Ensure VADER is downloaded for sentiment analysis
    try:
        nltk.data.find('sentiment/vader_lexicon')
    except LookupError:
        nltk.download('vader_lexicon')
    sia = SentimentIntensityAnalyzer()
except Exception as e:
    print(f"Warning: API Client initialization failed. Check keys. {e}")


# --- 1. HELPER FUNCTIONS (ATR, OBV, etc.) ---

def compute_atr(df, period=14):
    """Calculates Average True Range (ATR) manually."""
    df['high_low'] = df['high'] - df['low']
    df['high_close'] = abs(df['high'] - df['close'].shift())
    df['low_close'] = abs(df['low'] - df['close'].shift())
    df['tr'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
    # Use EWM for a more responsive ATR than SMA, common in TA
    df['atr'] = df['tr'].ewm(span=period, adjust=False).mean()
    return df.dropna(subset=['atr'])

def compute_obv(df):
    """Calculates On-Balance Volume (OBV)."""
    df['daily_ret_sign'] = np.sign(df['close'] - df['close'].shift(1))
    df['obv'] = (df['volume'] * df['daily_ret_sign']).cumsum()
    df['obv'] = df['obv'].fillna(0)
    return df

async def get_next_earnings(ticker):
    """
    PRIORITY WRAPPER: Gets the next earnings date using Finnhub, 
    falling back to Alpha Vantage if Finnhub fails.
    """
    today = datetime.date.today().isoformat()
    future = (datetime.date.today() + datetime.timedelta(days=90)).isoformat()

    async with aiohttp.ClientSession() as session:
        # 1. Try Finnhub (Preferred)
        try:
            url = f'https://finnhub.io/api/v1/calendar/earnings?symbol={ticker}&from={today}&to={future}&token={FINNHUB_KEY}'
            async with session.get(url, timeout=5) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    calendar = data.get('earningsCalendar', [])
                    if calendar and 'date' in calendar[0]:
                        return calendar[0]['date']
        except Exception as e:
            print(f"Finnhub failed for {ticker}: {e}. Falling back to Alpha Vantage.")

        # 2. Try Alpha Vantage (Fallback)
        try:
            alpha_url = f'https://www.alphavantage.co/query?function=EARNINGS_CALENDAR&symbol={ticker}&horizon=3month&apikey={ALPHA_KEY}'
            async with session.get(alpha_url, timeout=5) as resp:
                if resp.status == 200:
                    content = await resp.text()
                    lines = content.splitlines()
                    if len(lines) > 1:
                        # Assuming CSV structure
                        data = pd.read_csv(pd.io.common.StringIO(content))
                        if not data.empty and 'reportDate' in data.columns:
                            return data['reportDate'].iloc[0]
        except Exception as e:
            print(f"Alpha Vantage failed for {ticker}: {e}")

    return None

# --- 2. UNIVERSE BUILDING ---

async def build_universe(max_price=25, min_volume=100000) -> list:
    """Builds the target universe using Polygon SIC code filtering and snapshot checks."""
    sic_codes = '3674,7371,7372,3571'  # Semiconductors, Software, Computers
    keywords = ['AI', 'SENSOR', 'MIDDLEWARE', 'DEFENSE', 'MACHINE LEARNING', 'CLOUD', 'DATA']
    curated = ['SOUN', 'NRDY', 'BBAI', 'REKR', 'INOD', 'NVTS', 'AEYE', 'HLLY', 'SFT', 'CGNT', 'PLTR']

    sic_tickers = []
    
    async with aiohttp.ClientSession() as session:
        try:
            # Fetch broad list using SIC codes (efficient filtering at source)
            url = f'https://api.polygon.io/v3/reference/tickers?market=stocks&sic_code.in={sic_codes}&active=true&limit=1000&apiKey={POLYGON_KEY}'
            async with session.get(url, timeout=10) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    tickers_by_sic = data.get('results', [])
                    
                    # Filter by description keywords for AI relevance
                    for t in tickers_by_sic:
                        desc = t.get('name', '').upper()
                        if any(k in desc for k in keywords):
                            sic_tickers.append(t['ticker'])
                 
        except Exception as e:
            print(f"Error fetching SIC tickers from Polygon: {e}")
            # Fallback to curated if API fails
            return curated

    all_tickers = list(set(curated + sic_tickers))

    # Filter by Price and Liquidity (Batch Snapshot for efficiency)
    cheap_liquid_tickers = []
    chunk_size = 250  # Optimal for Polygon's limits
    for i in range(0, len(all_tickers), chunk_size):
        chunk = all_tickers[i:i + chunk_size]
        try:
            url = f'https://api.polygon.io/v3/snapshot?limit=250&ticker.any_of={",".join(chunk)}&apiKey={POLYGON_KEY}'
            async with session.get(url, timeout=10) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    snapshots = data.get('results', [])
                    for snap in snapshots:
                        session_data = snap.get('session', {})
                        # Check current close price and daily volume
                        if 'close' in session_data and session_data['close'] <= max_price and \
                           'volume' in session_data and session_data['volume'] >= min_volume:
                            cheap_liquid_tickers.append(snap['ticker'])
            
            await asyncio.sleep(1)  # Rate limit buffer
            
        except Exception as e:
            print(f"Error fetching snapshot for chunk: {e}")
            await asyncio.sleep(5)  # Backoff on error

    return cheap_liquid_tickers

# --- 3. SIGNAL DETECTION LOGIC ---

def detect_accumulation(df):
    """Q12 Core: High-Volume/Low-Volatility Pattern."""
    df = compute_atr(df.copy())
    # min_periods=20 ensures we use a full 20-day average
    df['vol_sma_20'] = df['volume'].rolling(20, min_periods=20).mean()
    df['daily_change'] = abs(df['close'] - df['open'])
    
    # Flag: High Volume (>1.1x SMA) AND Low Price Impact (<0.5x ATR)
    # Using shift(1) ensures the current day's volume is not factored into its own SMA test
    df['accum_day'] = (df['volume'] > df['vol_sma_20'].shift(1).fillna(0) * 1.10) & \
                      (df['daily_change'] < df['atr'] * 0.5)
    
    last5 = df['accum_day'].tail(5).sum() >= 3
    last15 = df['accum_day'].tail(15).sum() >= 7
    
    score = 40 if last5 else 20 if last15 else 0
    flags = ["Strong Recent (3/5)" if last5 else "Sustained (7/15)" if last15 else ""]
    return {'score': score, 'flags': flags, 'df': df} # Return DF for ML later

async def detect_short_drop(ticker, start_date, end_date, df):
    """Detects SIR drop >= 15% with flat price action."""
    async with aiohttp.ClientSession() as session:
        try:
            url = f'https://api.polygon.io/v1/short_interest/{ticker}?date.gte={start_date}&limit=10&apiKey={POLYGON_KEY}'
            async with session.get(url, timeout=5) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    shorts = data.get('results', [])
                    if len(shorts) < 2: return {'score': 0, 'flags': []}
                    
                    shorts = sorted(shorts, key=lambda s: s['settlement_date'], reverse=True)
                    current = shorts[0]; prev = shorts[1]
                    drop_ratio = (prev['days_to_cover'] - current['days_to_cover']) / prev['days_to_cover'] if prev['days_to_cover'] > 0 else 0
                    
                    # Check price flat (price change < 5% between settlement dates)
                    period_start = datetime.date.fromisoformat(prev['settlement_date'])
                    period_end = datetime.date.fromisoformat(current['settlement_date'])
                    df_period = df[(df['timestamp'] >= period_start) & (df['timestamp'] <= period_end)]
                    
                    price_flat = True
                    if not df_period.empty and df_period['close'].iloc[0] > 0:
                        price_change = abs(df_period['close'].iloc[-1] - df_period['close'].iloc[0]) / df_period['close'].iloc[0]
                        price_flat = price_change < 0.05
                    
                    if drop_ratio >= 0.15 and price_flat:
                        return {'score': 20, 'flags': [f'SIR drop {drop_ratio:.2f}']}
                    return {'score': 0, 'flags': []}
        except Exception as e:
            print(f"Error in short interest for {ticker}: {e}")
            return {'score': 0, 'flags': []}

async def detect_insider_buys(ticker, start_date):
    """Detects net open-market buys by insiders (Form 4). Wrapped sync call."""
    loop = asyncio.get_running_loop()
    data = await loop.run_in_executor(None, lambda: insider_api.get_data({'symbol': ticker, 'from': start_date}))
    buys = [trade for trade in data if trade['transactionType'] == 'P-Purchase']
    score = 15 if len(buys) > 0 else 0
    flags = [f"{len(buys)} P-Buys"] if score > 0 else []
    return {'score': score, 'flags': flags}

async def detect_13f_increase(ticker):
    """Detects increase in institutional holdings vs. prior quarter. Wrapped sync call."""
    loop = asyncio.get_running_loop()
    filings = await loop.run_in_executor(None, lambda: thirteenf_api.get_filings(ticker))
    if len(filings) < 2: return {'score': 0, 'flags': []}
    increase = filings[-1]['holdings'] > filings[-2]['holdings']
    score = 15 if increase else 0
    flags = ["13F Increase"] if increase else []
    return {'score': score, 'flags': flags}

async def detect_news_proxy(ticker, start_date):
    """Detects positive news with foreign/AI contract keywords."""
    # Prioritizing Finnhub for summaries/headlines
    end_date = datetime.date.today().isoformat()
    url = f'https://finnhub.io/api/v1/company-news?symbol={ticker}&from={start_date}&to={end_date}&token={FINNHUB_KEY}'
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url, timeout=5) as resp:
                news_data = await resp.json()
                summaries = [n['summary'] or n['headline'] for n in news_data if 'summary' in n or 'headline' in n]
        except:
            # Fallback to Polygon
            polygon_url = f'https://api.polygon.io/v2/reference/news?ticker={ticker}&published_utc.gte={start_date}&limit=50&apiKey={POLYGON_KEY}'
            async with session.get(polygon_url, timeout=5) as resp:
                data = await resp.json()
                news = data.get('results', [])
                summaries = [n['description'] or n['title'] for n in news if 'description' in n or 'title' in n]
    
    keywords = ["CFIUS", "foreign partnership", "AI contract", "trusted intermediary layer", "sensor infrastructure", "government contract", "AI deployment", "fulfillment contract", "large-scale deployment"]
    positive_news = [s for s in summaries if any(kw.lower() in s.lower() for kw in keywords) and sia.polarity_scores(s)['compound'] > 0.1]
    
    score = 10 if len(positive_news) >= 2 else 0
    flags = [f"{len(positive_news)} relevant news hits"] if score > 0 else []
    return {'score': score, 'flags': flags}

def ml_anomaly(df_accumulated):
    """ML Anomaly Detection using Isolation Forest on normalized features."""
    df = compute_obv(df_accumulated.copy())
    
    # 1. Feature Engineering
    df['volume_norm'] = df['volume'] / df['volume'].rolling(60, min_periods=1).mean()
    df['range_atr'] = (df['high'] - df['low']) / df['atr']
    df['accum_day_int'] = df['accum_day'].astype(int)
    
    # OBV Slope (20-day trend)
    df['obv_slope'] = 0.0
    for i in range(20, len(df)):
        try:
            slope, _ , _, _, _ = scipy.stats.linregress(range(20), df['obv'].iloc[i-20:i])
            df.loc[df.index[i], 'obv_slope'] = slope
        except ValueError:
            pass # Skip if insufficient data for linregress

    features = df[['volume_norm', 'range_atr', 'accum_day_int', 'obv_slope']].dropna()
    if features.empty: return 0
    
    # 2. Isolation Forest
    iso = IsolationForest(contamination=0.05, random_state=42)
    scores = iso.fit_predict(features)
    
    # If the latest score is an outlier (scores will be 1 for inliers, -1 for outliers)
    # Check if a recent day was flagged as an anomaly (-1)
    if scores[-5:].min() == -1:
        return 10
    return 0

# --- 4. SCORING AND MAIN EXECUTION ---

def compute_total_score(signals, tte_weeks, ml_bonus):
    """Applies base weights, time decay, and ML bonus."""
    base = sum(s['score'] for s in signals.values())
    
    # Time Decay Multiplier for TTE < 6 Weeks
    if tte_weeks < 6 and tte_weeks > 0:
        lambda_val = (6 - tte_weeks) / 6
        
        # Accumulation: 40% * lambda
        base += signals['accum']['score'] * lambda_val 
        
        # Short Interest: 20% * (lambda / 2)
        base += signals['short']['score'] * (lambda_val / 2)
        
    total = min(base + ml_bonus, 100)
    return total

class Q12Agent:
    async def get_data(self, ticker, start, end):
        """Async data fetch for aggregates using aiohttp (replaces sync Polygon client)."""
        async with aiohttp.ClientSession() as session:
            url = f'https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}?limit=200&apiKey={POLYGON_KEY}'
            async with session.get(url, timeout=10) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    aggs = data.get('results', [])
                    if not aggs or len(aggs) < 60: return None
                    df = pd.DataFrame(aggs)
                    df.rename(columns={'t': 'timestamp', 'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'}, inplace=True)
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms').dt.date
                    df = df.sort_values(by='timestamp').reset_index(drop=True)
                    return df
                else:
                    print(f"Error fetching aggregates for {ticker}: {resp.status}")
                    return None

    async def get_signals(self, ticker):
        """Async method to get signals for a single ticker."""
        edate_str = await get_next_earnings(ticker)
        if edate_str is None: return None
        
        try:
            today = datetime.date.today()
            edate = datetime.date.fromisoformat(edate_str)
            tte_weeks = (edate - today).days / 7
            if not (0 < tte_weeks <= 12): return None
        except ValueError:
            return None
            
        start = (today - datetime.timedelta(weeks=12)).isoformat()
        
        df = await self.get_data(ticker, start, today.isoformat())
        if df is None: return None
        
        accum_result = detect_accumulation(df.copy())
        signals = {
            'accum': accum_result,
            'short': await detect_short_drop(ticker, start, today.isoformat(), df.copy()),
            'insider': await detect_insider_buys(ticker, start),
            '13f': await detect_13f_increase(ticker),
            'news': await detect_news_proxy(ticker, start)
        }
        
        ml_bonus = ml_anomaly(accum_result['df'].copy())
        
        score = compute_total_score(signals, tte_weeks, ml_bonus)
        
        classification = "Strong Accumulation" if score >= 70 else "Moderate Accumulation" if score >= 50 else "No Accumulation"
        
        if classification != "No Accumulation":
            # Generate Chart (Volume)
            fig, ax = plt.subplots(figsize=(10, 4))
            df.plot(x='timestamp', y='volume', ax=ax, title=f'{ticker} Volume - TTE: {tte_weeks:.1f} Weeks')
            ax.axhline(df['volume'].tail(60).mean() * 1.10, color='r', linestyle='--', label='1.1x Avg Vol')
            plt.savefig(os.path.join(OUTPUT_DIR, f'{ticker}_volume.png'))
            plt.close()
            
            return {
                'ticker': ticker,
                'score': score,
                'classification': classification,
                'signals': {k: {key: v for key, v in val.items() if key != 'df'} for k, val in signals.items()},
                'ml_bonus': ml_bonus,
                'earnings_date': edate_str,
                'tte_weeks': tte_weeks
            }
        return None

    async def scan(self):
        """Main async Q12 Agent scan."""
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        universe = await build_universe()
        today = datetime.date.today()
        
        # Concurrent fetching with gather
        tasks = [self.get_signals(ticker) for ticker in universe]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter valid results
        valid_results = [r for r in results if r is not None and not isinstance(r, Exception)]
        
        # Final Report Output (Console)
        print("\n--- Q12 Agent Scan Report ---")
        if valid_results:
            valid_results = sorted(valid_results, key=lambda x: x['score'], reverse=True)
            for res in valid_results:
                print(f"\n[{res['classification']}] {res['ticker']} (Score: {res['score']})")
                print(f"  Earnings Date: {res['earnings_date']} (TTE: {res['tte_weeks']:.1f} Weeks)")
                print(f"  ML Bonus: +{res['ml_bonus']}")
                for sig_name, sig_data in res['signals'].items():
                    print(f"  - {sig_name.capitalize()}: Score {sig_data['score']}, Flags: {', '.join(sig_data['flags'])}")
                print(f"  Chart saved to {os.path.join(OUTPUT_DIR, f'{res['ticker']}_volume.png')}")
        else:
            print("No stocks flagged for accumulation in the current universe.")

        return valid_results
