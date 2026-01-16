import asyncio
import os
import datetime
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()  # Load .env for keys

# --- Configuration ---
API_TITLE = "Q12 Scan API"
API_VERSION = "1.0"
CORS_ORIGINS = ["*"]
OUTPUT_DIR = 'q12_reports'
# ---------------------------------------------

from q12_agent import Q12Agent 

# Initialize FastAPI App
app = FastAPI(
    title=API_TITLE,
    version=API_VERSION
)

# Initialize Agent
agent = Q12Agent()

# ---- CORS Middleware ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure directories exist
os.makedirs("static", exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---- STATIC FILES ----
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount(f"/{OUTPUT_DIR}", StaticFiles(directory=OUTPUT_DIR), name="reports")

# ---- ROUTES ----

@app.get("/", include_in_schema=False)
async def root():
    """Redirects to the static report viewer."""
    return {"status": "Q12 API running", "version": API_VERSION, "docs": "/docs", "report_viewer": "/static/index.html"}

@app.get("/tickers")
async def list_tickers() -> Dict[str, List[str]]:
    """Returns the list of tickers in the current scanning universe."""
    try:
        universe = await agent.build_universe()
        return {"tickers": universe}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to build universe: {str(e)}")

@app.get("/scan", response_model=List[Dict[str, Any]])
async def run_scan() -> List[Dict[str, Any]]:
    """Runs the full asynchronous Q12 accumulation scan on the universe."""
    try:
        data = await agent.scan()
        
        import json
        with open(os.path.join(OUTPUT_DIR, 'latest_report.json'), 'w') as f:
            json.dump(data, f, indent=4)
            
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scan failed: {str(e)}")

@app.get("/signals/{ticker}", response_model=Optional[Dict[str, Any]])
async def signals_for_ticker(ticker: str) -> Optional[Dict[str, Any]]:
    """Retrieves the latest calculated signals and score for a specific ticker."""
    try:
        signals = await agent.get_signals(ticker.upper())
        if signals is None:
            raise HTTPException(status_code=404, detail=f"Ticker {ticker} data not available or outside 12-week window.")
        return signals
    except Exception as e:
        print(f"Error fetching signals for {ticker}: {e}")
        raise HTTPException(status_code=500, detail="Internal error during signal computation.")
