import asyncio
import os
import datetime
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any, Optional

# Assuming settings.py and q12_agent.py are present
# Since we don't have settings.py in the current context, we use the values directly.

# --- Configuration (Based on original plan) ---
API_TITLE = "Q12 Scan API"
API_VERSION = "1.0"
CORS_ORIGINS = ["*"]
OUTPUT_DIR = 'q12_reports'
# ---------------------------------------------

# We need the full Q12Agent definition from q12_agent.py, which is implicitly present.
# We will use the synchronous dependencies and utility functions directly within the agent.
# For demonstration purposes, we import the class from the file name.
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

# Ensure the static directory exists before mounting
os.makedirs("static", exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True) # Ensure reports directory exists

# ---- STATIC FILES ----
# Mount the static directory to serve index.html and q12_report_template.js
app.mount("/static", StaticFiles(directory="static"), name="static")

# Mount the reports directory to serve the PNG charts
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
        # build_universe is an async method on the agent now
        universe = await agent.build_universe()
        return {"tickers": universe}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to build universe: {str(e)}")


@app.get("/scan", response_model=List[Dict[str, Any]])
async def run_scan() -> List[Dict[str, Any]]:
    """Runs the full asynchronous Q12 accumulation scan on the universe."""
    try:
        # agent.scan is an async method
        data = await agent.scan()
        
        # Save the result to a JSON file for easy access/debugging
        # Note: Saving to a file here is synchronous and should ideally be in a thread,
        # but for a simple final output, we do it here.
        # This mirrors the behavior of the original sync agent writing a report.
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
        # agent.get_signals is an async method
        signals = await agent.get_signals(ticker.upper())
        if signals is None:
            raise HTTPException(status_code=404, detail=f"Ticker {ticker} data not available or outside 12-week window.")
        return signals
    except Exception as e:
        # We don't want to expose raw API errors, but log them
        print(f"Error fetching signals for {ticker}: {e}")
        raise HTTPException(status_code=500, detail="Internal error during signal computation.")

