from fastapi import FastAPI
import asyncio
from q12_agent import Q12Agent

app = FastAPI()

agent = Q12Agent()

@app.get("/scan")
async def scan():
    return await agent.scan()

@app.get("/signals/{ticker}")
async def get_signals(ticker: str):
    return await agent.get_signals(ticker)
