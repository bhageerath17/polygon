from __future__ import annotations
import time
import requests
from app.config import settings


class PolygonClient:
    def __init__(self, rate_limit_sleep: float = 0.25):
        self._sleep = rate_limit_sleep
        self._session = requests.Session()

    def get(self, url: str, params: dict | None = None, _retries: int = 5) -> dict:
        p = {"apiKey": settings.api_key, **(params or {})}
        for attempt in range(_retries):
            resp = self._session.get(url, params=p)
            if resp.status_code == 429:
                wait = 12 * (attempt + 1)
                print(f"  Rate limit — waiting {wait}s …")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            time.sleep(self._sleep)
            return resp.json()
        resp.raise_for_status()
        return {}

    def get_paginated(self, url: str, params: dict | None = None) -> list[dict]:
        all_results: list[dict] = []
        current_url: str | None = url
        current_params = params

        while current_url:
            data = self.get(current_url, current_params)
            results = data.get("results", [])
            all_results.extend(results)
            print(f"  Fetched {len(results)} records (total: {len(all_results)})")
            current_url = data.get("next_url")
            current_params = None  # next_url already contains all query params

        return all_results


client = PolygonClient()
