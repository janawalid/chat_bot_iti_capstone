"""
Diagnostic script: shows exactly what response your machine gets back
from Wikipedia's API, so we can see why it's not valid JSON.

Usage:
    python diagnose_network.py
"""

import requests

API_URL = "https://en.wikipedia.org/w/api.php"
HEADERS = {
    "User-Agent": "RAG-Graduation-Project/1.0 (student project; contact: student@example.com)"
}

params = {
    "action": "query",
    "format": "json",
    "titles": "Machine learning",
    "prop": "extracts",
    "explaintext": 1,
}

print("Sending request to:", API_URL)
try:
    response = requests.get(API_URL, params=params, headers=HEADERS, timeout=15)
    print("Status code:", response.status_code)
    print("Response headers:", dict(response.headers))
    print("\n--- First 500 characters of raw response body ---")
    print(response.text[:500])
except Exception as e:
    print("Request itself failed with an exception:")
    print(type(e).__name__, "-", e)