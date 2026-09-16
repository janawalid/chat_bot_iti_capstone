"""
Collects a set of Wikipedia articles as plain-text source documents
for the RAG pipeline. Run this once before the notebook.

Uses Wikipedia's API directly via `requests` (instead of the unmaintained
`wikipedia` pip package, which breaks because it doesn't send a proper
User-Agent header and Wikipedia now blocks such requests).

Usage:
    pip install requests
    python collect_data.py
"""

import os
import time
import requests

API_URL = "https://en.wikipedia.org/w/api.php"

# Wikipedia requires a descriptive User-Agent identifying the app/contact.
# Feel free to edit the contact info below.
HEADERS = {
    "User-Agent": "RAG-Graduation-Project/1.0 (student project; contact: student@example.com)"
}

# Topic list -- edit this to change your domain.
# Keep 15-30 topics for a meaningful but manageable document collection.
TOPICS = [
    "Machine learning",
    "Supervised learning",
    "Unsupervised learning",
    "Neural network",
    "Deep learning",
    "Overfitting",
    "Gradient descent",
    "Support vector machine",
    "Decision tree learning",
    "Random forest",
    "Convolutional neural network",
    "Recurrent neural network",
    "Natural language processing",
    "Reinforcement learning",
    "Feature engineering",
    "Cross-validation (statistics)",
    "Bias-variance tradeoff",
    "Ensemble learning",
    "Transfer learning",
    "Autoencoder",
]

OUT_DIR = "data/raw"


def fetch_article(title: str) -> tuple[str, str]:
    """Fetch the plain-text extract of a Wikipedia article by title.

    Returns (resolved_title, full_text). Raises ValueError if not found.
    """
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "extracts",
        "explaintext": 1,
        "redirects": 1,
    }
    response = requests.get(API_URL, params=params, headers=HEADERS, timeout=15)
    response.raise_for_status()  # raise clearly on 403/404/etc instead of silently failing
    data = response.json()

    pages = data.get("query", {}).get("pages", {})
    for page_id, page in pages.items():
        if page_id == "-1" or "missing" in page:
            raise ValueError(f"Page not found: {title}")
        extract = page.get("extract", "").strip()
        if not extract:
            raise ValueError(f"Empty extract for: {title}")
        return page.get("title", title), extract

    raise ValueError(f"No page data returned for: {title}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    saved, failed = [], []

    for topic in TOPICS:
        try:
            resolved_title, text = fetch_article(topic)
            safe_name = topic.replace(" ", "_").replace("/", "_").replace(
                "(", ""
            ).replace(")", "")
            path = os.path.join(OUT_DIR, f"{safe_name}.txt")
            with open(path, "w", encoding="utf-8") as f:
                f.write(f"TITLE: {resolved_title}\n\n{text}")
            print(f"[OK] {topic} -> {path}")
            saved.append(topic)
        except Exception as e:
            print(f"[FAIL] {topic}: {e}")
            failed.append(topic)
        time.sleep(0.3)  # be polite to the API

    print(f"\nDone. Saved {len(saved)} articles, {len(failed)} failed.")
    if failed:
        print("Failed topics (try renaming or picking a close alternative):")
        for t in failed:
            print(f"  - {t}")


if __name__ == "__main__":
    main()