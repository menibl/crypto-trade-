# pip install feedparser python-dotenv
import json, re, time, os
from pathlib import Path
import feedparser

APP_DIR = Path(__file__).parent
OUT_FILE = APP_DIR / "recommendations.json"

SOURCES = [
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://cointelegraph.com/rss",
    "https://decrypt.co/feed",
    "https://www.theblock.co/rss",
    "https://bitcoinmagazine.com/.rss",
    "https://insights.glassnode.com/rss/",
    "https://santiment.net/blog/feed/",
    "https://cryptopanic.com/feed/",
]

# מיפוי שמות נפוצים -> סימבול/טיקר (אפשר להרחיב)
ALIAS = {
    "bitcoin": "BTC", "btc": "BTC",
    "ethereum": "ETH", "ether": "ETH", "eth": "ETH",
    # הוסף כאן מיפויי ALT פופולריים…
}

TICKER_RE = re.compile(r"\b([A-Z0-9]{2,10})\b")

def extract_symbols(title: str, summary: str) -> set[str]:
    text = f"{title} {summary}".lower()
    syms = set()
    # 1) אליאסים מילוליים (bitcoin -> BTC)
    for k, v in ALIAS.items():
        if k in text:
            syms.add(v)
    # 2) טיקרים באותיות גדולות בתוך סוגריים/טקסט (e.g., “AVAX”, “SOL”)
    uppers = set(TICKER_RE.findall(f"{title} {summary}"))
    # מסנן זבל קצר מדי (AI, TV) וכד'
    for u in uppers:
        if 3 <= len(u) <= 6:
            syms.add(u)
    return syms

def run_once(quote="USDT"):
    counts: dict[str, int] = {}
    for url in SOURCES:
        try:
            feed = feedparser.parse(url)
            for e in feed.entries[:50]:
                title = e.get("title", "")
                summary = e.get("summary", "") or e.get("description", "")
                for s in extract_symbols(title, summary):
                    counts[s] = counts.get(s, 0) + 1
        except Exception:
            continue

    # הופך טיקרים בסיסיים לרשימת צמדים מול QUOTE (הבוט ממילא בודק אם זוג קיים)
    ranked = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    recs = [{"symbol": f"{t}{quote}", "ticker": t, "score": c} for t, c in ranked]

    OUT_FILE.write_text(json.dumps({"generated_at": time.time(), "recs": recs}, ensure_ascii=False, indent=2))
    print(f"wrote {OUT_FILE} with {len(recs)} recs")

if __name__ == "__main__":
    # ריצה פשוטה; להפעלה כמה פעמים ביום – שים ב-cron / Task Scheduler
    run_once(quote=os.getenv("QUOTE_ASSET", "USDT"))
