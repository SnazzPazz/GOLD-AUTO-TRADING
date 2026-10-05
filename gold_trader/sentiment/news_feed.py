"""Free, no-API-key sentiment signal for gold.

Pulls headlines from public financial-news RSS feeds and scores them with
VADER (a lexicon-based sentiment model that needs no network download or
API key). This is deliberately a *secondary filter*, not a trigger: news
and social sentiment are noisy and often lag price, so the strategy engine
only ever uses this to confirm or dampen a price-action signal, never to
originate one by itself.

Swap in a paid feed (NewsAPI, X/Twitter API, Reddit API, a commercial
sentiment vendor) later by implementing the same `get_sentiment()`
interface — nothing else in the system needs to change.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import feedparser
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

FEED_FETCH_TIMEOUT_SECONDS = 10

GOLD_RSS_FEEDS = [
    "https://www.kitco.com/rss/KitcoNews.xml",
    "https://www.investing.com/rss/commodities_Gold.rss",
    "https://www.fxstreet.com/rss/news",
]

GOLD_KEYWORDS = (
    "gold", "xau", "bullion", "precious metal", "fed", "fomc", "interest rate",
    "inflation", "dollar", "treasury yield", "safe haven", "rate cut", "rate hike",
)


@dataclass
class SentimentReading:
    score: float          # -1 (very bearish) .. +1 (very bullish)
    headline_count: int
    as_of: float           # unix timestamp
    sample_headlines: list[str]


class NewsSentimentProvider:
    def __init__(self, cache_seconds: int = 900, feeds: list[str] | None = None):
        self._analyzer = SentimentIntensityAnalyzer()
        self._cache_seconds = cache_seconds
        self._feeds = feeds or GOLD_RSS_FEEDS
        self._cached: SentimentReading | None = None

    def get_sentiment(self) -> SentimentReading:
        if self._cached and (time.time() - self._cached.as_of) < self._cache_seconds:
            return self._cached

        headlines: list[str] = []
        for url in self._feeds:
            try:
                # feedparser.parse(url) hands the fetch to urllib, which has no
                # timeout and could hang the whole polling loop on a slow/dead
                # feed; fetch with requests (bounded) and hand it the bytes instead.
                resp = requests.get(url, timeout=FEED_FETCH_TIMEOUT_SECONDS)
                resp.raise_for_status()
                parsed = feedparser.parse(resp.content)
            except Exception:
                continue
            for entry in parsed.entries[:30]:
                title = getattr(entry, "title", "")
                if not title:
                    continue
                if any(k in title.lower() for k in GOLD_KEYWORDS):
                    headlines.append(title)

        if not headlines:
            reading = SentimentReading(score=0.0, headline_count=0, as_of=time.time(), sample_headlines=[])
        else:
            scores = [self._analyzer.polarity_scores(h)["compound"] for h in headlines]
            avg = sum(scores) / len(scores)
            reading = SentimentReading(
                score=avg,
                headline_count=len(headlines),
                as_of=time.time(),
                sample_headlines=headlines[:5],
            )

        self._cached = reading
        return reading
