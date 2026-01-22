import requests
import time
from bs4 import BeautifulSoup
from datetime import datetime

class BaseNewsScraper:
    def __init__(self, delay=2):
        self.delay = delay
        self.headers = {
            "User-Agent": "Mozilla/5.0 (FakeNewsDetector Academic Project)"
        }

    def fetch(self, url):
        response = requests.get(url, headers=self.headers, timeout=10)
        response.raise_for_status()
        time.sleep(self.delay)
        return BeautifulSoup(response.text, "html.parser")

class ReutersScraper(BaseNewsScraper):
    def scrape_article(self, url):
        soup = self.fetch(url)

        title = soup.find("h1").get_text(strip=True)

        paragraphs = soup.find_all("p")
        text = " ".join(p.get_text(strip=True) for p in paragraphs)

        return {
            "title": title,
            "text": text,
            "source": "Reuters",
            "date": datetime.now(),
            "dataset": "Scraped-Reuters",
            "label": 1  # REAL
        }

class BBCScraper(BaseNewsScraper):
    def scrape_article(self, url):
        soup = self.fetch(url)

        title = soup.find("h1").get_text(strip=True)

        article = soup.find("article")
        paragraphs = article.find_all("p") if article else soup.find_all("p")

        text = " ".join(p.get_text(strip=True) for p in paragraphs)

        return {
            "title": title,
            "text": text,
            "source": "BBC",
            "date": datetime.now(),
            "dataset": "Scraped-BBC",
            "label": 1  # REAL
        }

import pandas as pd

def scrape_bulk(scraper, urls):
    records = []

    for url in urls:
        try:
            records.append(scraper.scrape_article(url))
        except Exception as e:
            print(f"Skipping {url}: {e}")

    return pd.DataFrame(records)
