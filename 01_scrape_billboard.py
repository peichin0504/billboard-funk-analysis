"""
01_scrape_billboard.py
Scrapes Billboard Hot 100 year-end charts for 2010-2020.
Output: billboard_2010_2020.csv
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time


def scrape_billboard_year(year):
    """
    Scrape Billboard Hot 100 year-end chart for a given year.
    Returns a DataFrame with rank, title, and artist.
    """
    url = f"https://www.billboard.com/charts/year-end/{year}/hot-100-songs/"
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}

    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print(f"Failed to fetch {year}: status {response.status_code}")
        return None

    soup = BeautifulSoup(response.text, "html.parser")

    results = []
    entries = soup.select("div.o-chart-results-list-row-container")

    for entry in entries:
        rank_tag = entry.select_one("span.c-label")
        title_tag = entry.select_one("h3.c-title")
        artist_tag = entry.select_one("span.c-label.a-no-trucate")

        if title_tag and artist_tag:
            results.append({
                "year": year,
                "rank": rank_tag.text.strip() if rank_tag else None,
                "title": title_tag.text.strip(),
                "artist": artist_tag.text.strip()
            })

    return pd.DataFrame(results)


if __name__ == "__main__":
    all_charts = []

    for year in range(2010, 2021):
        print(f"Scraping {year}...")
        df = scrape_billboard_year(year)
        if df is not None and len(df) > 0:
            all_charts.append(df)
        time.sleep(2)  # be polite to the server

    billboard_df = pd.concat(all_charts, ignore_index=True)
    billboard_df.to_csv("billboard_2010_2020.csv", index=False)
    print(f"\nTotal songs scraped: {len(billboard_df)}")
    print(billboard_df.head(10))
