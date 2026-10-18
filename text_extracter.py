#import cloudscraper
import requests
from bs4 import BeautifulSoup
from supabase import create_client, Client
import streamlit as st
import os
#from sentence_model import process_single_document, results_to_dataframe

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-AU,en;q=0.9",
}

# Insert your Supabase URL and Key here
url = os.environ.get("SUPABASE_URL", "https://lshtgdpdskhqqxdcwpjo.supabase.co")
key = os.environ.get("SUPABASE_KEY","")
supabase = create_client(url, key)

def get_reports(ticker, year=None, source=None, ASX_200: int = True):
    if ASX_200:
        table_name = "Equity Reports ASX200"
    else:
        table_name = "Equity Reports ASX Small Cap"
    query = supabase.table(table_name).select("*").eq("ticker", ticker)
    if year:
        if isinstance(year, list):
            query = query.in_("year", year)
        else:
            query = query.eq("year", year)

    if source:
        if isinstance(source, list):
            query = query.in_("source", source)
        else:
            query = query.eq("source", source)

    return query.execute().data



CACHE_REPORT_COLUMNS = ["ticker", "url", "industry", "investment_team_industry", "year", "source"]


def get_cached_sentiment(ticker, url):
    """Look up a previously scored report in the ASX200 sentiment cache table, keyed by ticker + url.

    pos_sent/neu_sent/neg_sent are stored as bigint percentages (0-100), so they're converted back
    to 0-1 probabilities here so callers always work with the same scale FinBERT produces.
    """
    result = (
        supabase.table("Equity Reports ASX200_duplicate")
        .select("pos_sent, neu_sent, neg_sent")
        .eq("ticker", ticker)
        .eq("url", url)
        .limit(1)
        .execute()
    )
    if not result.data:
        return None
    row = result.data[0]
    return {
        "pos_sent": row["pos_sent"] / 100,
        "neu_sent": row["neu_sent"] / 100,
        "neg_sent": row["neg_sent"] / 100,
    }


def save_cached_sentiment(report, pos, neu, neg):
    """Store a scored report plus sentiment scores in the ASX200 sentiment cache table.

    Only the columns that exist on the cache table are copied from `report` (it may carry extra
    columns from the source table's `select("*")` that the cache table doesn't have). pos/neu/neg
    are 0-1 probabilities from FinBERT; pos_sent/neu_sent/neg_sent are bigint, so they're stored
    as rounded 0-100 percentages.
    """
    row = {col: report[col] for col in CACHE_REPORT_COLUMNS}
    row["pos_sent"] = round(pos * 100)
    row["neu_sent"] = round(neu * 100)
    row["neg_sent"] = round(neg * 100)
    supabase.table("Equity Reports ASX200_duplicate").upsert(row, on_conflict="ticker,url").execute()


def get_article_text(url, source: str):

    if source in ["bell_potter","buy_hold_sell","motley_fool", "live_wire", "ord_minnet","wilsonsadvisory", "morningstar"]:
        session = requests.Session()
        session.headers.update(HEADERS)
        r = session.get(url)
        
        r.raise_for_status()
    """elif source in ["money_of_mine"]:
        scraper = cloudscraper.create_scraper()
        r = scraper.get(url, headers=HEADERS, timeout=10)
        r.raise_for_status()
    """
    soup = BeautifulSoup(r.text, "html.parser")

    # Remove the footer
    footer = soup.find("footer")
    if footer:
        footer.decompose()  # removes the footer from the soup
    
    # Extract remaining <p> text
    paragraphs = soup.find_all("p")
    text = " ".join(p.get_text() for p in paragraphs)
    
    return text


#print(get_reports("JBH", year=2020, source="live_wire", ASX_200=True))
#print(get_article_text(get_reports("NAB", year=2025, source="bell_potter", ASX_200=True)[0]["url"], source="bell_potter"))


#print(get_article_text("https://www.morningstar.com.au/stocks/have-profits-peaked-for-the-big-four-banks", source="morningstar"))
#print(get_reports("NAB", source="wilsonsadvisory", ASX_200=True))
