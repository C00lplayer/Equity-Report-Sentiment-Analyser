#import cloudscraper
import requests
from bs4 import BeautifulSoup
from supabase import create_client, Client
import streamlit as st
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
url = ""
key = ""
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



def get_article_text(url, source: str):

    if source in ["bell_potter","Buy_hold_sell","motely_fool", "live_wire", "ord_minnet","wilsonsadvisory", "morningstar"]:
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
