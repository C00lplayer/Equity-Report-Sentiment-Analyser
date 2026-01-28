import requests
from bs4 import BeautifulSoup
from supabase import create_client, Client

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-AU,en;q=0.9",
}


def get_article_text(url):
    r  = requests.get(url, timeout=10)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")

    # Remove obvious junk
    for tag in soup(["script", "style", "nav", "footer", "aside"]):
        tag.decompose()

    paragraphs = soup.find_all("p")
    text = " ".join(p.get_text(strip=True) for p in paragraphs)

    return text



### STILL TO INTEGRATE THIS FUNCTION INTO MAIN PIPELINE 
def get_from_database(ticker: str,website_name:str, limit: int =10, ) -> list[str]:
    supabase_url: str = ""
    supabase_key: str = ""
    supabase: Client = create_client(supabase_url, supabase_key)

    table_name = "Equity Reports ASX Small Cap"


    response = supabase.table(table_name).select("url").eq("ticker", ticker).eq("source", website_name).limit(limit).execute()
    data = response.data
    text = get_article_text(data[0]['url'])
    return data[0]['url']



print(get_article_text("https://www.livewiremarkets.com/wires/is-now-the-time-to-buy-asx-uranium-stocks-analysis-of-the-latest-uranium-technical-and-fundamental-factors"))
