import streamlit as st
import pandas as pd
import numpy as np
import time
from sentence_model import process_single_document, results_to_dataframe
from text_extracter import get_reports, get_article_text


# Motley fool = data base error
# Buy hold sell =  either database error or not saved as BHS
# Livewire = cannot be scraped need to change scraper
# Money of mine = cloudscraper needed
# Morningstar = works but need to add rest period not to overwhelm server





report_sources = {
    "Bell Potter": "bell_potter",
    "Buy Hold Sell": "buy_hold_sell",
    "Motley Fool": "motel_fool",
    "Livewire": "live_wire",
    "Money of Mine": "money_of_mine",
    "Morningstar": "morningstar",
    "Ord Minnett": "ord_minnet",
    "Wilson Advisory": "wilsonsadvisory",
}
def get_key_by_value(d, value):
    for k, v in d.items():
        if v == value:
            return k
    return None  # if not found



st.set_page_config(page_title='Equity Reports Sentiment Analyser Dashboard', layout ='wide')
st.title("Equity Reports Sentiment Analyser Dashboard")

st.markdown("This dashboard allows you to analyse sentiment of equity reports using FinBERT model.")


tab_pre_scraped, tab_new_reports = st.tabs([
    "📚 Pre-scraped Reports",
    "🌐 Scrape New Report"
])


def process_next_batch(batch_size=10):
    start = st.session_state.current_index
    end = min(start + batch_size, len(st.session_state.all_reports))

    bar = st.progress(0)

    for i in range(start, end):
        report = st.session_state.all_reports[i]

        report_text = get_article_text(report["url"], source=report["source"])
        report_sentiment = process_single_document(text=report_text)

        report["sentiment"] = {
            "neg": report_sentiment["agg_probs"][0],
            "neu": report_sentiment["agg_probs"][1],
            "pos": report_sentiment["agg_probs"][2],
        }

        st.session_state.processed_results.append(
            {
                "year": report["year"],
                "source": get_key_by_value(report_sources, report["source"]),
                "ticker": report["ticker"],
                "link": report["url"],
                "industry": report["industry"],
                "team_industry": report["investment_team_industry"],
                "sentiment": report["sentiment"],
            }
        )

        bar.progress((i - start + 1) / (end - start))
        time.sleep(1.5) # Add buffer to avoid overwhelming the server and to simulate processing time

    st.session_state.current_index = end

with tab_pre_scraped:
    st.header("Pre-scraped Equity Reports Analysis")
    st.markdown("Select from the pre-scraped equity reports to view sentiment analysis results.")
    st.markdown("The following options will allow you to narrow down the reports to analyze. The ticker option is compulsory, and at least one of the year and the source must be selected.")
    

    ticker = st.text_input('Select a Ticker, e.g., CBA, BHP, TLS')
    ticker_clean = ticker.strip().upper() if ticker else ""
    ASX_200 = st.checkbox('Is the ticker part of ASX 200?', value=True)


    year = st.multiselect('Select Year, e.g., 2023, 2022 (optional)', options=list(range(2026, 2019, -1)))
    year_selected_flag = len(year) > 0


    st.warning('As of right now only the following sources are supported: Bell Potter, Wilson Advisory.')
    selected_label = st.multiselect(
    "Select report source:",
    list(report_sources.keys()))
    selected_source_value = [report_sources[label] for label in selected_label]
    source_selected_flag = len(selected_source_value) > 0




    # change to normal button state management once loading bar option is choosen
    if "analyze_clicked" not in st.session_state:
        st.session_state.analyze_clicked = False

    if "analysis_results" not in st.session_state:
        st.session_state.analysis_results = []

    if "num_results_to_show" not in st.session_state:
        st.session_state.num_results_to_show = 5
    
    if "all_reports" not in st.session_state:
        st.session_state.all_reports = []

    if "processed_results" not in st.session_state:
        st.session_state.processed_results = []

    if "current_index" not in st.session_state:
        st.session_state.current_index = 0


    if st.button("Analyze Selected Report"):
        st.session_state.analyze_clicked = True
        st.session_state.num_results_to_show = 5  # reset pagination
        st.session_state.processed_results = []
        st.session_state.current_index = 0


        # Validation
        if not ticker_clean:
            st.error("Ticker is required!")
        elif not (year_selected_flag or source_selected_flag):
            st.error("Please select at least one of Year or Source.")
        else:
            # Passed validation
            st.write(f"Analyzing sentiment for Ticker: {ticker_clean}")

            # show selected filters
            st.write(f"Year(s): {year if year_selected_flag else 'All'}")
            st.write(f"Source(s): {', '.join(selected_label) if source_selected_flag else 'All'}")
            #if "Bell Potter" in selected_label:
            reports = get_reports(ticker_clean, year=year if year_selected_flag else None, source=selected_source_value if source_selected_flag else None, ASX_200=ASX_200)
            if reports:
                st.write(f"Found {len(reports)} report(s) matching the criteria.")
                st.session_state.all_reports = reports
            else:
                st.warning("No reports found for the selected criteria.")
                st.session_state.all_reports = []
    
    
    if st.session_state.analyze_clicked and st.session_state.all_reports:
        # Process first batch automatically if nothing processed yet
        if st.session_state.current_index == 0:
            process_next_batch(10)
        
        for report in st.session_state.processed_results:
            with st.expander(f"{report['year']} - {report['source']} - {report['ticker']}"):
                st.markdown(f"[Original Report]({report['link']})")
                st.write(f"Industry: {report['industry']}")
                st.write(f"Investment Team Industry: {report['team_industry']}")
                st.write(
                    f"Sentiment: Pos: {report['sentiment']['pos']*100:.1f}%, "
                    f"Neu: {report['sentiment']['neu']*100:.1f}%, "
                    f"Neg: {report['sentiment']['neg']*100:.1f}%"
                )


        # Show more button
        if st.session_state.current_index < len(st.session_state.all_reports):
            if st.button("Show more"):
                process_next_batch(10)
                st.rerun()
                
        """
        results_to_show = st.session_state.analysis_results[:st.session_state.num_results_to_show]

        for report in results_to_show:
            with st.expander(f"{report['year']} - {report['source']} - {report['ticker']}"):
                st.markdown(f"[Original Report]({report['link']})")
                st.write(f"Industry: {report['industry']}")
                st.write(f"Investment Team Industry: {report['team_industry']}")
                st.write(
                    f"Sentiment: Pos: {report['sentiment']['pos']*100:.1f}%, "
                    f"Neu: {report['sentiment']['neu']*100:.1f}%, "
                    f"Neg: {report['sentiment']['neg']*100:.1f}%"
                )

        # Show more button
        if st.session_state.num_results_to_show < len(st.session_state.analysis_results):
            if st.button("Show more", key="show_more"):
                st.session_state.num_results_to_show += 10
        """



with tab_new_reports:
    st.header("Scrape and Analyze New Equity Report")
    st.markdown("Input a URL to scrape a new equity report and analyze its sentiment.")

    report_url = st.text_input('Enter the URL of the equity report to scrape and analyze:')
    if "scrape_done" not in st.session_state:
            st.session_state.scrape_done = False

    # Scrape button
    if st.button("Scrape and Analyze"):

        if not report_url.strip():
            st.error("Please enter a valid URL.")
        else:
            st.write(f"Scraping and analyzing report from URL: {report_url.strip()}")

            with st.spinner(text="Scraping and analyzing..."):
                time.sleep(5)  

            st.success("Scraping and sentiment analysis completed!")
            st.write("Sentiment Results:")
            st.write("Positive: 65%")
            st.write("Neutral: 25%")
            st.write("Negative: 10%")

            
            st.session_state.scrape_done = True

    # Only show additional inputs if scraping is done
    if st.session_state.scrape_done:
        st.markdown("Could you please fill in the following information to add this report to the database for future use.")

        
        ticker_input = st.text_input('Select the relevant tickers separated by a space, e.g. CBA BHP TLS')
        tickers = [t.upper() for t in ticker_input.strip().split() if t.strip()]

        year = st.number_input('Select Year, e.g., 2023, 2022', min_value=2000, max_value=2026, step=1)

        selected_label = st.text_input("Select the report source:")
        clean_label = selected_label.strip().lower().replace(" ", "_")


        st.write("Given inputs:")
        st.write(f"Ticker(s): {', '.join(tickers)}")
        st.write(f"Year: {year}")
        st.write(f"Source: {clean_label}")



            
"""
if reports:
    results = []
    for i, report in enumerate(reports):
        report_text = get_article_text(report["url"], source=report["source"])
        report_sentiment = process_single_document(text=report_text)
        report["sentiment"] = {
            "neg": report_sentiment["agg_probs"][0],
            "neu": report_sentiment["agg_probs"][1],
            "pos": report_sentiment["agg_probs"][2],
        }
        results.append(
            {"year": report["year"],
                "source": get_key_by_value(report_sources, report["source"]),
                "ticker": report["ticker"],
                "link": report["url"],
                "industry": report["industry"],
                "team_industry": report["investment_team_industry"],
                "sentiment": report["sentiment"]}
        )
        bar.progress((i + 1) / len(reports))

    st.session_state.analysis_results = sorted(results, key=lambda x: (-x["year"], x["source"]))
    st.success("Sentiment analysis completed!")
else:
    st.warning("No reports found for the selected criteria.")
    st.session_state.analysis_results = []
"""
