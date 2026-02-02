import streamlit as st
import pandas as pd
import numpy as np
import time


report_sources = {
    "Bell Potter": "bell_potter",
    "Buy Hold Sell": "buy_hold_sell",
    "Motley Fool": "motel_fool",
    "Livewire": "live_wire",
    "Money of Mine": "money_of_mine",
    "Morningstar": "morningstar",
    "Ord Minnett": "ord_minnet",
    "Wilson Advisory": "wilson_advisory",
}



st.set_page_config(page_title='Equity Reports Sentiment Analyser Dashboard', layout ='wide')
st.title("Equity Reports Sentiment Analyser Dashboard")

st.markdown("This dashboard allows you to analyse sentiment of equity reports using FinBERT model.")


tab_pre_scraped, tab_new_reports = st.tabs([
    "📚 Pre-scraped Reports",
    "🌐 Scrape New Report"
])


with tab_pre_scraped:
    st.header("Pre-scraped Equity Reports Analysis")
    st.markdown("Select from the pre-scraped equity reports to view sentiment analysis results.")
    st.markdown("The following options will allow you to narrow down the reports to analyze. The ticker option is compulsory, and at least one of the year and the source must be selected.")
    

    ticker = st.text_input('Select a Ticker, e.g., CBA, BHP, TLS')
    ticker_clean = ticker.strip().upper() if ticker else ""

    year = st.multiselect('Select Year, e.g., 2023, 2022 (optional)', options=list(range(2026, 2019, -1)))
    year_selected_flag = len(year) > 0

    selected_label = st.multiselect(
    "Select report source:",
    list(report_sources.keys()))
    selected_source_value = [report_sources[label] for label in selected_label]
    source_selected_flag = len(selected_source_value) > 0


    # change to normal button state management once loading bar option is choosen
    if "analyze_clicked" not in st.session_state:
        st.session_state.analyze_clicked = False

    if st.button("Analyze Selected Report"):
        st.session_state.analyze_clicked = True
        st.session_state.num_results_to_show = 5  # reset pagination on new analysis

    if st.session_state.analyze_clicked:

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

            option_1= st.button("option 1")
            option_2= st.button("option 2")
            actual_output = st.button("Actual_Output")

            if option_1:
                with st.spinner(text="In progress"):
                    time.sleep(3)
                    st.success("Done")
                
                with st.status("Authenticating...") as s:
                    time.sleep(2)
                    st.write("Some long response.")
                    s.update(label="Successfully authenticated!")
                time.sleep(1)
                    
                # Placeholder for your analysis logic
                st.success("Sentiment analysis completed!")
                st.write("Positive: 70%")
                st.write("Neutral: 20%")
                st.write("Negative: 10%")

            if option_2:
                # Show and update progress bar
                bar = st.progress(0)
                for percent_complete in range(100):
                    time.sleep(0.01)
                    bar.progress(percent_complete + 1)

                with st.status("Authenticating...") as s:
                    time.sleep(2)
                    st.write("Some long response.")
                    s.update(label="Successfully authenticated!")
                
                time.sleep(1)
                    
                # Placeholder for your analysis logic
                
                st.success("Sentiment analysis completed!")
                st.write("Positive: 70%")
                st.write("Neutral: 20%")
                st.write("Negative: 10%")

            if actual_output:
                st.success("Sentiment analysis completed!")
                example_reports = [
                {
                    "year": 2023,
                    "source": "Bell Potter",
                    "ticker": "CBA",
                    "link": "https://example.com/report1",
                    "industry": "Banking",
                    "team_industry": "Financial Services",
                    "sentiment": {"pos": 0.7, "neu": 0.2, "neg": 0.1}
                },
                {
                    "year": 2023,
                    "source": "Morningstar",
                    "ticker": "BHP",
                    "link": "https://example.com/report2",
                    "industry": "Mining",
                    "team_industry": "Resources",
                    "sentiment": {"pos": 0.4, "neu": 0.3, "neg": 0.3}
                },
                {
                    "year": 2022,
                    "source": "Motley Fool",
                    "ticker": "TLS",
                    "link": "https://example.com/report3",
                    "industry": "Telecommunications",
                    "team_industry": "Tech & Comms",
                    "sentiment": {"pos": 0.6, "neu": 0.2, "neg": 0.2}
                },
                {
                    "year": 2022,
                    "source": "Wilson Advisory",
                    "ticker": "CBA",
                    "link": "https://example.com/report4",
                    "industry": "Banking",
                    "team_industry": "Financial Services",
                    "sentiment": {"pos": 0.3, "neu": 0.5, "neg": 0.2}
                },
                {
                    "year": 2021,
                    "source": "Buy Hold Sell",
                    "ticker": "BHP",
                    "link": "https://example.com/report5",
                    "industry": "Mining",
                    "team_industry": "Resources",
                    "sentiment": {"pos": 0.5, "neu": 0.4, "neg": 0.1}
                },
                {
                    "year": 2021,
                    "source": "Buy Hold Sell",
                    "ticker": "BHP",
                    "link": "https://example.com/report5",
                    "industry": "Mining",
                    "team_industry": "Resources",
                    "sentiment": {"pos": 0.5, "neu": 0.4, "neg": 0.1}
                },
                {
                    "year": 2021,
                    "source": "Buy Hold Sell",
                    "ticker": "BHP",
                    "link": "https://example.com/report5",
                    "industry": "Mining",
                    "team_industry": "Resources",
                    "sentiment": {"pos": 0.5, "neu": 0.4, "neg": 0.1}
                },
                {
                    "year": 2021,
                    "source": "Buy Hold Sell",
                    "ticker": "BHP",
                    "link": "https://example.com/report5",
                    "industry": "Mining",
                    "team_industry": "Resources",
                    "sentiment": {"pos": 0.5, "neu": 0.4, "neg": 0.1}
                }]

                example_reports_sorted = sorted(example_reports, key=lambda x: (-x["year"], x["source"]))

                # Display the first n results
                for report in example_reports_sorted[:st.session_state.num_results_to_show]:
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
                if st.session_state.num_results_to_show < len(example_reports_sorted):
                    show_more = st.button("Show more", key="show_more")  # key ensures unique widget
                    if show_more:
                        st.session_state.num_results_to_show += 10


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



            

