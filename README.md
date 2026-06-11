# Equity Report Sentiment Analyser

This repository contains code for analysing sentiment in equity research reports. The main workflow is:

- Fetch report metadata and links from a Supabase database.
- Extract full report text from web articles and GPFS PDF resources.
- Split text into sentences and chunks to stay within FinBERT token limits.
- Run each chunk through a FinBERT sentiment model.
- Aggregate sentiment scores across the report and present positive/neutral/negative probabilities.

## What the code does

- `dashboard.py` defines a Streamlit dashboard interface for selecting equity reports by ticker, year, and source.
- `text_extracter.py` retrieves report links from Supabase and extracts article text from supported web sources.
- It also extracts and cleans GPFS PDF text, avoiding table content when possible.
- `sentence_model.py` handles sentence splitting, chunking, and FinBERT sentiment scoring.

## Data sources

- Report links are obtained from a Supabase database.
- Supported report sources include Bell Potter, Buy Hold Sell, Motley Fool, Livewire, Money of Mine, Morningstar, Ord Minnett, and Wilson Advisory.

## Project structure

- `dashboard.py` – dashboard logic and analysis flow.
- `text_extracter.py` – text retrieval from web articles and GPFS PDFs.
- `sentence_model.py` – FinBERT sentiment pipeline and aggregation.
- `requirements.txt` – project dependencies.
- `results/` – generated output or result files.
- `testing_reports/` – sample text reports used for testing.
- `prev/` – older pipeline scripts and experimental code.

## Notes

- This README describes the code base and how it processes report links and text.
- It does not assume the project is running as a production service.
- The implementation focuses on extracting links from the database and analysing report content with FinBERT.
