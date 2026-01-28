### Uses chunking then runs FinBERT on each chunk, aggregates results.
# finbert_pipeline.py
import re
import os
import json
import math
from pathlib import Path
from typing import List, Dict, Tuple, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import hashlib

import pandas as pd
import numpy as np
from diskcache import Cache

# Transformers imports
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# spaCy for sentence splitting
import spacy
nlp = spacy.load("en_core_web_sm")

# ------------------------
# Config
# ------------------------
FINBERT_MODEL = "yiyanghkust/finbert-tone"  # common FinBERT; swap if you use another
MAX_TOKENS_FINBERT = 512
TARGET_TOKENS = 450   # leave margin for special tokens
NUM_WORKERS = max(1, os.cpu_count() - 1)
CACHE_DIR = "./cache_finbert"
RESULTS_DIR = "./results"
os.makedirs(RESULTS_DIR, exist_ok=True)
cache = Cache(CACHE_DIR)

# Importance keywords and section headers (extend as needed)
SECTION_HEADERS = {
    r"^\s*(outlook|guidance|forecast|projections)\b": 3.0,
    r"^\s*(risks|risk factors|uncertainties|downside)\b": 3.0,
    r"^\s*(management discussion|management commentary|md&a|analysis)\b": 2.0,
    r"^\s*(conclusion|summary|final thoughts)\b": 1.2,
    r"^\s*(financials|results|earnings|performance)\b": 2.5,
}

# keywords that increase importance if present in chunk
IMPORTANCE_KEYWORDS = [
    "guidance", "outlook", "risk", "downgrade", "upgrade", "headwind", "tailwind",
    "forecast", "expect", "guidance", "projections", "beat", "miss", "margin",
    "revenue", "profit", "layoff", "acquisition", "M&A", "lawsuit", "regulatory",
]

# LLM compression settings (optional)
USE_LLM_COMPRESSION = True
LLM_MAX_TOKEN_TARGET = TARGET_TOKENS

# ------------------------
# Load FinBERT model and tokenizer and pipeline
# ------------------------
print("Loading FinBERT model (this may take time)...")
tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL)
finbert_pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True, device=-1)
print("FinBERT loaded.")

# ------------------------
# Utilities
# ------------------------
def sha1(s: str):
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def sentence_split(text: str) -> List[str]:
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

def tokens_length(text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=True))

def chunk_sentences(sentences: List[str], max_tokens: int) -> List[str]:
    """
    Greedy pack sentences until token limit reached. Returns list of chunk strings.
    """
    chunks = []
    current = []
    current_tokens = 0
    for s in sentences:
        s_tokens = tokens_length(s)
        if s_tokens > max_tokens:
            # sentence alone too long -> optionally compress with LLM or split
            # for now, split by sub-sentences (fallback)
            parts = re.split(r'(?<=[\.\?\!])\s+', s)
            for p in parts:
                if p.strip():
                    if tokens_length(p) > max_tokens:
                        # force truncate (last resort) - better to call LLM here.
                        p = " ".join(p.split()[:max_tokens//2])
                    if tokens_length(" ".join(current + [p])) <= max_tokens:
                        current.append(p)
                    else:
                        if current:
                            chunks.append(" ".join(current))
                        current = [p]
            continue

        if current_tokens + s_tokens <= max_tokens:
            current.append(s)
            current_tokens += s_tokens
        else:
            if current:
                chunks.append(" ".join(current))
            current = [s]
            current_tokens = s_tokens
    if current:
        chunks.append(" ".join(current))
    return chunks

def detect_section_weight(chunk_text: str) -> float:
    # header detection: look for section headings within or directly before chunk
    lines = chunk_text.splitlines()
    weight = 1.0
    # check first few lines for headings
    for i in range(min(3, len(lines))):
        l = lines[i].lower().strip()
        for pat, w in SECTION_HEADERS.items():
            if re.match(pat, l):
                weight = max(weight, w)
    # keyword boosting
    kcount = sum(1 for k in IMPORTANCE_KEYWORDS if k.lower() in chunk_text.lower())
    if kcount:
        weight *= (1.0 + min(kcount, 5) * 0.25)
    return weight

# LLM compression prompt (copy of the recommended one adapted for API)
LLM_COMPRESSION_PROMPT = """
Compress the following financial text to under {max_tokens} tokens while preserving all sentiment-bearing content.

STRICT REQUIREMENTS:
- Do NOT alter sentiment-loaded wording (e.g., "weak", "strong", "downgrade", "pressure", "improving", "uncertain").
- Do NOT merge or smooth conflicting sentiment statements; keep positives and negatives separate.
- Preserve: forward-looking statements, risk factors, earnings results, guidance revisions, analyst rating changes, management commentary, M&A, layoffs, regulatory actions.
- Allowed removals: legal disclaimers, repeated boilerplate, non-sentiment adjectives, excessive examples that don't change sentiment.
- Keep sentences factual; avoid paraphrases that soften meaning.

Output only the compressed text.
"""

def compress_with_llm(text: str, max_tokens: int = LLM_MAX_TOKEN_TARGET) -> str:
    """
    Placeholder for an LLM compression call. Replace with your LLM provider code.
    This function must return the compressed text (<= target tokens).
    Example: use OpenAI chat completions, Anthropic, or your in-house LLM.
    """
    # --- Example pseudocode for OpenAI chat completion (replace with real call) ---
    # import openai
    # openai.api_key = os.environ["OPENAI_API_KEY"]
    # prompt = LLM_COMPRESSION_PROMPT.format(max_tokens=max_tokens) + "\n\n" + text
    # resp = openai.ChatCompletion.create(model="gpt-4o-mini", messages=[{"role":"user","content":prompt}], max_tokens=400)
    # return resp["choices"][0]["message"]["content"].strip()
    #
    # For now, we implement a simple heuristic fallback (sentence filter) to avoid external call:
    sentences = sentence_split(text)
    # Keep sentences that include importance keywords or are short and likely sentiment-bearing
    kept = []
    for s in sentences:
        if any(k.lower() in s.lower() for k in IMPORTANCE_KEYWORDS):
            kept.append(s)
        elif len(kept) < 3 and len(s.split()) < 40:
            kept.append(s)
    compressed = " ".join(kept)
    if tokens_length(compressed) > max_tokens:
        # naive truncation (last resort)
        compressed = " ".join(compressed.split()[: max_tokens * 2])
    return compressed

# ------------------------
# FinBERT inference wrapper
# ------------------------
def finbert_predict(text: str) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Returns probabilities vector [neg, neutral, pos] and a dict map label->prob
    """
    res = finbert_pipe(text[:10000])  # pipeline can fail on extremely long; ensure text not huge
    # res is a list of dicts for each label with 'label' and 'score'
    # mapping depends on model labels - let's map common labels to neg/neutral/pos
    # We'll sort by label names if they exist
    labels = {d['label'].lower(): d['score'] for d in res[0]}
    # Attempt common label names
    def get_prob(name):
        return labels.get(name, 0.0)
    # Known variants:
    neg = get_prob("negative") or get_prob("neg") or get_prob("bearish") or 0.0
    neu = get_prob("neutral") or get_prob("neutral/0") or 0.0
    pos = get_prob("positive") or get_prob("pos") or get_prob("bullish") or 0.0
    probs = np.array([neg, neu, pos], dtype=float)
    # normalize if needed
    if probs.sum() <= 0:
        probs = np.array([0.33, 0.34, 0.33], dtype=float)
    else:
        probs = probs / probs.sum()
    label_map = {"neg": float(probs[0]), "neu": float(probs[1]), "pos": float(probs[2])}
    return probs, label_map

# ------------------------
# Processing one document
# ------------------------
def process_single_document(doc_id: str, text: str, use_compression_fallback=True) -> Dict[str, Any]:
    """
    Process one report: chunk -> (optionally compress) -> FinBERT -> aggregate.
    Returns a dict with chunk-level and aggregated results.
    """
    cache_key = f"res::{sha1(doc_id)}"
    if cache_key in cache:
        return cache[cache_key]

    # Pre-clean: remove obvious disclaimers using regex (e.g., "disclaimer" blocks). Adjust as needed.
    text_clean = re.sub(r"(?is)^.*?disclaimer:.*?$", "", text)

    # Sentence split
    sentences = sentence_split(text_clean)

    # Chunk into token-safe groups
    chunks = chunk_sentences(sentences, max_tokens=TARGET_TOKENS)

    # If some chunks still exceed token length (rare), try compression
    final_chunks = []
    for ch in chunks:
        if tokens_length(ch) > TARGET_TOKENS:
            if USE_LLM_COMPRESSION and use_compression_fallback:
                ch2 = compress_with_llm(ch, max_tokens=TARGET_TOKENS)
                final_chunks.append(ch2)
            else:
                # force split
                words = ch.split()
                for i in range(0, len(words), TARGET_TOKENS*4):
                    final_chunks.append(" ".join(words[i:i+TARGET_TOKENS*4]))
        else:
            final_chunks.append(ch)

    # Run FinBERT on chunks
    chunk_results = []
    for ch in final_chunks:
        weight = detect_section_weight(ch)
        probs, label_map = finbert_predict(ch)
        chunk_results.append({
            "text": ch,
            "tokens": tokens_length(ch),
            "weight": weight,
            "probs": probs.tolist(),
            "label_map": label_map
        })

    # Aggregate: weighted average of probabilities
    weights = np.array([cr["weight"] * math.log(1 + cr["tokens"]) for cr in chunk_results], dtype=float)
    prob_matrix = np.array([cr["probs"] for cr in chunk_results], dtype=float)
    if prob_matrix.shape[0] == 0:
        agg_probs = np.array([0.33, 0.34, 0.33])
    else:
        weighted = (weights[:, None] * prob_matrix).sum(axis=0)
        if weighted.sum() == 0:
            agg_probs = np.array([0.33, 0.34, 0.33])
        else:
            agg_probs = weighted / weighted.sum()

    # Simple aggregated label (max prob)
    labels = ["neg", "neu", "pos"]
    agg_label = labels[int(np.argmax(agg_probs))]

    result = {
        "doc_id": doc_id,
        "n_chunks": len(chunk_results),
        "chunks": chunk_results,
        "agg_probs": agg_probs.tolist(),
        "agg_label": agg_label
    }
    cache[cache_key] = result
    return result

# ------------------------
# Batch processing
# ------------------------
def process_documents_batch(docs: List[Tuple[str, str]], workers=NUM_WORKERS) -> List[Dict[str, Any]]:
    """
    docs: list of (doc_id, text)
    """
    results = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(process_single_document, doc_id, text): doc_id for doc_id, text in docs}
        for fut in tqdm(as_completed(futures), total=len(futures)):
            try:
                r = fut.result()
                results.append(r)
            except Exception as e:
                doc_id = futures[fut]
                print(f"Error processing {doc_id}: {e}")
    return results

# ------------------------
# Save results helpers
# ------------------------
def results_to_dataframe(results: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append({
            "doc_id": r["doc_id"],
            "agg_label": r["agg_label"],
            "agg_neg": r["agg_probs"][0],
            "agg_neu": r["agg_probs"][1],
            "agg_pos": r["agg_probs"][2],
            "n_chunks": r["n_chunks"]
        })
    return pd.DataFrame(rows)

# ------------------------
# Example use
# ------------------------
if __name__ == "__main__":
    # Example: load all text files from ./reports and run pipeline
    report_dir = Path("./reports")
    files = list(report_dir.glob("*.txt"))
    docs = []
    for f in files:
        doc_id = f.stem
        text = f.read_text(encoding="utf-8")
        docs.append((doc_id, text))

    # For demo, do sequential small-run (avoid spawning large processes in notebooks)
    results = []
    for doc_id, txt in tqdm(docs):
        results.append(process_single_document(doc_id, txt))

    df = results_to_dataframe(results)
    df.to_parquet(Path(RESULTS_DIR) / "finbert_aggregated.parquet")
    print("Done. Saved parquet to:", Path(RESULTS_DIR) / "finbert_aggregated.parquet")
