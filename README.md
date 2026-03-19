# Medical Conformer RAG (Conformer + FAISS + Groq)

This project implements a clinical speech-to-text and retrieval-augmented generation (RAG) system that converts consultation audio into structured transcripts and enables context-aware question answering.

# Overview

Transcribes clinical audio using a Conformer-based STT model

Segments and stores transcripts as chunked JSON

Indexes transcript embeddings using FAISS

Retrieves relevant context to generate LLM-based clinical responses

# Pipeline

Audio Input
Processes clinical audio files (e.g., audio/patient_diagnosis.wav)

Speech-to-Text (STT)
Transcribes audio in overlapping chunks and stores results in:
data/audio_chunks/patient_diagnosis_chunks.json

Embedding & Indexing
Converts transcript chunks into embeddings and indexes them in FAISS

Retrieval-Augmented QA
Accepts user queries and retrieves relevant transcript context

LLM Response Generation
Uses Groq LLM with a clinical prompt to generate grounded answers

# Prompting Strategy

STT (Conformer): No prompt conditioning (fully model-driven transcription)

LLM (Groq): Uses a strict clinical RAG prompt (CLINICAL_RAG_PROMPT_TEMPLATE) to ensure responses are based only on retrieved consultation data

# Key Features

End-to-end clinical audio understanding pipeline

Context-aware QA grounded in patient consultation data

Efficient semantic retrieval using FAISS

Modular design for experimenting with alternative STT architectures

## Project Structure

```text
.
├── app.py
├── audio/
│   └── patient_diagnosis.wav
├── data/
│   └── audio_chunks/
│       └── patient_diagnosis_chunks.json
├── faiss_store/
├── src/
│   ├── audio_to_json_chunks.py
│   ├── data_loader.py
│   ├── embedding.py
│   ├── eval_rag.py
│   ├── search.py
│   └── vectorstore.py
└── stt/
    ├── audio.py
    ├── conformer_ctc_best.pth
    ├── inference.py
    └── model.py
```

## Requirements

- Python `>=3.10,<3.13`
- Groq API key (for LLM answers)

## Setup

### Option A: Using `uv` (recommended)

```bash
uv sync
```

### Option B: Using `pip`

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Environment Variables

Create `.env` in project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

## Run

```bash
.venv/bin/python app.py
```

Then ask questions at:

```text
Your question:
```

Type `exit` to stop.

## Output JSON Format

Each line in `data/audio_chunks/patient_diagnosis_chunks.json` is one chunk:

```json
{"chunk_id": 0, "start_sec": 0.0, "end_sec": 30.0, "text": "...", "confidence": 0.75}
```

## Change Audio Input

Edit in `app.py`:

- `audio_path`
- `json_output_path` (optional destination)
- `chunk_sec` and `overlap_sec` (if needed)

## Troubleshooting

- If model download fails once due network restrictions, rerun when network is available.
- If LLM answers fail, check `GROQ_API_KEY` in `.env`.
- If transcript quality is low, retrain/fine-tune Conformer or adjust chunking.

## GitHub Push Guide

If this is your first push for this folder:

```bash
git init
git add .
git commit -m "Initial clinical audio RAG pipeline"
git branch -M main
git remote add origin <YOUR_GITHUB_REPO_URL>
git push -u origin main
```

If repo is already initialized (this project is):

```bash
git add .
git commit -m "Clinical audio RAG setup + README"
git push -u origin main
```

If no remote is set yet, add one first:

```bash
git remote add origin <YOUR_GITHUB_REPO_URL>
git push -u origin main
```
