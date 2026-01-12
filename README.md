# Sustainability Report Analysis (CSRD)

Python script to automatically analyze sustainability reports in PDF format.

## Purpose

This code extracts textual metrics to compare reports **before** (2023) and **after** (2024) the EU's CSRD directive came into effect.

## What the script measures

| Dimension | Metrics | Research question |
|-----------|---------|-------------------|
| **Complexity** | Fog Index, flesch_reading_ease, flesch_kincaid_grade | Is the text harder to read? |
| **Length** | ln_pages, ln_chars | Are reports getting longer? |
| **Tone** | Loughran-McDonald, FinBERT, ClimateBERT | Is the tone more positive or negative? |

## Getting Started (Step by Step)

### Step 1: Install Python

If you don't have Python installed:
- Download it from [python.org](https://www.python.org/downloads/)
- During installation, **check the box "Add Python to PATH"**

### Step 2: Install required libraries

Open a terminal (Command Prompt on Windows, Terminal on Mac) and run:
```bash
pip install pandas PyMuPDF pdfminer.six transformers torch tqdm openpyxl
```

> This may take a few minutes (especially for `torch` and `transformers`)

### Step 3: Set up your folders

On your Desktop, create the following folders:
```
Desktop/
├── PDFs_a_analyser_2024/    ← Put your PDF reports here
└── lexicons/                 ← Put the Loughran-McDonald dictionary here
```

### Step 4: Download the Loughran-McDonald dictionary

1. Go to [Loughran-McDonald website](https://sraf.nd.edu/loughranmcdonald-master-dictionary/)
2. Download the Master Dictionary CSV file
3. Place it in `Desktop/lexicons/`

### Step 5: Run the script

In your terminal, navigate to where the script is located and run:
```bash
python analyse_rapports_2.py
```

### Step 6: Get your results

Once finished, open `Desktop/analyse_rapports_UNIFIED.xlsx` all metrics are there! 

## Notes

- Minimum threshold: 3,000 words and 100 sentences per report
- FinBERT and ClimateBERT models are downloaded automatically
---

*Developed as part of a Master's thesis in sustainability accounting.*
