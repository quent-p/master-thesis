from pathlib import Path
from typing import Tuple
import re, math, sys, os
import pandas as pd
import fitz  # PyMuPDF
from pdfminer.high_level import extract_text as pdfminer_extract_text
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# --- Transformers (optionnel) ---
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    from transformers import logging as hf_logging
    import torch
    _TRANSFORMERS_OK = True
    hf_logging.set_verbosity_error()
except Exception:
    _TRANSFORMERS_OK = False

# ================== CONFIG ==================
DESKTOP = Path.home() / "Desktop"
DESKTOP.mkdir(exist_ok=True)

INPUT_DIR = DESKTOP / "PDFs_a_analyser_2024"
EXCEL_PATH = DESKTOP / "analyse_rapports_UNIFIED.xlsx"
LEXICON_DIR = DESKTOP / "lexicons"
LM_MASTER_PATH = LEXICON_DIR / "Loughran-McDonald(positive & negative tone)_MasterDictionary_1993-2024.csv"

# FinBERT & ClimateBERT
FINBERT_ENABLED = True  # ← Active/désactive FinBERT
CLIMATE_ENABLED = True   # ← Active/désactive ClimateBERT

FINBERT_MODEL_NAME = "ProsusAI/finbert"
CLIMATE_MODEL_NAME = "climatebert/distilroberta-base-climate-sentiment"

MAX_TOKENS = 256
CHUNK_OVERLAP = 32
BATCH_SIZE = 8
SKIP_FINBERT_IF_DONE = True  # Ne recalcule pas FinBERT si déjà fait

# Colonnes
MANUAL_COLS = ["firm_name", "country", "sector", "year"]
AUTO_COLS = [
    "pages", "chars", "words", "sentences", "avg_words_per_sentence",
    "syllables_total","avg_syllables_per_word",
    "complex_word_ratio", "fog_index", "flesch_reading_ease", "flesch_kincaid_grade",
    "ln_pages", "ln_words", "ln_chars",
    "_sep_complexity_tone",
    "lm_pos", "lm_neg", "lm_pos_rate", "lm_neg_rate", "lm_sentiment", "lm_balance",
    "_sep_finbert",
    "finbert_positive", "finbert_negative", "finbert_neutral", "finbert_sentiment", "finbert_chunks",
    "_sep_climatebert",
    "climate_risk", "climate_neutral", "climate_opportunity", "climate_sentiment", "climate_chunks",
]

WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]+(?:-[A-Za-zÀ-ÖØ-öø-ÿ]+)*")
VOWELS = "aeiouyàâäéèêëîïôöùûüAEIOUYÀÂÄÉÈÊËÎÏÔÖÙÛÜ"

# ================== UTILS ==================
def count_syllables(word: str) -> int:
    """Compteur de syllabes amélioré pour l'anglais"""
    word = word.lower().strip()
    
    if len(word) <= 2:
        return 1
    
    # Supprimer 'e' final muet SAUF si précédé de certaines lettres
    if word.endswith('e') and len(word) > 3:
        if len(word) > 3 and word[-2] in 'lr' and word[-3] not in 'aeiouy':
            pass
        else:
            word = word[:-1]
    
    # Supprimer 'es' final
    if word.endswith('es') and len(word) > 3:
        word = word[:-2]
    
    # NE PAS supprimer -ed si précédé de 't' ou 'd' (ex: related, created)
    if word.endswith('ed') and len(word) > 4:
        if word[-3] not in 'td':
            word = word[:-2]
    
    # Compter groupes de voyelles
    vowels = "aeiouyàâäéèêëîïôöùûü"
    count = 0
    prev_was_vowel = False
    
    for char in word:
        is_vowel = char in vowels
        if is_vowel and not prev_was_vowel:
            count += 1
        prev_was_vowel = is_vowel
    
    return max(1, count)

def finbert_already_done(row: pd.Series) -> bool:
    """Retourne True si FinBERT a déjà été calculé pour cette ligne."""
    chunks = row.get("finbert_chunks")
    sent = row.get("finbert_sentiment")
    return (pd.notna(chunks) and (chunks or 0) > 0 and pd.notna(sent))

def advanced_clean_text(raw_text: str, min_words: int = 3000, min_lines: int = 100):
    """Nettoie le texte selon Li (2008) adapté"""
    
    stats = {
        'original_chars': len(raw_text),
        'original_words': len(raw_text.split()),
        'removed_headers': 0,
        'removed_short_lines': 0,
        'removed_numeric_paragraphs': 0,
        'removed_table_patterns': 0,
        'final_chars': 0,
        'final_words': 0,
        'final_lines': 0,
        'kept': True
    }
    
    text = raw_text
    text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'\r', '\n', text)
    
    lines = text.split('\n')
    
    # Détecter headers/footers répétitifs
    line_counter = {}
    for line in lines:
        cleaned_line = line.strip()
        if len(cleaned_line) > 3 and len(cleaned_line) < 100:
            line_counter[cleaned_line] = line_counter.get(cleaned_line, 0) + 1
    
    repetitive_lines = {line for line, count in line_counter.items() if count > 5}
    stats['removed_headers'] = len(repetitive_lines)

    # Filtrage ligne par ligne
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped in repetitive_lines:
            continue
        if len(stripped) < 10:
            stats['removed_short_lines'] += 1
            continue
        if re.match(r'^[\s\d\.\-\|]+$', stripped):
            stats['removed_short_lines'] += 1
            continue
        if stripped.startswith('http') or stripped.startswith('www.'):
            stats['removed_short_lines'] += 1
            continue
        cleaned_lines.append(line)
    
    # Regrouper en paragraphes
    paragraphs = []
    current_para = []
    
    for line in cleaned_lines:
        stripped = line.strip()
        if stripped:
            current_para.append(stripped)
            if stripped.endswith('.') and len(stripped) > 50:
                if len(current_para) >= 2:
                    paragraphs.append(' '.join(current_para))
                    current_para = []
        else:
            if len(current_para) >= 2:
                paragraphs.append(' '.join(current_para))
            current_para = []
    
    if len(current_para) >= 2:
        paragraphs.append(' '.join(current_para))
    
    print(f"     DEBUG: {len(paragraphs)} paragraphes créés")
    
    # Filtrer paragraphes (>30% alphabétique)
    final_paragraphs = []
    for para in paragraphs:
        alpha_chars = sum(1 for c in para if c.isalpha())
        total_chars = len(para)
        if total_chars == 0:
            continue
        alpha_ratio = alpha_chars / total_chars
        if alpha_ratio > 0.30:
            final_paragraphs.append(para)
        else:
            stats['removed_numeric_paragraphs'] += 1
    
    print(f"     DEBUG: {len(paragraphs)} paragraphes → {len(final_paragraphs)} gardés après filtre alphabétique")
    
    # Supprimer patterns de tableaux
    ultra_cleaned = []
    for para in final_paragraphs:
        separator_chars = para.count('|') + para.count('─') + para.count('_') + para.count('=')
        if separator_chars > len(para) * 0.3:
            stats['removed_table_patterns'] += 1
            continue
        
        digit_sequences = re.findall(r'\d[\d,\.\s]+\d', para)
        if len(digit_sequences) > 3:
            stats['removed_table_patterns'] += 1
            continue
        
        words = para.split()
        if len(words) > 0:
            numeric_words = sum(1 for w in words if any(c.isdigit() for c in w))
            if numeric_words / len(words) > 0.4:
                stats['removed_table_patterns'] += 1
                continue
        
        ultra_cleaned.append(para)
    
    print(f"     DEBUG: {len(final_paragraphs)} paragraphes → {len(ultra_cleaned)} après suppression tableaux")
    
    final_text = '\n\n'.join(ultra_cleaned)
    final_text = re.sub(r' +', ' ', final_text)
    final_text = re.sub(r'\n{3,}', '\n\n', final_text)
    
    stats['final_chars'] = len(final_text)
    stats['final_words'] = len(final_text.split())
    stats['final_lines'] = len([p for p in ultra_cleaned if p.strip()])
    
    # Fallback si tout supprimé
    if stats['final_words'] == 0:
        print(f"     ⚠️ Nettoyage avancé a tout supprimé, fallback sur nettoyage simple")
        simple_clean = raw_text.replace("-\n", "")
        simple_clean = re.sub(r"\n+", " ", simple_clean)
        simple_clean = re.sub(r"\s{2,}", " ", simple_clean)
        stats['final_words'] = len(simple_clean.split())
        stats['final_chars'] = len(simple_clean)
        stats['kept'] = True
        return simple_clean.strip(), stats
    
    if stats['final_words'] < min_words and stats['final_lines'] < min_lines:
        stats['kept'] = False
    
    return final_text, stats

def clean_text(raw_text: str) -> str:
    """Nettoyage avancé selon Li (2008)"""
    cleaned, stats = advanced_clean_text(raw_text)
    return cleaned

def extract_text_from_pdf(pdf_path: Path) -> tuple[str, int]:
    texte, pages = "", 0
    try:
        with fitz.open(str(pdf_path)) as doc:
            pages = doc.page_count
            for page in doc:
                texte += page.get_text("text") or ""
    except Exception:
        pass
    if not texte.strip():
        try:
            texte = pdfminer_extract_text(str(pdf_path)) or ""
        except Exception:
            pass
    return texte, pages  # Ne pas nettoyer ici

def compute_metrics(text: str) -> dict:
    words = WORD_RE.findall(text)
    wc = len(words)
    sc = max(1, len(re.findall(r"[\.!?](?:\s|$)", text)))
    chars = len(text)
    
    syllables_total = sum(count_syllables(w) for w in words) if wc > 0 else 0
    avg_syllables_per_word = (syllables_total / wc) if wc > 0 else 0.0
    
    avg_wps = round(wc / sc, 2) if sc else 0.0
    complex_words = sum(1 for w in words if count_syllables(w) >= 3)
    complex_ratio = (complex_words / wc) if wc else 0.0
    
    fog = 0.4 * ((wc / sc) + 100 * complex_ratio) if sc else 0.0
    
    if wc > 0 and sc > 0:
        wps = wc / sc
        syl_per_word = syllables_total / wc
        fre = round(206.835 - 1.015 * wps - 84.6 * syl_per_word, 2)
        fkg = round(0.39 * wps + 11.8 * syl_per_word - 15.59, 2)
    else:
        fre = None
        fkg = None

    return {
        "chars": chars,
        "words": wc,
        "sentences": sc,
        "avg_words_per_sentence": avg_wps,
        "complex_word_ratio": round(100 * complex_ratio, 2),
        "fog_index": round(fog, 2),
        "flesch_reading_ease": fre,
        "flesch_kincaid_grade": fkg,
        "ln_words": round(math.log(wc), 4) if wc > 0 else None,
        "ln_chars": round(math.log(chars), 4) if chars > 0 else None,
        "syllables_total": syllables_total,
        "avg_syllables_per_word": round(avg_syllables_per_word, 4) if wc > 0 else None,
    }

# ---------- Loughran & McDonald ----------
def load_lm_lexicons_from_master(csv_path: Path) -> tuple[set, set]:
    if not csv_path.exists():
        return set(), set()
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        df = pd.read_csv(csv_path, encoding="latin1", engine="python")
    df.columns = [c.strip().lower() for c in df.columns]
    word_col = "word" if "word" in df.columns else ("term" if "term" in df.columns else df.columns[0])
    pos_col = [c for c in df.columns if "positive" in c][0]
    neg_col = [c for c in df.columns if "negative" in c][0]
    positive = set(df.loc[df[pos_col] != 0, word_col].astype(str).str.lower().str.strip())
    negative = set(df.loc[df[neg_col] != 0, word_col].astype(str).str.lower().str.strip())
    return positive, negative

def lm_counts(text: str, pos_words: set, neg_words: set) -> dict:
    tokens = [w.lower() for w in WORD_RE.findall(text)]
    wc = len(tokens)
    if wc == 0:
        return {"lm_pos": 0, "lm_neg": 0, "lm_pos_rate": None, "lm_neg_rate": None, "lm_sentiment": None, "lm_balance": None}
    pos = sum(1 for w in tokens if w in pos_words)
    neg = sum(1 for w in tokens if w in neg_words)
    total = pos + neg
    pos_rate = round(1000 * pos / wc, 4)
    neg_rate = round(1000 * neg / wc, 4)
    sentiment = round((pos - neg) / total, 4) if total > 0 else None
    balance = round(1 - abs((pos - neg) / total), 4) if total > 0 else None
    return {
        "lm_pos": pos,
        "lm_neg": neg,
        "lm_pos_rate": pos_rate,
        "lm_neg_rate": neg_rate,
        "lm_sentiment": sentiment,
        "lm_balance": balance
    }

# ---------- FinBERT ----------
def try_load_finbert():
    if not FINBERT_ENABLED or not _TRANSFORMERS_OK:
        print("💼 FinBERT désactivé ou dépendances absentes → on continue sans.")
        sys.stdout.flush()
        return None, None
    try:
        print("💼 Chargement FinBERT...")
        sys.stdout.flush()
        tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL_NAME, use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL_NAME)
        device = 0 if torch.cuda.is_available() else -1
        clf = pipeline("text-classification", model=model, tokenizer=tokenizer, device=device, return_all_scores=True)
        print(f" ✓ FinBERT prêt (Device: {'GPU' if device == 0 else 'CPU'})")
        sys.stdout.flush()
        return clf, tokenizer
    except Exception as e:
        print(f"❌ Impossible de charger FinBERT: {e}")
        sys.stdout.flush()
        return None, None

# ---------- ClimateBERT ----------
def try_load_climatebert():
    if not CLIMATE_ENABLED or not _TRANSFORMERS_OK:
        print("🌍 ClimateBERT désactivé ou dépendances absentes → on continue sans.")
        sys.stdout.flush()
        return None, None
    try:
        print("🌍 Chargement ClimateBERT...")
        sys.stdout.flush()
        tokenizer = AutoTokenizer.from_pretrained(CLIMATE_MODEL_NAME, use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(CLIMATE_MODEL_NAME)
        device = 0 if torch.cuda.is_available() else -1
        clf = pipeline("text-classification", model=model, tokenizer=tokenizer, device=device, return_all_scores=True)
        print(f" ✓ ClimateBERT prêt (Device: {'GPU' if device == 0 else 'CPU'})")
        sys.stdout.flush()
        return clf, tokenizer
    except Exception as e:
        print(f"❌ Impossible de charger ClimateBERT: {e}")
        sys.stdout.flush()
        return None, None

def split_into_chunks(text: str, tokenizer, max_length: int = MAX_TOKENS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    ids = tokenizer.encode(text, add_special_tokens=False, truncation=False)
    chunks, start = [], 0
    while start < len(ids):
        end = start + max_length
        chunk_ids = ids[start:end]
        chunks.append(tokenizer.decode(chunk_ids, skip_special_tokens=True))
        if end >= len(ids):
            break
        start = end - overlap
    return chunks

def analyze_finbert_sentiment(text: str, clf, tokenizer) -> dict:
    """Analyse FinBERT avec barre de progression."""
    if not text.strip():
        return {
            "finbert_positive": None,
            "finbert_negative": None,
            "finbert_neutral": None,
            "finbert_sentiment": None,
            "finbert_chunks": 0
        }
    chunks = split_into_chunks(text, tokenizer)
    if not chunks:
        return {
            "finbert_positive": None,
            "finbert_negative": None,
            "finbert_neutral": None,
            "finbert_sentiment": None,
            "finbert_chunks": 0
        }
    weights = [max(1, len(tokenizer.encode(c, add_special_tokens=False))) for c in chunks]
    wsum = float(sum(weights))

    outs_all = []
    with tqdm(total=len(chunks), desc="💼 FinBERT chunks", unit="chunk", leave=False) as pbar:
        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i:i + BATCH_SIZE]
            try:
                outs_batch = clf(batch, truncation=True, max_length=MAX_TOKENS, batch_size=BATCH_SIZE)
            except Exception:
                outs_batch = [clf(x, truncation=True, max_length=MAX_TOKENS) for x in batch]
            outs_all.extend(outs_batch)
            pbar.update(len(batch))
            sys.stdout.flush()
    pos_w = neg_w = neu_w = 0.0
    for out, w in zip(outs_all, weights):
        d = {d["label"]: d["score"] for d in out}
        pos_w += w * d.get("positive", 0.0)
        neg_w += w * d.get("negative", 0.0)
        neu_w += w * d.get("neutral", 0.0)
    avg_pos = round(pos_w / wsum, 4)
    avg_neg = round(neg_w / wsum, 4)
    avg_neu = round(neu_w / wsum, 4)
    sentiment = round(avg_pos - avg_neg, 4)
    return {
        "finbert_positive": avg_pos,
        "finbert_negative": avg_neg,
        "finbert_neutral": avg_neu,
        "finbert_sentiment": sentiment,
        "finbert_chunks": len(chunks)
    }

def analyze_climate_sentiment(text: str, clf, tokenizer) -> dict:
    """Analyse ClimateBERT avec barre de progression."""
    if not text.strip():
        return {
            "climate_risk": None,
            "climate_neutral": None,
            "climate_opportunity": None,
            "climate_sentiment": None,
            "climate_chunks": 0
        }
    chunks = split_into_chunks(text, tokenizer)
    if not chunks:
        return {
            "climate_risk": None,
            "climate_neutral": None,
            "climate_opportunity": None,
            "climate_sentiment": None,
            "climate_chunks": 0
        }
    
    weights = [max(1, len(tokenizer.encode(c, add_special_tokens=False))) for c in chunks]
    wsum = float(sum(weights))

    outs_all = []
    with tqdm(total=len(chunks), desc="🌍 ClimateBERT chunks", unit="chunk", leave=False) as pbar:
        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i:i + BATCH_SIZE]
            try:
                outs_batch = clf(batch, truncation=True, max_length=MAX_TOKENS, batch_size=BATCH_SIZE)
            except Exception:
                outs_batch = [clf(x, truncation=True, max_length=MAX_TOKENS) for x in batch]
            outs_all.extend(outs_batch)
            pbar.update(len(batch))
            sys.stdout.flush()
    
    risk_w = neutral_w = opp_w = 0.0
    for out, w in zip(outs_all, weights):
        d = {d["label"]: d["score"] for d in out}
        risk_w += w * d.get("risk", 0.0)
        neutral_w += w * d.get("neutral", 0.0)
        opp_w += w * d.get("opportunity", 0.0)
    
    avg_risk = round(risk_w / wsum, 4)
    avg_neutral = round(neutral_w / wsum, 4)
    avg_opp = round(opp_w / wsum, 4)
    sentiment = round(avg_opp - avg_risk, 4)
    
    return {
        "climate_risk": avg_risk,
        "climate_neutral": avg_neutral,
        "climate_opportunity": avg_opp,
        "climate_sentiment": sentiment,
        "climate_chunks": len(chunks)
    }

# ---------- Guess firm/year ----------
MARKERS = [
    "annual report", "integrated report", "universal registration document",
    "registration document", "sustainability report", "csr report", "esg report",
    "rapport annuel", "document d'enregistrement universel", "rapport intégré",
    "annual", "report", "document", "rapport"
]
JUNK_LEGAL = r"\b(plc|s\.?a\.?|sa|se|ag|nv|ltd|inc|llc)\b\.?"

def smart_title(t: str) -> str:
    parts, out = t.split(), []
    for p in parts:
        out.append(p if p.isupper() else (p.upper() if len(p) <= 3 else p.capitalize()))
    return " ".join(out)

def guess_firm_and_year(file_name: str) -> tuple[str | None, int | None]:
    stem = Path(file_name).stem
    s = re.sub(r"[_\-.]+", " ", stem)
    s = re.sub(r"\s+", " ", s).strip()
    m = re.search(r"(20\d{2}|19\d{2})", s)
    year = int(m.group(1)) if m else None
    cut = len(s)
    if m:
        cut = min(cut, m.start())
    low = s.lower()
    for mk in MARKERS:
        idx = low.find(mk)
        if idx != -1:
            cut = min(cut, idx)
    firm = s[:cut].strip()
    firm = re.sub(JUNK_LEGAL, "", firm, flags=re.I).strip()
    firm = smart_title(firm)
    return (firm or None), year

# ---------- DataFrame helpers ----------
def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    base_cols = ["file_name", "file_path", *MANUAL_COLS, *AUTO_COLS]
    for c in base_cols:
        if c not in df.columns:
            df[c] = None
    return df[base_cols]

# ---------- Main ----------
def main():
    # Créer répertoires
    DESKTOP.mkdir(exist_ok=True)
    INPUT_DIR.mkdir(exist_ok=True)
    LEXICON_DIR.mkdir(exist_ok=True)
    
    print(f"📁 Configuration:")
    print(f"   PDFs: {INPUT_DIR}")
    print(f"   Excel: {EXCEL_PATH}")
    print(f"   FinBERT: {'✅ Activé' if FINBERT_ENABLED else '❌ Désactivé'}")
    print(f"   ClimateBERT: {'✅ Activé' if CLIMATE_ENABLED else '❌ Désactivé'}")
    sys.stdout.flush()
    
    # Charger Excel
    if EXCEL_PATH.exists():
        df = pd.read_excel(EXCEL_PATH)
    else:
        df = pd.DataFrame(columns=["file_name", "file_path", *MANUAL_COLS, *AUTO_COLS])
    df = ensure_columns(df)
    
    # L&M
    print("\n📚 Chargement Loughran-McDonald...")
    lm_pos_words, lm_neg_words = load_lm_lexicons_from_master(LM_MASTER_PATH)
    print(f" ✓ LM loaded: +{len(lm_pos_words)} / -{len(lm_neg_words)} words")
    sys.stdout.flush()
    
    # Charger modèles ML
    finbert_clf, finbert_tok = try_load_finbert()
    climate_clf, climate_tok = try_load_climatebert()
    
    # PDFs
    print(f"\n🔍 Recherche PDFs dans : {INPUT_DIR}")
    pdfs = sorted([p for p in INPUT_DIR.rglob('*') if p.suffix.lower() == '.pdf'])
    print(f"📄 PDF trouvés : {len(pdfs)}")
    sys.stdout.flush()
    
    if not pdfs:
        print("⚠️ Aucun PDF trouvé")
        return
    
    for idx, pdf in enumerate(pdfs, 1):
        file_path = str(pdf.resolve())
        file_name = pdf.name
        
        name_mask = (df["file_name"] == file_name)
        exists_by_name = name_mask.any()
        process_finbert = FINBERT_ENABLED and finbert_clf and finbert_tok
        
        if SKIP_FINBERT_IF_DONE and exists_by_name and process_finbert:
            row = df.loc[name_mask].iloc[0]
            if finbert_already_done(row):
                process_finbert = False
                print(f"\n[{idx}/{len(pdfs)}] 🔁 {file_name}")
                print(f"   FinBERT: ✅ Déjà fait | ClimateBERT: {'⏳ À faire' if CLIMATE_ENABLED else '❌ Désactivé'}")
            else:
                print(f"\n[{idx}/{len(pdfs)}] 📖 {file_name}")
        else:
            print(f"\n[{idx}/{len(pdfs)}] 📖 {file_name}")
        sys.stdout.flush()
        
        # Extraction
        text_raw, pages = extract_text_from_pdf(pdf)
        print(f"   Pages: {pages}, Chars bruts: {len(text_raw)}")
        sys.stdout.flush()
        
        text = clean_text(text_raw)
        print(f"   Chars nettoyés: {len(text)}")
        
        # Métriques de base
        m = compute_metrics(text)
        m["ln_pages"] = round(math.log(pages), 4) if pages > 0 else None
        
        # Filtre 3000 mots / 100 phrases
        if m['words'] < 3000 or m['sentences'] < 100:
            print(f"   ⚠️ IGNORÉ : {m['words']} mots / {m['sentences']} phrases (seuil: 3000/100)")
            continue
        
        # L&M
        lm = lm_counts(text, lm_pos_words, lm_neg_words)
        m.update(lm)
        
        # Guess firm/year
        g_name, g_year = guess_firm_and_year(file_name)
        
        # Upsert
        if exists_by_name:
            base_idx = df.index[name_mask][0]
        else:
            new_row = {
                "file_name": file_name,
                "file_path": file_path,
                "firm_name": g_name,
                "country": None,
                "sector": None,
                "year": g_year,
                "pages": pages,
                **{k: m.get(k) for k in AUTO_COLS if k in m},
            }
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            base_idx = df.index[df["file_name"] == file_name][0]
        
        # MAJ colonnes auto
        df.at[base_idx, "pages"] = pages
        df.at[base_idx, "file_path"] = file_path
        for col in AUTO_COLS:
            if col in m:
                df.at[base_idx, col] = m[col]
        
        # FinBERT
        if process_finbert:
            print("   💼 Analyse FinBERT...")
            sys.stdout.flush()
            finbert = analyze_finbert_sentiment(text, finbert_clf, finbert_tok)
            for k in ["finbert_positive", "finbert_negative", "finbert_neutral",
                      "finbert_sentiment", "finbert_chunks"]:
                df.at[base_idx, k] = finbert[k]
            print(f"   ✓ FinBERT: Sentiment={finbert['finbert_sentiment']}, Chunks={finbert['finbert_chunks']}")
        
        # ClimateBERT
        if CLIMATE_ENABLED and climate_clf and climate_tok:
            print("   🌍 Analyse ClimateBERT...")
            sys.stdout.flush()
            climate = analyze_climate_sentiment(text, climate_clf, climate_tok)
            for k in ["climate_risk", "climate_neutral", "climate_opportunity",
                      "climate_sentiment", "climate_chunks"]:
                df.at[base_idx, k] = climate[k]
            print(f"   ✓ ClimateBERT: Sentiment={climate['climate_sentiment']}, Chunks={climate['climate_chunks']}")
    
    # Sauvegarde
    print(f"\n💾 Sauvegarde dans {EXCEL_PATH}...")
    sys.stdout.flush()
    
    df = ensure_columns(df)
    
    with pd.ExcelWriter(EXCEL_PATH, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Data base")
    
    print(f"✅ Terminé : {len(pdfs)} fichiers traités.")
    print(f"➡️ Résultats : {EXCEL_PATH}")
    sys.stdout.flush()

if __name__ == "__main__":
    main()
