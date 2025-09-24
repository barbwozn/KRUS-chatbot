
# -*- coding: utf-8 -*-
import os, re, unicodedata
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
import torch


try:
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
except Exception:
    pass

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever

CSV_DIR = "data/csv"
#MultiQuery
LLM_MQ_ENABLED = True
MQ_BASE_VARIANTS = 4
MQ_LLM_VARIANTS = 8

# HyDE
USE_HYDE = True
HYDE_VARIANTS_BASE = 3
HYDE_VARIANTS_LLM = 3
HYDE_STRIP_NUMBERS = True

# Wagi 
RRF_WEIGHTS = {"dense_q": 1.0, "dense_mq": 0.95, "hyde": 0.7, "bm25": 1.0}

MIN_CE_FOR_CONTEXT = 0.06        # score min

USE_LLM = True

NATIONAL_TOKENS = {
    "ogolem", "ogółem", "-", "", "polska", "kraj", "poland",
    "cały kraj", "cala polska", "cała polska", "caly kraj"
}
FORCE_NATIONAL_IF_REGION_UNSPECIFIED = True 
PREFER_TYP_FROM_QUERY = True                 

def strip_acc(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", str(s)) if unicodedata.category(c) != "Mn")

def norm_text(s: str) -> str:
    s = strip_acc(str(s)).lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _is_national(region: Optional[str]) -> bool:
    if region is None:
        return True
    s = norm_text(region)
    s2 = re.sub(r"[^\w ]", "", s)
    return (s2 in NATIONAL_TOKENS) or s2.startswith("ogolem")


FIELD_VOCAB: Dict[str, set[str]] = {"dataset": set(), "measure": set(), "region": set(), "typ": set()}
VOIVODESHIP_CANON = {
    "dolnoslaskie","kujawsko-pomorskie","lubelskie","lubuskie","lodzkie","malopolskie",
    "mazowieckie","opolskie","podkarpackie","podlaskie","pomorskie","slaskie",
    "swietokrzyskie","warminsko-mazurskie","wielkopolskie","zachodniopomorskie"
}

REGION_INDEX = []  

_REGION_STOPWORDS = {"woj", "woj.", "wojewodztwo", "region", "kraj", "panstwo", "panstwie", "ogolem", "razem"}
_PL_VOWELS = set("aeiouy") 
_PL_SUFFIXES = [
    "skiego","skimi","skich","skim","skie","ckiego","ckimi","ckich","ckim","ckie",
    "owego","owemu","owych","owym","owa","owe","owy",
    "iego","iemu","owej","owe","owy","owa","owym","owych",
    "ami","ach","owi","ego","emu","om","ym","im","em","ie","ia","iu","a","e","i","y","o","u","ą","ę"
]

def _is_voivodeship_token(s: Optional[str]) -> bool:
    if not s: return False
    return norm_text(s) in VOIVODESHIP_CANON

def _is_special_internal_region(s: Optional[str]) -> bool:
    if not s: return False
    t = norm_text(s)
    return t in {"mswia","ms","msw","msz"} or t.startswith("ms ")

REGION_TAXONOMY = {"national": set(), "voivodeship": set(), "country": set()}

def rebuild_region_taxonomy() -> None:
    regs = {norm_text(r) for r in FIELD_VOCAB["region"] if r is not None}
    national = {r for r in regs if _is_national(r)}
    voiv     = {r for r in regs if _is_voivodeship_token(r)}
    specials = {r for r in regs if _is_special_internal_region(r)}
    country  = {r for r in regs if r not in national and r not in voiv and r not in specials and r not in {"", "-"}}
    REGION_TAXONOMY["national"]    = national
    REGION_TAXONOMY["voivodeship"] = voiv
    REGION_TAXONOMY["country"]     = country

def _is_country_name(name: Optional[str]) -> bool:
    return norm_text(name or "") in REGION_TAXONOMY["country"]
def _tokenize_words(s: str) -> list[str]:
    s = norm_text(s)
    s = re.sub(r"[-_/]", " ", s)
    toks = re.findall(r"[a-z0-9]+", s)
    return [t for t in toks if t and t not in _REGION_STOPWORDS]

def _strip_pl_suffix(tok: str) -> str:
    t = tok
    for suf in sorted(_PL_SUFFIXES, key=len, reverse=True):
        if t.endswith(suf) and len(t) - len(suf) >= 3:
            return t[: -len(suf)]
    return t

def _approx_syllables(word: str) -> list[str]:
    w = re.sub(r"[^a-z]", "", norm_text(word))
    if len(w) <= 3:
        return [w]
    syl, cur = [], ""
    for ch in w:
        if ch in _PL_VOWELS and cur and cur[-1] not in _PL_VOWELS:
            syl.append(cur); cur = ch
        else:
            cur += ch
    if cur: syl.append(cur)
    ngr = set()
    for n in (3, 4):
        for i in range(0, max(0, len(w) - n + 1)):
            ngr.add(w[i:i+n])
    return list(set(syl) | ngr)

def _roots_from_text(s: str) -> set[str]:
    return { _strip_pl_suffix(t) for t in _tokenize_words(s) if len(_strip_pl_suffix(t)) >= 3 }

def _syllset_from_text(s: str) -> set[str]:
    out = set()
    for t in _tokenize_words(s):
        for seg in _approx_syllables(t):
            if len(seg) >= 2:
                out.add(seg)
    return out

def _chargrams_from_text(s: str) -> set[str]:
    w = re.sub(r"[^a-z]", "", norm_text(s))
    out = set()
    for n in (3, 4):
        for i in range(0, max(0, len(w) - n + 1)):
            out.add(w[i:i+n])
    return out

def _add_to_vocab(val, field):
    if val is None: return
    s = str(val).strip()
    if not s: return
    FIELD_VOCAB[field].add(norm_text(s))

NUM_PAT = re.compile(r"(?<![A-Za-z])[-+]?\d[\d\s,./%]*")
def _strip_numbers(text: str) -> str:
    return NUM_PAT.sub("<X>", str(text))

def _read_csv_smart(path: str) -> pd.DataFrame:
    for enc in ["utf-8-sig", "cp1250", "iso-8859-2", "latin-1"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    return pd.read_csv(path)

def parse_value(x) -> Optional[float]:
    if x is None: return None
    s = str(x).strip()
    if s == "" or norm_text(s) in {"nan", "brak", "null"}:
        return None
    s = s.replace(" ", "").replace("\u00A0", "").replace(",", ".")
    try:
        return float(s)
    except Exception:
        return None

def norm_period(p: Optional[str]) -> Optional[str]:
    if not p or str(p).strip()=="":
        return None
    s = str(p).replace("\u00A0"," ").strip()
    s = s.upper()
    s = re.sub(r"[R]\.?$", "", s)
    s = re.sub(r"\s+", "", s)
    s = s.replace("/", "-").replace("_","-")
    if re.match(r"^20\d{2}-Q[1-4]$", s):
        return s
    m = re.match(r"^(20\d{2})Q([1-4])$", s)
    if m:
        return f"{m.group(1)}-Q{m.group(2)}"
    m = re.match(r"^Q([1-4])[- ]?(20\d{2})$", s)
    if m:
        return f"{m.group(2)}-Q{m.group(1)}"
    return s  

_QUARTER_WORDS = {
    "1":"1","i":"1","pierwszy":"1","pierwszym":"1",
    "2":"2","ii":"2","drugi":"2","drugim":"2",
    "3":"3","iii":"3","trzeci":"3","trzecim":"3",
    "4":"4","iv":"4","czwarty":"4","czwartym":"4",
}
_KW_PAT = r"kw(?:\.|art(?:ał|ale|al|)|)"

def _extract_pl_quarter_period(qn: str) -> Optional[str]:
    t = norm_text(qn)
    m = re.search(rf"\b({'|'.join(_QUARTER_WORDS.keys())})\s*{_KW_PAT}\b.*?\b(20\d{{2}})\b", t)
    if m:
        q = _QUARTER_WORDS.get(m.group(1), None); y = m.group(2)
        return f"{y}-Q{q}" if q else None
    m = re.search(rf"\b(20\d{{2}})\b.*?\b({'|'.join(_QUARTER_WORDS.keys())})\s*{_KW_PAT}\b", t)
    if m:
        y = m.group(1); q = _QUARTER_WORDS.get(m.group(2), None)
        return f"{y}-Q{q}" if q else None
    return None

def clean_measure_and_type(measure: str, typ: Optional[str]) -> Tuple[str, Optional[str]]:
    m = str(measure or "").strip()
    t = None if (typ is None or str(typ).strip()=="") else str(typ).strip()
    m = m.replace("przciętna", "przeciętna").replace(" w zl", " w zł")
    paren = re.search(r"\(([^)]+)\)", m)
    if paren and (t is None or t==""):
        t = paren.group(1).strip()
    m = re.sub(r"\s*\([^)]+\)\s*", " ", m)
    m = re.sub(r"\s+", " ", m).strip()
    return m, t

def make_page_text(row: dict) -> str:
    return " | ".join([
        f"dataset: {row.get('dataset')}",
        f"measure: {row.get('measure_clean')}",
        f"value: {row.get('value_float')}",
        f"region: {row.get('region') or '-'}",
        f"period: {row.get('period_norm') or row.get('period_raw') or '-'}",
        f"typ: {row.get('typ_clean') or '-'}",
    ])

def build_documents_from_standard_csv(dataset_name: str, csv_path: str) -> List[Document]:
    df = _read_csv_smart(csv_path)
    required = ["dataset","measure","value","region","period","typ"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Brak wymaganej kolumny: {c}")
    docs: List[Document] = []
    for i, row in df.iterrows():
        dataset = str(row.get("dataset") or "").strip()
        measure_raw = row.get("measure")
        typ_raw     = row.get("typ")
        measure_clean, typ_clean = clean_measure_and_type(measure_raw, typ_raw)

        value_float = parse_value(row.get("value"))
        region_raw  = None if pd.isna(row.get("region")) else str(row.get("region")).strip()
        period_raw  = None if pd.isna(row.get("period")) else str(row.get("period")).strip()
        period_norm_ = norm_period(period_raw)

        rec = {
            "dataset": dataset or dataset_name,
            "measure_clean": measure_clean,
            "typ_clean": typ_clean,
            "value_float": value_float,
            "region": region_raw,
            "period_raw": period_raw,
            "period_norm": period_norm_,
        }
        page_text = make_page_text(rec)
        meta: Dict[str, Any] = {
            "dataset": dataset or dataset_name,
            "source_file": os.path.basename(csv_path),
            "row_index": int(i),
            "okres": period_norm_ or period_raw,
            "region": region_raw,
            "type": typ_clean,
            "measure": measure_clean,
            "value": value_float,
        }
        docs.append(Document(page_content=page_text, metadata=meta))

        # uczymy słownik
        _add_to_vocab(meta["dataset"], "dataset")
        _add_to_vocab(meta["measure"], "measure")
        _add_to_vocab(meta["region"], "region")
        _add_to_vocab(meta["type"], "typ")
    return docs


def _looks_standard(df: pd.DataFrame) -> bool:
    needed = {"dataset","measure","value","region","period","typ"}
    return needed.issubset(set(df.columns))

CSV_SOURCES = {}
if os.path.isdir(CSV_DIR):
    for fn in os.listdir(CSV_DIR):
        if fn.lower().endswith(".csv"):
            CSV_SOURCES[os.path.splitext(fn)[0]] = os.path.join(CSV_DIR, fn)
elif os.path.isfile(CSV_DIR) and CSV_DIR.lower().endswith(".csv"):
    CSV_SOURCES[os.path.splitext(os.path.basename(CSV_DIR))[0]] = CSV_DIR
else:
    print(f"[UWAGA] {CSV_DIR} nie jest ani katalogiem z CSV, ani plikiem CSV.")

all_docs: List[Document] = []
for name, path in CSV_SOURCES.items():
    try:
        df_head = _read_csv_smart(path).head(1)
        if _looks_standard(df_head):
            docs = build_documents_from_standard_csv(name, path)
            all_docs.extend(docs)
        else:
            print(f"[UWAGA] Pomijam '{path}' — niestandardowy schemat kolumn.")
    except Exception as e:
        print(f"[BŁĄD] {path}: {e}")

print(f"Zbudowano dokumentów: {len(all_docs)}")

rebuild_region_taxonomy()
def build_region_match_index() -> None:
    """Indeksuje wszystkie unikatowe wartości 'region' z CSV (kraje, województwa, itp.)."""
    REGION_INDEX.clear()
    seen = set()
    for reg in FIELD_VOCAB["region"]:
        if not reg:
            continue
        canon = str(reg)
        if canon in seen:
            continue
        seen.add(canon)
        REGION_INDEX.append({
            "canon": canon,
            "norm": norm_text(canon),
            "roots": _roots_from_text(canon),
            "syll":  _syllset_from_text(canon),
            "chargrams": _chargrams_from_text(canon),
        })

def match_region_text(q: str, min_score: float = 0.58) -> Optional[str]:
    """
    Zwraca kanoniczną nazwę regionu (dokładnie jak w CSV) dla krajów i województw.
    Obsługuje odmiany dzięki 'sylabom' i n-gramom. Gdy brak dobrego trafienia — None.
    """
    if not REGION_INDEX:
        return None

    q_norm  = norm_text(q)
    q_roots = _roots_from_text(q_norm)
    q_syll  = _syllset_from_text(q_norm)
    q_ngr   = _chargrams_from_text(q_norm)

    for e in REGION_INDEX:
        pat = r"\b" + re.escape(e["norm"]).replace(r"\ ", r"[\s_-]+") + r"\b"
        if re.search(pat, q_norm):
            return e["canon"]

    def jacc(a: set[str], b: set[str]) -> float:
        if not a or not b: return 0.0
        i = len(a & b); u = len(a | b)
        return i / u if u else 0.0

    best_canon, best_score = None, 0.0
    for e in REGION_INDEX:
        s_syll = jacc(q_syll, e["syll"])
        s_root = jacc(q_roots, e["roots"])
        s_ngr  = jacc(q_ngr,  e["chargrams"])
        score  = 0.5 * s_syll + 0.35 * s_root + 0.15 * s_ngr
        if score > best_score:
            best_score, best_canon = score, e["canon"]

    return best_canon if best_score >= min_score else None
build_region_match_index()

#Chroma

from resources import vectorstore
vectorstore =   vectorstore

def _collection_count(vs) -> int:
    coll = getattr(vs, "_collection", None)
    try: return coll.count() if coll is not None else 0
    except Exception: return 0

if _collection_count(vectorstore) == 0 and all_docs:
    vectorstore.add_documents(all_docs)
    getattr(vectorstore, "persist", lambda: None)()

#retriever
bm25 = BM25Retriever.from_documents(all_docs)
bm25.k = 80

# MultiQuery + HyDE
def _base_mq_variants(q: str, n: int = MQ_BASE_VARIANTS) -> List[str]:
    qn = norm_text(q)
    base = {
        q.strip(),
        strip_acc(q),
        qn.replace("ile", "jaka jest liczba"),
        qn.replace("ile wynosi", "podaj wartość"),
        qn.replace("emerytura", "przeciętne świadczenie emerytalne"),
    }
    if "ubezpieczonych" in qn:
        base.update({
            "liczba ubezpieczonych ogółem",
            "liczba ubezpieczonych w krus ogółem",
            "ubezpieczeni w krus ogółem"
        })
    if "płatnik" in qn or "platnik" in qn:
        base.update({"liczba płatników składek ogółem"})
    if "macierzyńsk" in qn or "macierzynsk" in qn:
        base.update({"przeciętne świadczenie zasiłek macierzyński ogółem"})
    if "emerytur" in qn or "emeryturę" in qn or "emerytury" in qn:
        base.update({"przeciętne świadczenie emerytalne ogółem"})
    if FORCE_NATIONAL_IF_REGION_UNSPECIFIED and "wojew" not in qn and "region" not in qn:
        base.add(q.strip() + " ogółem")
    return list(base)[:n]

def _take_closest_vocab(target: Optional[str], pool: set[str], k: int = 20) -> List[str]:
    if not target or not pool: return []
    scored = []
    for cand in pool:
        scored.append((cand, _ratio(target, cand)))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [c for c, _ in scored[:k]]

def _safe_unique(xs: List[str], limit: int = 20) -> List[str]:
    seen, out = set(), []
    for x in xs:
        s = norm_text(x)
        if s and s not in seen:
            seen.add(s); out.append(x.strip())
        if len(out) >= limit: break
    return out

def _pick_or_default(val: Optional[str], pool: set[str], k: int = 3) -> List[str]:
    base = [val] if val else []
    rest = _take_closest_vocab(val, pool, k=k) if pool else []
    out = [x for x in base + rest if x]
    return _safe_unique(out, limit=max(1, k+1)) or ["-"]

def _llm_generate_query_expansions(question: str, measure_hint: Optional[str], typ_hint: Optional[str], dataset_hint: Optional[str], k_variants: int = MQ_LLM_VARIANTS) -> List[str]:
    if not (USE_LLM and llm_pllum and LLM_MQ_ENABLED):
        return []

    measure_cands = _take_closest_vocab(measure_hint, FIELD_VOCAB["measure"], k=20)
    typ_cands     = _take_closest_vocab(typ_hint,     FIELD_VOCAB["typ"],     k=20)
    dataset_cands = _take_closest_vocab(dataset_hint, FIELD_VOCAB["dataset"], k=20)

    sys_rules = (
        "Jesteś generatorem wariantów zapytań do wyszukiwarki danych KRUS.\n"
        "Zasady:\n"
        " Zwięzłe warianty PL (jedna linia = jedno zapytanie).\n"
        " Używaj TYLKO podanych etykiet measure/typ/dataset (jeśli podane).\n"
        " Zachowuj znaczenie pytania.\n"
        " Nie zgaduj, gdy brak kandydatów.\n"
    )
    m_line = " | ".join(measure_cands) if measure_cands else "(brak)"
    t_line = " | ".join(typ_cands) if typ_cands else "(brak)"
    d_line = " | ".join(dataset_cands) if dataset_cands else "(brak)"

    user_block = (
        f"Pytanie:\n{question}\n\n"
        f"Kandydaci 'measure': {m_line}\n"
        f"Kandydaci 'typ': {t_line}\n\n"
        f"Kandydaci 'dataset': {d_line}\n"
        f"Wygeneruj do {k_variants} wariantów. Zwróć TYLKO linie zapytań."
    )
    prompt = f"<s>[INST] <<SYS>>{sys_rules}<</SYS>>\n{user_block}\n[/INST]"
    try:
        out = llm_pllum.invoke(prompt)
        lines = [ln.strip(" -•\t") for ln in str(out).splitlines()]
        lines = [ln for ln in lines if ln and not ln.lower().startswith(("[źródła", "zrodla", "sources", "<", "["))]
        return _safe_unique(lines, limit=k_variants)
    except Exception:
        return []

def make_mq_prompts_llm(q: str) -> List[str]:
    parsed = parse_query_fields(q)
    base = _safe_unique(_base_mq_variants(q, n=MQ_BASE_VARIANTS))
    gen  = _llm_generate_query_expansions(
        question=q,
        measure_hint=parsed.get("measure"),
        typ_hint=parsed.get("typ"),
        dataset_hint=parsed.get("dataset"),
        k_variants=MQ_LLM_VARIANTS
    )
    return _safe_unique(base + gen, limit=MQ_BASE_VARIANTS + MQ_LLM_VARIANTS)

def make_hyde_texts(question: str,n_base: int = HYDE_VARIANTS_BASE,n_llm:  int = HYDE_VARIANTS_LLM) -> List[str]:
    parsed = parse_query_fields(question)
    m  = parsed.get("measure")
    t  = parsed.get("typ")
    ds = parsed.get("dataset")
    rg = parsed.get("region") or ("ogółem" if FORCE_NATIONAL_IF_REGION_UNSPECIFIED else "-")
    pr = parsed.get("period") or "-"

    cand_meas = _pick_or_default(m,  FIELD_VOCAB["measure"], k=3)
    cand_typ  = _pick_or_default(t,  FIELD_VOCAB["typ"],     k=3)
    cand_ds   = _pick_or_default(ds, FIELD_VOCAB["dataset"], k=3)

    base: List[str] = []
    for i in range(max(1, n_base)):
        base.append(" | ".join([
            f"dataset: {cand_ds[i % len(cand_ds)]}",
            f"measure: {cand_meas[i % len(cand_meas)]}",
            "value: <X>",
            f"region: {rg}",
            f"period: {pr}",
            f"typ: {cand_typ[i % len(cand_typ)]}",
        ]))

    gen: List[str] = []
    if USE_LLM and llm_pllum and n_llm > 0:
        sys = (
            "Jesteś pomocnikiem generującym hipotetyczne opisy danych KRUS. "
            "Napisz 1–2 zdania po polsku, bez żadnych liczb/%, bez konkretnych wartości. "
            "Jeśli pojawia się liczba, zastąp ją tokenem <X>."
        )
        user = (
            f"Pytanie: {question}\n"
            f"Wskazówki: dataset={ds or '-'}, measure={m or '-'}, typ={t or '-'}, region={rg}, period={pr}.\n"
            f"Wypisz {n_llm} wariantów, każdy w osobnej linii."
        )
        prompt = f"<s>[INST] <<SYS>>{sys}<</SYS>>\n{user}\n[/INST]"
        try:
            raw = str(llm_pllum.invoke(prompt))
            lines = [ln.strip(" -•\t") for ln in raw.splitlines() if ln.strip()]
            gen = lines[:n_llm]
        except Exception:
            gen = []

    texts = base + gen
    if HYDE_STRIP_NUMBERS:
        texts = [_strip_numbers(t) for t in texts]
    return _safe_unique(texts, limit=n_base + n_llm)

# RRF
def rrf_merge_with_support(runs: List[List[Document]], k: int = 120, k_rrf: int = 60) -> List[Document]:
    scores, support, best = {}, {}, {}
    pos_maps: List[Dict[Tuple[str,int], int]] = []
    for run in runs:
        m = {}
        for i, d in enumerate(run):
            key = (d.metadata.get("source_file","?"), d.metadata.get("row_index",-1))
            m[key] = i
        pos_maps.append(m)
    keys = set().union(*[set(m.keys()) for m in pos_maps])
    for key in keys:
        rrf = 0.0; sup = 0
        for m in pos_maps:
            if key in m:
                rrf += 1.0 / (k_rrf + m[key] + 1)
                sup += 1
        scores[key]  = rrf
        support[key] = sup
    for run in runs:
        for d in run:
            key = (d.metadata.get("source_file","?"), d.metadata.get("row_index",-1))
            if key not in best: best[key] = d
    merged = sorted(best.values(),
                    key=lambda d: scores[(d.metadata.get("source_file","?"), d.metadata.get("row_index",-1))],
                    reverse=True)
    for d in merged:
        key = (d.metadata.get("source_file","?"), d.metadata.get("row_index",-1))
        d.metadata["rrf_support"] = support[key]
        d.metadata["rrf_score"]   = scores[key]
    return merged[:k]

def rrf_merge_with_support_weighted(runs: List[List[Document]],labels: List[str],weights: Dict[str, float],k: int = 120,k_rrf: int = 60) -> List[Document]:
    scores, support_w, best = {}, {}, {}
    pos_maps: List[Dict[Tuple[str,int], int]] = []

    for run in runs:
        m = {}
        for i, d in enumerate(run):
            key = (d.metadata.get("source_file","?"), d.metadata.get("row_index",-1))
            m[key] = i
        pos_maps.append(m)

    keys = set().union(*[set(m.keys()) for m in pos_maps])
    for key in keys:
        s = 0.0; supw = 0.0
        for m, lab in zip(pos_maps, labels):
            if key in m:
                w = float(weights.get(lab, 1.0))
                s   += w * (1.0 / (k_rrf + m[key] + 1))
                supw += w
        scores[key]    = s
        support_w[key] = supw

    for run in runs:
        for d in run:
            key = (d.metadata.get("source_file","?"), d.metadata.get("row_index",-1))
            if key not in best:
                best[key] = d

    merged = sorted(best.values(),key=lambda d: scores[(d.metadata.get("source_file","?"), d.metadata.get("row_index",-1))],reverse=True)
    for d in merged:
        key = (d.metadata.get("source_file","?"), d.metadata.get("row_index",-1))
        d.metadata["rrf_score"]   = scores[key]
        d.metadata["rrf_support"] = support_w[key]  
    return merged[:k]


# Dense search helpers
def dense_search(query: str, k: int = 80) -> List[Document]:
    retr = vectorstore.as_retriever(search_kwargs={"k": k})
    return retr.invoke(query)

def dense_search_on(texts: List[str], k: int = 40) -> List[Document]:
    retr = vectorstore.as_retriever(search_kwargs={"k": k})
    out = []
    for t in texts:
        out.extend(retr.invoke(t))
    seen = set()
    uniq = []
    for d in out:
        key = (d.metadata.get("source_file","?"), d.metadata.get("row_index",-1))
        if key in seen: continue
        seen.add(key); uniq.append(d)
    return uniq

# Fuzzy 
try:
    from rapidfuzz import fuzz
    def _ratio(a,b): return fuzz.token_set_ratio(a,b) / 100.0
except Exception:
    import difflib
    def _ratio(a,b): return difflib.SequenceMatcher(None, a, b).ratio()

def best_match(token: str, candidates: set[str], min_ratio: float = 0.62) -> Optional[str]:
    if not token or not candidates: return None
    tn = norm_text(token)
    best, score = None, 0.0
    for c in candidates:
        r = _ratio(tn, c)
        if r > score:
            best, score = c, r
    return best if score >= min_ratio else None

def parse_query_fields(q: str) -> Dict[str, Optional[str]]:
    qn = norm_text(q)

    m = re.search(r"(20\d{2}\s*[-_/ ]?\s*q[1-4])", qn)
    period = norm_period(m.group(1)) if m else None
    if period is None:
        period = _extract_pl_quarter_period(qn)

    if period is None:
        years = set()
        years.update(re.findall(r"\b(20\d{2})\b", qn))
        years.update(re.findall(r"\b(20\d{2})\s*r\.?\b", qn))  
        years.update(re.findall(r"\b(20\d{2})r\.?\b", qn))    
        if years:
            period = str(sorted({int(y) for y in years})[-1])

    dataset = best_match(qn, FIELD_VOCAB["dataset"])
    measure = best_match(qn, FIELD_VOCAB["measure"])
    typ     = best_match(qn, FIELD_VOCAB["typ"])
    region  = match_region_text(q)

    if typ and measure and norm_text(typ) == norm_text(measure):
        typ = None

    return {"dataset": dataset, "measure": measure, "region": region, "typ": typ, "period": period}

# Reranker
from resources import reranker_ce
reranker_ce = reranker_ce

def _ce_text(d: Document, n: int = 800) -> str:
    s = d.page_content or ""
    return s if len(s) <= n else s[:n]

def _to_float_score(sc) -> float:
    try:
        import numpy as np
        if hasattr(sc, "shape"):
            return float(np.squeeze(sc))
    except Exception:
        pass
    if isinstance(sc, (list, tuple)):
        return 0.0 if not sc else _to_float_score(sc[0])
    try:
        return float(sc)
    except Exception:
        return 0.0
from resources import emb
emb = emb
def _fallback_dense_similarity(query: str, docs: List[Document]) -> List[Tuple[Document,float]]:
    qv = torch.tensor(emb.embed_query(query))
    dvs = torch.tensor([emb.embed_documents([d.page_content])[0] for d in docs])
    sims = torch.nn.functional.cosine_similarity(qv[None, :], dvs, dim=1).tolist()
    ranked = sorted([(d, s) for d, s in zip(docs, sims)], key=lambda x: x[1], reverse=True)
    for d, s in ranked:
        try: d.metadata["ce_score"] = float(s)
        except Exception: pass
    return ranked

def rerank_with_scores(query: str, docs: List[Document]) -> List[Tuple[Document, float]]:
    if reranker_ce is None or not docs:
        for d in docs:
            try: d.metadata["ce_score"] = 0.0
            except Exception: pass
        return [(d, 0.0) for d in docs]

    pairs = [(query, _ce_text(d)) for d in docs]
    scores = reranker_ce.predict(pairs)
    out = []
    for d, sc in zip(docs, scores):
        scf = _to_float_score(sc)
        try: d.metadata["ce_score"] = scf
        except Exception: pass
        out.append((d, scf))

    ranked = sorted(out, key=lambda x: x[1], reverse=True)

    vals = [s for _, s in ranked]
    if len(vals) >= 3 and (max(vals) - min(vals) < 1e-6):
        ranked = _fallback_dense_similarity(query, docs)

    return ranked
# sortowanie
def period_key(p: Optional[str]) -> Tuple[int,int,int]:
    if not p or str(p).strip()=="": return (0,0,0)
    s = str(p).lower().strip()
    m = re.match(r"^(20\d{2})[-_/ ]?q([1-4])$", s)
    if m: return (int(m.group(1)), int(m.group(2)), 0)
    m = re.match(r"^q([1-4])[-_/ ]?(20\d{2})$", s)
    if m: return (int(m.group(2)), int(m.group(1)), 0)
    m = re.match(r"^(20\d{2})[-_/](\d{1,2})[-_/](\d{1,2})$", s)
    if m:
        y, mo, d = map(int, m.groups())
        daynum = (mo-1)*31 + d
        q = (mo-1)//3 + 1
        return (y, q, daynum)
    m = re.match(r"^(20\d{2})$", s)
    if m: return (int(m.group(1)), 0, 0)
    return (0,0,0)

# Klastrowanie 
def _key4(d: Document) -> Tuple[str,str,str,str]:
    m = d.metadata or {}
    return (
        norm_text(m.get("dataset") or ""),
        norm_text(m.get("measure") or ""),
        norm_text(m.get("type") or ""),
        norm_text(m.get("region") or ""),
    )
def _valid_value(d: Document) -> bool:
    v = d.metadata.get("value", None)
    try:
        return (v is not None) and (not (isinstance(v,float) and (pd.isna(v))))
    except Exception:
        return v is not None

def pick_latest_per_cluster(
    docs_scored: List[Tuple[Document, float]],
    k_clusters: int = 6,
    prefer_typ: Optional[str] = None,
    prefer_national_first: bool = False
) -> List[Document]:
    bykey: Dict[Tuple[str,str,str,str], List[Tuple[Document,float,int]]] = {}
    for d, s in docs_scored:
        sup = int(d.metadata.get("rrf_support", 1))
        bykey.setdefault(_key4(d), []).append((d, s, sup))

    best_per: List[Tuple[Document,float,int]] = []
    for key, items in bykey.items():
        items.sort(key=lambda x: (x[2], x[1], period_key(x[0].metadata.get("okres"))), reverse=True)
        items_valid = [it for it in items if _valid_value(it[0])]
        chosen = (sorted(items_valid, key=lambda x: period_key(x[0].metadata.get("okres")), reverse=True)[0]
                  if items_valid else items[0])
        best_per.append(chosen)

    def _prio(meta: Dict[str, Any]) -> Tuple[int, int]:
        region_prio = 1 if (prefer_national_first and _is_national(meta.get("region"))) else 0
        typ_prio = 1 if (prefer_typ and _soft_match(prefer_typ, meta.get("type"))) else 0
        return (region_prio, typ_prio)
    best_per.sort(
        key=lambda x: (
            _prio(x[0].metadata),
            x[2],                                  # rrf_support
            x[1],                                  # score = CE
            period_key(x[0].metadata.get("okres")) # świeżość
        ),
        reverse=True
    )
    return [d for d,_,_ in best_per[:k_clusters]]

#utils
DATASET_PENALTIES: List[Tuple[re.Pattern, float]] = [
    (re.compile(r"jednoczesnym|jednoczesne|zus", re.I), 0.35),
    (re.compile(r"fundusz\s+składkowy", re.I), 0.30),
    (re.compile(r"transferowan", re.I), 0.40),
    (re.compile(r"ue|efta|wielk(?:a|iej)\s+bryt", re.I), 0.40),
]

_FOREIGN_PAT = re.compile(r"\b(ue|efta|wielk(?:a|iej)\s*bryt|dwustronn|transferowan)\b", re.I)
def _dataset_is_foreign(name: Optional[str]) -> bool:
    return bool(_FOREIGN_PAT.search(str(name or "")))
def _query_mentions_foreign(q: str) -> bool:
    return bool(_FOREIGN_PAT.search(norm_text(q)))

def _soft_match(a: Optional[str], b: Optional[str], thr: float = 0.72) -> bool:
    if not a:
        return True
    if not b:
        return False
    return _ratio(norm_text(a), norm_text(b)) >= thr

def derive_preferences(query: str, parsed: Dict[str, Optional[str]]) -> Dict[str, Any]:
    qn = norm_text(query)
    prefer_typ, force_typ = None, False
    if "emerytaln" in qn or "emerytur" in qn or "emerytura" in qn:
        prefer_typ = "emerytury"
        force_typ = not ("rent" in qn or "renty" in qn or "rentę" in qn or "rentowe" in qn)

    is_country_query = _is_country_name(parsed.get("region"))
    force_national = FORCE_NATIONAL_IF_REGION_UNSPECIFIED and (parsed.get("region") is None) and (not is_country_query)

    return {
        "prefer_typ": prefer_typ,
        "force_typ": force_typ,
        "force_national": force_national,
        "is_country_query": is_country_query
    }

def _adjusted_rank_score(d: Document, base_ce: float, prefs: Dict[str, Any], parsed: Dict[str, Optional[str]], query: str) -> float:
    score = float(base_ce or 0.0)
    name  = str((d.metadata or {}).get("dataset") or "")
    for pat, pen in DATASET_PENALTIES:
        if pat.search(name):
            score *= (1.0 - pen)
    if prefs.get("force_national") and not _is_national((d.metadata or {}).get("region")):
        score *= 0.62
    qn = norm_text(query)
    if "krus" in qn and (" w krus" in qn or "wkrus" in qn):
        if re.search(r"\bw\s+krus\b", norm_text(name)):
            score *= 1.1
    return score

def _choose_best_doc(docs: List[Document], query: str, parsed: Dict[str, Optional[str]], prefs: Dict[str, Any]) -> Optional[Document]:
    if not docs:
        return None
    pool = list(docs)
    if prefs.get("force_national") and not prefs.get("is_country_query"):
        nat = [d for d in pool if _is_national((d.metadata or {}).get("region"))]
        if nat:
            pool = nat

    def _key(d: Document):
        ce = float((d.metadata or {}).get("ce_score") or 0.0)
        meas = 1.0 if _soft_match(parsed.get("measure"), (d.metadata or {}).get("measure")) else 0.0
        typm = 1.0 if _soft_match(parsed.get("typ"),     (d.metadata or {}).get("type"))    else 0.0
        return (ce, meas, typm, period_key((d.metadata or {}).get("okres")))

    return max(pool, key=_key) if pool else None

# SMART RETRIEVE 

def retrieve(query: str, k_final: int = 24) -> List[Document]:
    global LAST_DEBUG

    parsed = parse_query_fields(query)
    prefs = derive_preferences(query, parsed)

    debug_info: Dict[str, Any] = {"query": query, "parsed": parsed, "prefs": prefs}

    mq_variants = make_mq_prompts_llm(query)
    d1 = dense_search_on(mq_variants, k=60)

    d2 = []
    if USE_HYDE:
        hyde_texts = make_hyde_texts(query, n_base=HYDE_VARIANTS_BASE, n_llm=HYDE_VARIANTS_LLM)
        d2 = dense_search_on(hyde_texts, k=60)

    d0 = dense_search(query, k=100)
    bm25.k = 100
    d3 = bm25.invoke(query)

    debug_info["hits"] = {"dense_q": len(d0), "dense_mq": len(d1), "hyde": len(d2), "bm25": len(d3)}

    fused = rrf_merge_with_support_weighted(
        [d0, d1, d2, d3],
        labels=["dense_q", "dense_mq", "hyde", "bm25"],
        weights=RRF_WEIGHTS,
        k=160,
        k_rrf=60
    )
    docs_scored = rerank_with_scores(query, fused) 
    if PREFER_TYP_FROM_QUERY and prefs.get("force_typ") and prefs.get("prefer_typ"):
        parsed["typ"] = prefs["prefer_typ"]

    cand_for_threshold = docs_scored

    if prefs.get("force_national"):
        nat_only = [(d, s) for (d, s) in cand_for_threshold if _is_national((d.metadata or {}).get("region"))]
        if nat_only:
            cand_for_threshold = nat_only

    if not _query_mentions_foreign(query):
        cand_for_threshold = [
            (d, s) for (d, s) in cand_for_threshold
            if not _dataset_is_foreign((d.metadata or {}).get("dataset"))
        ]

    cand_pairs: List[Tuple[Document, float]] = []
    for d, s in cand_for_threshold:
        ce = float(s or 0.0)
        d.metadata["ce_score"] = ce
        d.metadata["ce_adj"]   = _adjusted_rank_score(d, ce, prefs, parsed, query)  # tylko dla debug/analizy
        cand_pairs.append((d, ce))
    cand_pairs.sort(key=lambda x: x[1], reverse=True)

    cand_pairs = [(d, ce) for (d, ce) in cand_pairs if ce >= MIN_CE_FOR_CONTEXT]
    if not cand_pairs:
        debug_info["decision"] = "unanswerable_by_ce_ctx_threshold"
        LAST_DEBUG = debug_info
        return []

    req_period = parsed.get("period")
    if req_period:
        cand_p = [(d, ce) for (d, ce) in cand_pairs
                  if norm_period((d.metadata or {}).get("okres")) == req_period]
        if cand_p:
            cand_pairs = cand_p
        else:
            debug_info["decision"] = "no_data_for_requested_period"
            debug_info["requested_period"] = req_period
            LAST_DEBUG = debug_info
            return []

    if prefs.get("is_country_query") and parsed.get("region"):
        want_country = parsed["region"]
        cand_country = [(d, ce) for (d, ce) in cand_pairs
                        if _soft_match(want_country, (d.metadata or {}).get("region"))]
        if cand_country:
            cand_pairs = cand_country
        else:
            debug_info["decision"] = "no_data_for_requested_country"
            debug_info["requested_country"] = want_country
            LAST_DEBUG = debug_info
            return []

    if prefs.get("force_national") and not prefs.get("is_country_query"):
        nat_pairs = [(d, ce) for (d, ce) in cand_pairs if _is_national((d.metadata or {}).get("region"))]
        if nat_pairs:
            cand_pairs = nat_pairs

    docs_for_llm = pick_latest_per_cluster(
        cand_pairs,
        k_clusters=max(1, k_final),
        prefer_typ=prefs.get("prefer_typ"),
        prefer_national_first=prefs.get("force_national", False)
    )

    docs_for_llm.sort(key=lambda d: period_key(d.metadata.get("okres")), reverse=True)

    return docs_for_llm[:k_final]

# LLM 
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline
from resources import BASE_MODEL_ID
LLM = BASE_MODEL_ID 
def build_pllum_llm():
    from resources import gen_model
    llm_model = gen_model
    from resources import tokenizer
    tok = tokenizer

    gen_pipe = pipeline(
        "text-generation",
        model=llm_model,
        tokenizer=tok,
        max_new_tokens = 220,
        do_sample=False, 
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
        return_full_text=False,
    )
    return HuggingFacePipeline(pipeline=gen_pipe)

llm_pllum = build_pllum_llm() if USE_LLM else None

def _trim(s: str, n: int = 500) -> str:
    s = str(s)
    return s if len(s) <= n else s[:n].rstrip() + "…"

def _fmt_num(x, nd=3):
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x) if x is not None else "-"

def format_context_for_llm(docs: List[Document], max_docs: int = 6, max_snip_chars: int = 220) -> str:
    lines = []
    for i, d in enumerate(docs[:max_docs], 1):
        meta = d.metadata or {}
        header = (
            f"[{i}] dataset={meta.get('dataset','')} | okres={meta.get('okres') or '-'} | region={meta.get('region') or '-'} "
            f"| row={meta.get('row_index','-')} | type={meta.get('type') or '-'} | measure={meta.get('measure','')} | value={meta.get('value','-')}"
        )
        body = _trim(d.page_content, max_snip_chars)
        lines.append(header + "\n" + body)
    return "\n\n".join(lines)

# format odpowiedzi
def _pl_number(x: float) -> str:
    if x is None: return "-"
    if abs(x - int(x)) < 1e-6:
        s = f"{int(round(x)):,}".replace(",", " ")
    else:
        s = f"{x:,.2f}".replace(",", " ").replace(".", ",")
    return s

def _unit_from_measure(measure: str) -> str:
    m = norm_text(measure or "")
    if " zł" in (measure or "") or "zł" in m:
        return " zł"
    if "osób" in m or "liczba" in m or "osoby" in m:
        return ""
    return ""

def build_sources_block(docs: List[Document]) -> str:
    lines = ["[ŹRÓDŁA]"]
    for i, d in enumerate(docs, 1):
        m = d.metadata or {}
        lines.append(
            f"- #[{i}] dataset={m.get('dataset')} | measure={m.get('measure')}|"
            f"okres={m.get('okres') or '-'} | region={m.get('region') or '-'}  | typ={m.get('type') or '-'} | value={m.get('value')}| "
            f"ce={_fmt_num(m.get('ce_score'))} | ce_adj={_fmt_num(m.get('ce_adj'))} | rrf_sup={m.get('rrf_support','-')} | rrf={_fmt_num(m.get('rrf_score'),4)} | "
            f"source={m.get('source_file')} | row={m.get('row_index','-')}"
        )
    return "\n".join(lines)
# ASCII
def _to_str(x):
    return "-" if x is None else str(x)

def ascii_table_from_docs(docs, max_rows=6):
    cols = [
        ("wartość",  lambda i,m: _to_str(m.get("value"))),
        ("tabela",   lambda i,m: _to_str(m.get("dataset"))),
        ("miara",    lambda i,m: _to_str(m.get("measure"))),
        ("typ",      lambda i,m: _to_str(m.get("type") or "-")),
        ("okres",    lambda i,m: _to_str(m.get("okres") or "-")),
        ("region",   lambda i,m: _to_str(m.get("region") or "-"))
    ]

    rows = []
    for i, d in enumerate(docs[:max_rows], 1):
        m = d.metadata or {}
        rows.append([fmt(i, m) for _, fmt in cols])

    headers = [name for name, _ in cols]
    widths = [len(h) for h in headers]
    for r in rows:
        for j, cell in enumerate(r):
            widths[j] = max(widths[j], len(str(cell)))

    def hline(sep_left="-", sep_mid="-", sep_right="-", fill="-"):
        return sep_left + sep_mid.join(fill * (w + 2) for w in widths) + sep_right

    def fmt_row(cells):
        parts = []
        for j, cell in enumerate(cells):
            s = str(cell)
            if headers[j] in ("value", "row", "#"):
                s = s.rjust(widths[j])
            else:
                s = s.ljust(widths[j])
            parts.append(f" {s} ")
        return "|" + "|".join(parts) + "|"

    top = hline()
    head = fmt_row(headers)
    mid = hline(sep_left="-", sep_mid="-", sep_right="-", fill="-")
    body = "\n".join(fmt_row(r) for r in rows) if rows else fmt_row(["(brak danych)"] + [""]*(len(widths)-1))
    bot = hline()

    return "\n".join([top, head, mid, body, bot])

# Odpowiadanie
def answer(question: str, k_ctx: int = 8) -> str:
    docs = retrieve(question, k_final=max(12, k_ctx))
    if not docs:
        parsed = parse_query_fields(question)
        msgs = []
        if parsed.get("period"):
            msgs.append(f"dla okresu „{parsed['period']}”")
        if _is_country_name(parsed.get("region")):
            msgs.append(f"dla państwa „{parsed['region']}”")
        if msgs:
            return f"Brak danych {' i '.join(msgs)}.\n[ŹRÓDŁA]\n– brak"
        return "Brak danych pasujących do pytania.\n[ŹRÓDŁA]\n– brak"

    parsed = parse_query_fields(question)
    prefs  = derive_preferences(question, parsed)

    docs_sorted = sorted(docs, key=lambda d: period_key(d.metadata.get("okres")), reverse=True)

    best = _choose_best_doc(docs_sorted, question, parsed, prefs)
    if not best:
        return "Brak danych w dostarczonej dokumentacji.\n[ŹRÓDŁA]\n– brak"

    mv = best.metadata or {}
    val = mv.get("value")
    measure = mv.get("measure") or ""
    okres = mv.get("okres") or "-"
    region = mv.get("region") or "ogółem"
    unit = _unit_from_measure(measure)
    value_text = _pl_number(val) + unit

    docs_top = [best] + [d for d in docs_sorted if d is not best][:5]
    table_txt = ascii_table_from_docs(docs_top, max_rows=6)

    answer_txt = f"{value_text} ({okres}, {region})."
    return answer_txt + "\n\n" + table_txt + "\n\n"