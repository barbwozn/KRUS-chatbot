
import os
os.environ["LANGCHAIN_API_KEY"] = ""
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "KRUS-debug"

from huggingface_hub import login

login("") #trzeba dac klucz z hf


sorDEBUG = False
DEBUG = sorDEBUG
RETURN_STRING_WHEN_DEBUG_FALSE = True

# EMBEDDER_MODEL   = "intfloat/multilingual-e5-small"
# RERANKER_MODEL   = "radlab/polish-cross-encoder"
# CLASSIFIER_MODEL = "klasyfikator" 
# BASE_MODEL_ID    = "speakleash/Bielik-11B-v2.6-Instruct"
#"CYFRAGOVPL/PLLuM-12B-chat"

RERANK_THRESHOLD = 0.30
K_SIM   = 10
K_FINAL = 3

import os, torch, platform
os.environ["TOKENIZERS_PARALLELISM"] = "true"   
torch.backends.cuda.matmul.allow_tf32 = True    


print("PyTorch:", torch.__version__, "| CUDA available:", torch.cuda.is_available(), "| GPUs:", torch.cuda.device_count())
print("Platform:", platform.platform()) 


from typing import List
from langchain.docstore.document import Document


from resources import embedder, gen_model, tokenizer
embedder = embedder
from resources import db
db = db


K_SIM = globals().get("K_SIM", 12)
K_FINAL = globals().get("K_FINAL", 6)
RERANK_THRESHOLD = globals().get("RERANK_THRESHOLD", None) 

import os, re, shutil, unicodedata, asyncio
from typing import List, Optional, Dict, Any, Tuple, Callable
import numpy as np
import torch



from langchain.memory import ConversationBufferWindowMemory
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain_core.callbacks import CallbackManager
from langchain.callbacks.tracers.langchain import LangChainTracer
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document  

# Reranker + LLM

from transformers import pipeline

from pydantic import PrivateAttr



def strip_accents_lower(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s.lower().strip()


if "db" not in globals():
    raise RuntimeError("Brak globalnej bazy `db` (Chroma). Zainicjalizuj ją przed załadowaniem skryptu.")

model = gen_model

ROLE_CUT_RE = re.compile(
    r"(?i)"                              
    r"(###\s*(?:user|asystent|dokumenty|system)?\s*:?" 
    r"|(?:^|\s)(?:user|asystent|dokumenty|system)\s*:" 
    r"|<\s*(?:user|assistant|docs?|system)\s*>)"      
)

def cut_after_role_markers(s: str) -> str:
    if not s:
        return s
    m = ROLE_CUT_RE.search(s)
    return s[:m.start()].rstrip() if m else s


if globals().get("model", None) is None or globals().get("tokenizer", None) is None:
    raise RuntimeError("Załaduj wcześniej LLM do zmiennych `model` i `tokenizer`.")

# reranker z ONNX na CPU
from resources import cross_encoder_ustawa
cross_encoder = cross_encoder_ustawa


REF_RE_EXT = re.compile(
    r"(?:art\.?\s*(?P<art>[0-9]+[a-z]?))"
    r"(?:\s*(?:ust(?:\.|ęp)?|ustep)\s*(?P<ust>[0-9]+[a-z]?))?"
    r"(?:\s*(?:pkt\.?)\s*(?P<pkt>[0-9]+[a-z]?))?"
    r"(?:\s*(?:lit\.?)\s*(?P<lit>[a-z]))?",
    re.IGNORECASE
)

def parse_ref_ext(query: str) -> Optional[Dict[str, str]]:
    m = REF_RE_EXT.search(query or "")
    if not m:
        return None
    ref = {}
    if m.group("art"): ref["article"]   = m.group("art").lower()
    if m.group("ust"): ref["paragraph"] = m.group("ust").lower()
    if m.group("pkt"): ref["punkt"]     = m.group("pkt").lower()
    if m.group("lit"): ref["litera"]    = m.group("lit").lower()
    return ref if ref else None

#retriever + reranker
def retrieve_basic(query: str, k_sim: int = K_SIM, k_final: int = K_FINAL, rerank_threshold: float | None = RERANK_THRESHOLD) -> List[Document]:
    ref = parse_ref_ext(query)
    filter_dict = {}
    if ref:
        if "article" in ref:   filter_dict["article"]   = ref["article"]
        if "paragraph" in ref: filter_dict["paragraph"] = ref["paragraph"]
    if not filter_dict:
        filter_dict = None

    docs = db.similarity_search(query, k=k_sim, filter=filter_dict)
    if not docs:
        return []

    pairs = [(query, d.page_content) for d in docs]
    scores = cross_encoder.predict(pairs, batch_size=32) 
    scores = np.asarray(scores, dtype=float)

    scored_docs = [(d, s) for d, s in zip(docs, scores)]
    if rerank_threshold is not None:
        scored_docs = [(d, s) for d, s in scored_docs if s >= rerank_threshold]
        if not scored_docs:
            return []

    scored_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)[:k_final]
    out: List[Document] = []
    for d, s in scored_docs:
        md = dict(d.metadata or {})
        md["rerank_score"] = float(s)
        d.metadata = md
        out.append(d)
    return out

#memory + LLM
memory = ConversationBufferWindowMemory(
    k=3, memory_key="chat_history", return_messages=True, output_key="answer"
)

hf_pipe = pipeline(
    "text-generation",
    model=globals().get("model"),
    tokenizer=globals().get("tokenizer"),
    max_new_tokens=512,
    do_sample=True,
    temperature=0.35,
    top_p=0.95,
    top_k=50,
    repetition_penalty=1.0,
    pad_token_id=(getattr(globals().get("tokenizer"), "eos_token_id", None)),
    eos_token_id=(getattr(globals().get("tokenizer"), "eos_token_id", None)),
    return_full_text=False,
    use_cache=True,
    batch_size=1,  
    num_beams=1,
)
llm = HuggingFacePipeline(pipeline=hf_pipe)

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "### System:\n"
        "Jesteś ekspertem prawa ubezpieczeń społecznych rolników. "
        "Odpowiadasz WYŁĄCZNIE na podstawie Dokumentów poniżej. "
        "Twoim zadaniem jest odpowiedzenie na pytanie najlepiej jak możesz wyłącznie na podstawie Dokumentów. "
        "Nie zmyślaj informacji i jeśli w dokumentach brak odpowiedzi, poinformuj o tym.\n"
        "Formatowanie: nie używaj formatowania Markdown (**…**) ani __…__.\n"
        "Napisz maksymalnie 6-8 zdań lub 8 punktów."
        "### User:\n{question}\n\n"
        "### Dokumenty:\n{context}\n"
        "### Asystent:\n"
    )
)

tracer = LangChainTracer()
callback_manager = CallbackManager([tracer])

class FunctionRetriever(BaseRetriever):
    k_sim: int
    k_final: int
    rerank_threshold: Optional[float] = None
    _fn: Callable[..., List[Document]] = PrivateAttr()

    def __init__(self, fn: Callable[..., List[Document]], **data):
        super().__init__(**data)
        self._fn = fn

    def _get_relevant_documents(self, query: str) -> List[Document]:
        return self._fn(query, k_sim=self.k_sim, k_final=self.k_final, rerank_threshold=self.rerank_threshold)

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self._get_relevant_documents(query))

init_retriever = FunctionRetriever(fn=retrieve_basic, k_sim=K_SIM, k_final=K_FINAL, rerank_threshold=RERANK_THRESHOLD)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=init_retriever,
    memory=memory,
    combine_docs_chain_kwargs={"prompt": prompt},
    return_source_documents=True,
    output_key="answer",
    callback_manager=callback_manager
)

#formatowanie odpowiedzi
def _build_citations_block(docs: List[Document]) -> str:
    if not docs:
        return "Cytowane ustępy:\n(brak)\n"
    lines = ["Cytowane ustępy:"]
    for d in docs:
        md = d.metadata or {}
        rozdz = md.get("rozdzial", md.get("chapter"))
        art   = md.get("artykul",  md.get("article"))
        ust   = md.get("ust",      md.get("paragraph"))
        pid   = md.get("id", f"ch{rozdz}-art{art}-ust{ust}")
        lines.append(f"- [{pid}] Rozdz.{rozdz} Art.{art} Ust.{ust}")
    return "\n".join(lines) + "\n"

def format_docs_for_prompt(docs: List[Document]) -> str:
    blocks = []
    for d in docs:
        md = d.metadata or {}
        rozdz = md.get("rozdzial", md.get("chapter"))
        art   = md.get("artykul",  md.get("article"))
        ust   = md.get("ust",      md.get("paragraph"))
        pid   = md.get("id", f"ch{rozdz}-art{art}-ust{ust}")
        blocks.append(f"[{pid}]\n{d.page_content}")
    return "\n\n---\n\n".join(blocks)

def log_rerank_scores(docs: List[Document], header: str = "DEBUG rerank scores") -> None:
    if not DEBUG: return
    print(header)
    if not docs:
        print("(brak dokumentów)"); return
    for d in docs:
        md = d.metadata or {}
        pid = md.get("id") or f"ch{md.get('chapter')}-art{md.get('article')}-ust{md.get('paragraph')}"
        sc  = md.get("rerank_score")
        print(f"- {pid}: score={sc:.6f}" if isinstance(sc, (int, float)) else f"- {pid}: score=(brak)")


def strip_markdown_bold(text: str) -> str:
    if not text:
        return text
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"__(.*?)__", r"\1", text, flags=re.DOTALL)
    text = text.replace("**", "").replace("__", "")
    return text

_SENT_ENDERS = ".?!…"
_CLOSERS = "”»\")]’"

def _rstrip_u(s: str) -> str:
    return s.rstrip(" \t\r\n\u00A0")

_LIST_MARKER_ONLY_RE = re.compile(
    r"""^[\s\u00A0]*(
            [-*+•·—–]
          | (?:\(?\d+[a-z]?\)|\d+[a-z]?[.)])
          | (?:\(?[ivxlcdm]+\)|[ivxlcdm]+[.)])
          | (?:[a-z][.)])
        )[\s\u00A0]*$""",
    re.IGNORECASE | re.VERBOSE
)

def _strip_trailing_empty_list_item(s: str) -> tuple[str, bool]:
    if not s:
        return s, False
    s2 = _rstrip_u(s)
    if not s2:
        return s2, (s2 != s)
    lines = s2.splitlines()
    last = _rstrip_u(lines[-1])
    if _LIST_MARKER_ONLY_RE.match(last or ""):
        return _rstrip_u("\n".join(lines[:-1])), True
    return s2, False

def _ends_with_full_stop(s: str) -> bool:
    return re.search(rf"[{re.escape(_SENT_ENDERS)}][{re.escape(_CLOSERS)}]*[\s\u00A0]*$", s) is not None

_ABBR_SET = {"art","ust","pkt","lit","tj","tzw","np","itd","itp","m.in","prof","dr","nr","poz","cd","al","ul","pl","św","sw"}

def _last_token_before_dot_for_trim(buf: str) -> str:
    m = re.search(r"([A-Za-zÀ-ÖØ-öø-ÿŁŚŻŹĆŃÓÄÖÜĄĘłśżźćńóäöüąę]+)\.\s*$", buf)
    return (m.group(1).lower() if m else "")

def _is_abbreviation_dot(text: str, dot_pos: int) -> bool:
    prev = text[:dot_pos+1]
    tok = _last_token_before_dot_for_trim(prev)
    return tok in _ABBR_SET

def _find_last_safe_boundary(s: str) -> int | None:
    i = len(s) - 1
    while i >= 0 and (s[i].isspace() or s[i] in _CLOSERS or s[i] == "\u00A0"):
        i -= 1
    while i >= 0:
        ch = s[i]
        if ch in _SENT_ENDERS:
            if ch == "." and _is_abbreviation_dot(s[:i+1], i):
                i -= 1
                continue
            return i + 1
        i -= 1
    return None

def trim_incomplete_sentences(text: str) -> str:
    if not text:
        return text
    text = cut_after_role_markers(text)
    s = _rstrip_u(text)
    changed = True
    while changed:
        s, changed = _strip_trailing_empty_list_item(s)
    if s.endswith(":"):
        last_nl = s.rfind("\n")
        s = _rstrip_u(s[:last_nl]) if last_nl != -1 else ""
    if not s:
        return s
    if _ends_with_full_stop(s):
        return s
    cut = _find_last_safe_boundary(s)
    if cut is None:
        return s
    return _rstrip_u(s[:cut])

def _finalize_return(text: str, docs: List[Document], mode: str):
    debug = [{"id": (d.metadata or {}).get("id"),
              "score": (d.metadata or {}).get("rerank_score")} for d in (docs or [])]
    payload = {"answer": text, "source_documents": docs, "debug": {"mode": mode, "rerank": debug}}

    if DEBUG:
        print("\nODPOWIEDŹ\n", text,
              "\n\n *Mogę popełniać błędy, skonsultuj się z placówką KRUS w celu potwierdzenia informacji.*\n")
        return payload

    if RETURN_STRING_WHEN_DEBUG_FALSE:
        return text
    return payload

_SMALLTALK_RULES = [
    (r"^(czesc|cze|hej|heja|hejka|witam|siema|elo|halo|dzien dobry|dobry wieczor)\b",
     "Cześć! W czym mogę pomóc w sprawie KRUS/ustawy?"),
    (r"\b(dzieki|dziekuje|dzieki wielkie|dziekuje bardzo|thx|thanks)\b",
     "Nie ma sprawy! Jeśli chcesz, podaj kolejne pytanie."),
]
def smalltalk_reply(user_q: str) -> Optional[str]:
    qn = strip_accents_lower(user_q)
    for pat, resp in _SMALLTALK_RULES:
        if re.search(pat, qn, flags=re.IGNORECASE):
            return resp
    return None

#context judge
from resources import ctx_judge
ctx_judge = ctx_judge

CTX_YES_MIN = 1.00
CTX_NO_MIN  = 0.00
CTX_MARGIN  = 0.12

def judge_contextual(prev_q: str, curr_q: str) -> Tuple[Optional[bool], float, Dict[str, float]]:
    if not prev_q or not curr_q:
        return None, 0.0, {}
    seq = (
        f"Pytanie A: {prev_q}\n"
        f"Pytanie B: {curr_q}\n"
        f"Czy Pytanie B jest dopytaniem/uzupełnieniem/jest powiązane do Pytania A?"
    )
    out = ctx_judge(
        sequences=seq,
        candidate_labels=["TAK", "NIE"],
        hypothesis_template="To jest {}."
    )
    dist = {lbl.upper(): float(score) for lbl, score in zip(out["labels"], out["scores"])}
    yes, no = dist.get("TAK", 0.0), dist.get("NIE", 0.0)
    margin = abs(yes - no)
    if DEBUG:
        print(f"[CTX-JUDGE] dist={dist} margin={margin:.2f}")
    if yes >= CTX_YES_MIN and margin >= CTX_MARGIN:
        return True, yes, dist
    if no >= CTX_NO_MIN and margin >= CTX_MARGIN:
        return False, no, dist
    return None, max(yes, no), dist

#conversation state
class ConversationState:
    def __init__(self):
        self.last_article_num: Optional[str] = None
        self.last_paragraph_num: Optional[str] = None
        self.last_docs: List[Document] = []
        self.last_query: Optional[str] = None

STATE = ConversationState()

def _short_doc_label(d: Document) -> str:
    md = d.metadata or {}
    rozdz = md.get("rozdzial", md.get("chapter"))
    art   = md.get("artykul",  md.get("article"))
    ust   = md.get("ust",      md.get("paragraph"))
    pid   = md.get("id", f"ch{rozdz}-art{art}-ust{ust}")
    return f"{pid}"

def _update_state_from_docs(docs: List[Document], user_q: str):
    if not docs: return
    STATE.last_docs = docs[:]
    STATE.last_query = user_q
    md0 = docs[0].metadata or {}
    STATE.last_article_num   = (md0.get("artykul")  or md0.get("article"))
    STATE.last_paragraph_num = (md0.get("ust")      or md0.get("paragraph"))
    if DEBUG:
        print(f"[STATE] last_article={STATE.last_article_num} last_paragraph={STATE.last_paragraph_num}")

def rewrite_query(user_q: str) -> str:
    base = (STATE.last_query or "").strip()
    if not base: return user_q
    doc_labels = [_short_doc_label(d) for d in (STATE.last_docs or [])][:6]
    doc_part = f" (odnieś do: {', '.join(doc_labels)})" if doc_labels else ""
    return f"{user_q} — w kontekście poprzedniego pytania: '{base}'{doc_part}"

#router
def route_query(query: str):
    qn = strip_accents_lower(query)
    ref = parse_ref_ext(query)
    if DEBUG:
        print(f"[ROUTER] raw='{query}' | norm='{qn}' | ref={ref}")
    if ref:
        return "EXPLICIT_REF", {"ref": ref}
    return "GENERAL", {"query": query}

#restricted answer from docs
def _llm_generate(prompt_tmpl: PromptTemplate, **kwargs) -> str:
    text = prompt_tmpl.format(**kwargs)
    out = llm.invoke(text)
    return (out or "").strip()

def answer_from_docs(question: str, docs: List[Document]):
    ctx = format_docs_for_prompt(docs) if docs else "(brak dokumentów)"
    answer = _llm_generate(prompt, question=question, context=ctx)
    answer = trim_incomplete_sentences(strip_markdown_bold(answer)) or answer
    final_text = f"{_build_citations_block(docs)}\nOdpowiedź:\n{answer}"
    return _finalize_return(final_text, docs, mode="restricted")

#main function
def ask(q: str, reset_memory: bool=False):
    if reset_memory:
        qa_chain.memory.clear()

    st = smalltalk_reply(q)
    if st is not None:
        return _finalize_return(st, [], mode="smalltalk")

    action, payload = route_query(q)
    if action == "EXPLICIT_REF":
        ref = payload["ref"]
        flt = {}
        if "article" in ref:   flt["article"]   = ref["article"]
        if "paragraph" in ref: flt["paragraph"] = ref["paragraph"]
        docs = db.similarity_search("treść przepisu", k=5, filter=flt)
        if DEBUG:
            print("[ROUTER] EXPLICIT_REF → filter", flt, "→ docs:", [ (d.metadata or {}).get("id") for d in docs ])
        if docs:
            _update_state_from_docs(docs, q)
            content = strip_markdown_bold(docs[0].page_content or "")
            content = trim_incomplete_sentences(content) or content
            final_text = f"{_build_citations_block(docs)}\nOdpowiedź (pełny przepis):\n{content}"
            return _finalize_return(final_text, docs, mode="explicit")
        else:
            return _finalize_return("Nie znalazłem takiego artykułu/ustępu.", [], mode="explicit")

    verdict, conf, dist = judge_contextual(STATE.last_query or "", q)
    if DEBUG:
        print(f"[INTENT] judge_verdict={verdict} conf={conf:.2f}")

    mode = "follow_up" if verdict is True else "new_query"  

    if mode == "follow_up":
        rewritten = rewrite_query(q)
        if DEBUG: print(f"[REWRITE:follow_up] {rewritten}")

        docs_narrow: List[Document] = []
        if STATE.last_article_num:
            flt = {"article": STATE.last_article_num}
            docs_tmp = db.similarity_search(rewritten, k=K_SIM, filter=flt)
            if docs_tmp:
                pairs = [(rewritten, d.page_content) for d in docs_tmp]
                scores = cross_encoder.predict(pairs, batch_size=32)
                scored = sorted([(d, float(s)) for d, s in zip(docs_tmp, np.asarray(scores))],
                                key=lambda x: x[1], reverse=True)[:K_FINAL]
                for d, s in scored:
                    md = dict(d.metadata or {}); md["rerank_score"] = s; d.metadata = md
                    docs_narrow.append(d)

        if not docs_narrow and STATE.last_docs:
            STATE.last_query = q
            return answer_from_docs(rewritten, STATE.last_docs)

        if not docs_narrow:
            res = qa_chain.invoke({"question": rewritten})
            out_docs = res.get("source_documents", []) or []
            _update_state_from_docs(out_docs, q)
            log_rerank_scores(out_docs)
            citations_block = _build_citations_block(out_docs)
            raw_answer = (res.get("answer") or "").strip()
            raw_answer = trim_incomplete_sentences(strip_markdown_bold(raw_answer)) or raw_answer
            final_text = f"{citations_block}\nOdpowiedź:\n{raw_answer}"
            return _finalize_return(final_text, out_docs, mode="follow_up→global")

        _update_state_from_docs(docs_narrow, q)
        return answer_from_docs(rewritten, docs_narrow)

    res = qa_chain.invoke({"question": q})
    docs = res.get("source_documents", []) or []
    _update_state_from_docs(docs, q)
    log_rerank_scores(docs)

    citations_block = _build_citations_block(docs)
    raw_answer = (res.get("answer") or "").strip()
    raw_answer = trim_incomplete_sentences(strip_markdown_bold(raw_answer)) or raw_answer
    final_text = f"{citations_block}\nOdpowiedź:\n{raw_answer}"
    return _finalize_return(final_text, docs, mode="new_query")
