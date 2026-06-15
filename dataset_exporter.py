"""
Dataset Forge v1 - multi-model pipeline (router / generator / validator)

PIPELINE (per chunk):
  1. ROUTER   - small/fast model decides if a chunk is "usable" and which
                task type(s) fit it (summary / qa / explanation / application).
                Filters junk (references, headers, fragments) before any
                expensive generation happens.
  2. GENERATOR - alternates between your two "heavy" models per chunk.
                Produces ONE {"text": "User: ...\nAssistant: ..."} pair per
                task type, using the FULL chunk as context (no sentence-level
                pairing, real instructions, complete answers).
  3. VALIDATOR - the OTHER heavy model cross-checks the pair for grounding
                (no hallucinated facts) and completeness. Failing pairs are
                discarded.

Also: smart paragraph-merging chunker (de-hyphenates PDF wraps, drops
headers/page numbers/references), rolling near-duplicate filter, multi-file
+ pasted-text input, live log/preview, pause/stop, dynamic Ollama model list.
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk, scrolledtext
import json
import os
import re
import sys
import time
import queue
import difflib
import threading
import subprocess
import hashlib

# Attempt to import optional packages
try:
    import PyPDF2
    import docx
    import requests
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "PyPDF2", "python-docx", "requests"])
    import PyPDF2
    import docx
    import requests

OLLAMA_BASE = "http://localhost:11434"
OLLAMA_GENERATE = f"{OLLAMA_BASE}/api/generate"
OLLAMA_TAGS = f"{OLLAMA_BASE}/api/tags"


# ----------------------------------------------------------------------------
# 0. Quality-of-life helpers (safe keepers for missing libraries)
# ----------------------------------------------------------------------------

_LRU_AVAILABLE = False
SIMHASH_AVAILABLE = False

try:
    from functools import lru_cache
    _LRU_AVAILABLE = True
except ImportError:
    pass

# Optional: Install simhash for near-duplicate detection (semantic level)
try:
    from simhash import Simhash
    SIMHASH_AVAILABLE = True
except ImportError:
    pass

# Attempt to install simhash if user wants semantic dedup but doesn't have it
def _ensure_simhash():
    global SIMHASH_AVAILABLE, Simhash
    if not SIMHASH_AVAILABLE:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "simhash"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            from simhash import Simhash
            SIMHASH_AVAILABLE = True
        except Exception:
            pass


# ----------------------------------------------------------------------------
# 1. Text extraction (unchanged + optional OCR fallback)
# ----------------------------------------------------------------------------

def extract_text(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".txt":
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    elif ext == ".pdf":
        with open(filepath, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            pages = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(pages)
    elif ext == ".docx":
        d = docx.Document(filepath)
        return "\n\n".join(p.text for p in d.paragraphs)
    else:
        raise ValueError(f"Unsupported format: {ext}")


# ----------------------------------------------------------------------------
# 2. Smart chunking: merge paragraphs into coherent ~target-word blocks,
#    de-hyphenate PDF line wraps, drop noise (headers/refs/page numbers)
# ----------------------------------------------------------------------------

_NOISE_PATTERNS = [
    re.compile(r"^\s*(page\s*\d+|\d+)\s*$", re.I),
    re.compile(r"^\s*(references|bibliography|acknowledg(e)?ments?|appendix)\s*$", re.I),
    re.compile(r"^\s*(table|figure)\s*\d+", re.I),
]


def _is_noise(paragraph):
    p = paragraph.strip()
    # Lowered from 30 to 15 to allow shorter, messy, or conversational fragments
    if len(p) < 15:
        return True
    # Raised from 0.3 to 0.45 to preserve data-heavy tables or imperfect OCR texts
    digit_ratio = sum(c.isdigit() for c in p) / max(len(p), 1)
    if digit_ratio > 0.45:
        return True
    for pat in _NOISE_PATTERNS:
        if pat.match(p):
            return True
    return False


def smart_chunk(text, target_words=250, max_words=450, min_chunk_words=15):
    """
    Split *text* into coherent multi-paragraph chunks.
    Paragraphs are joined with double newlines so downstream tasks can still
    see the paragraph boundaries (useful for citation / multi-section work).
    """
    if not text or not text.strip():
        return []

    # De-hyphenate words split across line wraps: "indus-\ntrial" -> "industrial"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # Also handle em-dash and en-dash line breaks (OCR / PDF artefact)
    text = re.sub(r"(\w)\u2013?\n(\w)", r"\1\2", text)
    text = re.sub(r"(\w)\u2014\n(\w)", r"\1\2", text)
    # Collapse single newlines (line wraps inside a paragraph) to spaces,
    # but keep blank-line (\n\n) paragraph separators intact.
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

    raw_paras = re.split(r"\n\s*\n", text)
    paras = []
    for p in raw_paras:
        p = re.sub(r"\s+", " ", p).strip()
        if p and not _is_noise(p):
            paras.append(p)

    chunks = []
    current, current_words = [], 0
    for p in paras:
        wc = len(p.split())
        if current and current_words + wc > max_words:
            chunks.append("\n\n".join(current))  # Keep paragraph separators intact
            current, current_words = [p], wc
        else:
            current.append(p)
            current_words += wc
            if current_words >= target_words:
                chunks.append("\n\n".join(current))
                current, current_words = [], 0
    if current:
        chunks.append("\n\n".join(current))

    # Lowered minimum chunk size from 40 to 15 to catch brief/ambiguous edge cases
    return [c for c in chunks if len(c.split()) >= min_chunk_words]


# ----------------------------------------------------------------------------
# 3. Ollama helpers (unchanged signature, + retry & timeout back-off)
# ----------------------------------------------------------------------------

def list_ollama_models():
    try:
        r = requests.get(OLLAMA_TAGS, timeout=5)
        r.raise_for_status()
        return sorted(m["name"] for m in r.json().get("models", []))
    except Exception:
        return []


def parse_model_size(name):
    m = re.search(r"(\d+(?:\.\d+)?)\s*b\b", name.lower())
    return float(m.group(1)) if m else 0.0


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"^```[a-zA-Z0-9]*\n?|```\s*$")


def clean_json_response(raw):
    """Strip <think>...</think> blocks and markdown code fences that some
    reasoning models (e.g. qwen3) emit even when format=json is requested."""
    if not raw:
        return raw
    raw = _THINK_RE.sub("", raw).strip()
    raw = _FENCE_RE.sub("", raw).strip()
    return raw


def ollama_generate(model, prompt, system, temperature=0.2, top_p=0.9, timeout=300, retries=2):
    """
    Generate text via Ollama with automatic retry on transient failures.
    Retries use a small back-off to avoid hammering a cold Ollama instance.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "format": "json",
        "think": False,
        "options": {"temperature": temperature, "top_p": top_p},
    }
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            r = requests.post(OLLAMA_GENERATE, json=payload, timeout=timeout)
            r.raise_for_status()
            raw = r.json().get("response", "")
            return clean_json_response(raw)
        except requests.exceptions.HTTPError as exc:
            last_err = exc
            # Older Ollama versions may reject the "think" field - retry without it.
            payload.pop("think", None)
            try:
                r = requests.post(OLLAMA_GENERATE, json=payload, timeout=timeout)
                r.raise_for_status()
                raw = r.json().get("response", "")
                return clean_json_response(raw)
            except Exception:
                pass
        except requests.exceptions.ConnectionError as exc:
            last_err = exc
            time.sleep(attempt * 2)
            continue
        except Exception as exc:
            last_err = exc
            time.sleep(attempt)
    raise last_err


# ----------------------------------------------------------------------------
# 4. Prompt templates (TASK_INSTRUCTIONS expanded with SOTA techniques)
# ----------------------------------------------------------------------------

ROUTER_SYSTEM_TMPL = """You are a dataset-curation assistant building a high-quality instruction-tuning \
dataset about {domain}.

Given a SOURCE TEXT chunk, decide:
1. "usable": true if the chunk contains enough self-contained, substantive information about {domain} \
to build a good training example. It must NOT be a reference list, table of contents, page header/footer, \
caption, or meaningless fragment. Otherwise false.
2. "difficulty": estimate the complexity of the chunk on a 1-3 scale:
   - 1 = basic / accessible (introductory concepts, definitions)
   - 2 = intermediate (technical details, trade-offs, implementation)
   - 3 = advanced (complex reasoning, multi-step analysis, edge cases)
3. "tasks": pick 1 to {max_tasks} task type(s) from this list that best fit the content: {task_list}

Respond with ONLY valid JSON, no extra text, in exactly this shape:
{{"usable": true, "difficulty": 2, "tasks": ["summary"]}}"""

TASK_INSTRUCTIONS = {
    "summary": (
        "Ask for a quick overview. Write the user prompt in a casual, slightly messy, or brief way "
        "(e.g., 'summarize fast plz' or 'what are the main takeaways here?'). The assistant response "
        "MUST be a pure, plain-text paragraph with absolutely NO markdown, bolding, lists, or bullets."
    ),
    "qa": (
        "Formulate a direct question about a specific fact. Alternate between a clean question and a messy, "
        "conversational user prompt (e.g., typos or conversational filler). Provide a short, direct plain-text answer."
    ),
    "explanation": (
        "Identify a core technical concept or term. Ask a user question explaining it. "
        "The assistant response must be highly structured, using markdown sections with clear headers to break down the concept."
    ),
    "application": (
        "Ask how the text applies to real-world workflows. The assistant response should provide a practical example "
        "written as a clean, multi-sentence plain-text paragraph without markdown formatting."
    ),
    "compare": (
        "Identify two distinct items, materials, or methods mentioned. Formulate a user prompt asking to evaluate "
        "the TRADEOFFS between them under specific constraints (e.g., 'Why choose A over B when time/cost is tight?'). "
        "The prompt can be informal or imperfect. The response must clearly outline the structural tradeoffs or pros/cons."
    ),
    "critique": (
        "Formulate a hard reasoning task where the user must RANK options, JUSTIFY a complex choice, or CHOOSE THE BEST option "
        "based strictly on the text criteria (e.g., 'Based on these limitations, pick the best layout path and justify it'). "
        "The assistant must provide a highly rigorous, logical evaluation choosing or ranking the items cleanly."
    ),
    "classify": (
        "Create a simple instruction asking to categorize or tag a process, tone, or theme from the text. "
        "The assistant response must be highly standard and brief, starting directly with the category name followed by a plain-text reason."
    ),
    "edge_case": (
        "Craft a deliberately ambiguous, incomplete, or contradictory user prompt (e.g., 'this text seems to contradict itself, "
        "explain why' or 'there is missing info, point it out'). The assistant must detect the issue, acknowledge the "
        "limitation, and provide the best possible answer despite the imperfection in the source."
    ),
    "step_by_step": (
        "Formulate a user request asking for a step-by-step procedure or sequential breakdown derived from the text. "
        "The assistant response must use a numbered list format (1., 2., 3.) with clear, self-contained steps."
    ),
    "creative_rewrite": (
        "Ask the assistant to re-express the same information in a totally different register or format "
        "(e.g., 'rewrite this as a brief email to a client', 'turn this into a checklist', 'make this sound like a tweet thread'). "
        "The user prompt may be brief or slightly messy. The assistant must faithfully preserve all factual content while adopting the requested form."
    ),
}

GENERATOR_SYSTEM_TMPL = """You are an expert dataset generator creating instruction-tuning examples about {domain}.

PRIME DIRECTIVE ON USER PROMPTS: Real users write imperfectly. Dynamically mix clean instructions with messy, unorganized, colloquial, or grammatically loose user entries (e.g., using abbreviations, fragmented phrasing, or casual demand styles).

STRICT ASSISTANT STYLE RULES:
1. NEVER use repetitive, robotic framing phrases. DO NOT start assistant responses with 'Based on the text...', 'The passage discusses...', 'According to the provided excerpt...', 'The author states...', 'This excerpt describes...', or 'This passage covers...'. Dive completely and directly into the answer.
2. Enforce the specific layout requested in the task instructions (some tasks demand rich markdown, while others MANDATE pure plain-text paragraphs with zero markdown syntax).
3. Vary complexity: alternate between direct fact retrieval and hard, analytical reasoning tasks (tradeoffs, constraints, rankings).
4. If the source text is ambiguous, incomplete, or contradictory, acknowledge it explicitly rather than fabricating information.

TASK: {task_instruction}

Output format rules:
- "User" must be a natural, diverse instruction (often informal, messy, or conversational).
- "Assistant" must be a complete, self-contained response following the formatting layouts specified.
- Respond with ONLY valid JSON in exactly this shape:
{{"text": "User: <instruction>\\nAssistant: <full response>"}}"""

# Rubric-based validator (scoring approach borrowed from Prometheus / HealthBench)
VALIDATOR_SYSTEM_TMPL = """You are a fact-checking auditor for a training dataset about {domain}.

You are given SOURCE TEXT and a candidate User/Assistant training pair generated from it.

Paraphrasing, summarizing, reordering, and rewording the SOURCE TEXT in different language is GOOD \
and expected - never reject for that alone.

Evaluate the candidate pair against these rubric criteria (score each 0-2):
- grounded: 2 = all facts traceable to source, 1 = minor paraphrase OK, 0 = hallucinated or unsupported claims
- complete: 2 = fully answers the user instruction, 1 = partial answer, 0 = incomplete or off-topic
- style_adherence: 2 = follows requested format perfectly, 1 = minor deviation, 0 = wrong format

Mark "grounded": false ONLY if the Assistant's response:
- States a specific fact, number, name, date, or claim that does NOT appear in and is not a \
direct restatement/implication of SOURCE TEXT, or
- Is incomplete, cut off, or not a real answer to the User instruction.

If the response is a reasonable paraphrase or summary of the SOURCE TEXT with no invented \
specifics, mark "grounded": true.

Respond with ONLY valid JSON, in exactly this shape:
{{"grounded": true, "score": 2, "reason": "short reason"}}"""


# ----------------------------------------------------------------------------
# 5. Semantic deduplication helpers (optional simhash, fallback to difflib)
# ----------------------------------------------------------------------------

def _get_tokens(text):
    """Simple tokenisation for simhash deduplication."""
    return re.findall(r"\b\w+\b", text.lower())

def _tokens_to_features(tokens):
    """Build a frequency dict from tokens for simhash input."""
    features = {}
    for t in tokens:
        features[t] = features.get(t, 0) + 1
    return features

def semantic_near_duplicate(text_a, text_b, threshold=3):
    """
    Near-duplicate detection using simhash Hamming distance if available,
    otherwise falls back to difflib SequenceMatcher.
    threshold=3 Hamming distance (~0.85 similarity) is the default cut-off.
    """
    if SIMHASH_AVAILABLE and 'Simhash' in globals():
        features_a = _tokens_to_features(_get_tokens(text_a))
        features_b = _tokens_to_features(_get_tokens(text_b))
        if not features_a or not features_b:
            return False
        h_a = Simhash(features_a)
        h_b = Simhash(features_b)
        return h_a.distance(h_b) <= threshold
    else:
        return difflib.SequenceMatcher(None, text_a, text_b).ratio() > 0.85


# ----------------------------------------------------------------------------
# 6. Source gathering (unchanged + provenance tracking + curriculum ordering)
# ----------------------------------------------------------------------------

def gather_chunks(cfg, log_fn):
    import random
    combined = []
    for path in cfg["input_files"]:
        try:
            raw = extract_text(path)

            # 25% chance to create a massive multi-section chunk instead of standard sizes
            if random.random() < 0.25:
                t_words = cfg["target_words"] * 2
                m_words = cfg["max_words"] * 2
            else:
                t_words = cfg["target_words"]
                m_words = cfg["max_words"]

            cks = smart_chunk(raw, t_words, m_words)
            log_fn(f"Extracted {len(cks)} chunk(s) from {os.path.basename(path)} (Target size: {t_words} words)")
            for c in cks:
                # Compute a content hash for provenance tracking
                chunk_hash = hashlib.sha256(c.encode("utf-8")).hexdigest()[:16]
                combined.append((os.path.basename(path), c, chunk_hash))
        except Exception as e:
            log_fn(f"[ERROR] Failed to read {os.path.basename(path)}: {e}")

    pasted = cfg.get("pasted_text", "").strip()
    if pasted:
        cks = smart_chunk(pasted, cfg["target_words"], cfg["max_words"])
        log_fn(f"Extracted {len(cks)} chunk(s) from pasted text")
        for c in cks:
            chunk_hash = hashlib.sha256(c.encode("utf-8")).hexdigest()[:16]
            combined.append(("Pasted text", c, chunk_hash))

    # Optional curriculum ordering: shuffle with a slight bias toward keeping
    # adjacent chunks from the same document together (multi-section continuity)
    if cfg.get("curriculum_order", False):
        log_fn("Curriculum ordering enabled: sorting by source continuity")
        combined.sort(key=lambda x: x[0])

    return combined


# ----------------------------------------------------------------------------
# 7. Pipeline (runs in a background thread)
# ----------------------------------------------------------------------------

def run_pipeline(app, cfg):
    log = app.log
    log(f"Domain: {cfg['domain']}")
    log(f"Router: {cfg['router_model']}  |  Generators: {cfg['model_a']} <-> {cfg['model_b']}  "
        f"|  Validation: {'ON' if cfg['validate'] else 'OFF'}")
    if cfg["model_a"] == cfg["model_b"]:
        log("Note: Generator A and B are the same model - cross-validation becomes self-validation.")

    # Ensure optional simhash is available if user enabled semantic dedup
    if cfg.get("semantic_dedup", False):
        _ensure_simhash()
        log("Semantic deduplication enabled (simhash)")

    chunks = gather_chunks(cfg, log)
    if not chunks:
        log("[ERROR] No usable text chunks found. Check your input files/text.")
        app.finish(written=0)
        return

    app.set_total(len(chunks))
    task_list_str = ", ".join(cfg["task_types"])
    verbose = cfg.get("verbose", False)

    def vlog(msg):
        if verbose:
            log(msg)

    written = rejected = duplicates = skipped = 0
    recent_texts = []

    # Open output file with metadata header comment
    out_f = open(cfg["output_path"], "w", encoding="utf-8")
    try:
        # Write a metadata header as a JSON comment line
        meta = {
            "metadata": {
                "domain": cfg["domain"],
                "task_types": cfg["task_types"],
                "models": {"router": cfg["router_model"], "generator_a": cfg["model_a"], "generator_b": cfg["model_b"]},
                "validation": cfg["validate"],
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
        }
        out_f.write(json.dumps(meta, ensure_ascii=False) + "\n")

        for i, item in enumerate(chunks):
            # Handle both old 2-tuple and new 3-tuple formats gracefully
            if len(item) == 3:
                source_name, chunk, chunk_hash = item
            else:
                source_name, chunk = item
                chunk_hash = hashlib.sha256(chunk.encode("utf-8")).hexdigest()[:16]

            if app.stop_event.is_set():
                break
            while app.pause_event.is_set() and not app.stop_event.is_set():
                time.sleep(0.2)
            if app.stop_event.is_set():
                break

            app.set_progress(i + 1, f"routing ({source_name})")

            # --- ROUTER ---
            router_sys = ROUTER_SYSTEM_TMPL.format(
                domain=cfg["domain"], max_tasks=cfg["max_tasks_per_chunk"], task_list=task_list_str
            )
            raw = ""
            try:
                raw = ollama_generate(
                    cfg["router_model"],
                    f"SOURCE TEXT:\n{chunk}\n\nRespond with JSON only.",
                    router_sys,
                    temperature=0.0,
                )
                vlog(f"[{i+1}/{len(chunks)}] Router raw: {raw[:150]!r}")
                router_data = json.loads(raw)
            except Exception as e:
                log(f"[{i+1}/{len(chunks)}] Router error: {e} -> skipping chunk (raw: {raw[:150]!r})")
                skipped += 1
                continue

            if not router_data.get("usable", False):
                log(f"[{i+1}/{len(chunks)}] Skipped (router marked low-value)")
                skipped += 1
                continue

            # Extract difficulty level from router (default to 2 for backward compatibility)
            difficulty = router_data.get("difficulty", 2)
            tasks = [t for t in router_data.get("tasks", []) if t in cfg["task_types"]]
            if not tasks:
                tasks = [cfg["task_types"][0]]
            tasks = tasks[: cfg["max_tasks_per_chunk"]]

            gen_model = cfg["model_a"] if i % 2 == 0 else cfg["model_b"]
            val_model = cfg["model_b"] if i % 2 == 0 else cfg["model_a"]

            for task in tasks:
                if app.stop_event.is_set():
                    break

                app.set_progress(i + 1, f"generating [{task}] via {gen_model}")
                task_instr = TASK_INSTRUCTIONS[task].format(domain=cfg["domain"])
                gen_sys = GENERATOR_SYSTEM_TMPL.format(domain=cfg["domain"], task_instruction=task_instr)

                # --- GENERATOR ---
                raw = ""
                try:
                    raw = ollama_generate(
                        gen_model,
                        f"SOURCE TEXT:\n{chunk}\n\nGenerate the JSON training pair now.",
                        gen_sys,
                        temperature=cfg["temperature"],
                        top_p=cfg["top_p"],
                    )
                    vlog(f"[{i+1}/{len(chunks)}] Generator raw [{task}]: {raw[:150]!r}")
                    if not raw or raw.strip() == "{}":
                        log(f"[{i+1}/{len(chunks)}] Generator returned nothing for [{task}] "
                            f"via {gen_model} (model says no usable content)")
                        continue
                    gen_data = json.loads(raw)
                    pair_text = gen_data.get("text", "").strip()
                    if not pair_text or "User:" not in pair_text or "Assistant:" not in pair_text:
                        log(f"[{i+1}/{len(chunks)}] Generator output missing User/Assistant for [{task}] "
                            f"via {gen_model} (raw: {raw[:150]!r})")
                        continue
                except Exception as e:
                    log(f"[{i+1}/{len(chunks)}] Generation error [{task}] via {gen_model}: {e} "
                        f"(raw: {raw[:150]!r})")
                    continue

                # --- VALIDATOR (rubric-based scoring) ---
                quality_score = 2  # default assumption
                if cfg["validate"]:
                    app.set_progress(i + 1, f"validating [{task}] via {val_model}")
                    val_sys = VALIDATOR_SYSTEM_TMPL.format(domain=cfg["domain"])
                    raw = ""
                    try:
                        raw = ollama_generate(
                            val_model,
                            f"SOURCE TEXT:\n{chunk}\n\nCANDIDATE PAIR:\n{pair_text}\n\nRespond with JSON only.",
                            val_sys,
                            temperature=0.0,
                        )
                        vlog(f"[{i+1}/{len(chunks)}] Validator raw [{task}]: {raw[:150]!r}")
                        val_data = json.loads(raw)
                        quality_score = val_data.get("score", 2)
                        if not val_data.get("grounded", False):
                            reason = val_data.get("reason", "no reason given")
                            log(f"[{i+1}/{len(chunks)}] Rejected [{task}] by {val_model}: {reason}")
                            rejected += 1
                            continue
                    except Exception as e:
                        log(f"[{i+1}/{len(chunks)}] Validation error [{task}] via {val_model}: {e} "
                            f"(raw: {raw[:150]!r}) -> keeping pair")

                # --- DEDUPLICATION ---
                is_dup = False
                if cfg.get("semantic_dedup", False) and SIMHASH_AVAILABLE:
                    is_dup = any(
                        semantic_near_duplicate(prev, pair_text)
                        for prev in recent_texts
                    )
                else:
                    is_dup = any(
                        difflib.SequenceMatcher(None, prev, pair_text).ratio() > 0.85
                        for prev in recent_texts
                    )

                if is_dup:
                    duplicates += 1
                    log(f"[{i+1}/{len(chunks)}] Duplicate [{task}] skipped")
                    continue

                recent_texts.append(pair_text)
                if len(recent_texts) > 40:
                    recent_texts.pop(0)

                # --- WRITE OUTPUT (with metadata) ---
                output_record = {
                    "text": pair_text,
                    "meta": {
                        "task": task,
                        "difficulty": difficulty,
                        "source": source_name,
                        "chunk_hash": chunk_hash,
                        "quality_score": quality_score,
                        "generator": gen_model,
                    }
                }
                out_f.write(json.dumps(output_record, ensure_ascii=False) + "\n")
                out_f.flush()
                written += 1
                app.preview(task, pair_text)
                app.set_stats(written, rejected, duplicates, skipped)
    finally:
        out_f.close()

    if app.stop_event.is_set():
        log(f"\nStopped early. {written} pairs written, {rejected} rejected, "
            f"{duplicates} duplicates, {skipped} chunks skipped.")
    else:
        log(f"\nDone. {written} pairs written, {rejected} rejected, "
            f"{duplicates} duplicates, {skipped} chunks skipped.")
    app.finish(written=written)


# ----------------------------------------------------------------------------
# 8. GUI
# ----------------------------------------------------------------------------

TASK_TYPES = ["summary", "qa", "explanation", "application", "compare", "critique", "classify", "edge_case", "step_by_step", "creative_rewrite"]


class App:
    def __init__(self, root):
        self.root = root
        root.title("AI Dataset Forge - Multi-Model Pipeline (Ollama)")
        root.geometry("840x700")
        root.minsize(760, 620)

        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("Start.TButton", foreground="white", background="#2e7d32")
        style.configure("Stop.TButton", foreground="white", background="#c62828")
        style.configure("Header.TLabel", font=("Segoe UI", 10, "bold"))

        self.input_files = []
        self.queue = queue.Queue()
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self.worker = None

        self._build_ui()
        self.root.after(100, self._poll_queue)
        threading.Thread(target=self._refresh_models, daemon=True).start()

    # -- UI construction -----------------------------------------------------

    def _build_ui(self):
        nb = ttk.Notebook(self.root)
        nb.pack(fill="both", expand=True, padx=8, pady=8)

        self.tab_input = ttk.Frame(nb, padding=10)
        self.tab_pipeline = ttk.Frame(nb, padding=10)
        self.tab_run = ttk.Frame(nb, padding=10)
        nb.add(self.tab_input, text="1. Input / Output")
        nb.add(self.tab_pipeline, text="2. Models & Pipeline")
        nb.add(self.tab_run, text="3. Run")

        self._build_input_tab()
        self._build_pipeline_tab()
        self._build_run_tab()

    def _build_input_tab(self):
        f = self.tab_input

        ttk.Label(f, text="Source files (txt / pdf / docx)", style="Header.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 4)
        )

        list_frame = ttk.Frame(f)
        list_frame.grid(row=1, column=0, columnspan=2, sticky="nsew")
        f.rowconfigure(1, weight=1)
        f.columnconfigure(0, weight=1)

        self.file_listbox = tk.Listbox(list_frame, height=6, selectmode="extended")
        self.file_listbox.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(list_frame, command=self.file_listbox.yview)
        sb.pack(side="right", fill="y")
        self.file_listbox.config(yscrollcommand=sb.set)

        btns = ttk.Frame(f)
        btns.grid(row=1, column=2, sticky="n", padx=(8, 0))
        ttk.Button(btns, text="Add Files...", command=self._add_files).pack(fill="x", pady=2)
        ttk.Button(btns, text="Remove Selected", command=self._remove_files).pack(fill="x", pady=2)
        ttk.Button(btns, text="Clear All", command=self._clear_files).pack(fill="x", pady=2)

        ttk.Label(f, text="Or paste raw text (optional - treated as an extra source)",
                  style="Header.TLabel").grid(row=2, column=0, columnspan=3, sticky="w", pady=(12, 4))
        self.paste_text = scrolledtext.ScrolledText(f, height=6, wrap="word")
        self.paste_text.grid(row=3, column=0, columnspan=3, sticky="nsew")
        f.rowconfigure(3, weight=1)

        ttk.Label(f, text="Output dataset (.jsonl)", style="Header.TLabel").grid(
            row=4, column=0, columnspan=3, sticky="w", pady=(12, 4)
        )
        out_frame = ttk.Frame(f)
        out_frame.grid(row=5, column=0, columnspan=3, sticky="ew")
        out_frame.columnconfigure(0, weight=1)
        self.output_var = tk.StringVar()
        ttk.Entry(out_frame, textvariable=self.output_var).grid(row=0, column=0, sticky="ew")
        ttk.Button(out_frame, text="Browse...", command=self._select_output).grid(row=0, column=1, padx=4)
        ttk.Button(out_frame, text="Open Folder", command=self._open_output_folder).grid(row=0, column=2)

        ttk.Label(f, text="Dataset domain / topic", style="Header.TLabel").grid(
            row=6, column=0, columnspan=3, sticky="w", pady=(12, 4)
        )
        self.domain_var = tk.StringVar(value="Industrial Design")
        domain_combo = ttk.Combobox(
            f, textvariable=self.domain_var,
            values=["Industrial Design", "Product Design", "UX/UI Design", "Engineering Design", "Architecture"],
        )
        domain_combo.grid(row=7, column=0, columnspan=3, sticky="ew")

    def _build_pipeline_tab(self):
        f = self.tab_pipeline
        f.columnconfigure(1, weight=1)

        # Models
        ttk.Label(f, text="Ollama Models", style="Header.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 4))
        ttk.Button(f, text="Refresh Models", command=lambda: threading.Thread(
            target=self._refresh_models, daemon=True).start()).grid(row=0, column=2, sticky="e")
        self.ollama_status_var = tk.StringVar(value="Checking Ollama...")
        ttk.Label(f, textvariable=self.ollama_status_var, foreground="gray").grid(
            row=0, column=1, sticky="w", padx=8)

        ttk.Label(f, text="Router (fast, runs every chunk):").grid(row=1, column=0, sticky="w", pady=4)
        self.router_var = tk.StringVar()
        self.router_combo = ttk.Combobox(f, textvariable=self.router_var, state="readonly")
        self.router_combo.grid(row=1, column=1, columnspan=2, sticky="ew", pady=4)

        ttk.Label(f, text="Generator A:").grid(row=2, column=0, sticky="w", pady=4)
        self.model_a_var = tk.StringVar()
        self.model_a_combo = ttk.Combobox(f, textvariable=self.model_a_var, state="readonly")
        self.model_a_combo.grid(row=2, column=1, columnspan=2, sticky="ew", pady=4)

        ttk.Label(f, text="Generator B:").grid(row=3, column=0, sticky="w", pady=4)
        self.model_b_var = tk.StringVar()
        self.model_b_combo = ttk.Combobox(f, textvariable=self.model_b_var, state="readonly")
        self.model_b_combo.grid(row=3, column=1, columnspan=2, sticky="ew", pady=4)

        ttk.Label(f, text="A and B alternate per chunk (generator <-> validator) for "
                          "diversity and cross-model grounding checks.",
                  foreground="gray", wraplength=600, justify="left").grid(
            row=4, column=0, columnspan=3, sticky="w", pady=(0, 10))

        self.validate_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(f, text="Enable cross-model grounding validation (recommended)",
                         variable=self.validate_var).grid(row=5, column=0, columnspan=3, sticky="w")

        self.semantic_dedup_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(f, text="Use semantic deduplication (simhash) instead of string matching",
                         variable=self.semantic_dedup_var).grid(row=6, column=0, columnspan=3, sticky="w")

        self.curriculum_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(f, text="Enable curriculum ordering (group chunks by source continuity)",
                         variable=self.curriculum_var).grid(row=7, column=0, columnspan=3, sticky="w")

        ttk.Separator(f, orient="horizontal").grid(row=8, column=0, columnspan=3, sticky="ew", pady=10)

        # Task types
        ttk.Label(f, text="Task types to generate", style="Header.TLabel").grid(
            row=9, column=0, columnspan=3, sticky="w", pady=(0, 4))
        self.task_vars = {}
        defaults = {"summary": True, "qa": True, "explanation": True, "application": False,
                    "compare": True, "critique": True, "classify": False,
                    "edge_case": True, "step_by_step": True, "creative_rewrite": True}
        # Place checkbuttons in a 5-column grid to fit all 10 task types
        for idx, t in enumerate(TASK_TYPES):
            row = 10 + (idx // 5)
            col = idx % 5
            var = tk.BooleanVar(value=defaults[t])
            self.task_vars[t] = var
            ttk.Checkbutton(f, text=t, variable=var).grid(row=row, column=col, sticky="w", padx=(0, 8))

        ttk.Label(f, text="Max task types per chunk:").grid(row=12, column=0, sticky="w", pady=(10, 4))
        self.max_tasks_var = tk.IntVar(value=2)
        ttk.Spinbox(f, from_=1, to=4, textvariable=self.max_tasks_var, width=6).grid(
            row=12, column=1, sticky="w", pady=(10, 4))

        ttk.Separator(f, orient="horizontal").grid(row=13, column=0, columnspan=3, sticky="ew", pady=10)

        # Chunking + sampling
        ttk.Label(f, text="Chunking & sampling", style="Header.TLabel").grid(
            row=14, column=0, columnspan=3, sticky="w", pady=(0, 4))

        ttk.Label(f, text="Target chunk size (words):").grid(row=15, column=0, sticky="w", pady=2)
        self.target_words_var = tk.IntVar(value=250)
        ttk.Spinbox(f, from_=80, to=800, increment=10, textvariable=self.target_words_var, width=8).grid(
            row=15, column=1, sticky="w", pady=2)

        ttk.Label(f, text="Max chunk size (words):").grid(row=16, column=0, sticky="w", pady=2)
        self.max_words_var = tk.IntVar(value=450)
        ttk.Spinbox(f, from_=120, to=1200, increment=10, textvariable=self.max_words_var, width=8).grid(
            row=16, column=1, sticky="w", pady=2)

        ttk.Label(f, text="Generation temperature:").grid(row=17, column=0, sticky="w", pady=2)
        self.temperature_var = tk.DoubleVar(value=0.1)
        ttk.Spinbox(f, from_=0.0, to=1.0, increment=0.05, textvariable=self.temperature_var, width=8).grid(
            row=17, column=1, sticky="w", pady=2)

        ttk.Label(f, text="Generation top_p:").grid(row=18, column=0, sticky="w", pady=2)
        self.top_p_var = tk.DoubleVar(value=0.9)
        ttk.Spinbox(f, from_=0.1, to=1.0, increment=0.05, textvariable=self.top_p_var, width=8).grid(
            row=18, column=1, sticky="w", pady=2)

    def _build_run_tab(self):
        f = self.tab_run
        f.columnconfigure(0, weight=1)
        f.rowconfigure(3, weight=1)

        ctrl = ttk.Frame(f)
        ctrl.grid(row=0, column=0, sticky="ew")
        self.start_btn = ttk.Button(ctrl, text="Start", style="Start.TButton", command=self._start)
        self.start_btn.pack(side="left", padx=(0, 6))
        self.pause_btn = ttk.Button(ctrl, text="Pause", command=self._toggle_pause, state="disabled")
        self.pause_btn.pack(side="left", padx=6)
        self.stop_btn = ttk.Button(ctrl, text="Stop", style="Stop.TButton", command=self._stop, state="disabled")
        self.stop_btn.pack(side="left", padx=6)

        self.progress = ttk.Progressbar(f, mode="determinate")
        self.progress.grid(row=1, column=0, sticky="ew", pady=(10, 2))

        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(f, textvariable=self.status_var, foreground="gray").grid(row=2, column=0, sticky="w")

        stats = ttk.Frame(f)
        stats.grid(row=2, column=0, sticky="e")
        self.stats_var = tk.StringVar(value="Written: 0  |  Rejected: 0  |  Duplicates: 0  |  Skipped: 0")
        ttk.Label(stats, textvariable=self.stats_var, foreground="gray").pack()

        nb = ttk.Notebook(f)
        nb.grid(row=3, column=0, sticky="nsew", pady=(8, 0))

        log_frame = ttk.Frame(nb, padding=4)
        nb.add(log_frame, text="Log")
        self.log_box = scrolledtext.ScrolledText(log_frame, wrap="word", state="disabled")
        self.log_box.pack(fill="both", expand=True)

        preview_frame = ttk.Frame(nb, padding=4)
        nb.add(preview_frame, text="Preview")
        self.preview_box = scrolledtext.ScrolledText(preview_frame, wrap="word", state="disabled")
        self.preview_box.pack(fill="both", expand=True)

    # -- Input tab actions -----------------------------------------------------

    def _add_files(self):
        paths = filedialog.askopenfilenames(
            title="Select Source Documents",
            filetypes=[("Supported", "*.txt *.pdf *.docx"), ("Text", "*.txt"),
                       ("PDF", "*.pdf"), ("Word", "*.docx")],
        )
        for p in paths:
            if p not in self.input_files:
                self.input_files.append(p)
                self.file_listbox.insert("end", os.path.basename(p))

    def _remove_files(self):
        for idx in reversed(self.file_listbox.curselection()):
            self.file_listbox.delete(idx)
            del self.input_files[idx]

    def _clear_files(self):
        self.file_listbox.delete(0, "end")
        self.input_files.clear()

    def _select_output(self):
        path = filedialog.asksaveasfilename(
            title="Save Dataset As", defaultextension=".jsonl",
            filetypes=[("JSON Lines", "*.jsonl")],
        )
        if path:
            self.output_var.set(path)

    def _open_output_folder(self):
        path = self.output_var.get()
        if not path:
            return
        folder = os.path.dirname(os.path.abspath(path))
        try:
            if sys.platform.startswith("win"):
                os.startfile(folder)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", folder])
            else:
                subprocess.Popen(["xdg-open", folder])
        except Exception as e:
            messagebox.showerror("Error", str(e))

    # -- Model refresh -----------------------------------------------------

    def _refresh_models(self):
        models = list_ollama_models()
        self.queue.put(("models", models))

    def _apply_models(self, models):
        if not models:
            self.ollama_status_var.set("Ollama not reachable - start it and click Refresh.")
            return
        self.ollama_status_var.set(f"{len(models)} model(s) found.")
        for combo in (self.router_combo, self.model_a_combo, self.model_b_combo):
            combo["values"] = models

        by_size = sorted(models, key=parse_model_size)
        if not self.router_var.get():
            self.router_var.set(by_size[0])
        if not self.model_a_var.get():
            self.model_a_var.set(by_size[-1] if len(by_size) >= 1 else by_size[0])
        if not self.model_b_var.get():
            self.model_b_var.set(by_size[-2] if len(by_size) >= 2 else by_size[0])

    # -- Run controls -----------------------------------------------------

    def _collect_config(self):
        if not self.input_files and not self.paste_text.get("1.0", "end").strip():
            messagebox.showerror("Error", "Add at least one source file or paste some text.")
            return None
        if not self.output_var.get():
            messagebox.showerror("Error", "Choose an output .jsonl file.")
            return None
        task_types = [t for t, v in self.task_vars.items() if v.get()]
        if not task_types:
            messagebox.showerror("Error", "Enable at least one task type.")
            return None
        if not (self.router_var.get() and self.model_a_var.get() and self.model_b_var.get()):
            messagebox.showerror("Error", "Select Router, Generator A and Generator B models.")
            return None

        return {
            "input_files": list(self.input_files),
            "pasted_text": self.paste_text.get("1.0", "end"),
            "output_path": self.output_var.get(),
            "domain": self.domain_var.get().strip() or "Design",
            "router_model": self.router_var.get(),
            "model_a": self.model_a_var.get(),
            "model_b": self.model_b_var.get(),
            "validate": self.validate_var.get(),
            "semantic_dedup": self.semantic_dedup_var.get(),
            "curriculum_order": self.curriculum_var.get(),
            "task_types": task_types,
            "max_tasks_per_chunk": self.max_tasks_var.get(),
            "target_words": self.target_words_var.get(),
            "max_words": self.max_words_var.get(),
            "temperature": self.temperature_var.get(),
            "top_p": self.top_p_var.get(),
        }

    def _start(self):
        cfg = self._collect_config()
        if cfg is None:
            return

        self.log_box.config(state="normal")
        self.log_box.delete("1.0", "end")
        self.log_box.config(state="disabled")
        self.preview_box.config(state="normal")
        self.preview_box.delete("1.0", "end")
        self.preview_box.config(state="disabled")

        self.stop_event.clear()
        self.pause_event.clear()
        self.pause_btn.config(text="Pause", state="normal")
        self.stop_btn.config(state="normal")
        self.start_btn.config(state="disabled")
        self.progress["value"] = 0
        self.status_var.set("Starting...")
        self.stats_var.set("Written: 0  |  Rejected: 0  |  Duplicates: 0  |  Skipped: 0")

        self.worker = threading.Thread(target=run_pipeline, args=(self, cfg), daemon=True)
        self.worker.start()

    def _toggle_pause(self):
        if self.pause_event.is_set():
            self.pause_event.clear()
            self.pause_btn.config(text="Pause")
            self.status_var.set("Resumed.")
        else:
            self.pause_event.set()
            self.pause_btn.config(text="Resume")
            self.status_var.set("Paused.")

    def _stop(self):
        self.stop_event.set()
        self.pause_event.clear()
        self.status_var.set("Stopping...")

    # -- Thread-safe callbacks (called from worker thread) -----------------

    def log(self, msg):
        self.queue.put(("log", msg))

    def preview(self, task, text):
        self.queue.put(("preview", (task, text)))

    def set_total(self, total):
        self.queue.put(("total", total))

    def set_progress(self, i, stage):
        self.queue.put(("progress", (i, stage)))

    def set_stats(self, written, rejected, duplicates, skipped):
        self.queue.put(("stats", (written, rejected, duplicates, skipped)))

    def finish(self, written):
        self.queue.put(("finish", written))

    # -- Queue processing (runs on main thread) -----------------------------

    def _poll_queue(self):
        try:
            while True:
                kind, payload = self.queue.get_nowait()
                if kind == "log":
                    self._append_log(payload)
                elif kind == "preview":
                    task, text = payload
                    self._append_preview(task, text)
                elif kind == "total":
                    self.progress["maximum"] = payload
                elif kind == "progress":
                    i, stage = payload
                    self.progress["value"] = i
                    self.status_var.set(f"Chunk {i}/{int(self.progress['maximum'])} - {stage}")
                elif kind == "stats":
                    w, rj, d, s = payload
                    self.stats_var.set(
                        f"Written: {w}  |  Rejected: {rj}  |  Duplicates: {d}  |  Skipped: {s}")
                elif kind == "models":
                    self._apply_models(payload)
                elif kind == "finish":
                    self.start_btn.config(state="normal")
                    self.pause_btn.config(state="disabled", text="Pause")
                    self.stop_btn.config(state="disabled")
                    self.status_var.set(f"Finished - {payload} pairs written.")
                    if payload:
                        messagebox.showinfo("Done", f"Generated {payload} training pairs.")
        except queue.Empty:
            pass
        self.root.after(100, self._poll_queue)

    def _append_log(self, msg):
        self.log_box.config(state="normal")
        self.log_box.insert("end", msg + "\n")
        self.log_box.see("end")
        self.log_box.config(state="disabled")

    def _append_preview(self, task, text):
        self.preview_box.config(state="normal")
        self.preview_box.insert("end", f"--- [{task}] ---\n{text}\n\n")
        self.preview_box.see("end")
        self.preview_box.config(state="disabled")


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
