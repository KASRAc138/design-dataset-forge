"""
Dataset Forge - builds instruction-tuning datasets from design documents
using local Ollama models.

Pipeline per chunk: analyze -> route -> generate -> validate -> clean ->
dedupe -> export (ShareGPT / Alpaca, train/val/test split).

Outputs: dataset.jsonl, dataset_sharegpt.jsonl, dataset_alpaca.jsonl,
quality_report.json
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
import random
import string
from collections import Counter, defaultdict
from datetime import datetime

# ---------------------------------------------------------------------------
# 0. Optional dependency handling
# ---------------------------------------------------------------------------

def _ensure(package, import_name=None):
    """Install a package if missing. Returns the imported module."""
    import_name = import_name or package
    try:
        return __import__(import_name)
    except ImportError:
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", package],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            return __import__(import_name)
        except Exception:
            return None

PyPDF2 = _ensure("PyPDF2", "PyPDF2")
docx_module = _ensure("python-docx", "docx")
requests = _ensure("requests", "requests")

OLLAMA_BASE = "http://localhost:11434"
OLLAMA_GENERATE = f"{OLLAMA_BASE}/api/generate"
OLLAMA_TAGS = f"{OLLAMA_BASE}/api/tags"

# Optional simhash for semantic deduplication
SIMHASH_AVAILABLE = False
try:
    from simhash import Simhash
    SIMHASH_AVAILABLE = True
except ImportError:
    pass


def _ensure_simhash():
    global SIMHASH_AVAILABLE, Simhash
    if not SIMHASH_AVAILABLE:
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "simhash"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            from simhash import Simhash
            SIMHASH_AVAILABLE = True
        except Exception:
            pass


# ===========================================================================
# 1. CHUNK ANALYZER - extracts concrete anchors from text (no NLP deps)
# ===========================================================================

class ChunkAnalyzer:
    """
    Extracts concrete content elements from a chunk so prompts can reference
    SPECIFIC material rather than asking generic questions.

    Anchor types extracted:
      - key_terms: Capitalized technical phrases (e.g., "Universal Design")
      - definitions: "X is Y" / "X refers to Y" patterns
      - comparisons: A vs B patterns, contrast words
      - lists: Enumerated or bulleted items
      - metrics: Numbers, percentages, dimensions
      - named_entities: Proper names, product names, author names
    """

    # Patterns for definition extraction
    _DEF_PATTERNS = [
        re.compile(r"([A-Z][A-Za-z\s\-]{2,50})\s+(?:is|are|refers to|can be defined as|means|denotes)\s+([^.;]{10,200})", re.I),
        re.compile(r"(?:we define|by)\s+([A-Z][A-Za-z\s\-]{2,50})\s+(?:as|to mean)\s+([^.;]{10,200})", re.I),
    ]

    # Comparison markers
    _COMPARE_WORDS = re.compile(
        r"\b(versus|vs\.?|compared to|in contrast|unlike|whereas|while|better than|"
        r"worse than|rather than|instead of|on the other hand|alternatively)\b",
        re.I
    )

    # Metric patterns (numbers with units/context)
    _METRIC_RE = re.compile(
        r"\b\d+(?:[.,]\d+)?\s*(?:percent|%|mm|cm|m|kg|g|lbs|hours?|minutes?|"
        r"days?|years?|\$|€|£|USD|EUR|times?|x|fold|°[CF])?\b",
        re.I
    )

    # Multi-word capitalized terms (potential key concepts)
    _KEY_TERM_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+(?:of|the|in|for|and|&)\s+)?[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b")

    # List item patterns
    _LIST_RE = re.compile(r"^(?:[•\-\*]|\d+[.):])\s+(.{10,150})$", re.M)

    @classmethod
    def analyze(cls, text):
        """Return a dict of anchor elements found in the text."""
        anchors = {
            "key_terms": cls._extract_key_terms(text),
            "definitions": cls._extract_definitions(text),
            "comparisons": cls._extract_comparisons(text),
            "lists": cls._extract_lists(text),
            "metrics": cls._extract_metrics(text),
            "entities": cls._extract_entities(text),
            "sample_sentences": cls._extract_sample_sentences(text),
        }
        # Remove empty categories
        return {k: v for k, v in anchors.items() if v}

    @classmethod
    def _extract_key_terms(cls, text):
        terms = cls._KEY_TERM_RE.findall(text)
        # Filter out common false positives
        stopwords = {"The", "This", "That", "These", "Those", "There", "They",
                     "Figure", "Table", "Appendix", "Section", "Chapter", "Page"}
        terms = [t for t in terms if t not in stopwords and len(t) > 3]
        # Return top unique terms (most frequent first)
        counts = Counter(terms)
        return [term for term, _ in counts.most_common(8)]

    @classmethod
    def _extract_definitions(cls, text):
        defs = []
        for pat in cls._DEF_PATTERNS:
            for m in pat.finditer(text):
                term = m.group(1).strip()
                meaning = m.group(2).strip()
                if len(meaning) > 15:
                    defs.append(f"{term}: {meaning}")
        return defs[:5]

    @classmethod
    def _extract_comparisons(cls, text):
        sents = text.replace("\n", " ").split(". ")
        comp_sents = [s.strip() for s in sents if cls._COMPARE_WORDS.search(s)]
        return comp_sents[:4]

    @classmethod
    def _extract_lists(cls, text):
        items = cls._LIST_RE.findall(text)
        return [it.strip() for it in items[:6]]

    @classmethod
    def _extract_metrics(cls, text):
        metrics = cls._METRIC_RE.findall(text)
        seen = set()
        unique = []
        for m in metrics:
            key = re.sub(r"\s+", "", m.lower())
            if key not in seen:
                seen.add(key)
                unique.append(m.strip())
        return unique[:6]

    @classmethod
    def _extract_entities(cls, text):
        # Simple proper name extraction: sequences of capitalized words
        # that aren't sentence-start words
        sentences = text.replace("\n", " ").split(". ")
        entities = []
        for sent in sentences:
            words = sent.split()
            for i, w in enumerate(words):
                # Skip first word of sentence (likely just capitalized)
                if i == 0:
                    continue
                # Look for 2-3 capitalized words in a row
                if w[0].isupper() and len(w) > 2:
                    phrase = w
                    j = i + 1
                    while j < len(words) and words[j][0].isupper() and len(words[j]) > 2:
                        phrase += " " + words[j]
                        j += 1
                    if j - i >= 2 and len(phrase) > 5:
                        entities.append(phrase)
        counts = Counter(entities)
        return [e for e, _ in counts.most_common(6)]

    @classmethod
    def _extract_sample_sentences(cls, text):
        """Extract 3 diverse sentences that are substantive (not too short/long)."""
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.replace("\n", " "))]
        candidates = []
        for s in sentences:
            s = s.strip()
            if 40 < len(s) < 200 and s[0].isupper():
                candidates.append(s)
        # Pick diverse sentences (beginning, middle-like, end-like)
        if len(candidates) >= 3:
            return [candidates[0], candidates[len(candidates)//2], candidates[-1]]
        return candidates[:3]

    @classmethod
    def format_anchors(cls, anchors):
        """Format extracted anchors into a string for the generator prompt."""
        lines = []
        if anchors.get("key_terms"):
            lines.append("KEY TERMS: " + "; ".join(anchors["key_terms"]))
        if anchors.get("definitions"):
            lines.append("DEFINITIONS: " + " | ".join(anchors["definitions"]))
        if anchors.get("comparisons"):
            lines.append("COMPARISONS: " + " | ".join(anchors["comparisons"]))
        if anchors.get("lists"):
            lines.append("LISTS: " + "; ".join(anchors["lists"]))
        if anchors.get("metrics"):
            lines.append("METRICS: " + "; ".join(anchors["metrics"]))
        if anchors.get("entities"):
            lines.append("NAMES: " + "; ".join(anchors["entities"]))
        if anchors.get("sample_sentences"):
            lines.append("SAMPLE SENTENCES: " + " | ".join(
                s[:100] + "..." if len(s) > 100 else s for s in anchors["sample_sentences"]
            ))
        return "\n".join(lines) if lines else "(No specific anchors extracted - generate from the full text.)"


# ===========================================================================
# 2. TASK BALANCER - ensures even distribution across task types
# ===========================================================================

class TaskBalancer:
    """
    Tracks per-task counts and dynamically adjusts selection probability
    to prevent any single task type from dominating the dataset.
    """

    # Target distribution: tasks that should appear more/less frequently
    # Weights are relative - higher = more frequent. 1.0 = baseline.
    DEFAULT_TARGETS = {
        "summary": 1.0,
        "qa": 1.2,
        "explanation": 1.2,
        "application": 1.0,
        "compare": 0.9,
        "critique": 0.8,
        "classify": 0.8,
        "edge_case": 0.7,
        "step_by_step": 1.0,
        "creative_rewrite": 0.7,
    }

    def __init__(self, task_types, min_per_task=50):
        self.task_types = [t for t in task_types]
        self.counts = Counter()
        self.min_per_task = min_per_task
        self.targets = {t: self.DEFAULT_TARGETS.get(t, 1.0) for t in self.task_types}
        self.total_generated = 0

    def update(self, task):
        self.counts[task] += 1
        self.total_generated += 1

    def select_tasks(self, router_tasks, max_tasks=2):
        """
        Given tasks from the router, filter/reorder to improve balance.
        Returns a list of 1-max_tasks task names.
        """
        if not router_tasks:
            return [self.task_types[0]] if self.task_types else []

        # Filter to enabled tasks
        candidates = [t for t in router_tasks if t in self.task_types]
        if not candidates:
            candidates = list(self.task_types)

        # If we haven't hit minimum for any task, force-include underrepresented ones
        underrepresented = [t for t in self.task_types
                          if self.counts[t] < self.min_per_task]
        # Interleave underrepresented tasks with router suggestions
        prioritized = []
        for t in underrepresented:
            if t in candidates and t not in prioritized:
                prioritized.append(t)
        for t in candidates:
            if t not in prioritized:
                prioritized.append(t)

        # Score each candidate by how "under its target" it is
        def deficit_score(task):
            target_count = self.targets.get(task, 1.0) * max(self.total_generated / sum(self.targets.values()), 1)
            return target_count - self.counts[task]

        prioritized.sort(key=deficit_score, reverse=True)
        return prioritized[:max_tasks]

    def get_stats(self):
        return dict(self.counts)


# ===========================================================================
# 3. CLEANING UTILITIES - boilerplate stripper, truncation detector
# ===========================================================================

class ResponseCleaner:
    """
    Post-processes assistant responses to remove robotic boilerplate
    and detect incomplete/truncated outputs.
    """

    # Phrases that add no value and mark LLM-generated text
    BOILERPLATE_OPENERS = [
        r"^Based on the (?:text|provided text|excerpt|passage|chunk|source|document)[,\s]",
        r"^According to the (?:text|provided text|excerpt|passage|chunk|source|document)[,\s]",
        r"^The (?:text|excerpt|passage|chunk|source|document) (?:discusses|describes|explains|covers|states|mentions|highlights)[,\s]",
        r"^This (?:text|excerpt|passage|chunk|source|document) (?:discusses|describes|explains|covers|states|mentions|highlights)[,\s]",
        r"^The author (?:states|says|notes|argues|explains|describes|discusses)[,\s]",
        r"^In this (?:text|excerpt|passage|chunk)[,\s]",
        r"^From the (?:text|excerpt|passage|chunk|source)[,\s]",
        r"^The provided (?:text|excerpt|passage|chunk) (?:discusses|describes|explains|covers)[,\s]",
        r"^As (?:mentioned|stated|noted|described) in the (?:text|excerpt|passage|chunk)[,\s]",
    ]

    # Compile once
    _OPENER_RES = [re.compile(p, re.I) for p in BOILERPLATE_OPENERS]

    # Truncation signals: ending without proper punctuation
    _TRUNCATION_RE = re.compile(r"[^.!?;:]$")
    # Also check for mid-word cutoff at end
    _MIDWORD_RE = re.compile(r"\w{3,}$")

    @classmethod
    def strip_boilerplate(cls, text):
        """Remove robotic opening phrases from assistant responses."""
        cleaned = text.strip()
        for pattern in cls._OPENER_RES:
            cleaned = pattern.sub("", cleaned).strip()
        # Clean up leading lowercase after removal
        if cleaned and cleaned[0].islower():
            cleaned = cleaned[0].upper() + cleaned[1:]
        return cleaned.strip()

    @classmethod
    def is_truncated(cls, text, min_length=30):
        """
        Detect if response appears truncated (cut off mid-sentence or mid-word).
        Returns (is_truncated, reason).
        """
        if len(text.strip()) < min_length:
            return False, "too_short_to_check"

        last_100 = text.strip()[-100:]

        # Check if ends without terminal punctuation
        if cls._TRUNCATION_RE.search(text.strip()[-3:]):
            # But allow if it looks like a complete thought that just ends without punctuation
            if not text.strip()[-1].isalnum():
                return False, "ends_with_symbol"
            return True, "no_terminal_punctuation"

        # Check for mid-list truncation (ends with comma in a list context)
        if text.strip()[-1] == "," and any(c in last_100 for c in ["1.", "2.", "3.", "- ", "* "]):
            return True, "mid_list_truncation"

        return False, "complete"

    @classmethod
    def clean(cls, text):
        """Full cleaning pipeline. Returns (cleaned_text, was_truncated, truncation_reason)."""
        cleaned = cls.strip_boilerplate(text)
        truncated, reason = cls.is_truncated(cleaned)
        return cleaned, truncated, reason


# ===========================================================================
# 4. ENHANCED PROMPT TEMPLATES - chunk-aware, style-varied, task-specific
# ===========================================================================

ROUTER_SYSTEM_TMPL = """You are a dataset-curation assistant building a high-quality instruction-tuning dataset about {domain}.

Given a SOURCE TEXT chunk, decide:
1. "usable": true if the chunk contains enough self-contained, substantive information about {domain} to build a good training example. It must NOT be a reference list, table of contents, page header/footer, caption, or meaningless fragment. Otherwise false.
2. "difficulty": estimate the complexity on a 1-3 scale:
   - 1 = basic (introductory concepts, simple definitions)
   - 2 = intermediate (technical details, trade-offs, methods)
   - 3 = advanced (complex reasoning, multi-step analysis, synthesis, edge cases)
3. "tasks": pick 1 to {max_tasks} task type(s) from this list that BEST fit the content: {task_list}

Selection guidance:
- If the chunk defines/explains a concept → "explanation"
- If the chunk lists steps or a process → "step_by_step"
- If the chunk compares two or more things → "compare"
- If the chunk has specific facts/numbers → "qa"
- If the chunk discusses trade-offs or evaluation → "critique"
- If the chunk is a broad overview → "summary"
- If the chunk describes a method applied to a real situation → "application"
- If the chunk can be categorized/tagged → "classify"
- If the chunk has ambiguity or contradictions → "edge_case"
- If the chunk presents information that could be reformatted → "creative_rewrite"

Respond with ONLY valid JSON, no extra text:
{{"usable": true, "difficulty": 2, "tasks": ["explanation", "qa"]}}"""

# --- TASK INSTRUCTIONS: Each enforces chunk-specific prompt generation ---
# These are fed to the generator alongside extracted anchor elements.
# The generator MUST reference actual content from the chunk.

_TASK_NOTES = "IMPORTANT: The user prompt MUST specifically reference one or more ANCHOR ELEMENTS provided below. Do NOT write a generic question - it should only make sense for THIS chunk."

TASK_INSTRUCTIONS = {
    "summary": (
        "Generate a user request asking for a concise overview. "
        "The user prompt should reference a SPECIFIC concept, section, or theme from the ANCHOR ELEMENTS "
        "(e.g., 'What are the main points about [key term] mentioned here?' or 'gimme the gist of this [topic] stuff'). "
        "Vary the style: sometimes casual/messy ('sum this up quick'), sometimes detailed ('Provide a comprehensive overview of...'). "
        "The assistant response must be a pure plain-text paragraph with NO markdown, bold, lists, or bullets. "
        + _TASK_NOTES
    ),
    "qa": (
        "Generate a user asking a SPECIFIC question about a concrete fact from the ANCHOR ELEMENTS. "
        "The question must reference actual names, numbers, terms, or definitions extracted - not generic 'what is this about?'. "
        "Alternate styles: clean formal question, conversational with typos/filler ('umm so what does [term] actually mean?'), "
        "or imperative ('explain [concept]'). The assistant gives a direct, accurate plain-text answer. "
        + _TASK_NOTES
    ),
    "explanation": (
        "Pick a core technical concept from the KEY TERMS or DEFINITIONS in the ANCHOR ELEMENTS. "
        "Generate a user asking to explain it - referencing the actual term by name. "
        "The assistant response must use markdown headers to structure the explanation clearly. "
        + _TASK_NOTES
    ),
    "application": (
        "Generate a user asking how a method, principle, or finding from the ANCHOR ELEMENTS applies to a real-world design scenario. "
        "Reference the actual method/principle by name. "
        "The assistant provides a practical example as a clean multi-sentence plain-text paragraph. "
        + _TASK_NOTES
    ),
    "compare": (
        "Identify two distinct items, materials, methods, or approaches from the COMPARISONS or KEY TERMS in the ANCHOR ELEMENTS. "
        "Generate a user prompt asking to evaluate trade-offs between them under specific constraints. "
        "The assistant clearly outlines structural pros/cons or trade-offs in plain text. "
        + _TASK_NOTES
    ),
    "critique": (
        "Generate a hard reasoning task: rank options, justify a complex choice, or evaluate an approach "
        "based STRICTLY on criteria in the text. Reference specific items from the ANCHOR ELEMENTS. "
        "The assistant provides rigorous logical evaluation. "
        + _TASK_NOTES
    ),
    "classify": (
        "Generate a user instruction to categorize a process, approach, or theme from the ANCHOR ELEMENTS. "
        "Reference the actual item being classified. "
        "The assistant gives the category name followed by a brief plain-text justification. "
        + _TASK_NOTES
    ),
    "edge_case": (
        "Craft a user prompt that points out ambiguity, contradiction, or missing information in the chunk "
        "(e.g., 'This seems to contradict [X], explain why' or 'What's missing from this description of [Y]?'). "
        "Reference specific elements from the ANCHOR ELEMENTS. "
        "The assistant acknowledges limitations and provides the best possible answer. "
        + _TASK_NOTES
    ),
    "step_by_step": (
        "Generate a user request for a step-by-step procedure or sequential breakdown derived from the text. "
        "Reference the actual process or method name from the ANCHOR ELEMENTS. "
        "The assistant uses a numbered list (1., 2., 3.) with clear self-contained steps. "
        + _TASK_NOTES
    ),
    "creative_rewrite": (
        "Generate a user asking to re-express information in a different format or register "
        "(e.g., 'rewrite as an email to a client', 'turn into a checklist', 'make this a tweet thread', 'write as a design brief'). "
        "Reference the specific content to rewrite from the ANCHOR ELEMENTS. "
        "The assistant preserves all facts while adopting the requested form faithfully. "
        + _TASK_NOTES
    ),
}

# --- PROMPT LENGTH VARIATION ---
# Used to instruct the generator to produce short/medium/long user prompts
PROMPT_LENGTH_STYLES = [
    "The user prompt should be SHORT (5-15 words), casual or abrupt.",
    "The user prompt should be MEDIUM (15-40 words), natural conversational style.",
    "The user prompt should be DETAILED (40-100 words), specific and well-structured.",
]

GENERATOR_SYSTEM_TMPL = """You are an expert dataset generator creating instruction-tuning examples about {domain}.

ANCHOR ELEMENTS extracted from the source chunk:
{anchors}

PROMPT STYLE RULES:
- User prompts MUST reference specific anchor elements by name - never generic "summarize this" or "what is this about"
- Vary user style: sometimes messy/casual with typos, sometimes clean and formal, sometimes imperative
- NEVER use repetitive robotic framing in assistant responses
- STRICTLY follow the format rules for this task type (plain-text vs markdown vs numbered list)
- If source text is ambiguous, acknowledge it explicitly

{length_style}

TASK: {task_instruction}

Output ONLY valid JSON in exactly this shape:
{{"text": "User: <instruction>\nAssistant: <full response>"}}"""


# ===========================================================================
# 5. ENHANCED VALIDATOR - rubric-based scoring (0-6 composite)
# ===========================================================================

VALIDATOR_SYSTEM_TMPL = """You are a rigorous dataset quality auditor for {domain} instruction-tuning data.

Evaluate the candidate training pair against SOURCE TEXT using these rubric criteria.
Score EACH dimension 0-2:

--- RUBRIC ---
1. grounded (0-2):
   2 = All specific facts, numbers, names, and claims are directly traceable to SOURCE TEXT
   1 = Minor paraphrasing or reordering OK, no invented specifics
   0 = Contains hallucinated facts, numbers, names, or claims not in source

2. complete (0-2):
   2 = Fully answers the user's instruction with appropriate depth
   1 = Partial answer - misses some aspects but not wrong
   0 = Incomplete, off-topic, or evades the question

3. style_adherence (0-2):
   2 = Perfectly follows the requested format (plain-text vs markdown vs numbered list)
   1 = Minor format deviation
   0 = Wrong format entirely (e.g., uses markdown when plain-text required)

--- RULES ---
- Paraphrasing and rewording the SOURCE TEXT is GOOD - never reject for that alone
- The user prompt SHOULD reference specific content from the source (not generic) - if it doesn't, reduce "complete" by 1
- Mark "grounded": false if ANY specific fact/number/name is invented

Respond with ONLY valid JSON:
{{"grounded": true/false, "grounded_score": 0-2, "complete_score": 0-2, "style_score": 0-2, "composite_score": 0-6, "reason": "brief explanation"}}"""


# ===========================================================================
# 6. TEXT EXTRACTION (enhanced with source provenance)
# ===========================================================================

def extract_text(filepath):
    """Extract text from txt, pdf, or docx. Returns (text, source_meta) tuple."""
    ext = os.path.splitext(filepath)[1].lower()
    basename = os.path.basename(filepath)
    source_meta = {"filename": basename, "type": ext.lstrip(".")}

    if ext == ".txt":
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

    elif ext == ".pdf":
        if PyPDF2 is None:
            raise ValueError("PyPDF2 not installed. Run: pip install PyPDF2")
        with open(filepath, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            pages_text = []
            for i, page in enumerate(reader.pages):
                pt = page.extract_text() or ""
                pages_text.append(f"[PAGE_{i+1}]\n{pt}")
            text = "\n\n".join(pages_text)
            source_meta["pages"] = len(reader.pages)

    elif ext == ".docx":
        if docx_module is None:
            raise ValueError("python-docx not installed. Run: pip install python-docx")
        doc = docx_module.Document(filepath)
        paragraphs = []
        for i, p in enumerate(doc.paragraphs):
            if p.text.strip():
                paragraphs.append(p.text)
        text = "\n\n".join(paragraphs)

    else:
        raise ValueError(f"Unsupported format: {ext}")

    return text, source_meta


# ===========================================================================
# 7. SMART CHUNKING (enhanced - preserves page markers, better noise filtering)
# ===========================================================================

_NOISE_PATTERNS = [
    re.compile(r"^\s*(page\s*\d+|\d+)\s*$", re.I),
    re.compile(r"^\s*(references|bibliography|acknowledg(e)?ments?|appendix)\s*$", re.I),
    re.compile(r"^\s*(table|figure)\s*\d+", re.I),
    re.compile(r"^\s*\d+\s*$"),  # Lone numbers (page refs)
    re.compile(r"^\s*doi:\s*", re.I),
]


def _is_noise(paragraph):
    p = paragraph.strip()
    if len(p) < 15:
        return True
    digit_ratio = sum(c.isdigit() for c in p) / max(len(p), 1)
    if digit_ratio > 0.45:
        return True
    for pat in _NOISE_PATTERNS:
        if pat.match(p):
            return True
    return False


def smart_chunk(text, target_words=250, max_words=450, min_chunk_words=15):
    """
    Split text into coherent multi-paragraph chunks.
    Preserves [PAGE_N] markers for provenance tracking.
    """
    if not text or not text.strip():
        return []

    # De-hyphenate line wraps
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"(\w)\u2013?\n(\w)", r"\1\2", text)
    text = re.sub(r"(\w)\u2014\n(\w)", r"\1\2", text)
    # Preserve double-newlines (paragraph breaks), collapse single ones
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

    raw_paras = re.split(r"\n\s*\n", text)
    paras = []
    for p in raw_paras:
        p = re.sub(r"\s+", " ", p).strip()
        if p and not _is_noise(p):
            paras.append(p)

    chunks = []
    current, current_words = [], 0
    current_pages = set()

    for p in paras:
        # Track page markers
        page_markers = re.findall(r"\[PAGE_(\d+)\]", p)
        for pm in page_markers:
            current_pages.add(int(pm))
        # Remove page markers for word counting
        clean_p = re.sub(r"\[PAGE_\d+\]\n?", "", p).strip()
        wc = len(clean_p.split())

        if current and current_words + wc > max_words:
            chunk_text = "\n\n".join(current)
            chunks.append((chunk_text, sorted(current_pages)))
            current, current_words = [p], wc
            current_pages = set(int(m) for m in page_markers)
        else:
            current.append(p)
            current_words += wc
            if current_words >= target_words:
                chunk_text = "\n\n".join(current)
                chunks.append((chunk_text, sorted(current_pages)))
                current, current_words = [], 0
                current_pages = set()

    if current:
        chunk_text = "\n\n".join(current)
        chunks.append((chunk_text, sorted(current_pages)))

    # Filter by minimum word count (on cleaned text)
    result = []
    for chunk_text, pages in chunks:
        clean = re.sub(r"\[PAGE_\d+\]\n?", "", chunk_text).strip()
        if len(clean.split()) >= min_chunk_words:
            result.append((chunk_text, pages))
    return result


# ===========================================================================
# 8. OLLAMA HELPERS (enhanced with retry/backoff and model listing)
# ===========================================================================

def list_ollama_models():
    if requests is None:
        return []
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
    if not raw:
        return raw
    raw = _THINK_RE.sub("", raw).strip()
    raw = _FENCE_RE.sub("", raw).strip()
    return raw


def ollama_generate(model, prompt, system, temperature=0.2, top_p=0.9, timeout=300, retries=2):
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


# ===========================================================================
# 9. SEMANTIC DEDUPLICATION (simhash + difflib fallback)
# ===========================================================================

def _get_tokens(text):
    return re.findall(r"\b\w+\b", text.lower())


def _tokens_to_features(tokens):
    features = {}
    for t in tokens:
        features[t] = features.get(t, 0) + 1
    return features


def semantic_near_duplicate(text_a, text_b, threshold=3):
    if SIMHASH_AVAILABLE and "Simhash" in globals():
        features_a = _tokens_to_features(_get_tokens(text_a))
        features_b = _tokens_to_features(_get_tokens(text_b))
        if not features_a or not features_b:
            return False
        return Simhash(features_a).distance(Simhash(features_b)) <= threshold
    else:
        return difflib.SequenceMatcher(None, text_a, text_b).ratio() > 0.85


class DedupTracker:
    """
    Tracks recent texts for deduplication. Supports both exact and semantic dedup.
    Also detects near-duplicate PROMPTS (same intent, different wording).
    """

    def __init__(self, max_recent=50, semantic=True):
        self.max_recent = max_recent
        self.semantic = semantic
        self.recent_pairs = []   # full "User:... Assistant:..." strings
        self.recent_prompts = []  # just the User: portion

    def _extract_prompt(self, pair_text):
        """Extract just the user prompt portion for prompt-level dedup."""
        match = re.search(r"User:\s*(.+?)(?:\nAssistant:|$)", pair_text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return pair_text[:200]

    def is_duplicate(self, pair_text):
        """Check if pair_text is a duplicate (exact or semantic) of recent items."""
        # Exact duplicate of full pair
        if pair_text in self.recent_pairs:
            return True, "exact_pair"

        # Semantic duplicate of full pair
        if self.semantic:
            for prev in self.recent_pairs:
                if semantic_near_duplicate(prev, pair_text):
                    return True, "semantic_pair"

        # Near-duplicate prompt check (same question, different wording)
        prompt = self._extract_prompt(pair_text)
        for prev_prompt in self.recent_prompts:
            if difflib.SequenceMatcher(None, prev_prompt, prompt).ratio() > 0.82:
                return True, "near_duplicate_prompt"
            if self.semantic:
                if semantic_near_duplicate(prev_prompt, prompt):
                    return True, "semantic_prompt"

        return False, None

    def add(self, pair_text):
        self.recent_pairs.append(pair_text)
        self.recent_prompts.append(self._extract_prompt(pair_text))
        if len(self.recent_pairs) > self.max_recent:
            self.recent_pairs.pop(0)
            self.recent_prompts.pop(0)


# ===========================================================================
# 10. EXPORT MANAGER - ShareGPT, Alpaca, raw JSONL with train/val/test split
# ===========================================================================

class ExportManager:
    """
    Handles formatting and exporting the dataset in multiple formats
    with optional stratified train/validation/test splitting.
    """

    @staticmethod
    def parse_pair(text):
        """Parse 'User: ...\nAssistant: ...' into (prompt, response)."""
        match = re.search(r"User:\s*(.+?)\nAssistant:\s*(.+)", text, re.DOTALL)
        if match:
            return match.group(1).strip(), match.group(2).strip()
        return None, None

    @classmethod
    def to_sharegpt(cls, records, system_msg=None):
        """Convert records to ShareGPT format."""
        system_msg = system_msg or "You are a helpful assistant specializing in design."
        results = []
        for rec in records:
            prompt, response = cls.parse_pair(rec.get("text", ""))
            if not prompt or not response:
                continue
            msg = {
                "messages": [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response},
                ]
            }
            # Carry forward metadata in a separate field (training frameworks ignore it)
            msg["metadata"] = rec.get("meta", {})
            results.append(msg)
        return results

    @classmethod
    def to_alpaca(cls, records):
        """Convert records to Alpaca format."""
        results = []
        for rec in records:
            prompt, response = cls.parse_pair(rec.get("text", ""))
            if not prompt or not response:
                continue
            # If prompt has context/instruction split, try to separate
            parts = prompt.split("\n\n", 1)
            if len(parts) == 2 and len(parts[1]) > 20:
                instruction, input_text = parts[0], parts[1]
            else:
                instruction, input_text = prompt, ""
            results.append({
                "instruction": instruction,
                "input": input_text,
                "output": response,
                "metadata": rec.get("meta", {}),
            })
        return results

    @classmethod
    def stratified_split(cls, records, train=0.8, val=0.1, test=0.1):
        """
        Split records into train/val/test, stratified by task type.
        Returns dict with "train", "validation", "test" keys.
        """
        assert abs(train + val + test - 1.0) < 0.01, "Split ratios must sum to 1.0"

        # Group by task
        by_task = defaultdict(list)
        for rec in records:
            task = rec.get("meta", {}).get("task", "unknown")
            by_task[task].append(rec)

        splits = {"train": [], "validation": [], "test": []}

        for task, task_recs in by_task.items():
            random.shuffle(task_recs)
            n = len(task_recs)
            n_train = int(n * train)
            n_val = int(n * val)
            # Test gets remainder to avoid rounding issues
            splits["train"].extend(task_recs[:n_train])
            splits["validation"].extend(task_recs[n_train:n_train + n_val])
            splits["test"].extend(task_recs[n_train + n_val:])

        # Shuffle each split
        for k in splits:
            random.shuffle(splits[k])

        return splits

    @classmethod
    def write_jsonl(cls, records, filepath):
        """Write records as JSON Lines."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    @classmethod
    def export_all(cls, records, base_path, system_msg=None, do_split=True):
        """
        Export records in all formats. If do_split, creates train/val/test files.
        Returns a report dict with file paths and counts.
        """
        report = {"files": [], "counts": {}}
        base_dir = os.path.dirname(os.path.abspath(base_path))
        base_name = os.path.splitext(os.path.basename(base_path))[0]
        os.makedirs(base_dir, exist_ok=True)

        if do_split:
            splits = cls.stratified_split(records)
            for split_name, split_recs in splits.items():
                # Raw internal format
                raw_path = os.path.join(base_dir, f"{base_name}_{split_name}.jsonl")
                cls.write_jsonl(split_recs, raw_path)
                report["files"].append(raw_path)
                report["counts"][f"{split_name}_raw"] = len(split_recs)

                # ShareGPT format
                sg_recs = cls.to_sharegpt(split_recs, system_msg)
                sg_path = os.path.join(base_dir, f"{base_name}_{split_name}_sharegpt.jsonl")
                cls.write_jsonl(sg_recs, sg_path)
                report["files"].append(sg_path)
                report["counts"][f"{split_name}_sharegpt"] = len(sg_recs)

                # Alpaca format
                al_recs = cls.to_alpaca(split_recs)
                al_path = os.path.join(base_dir, f"{base_name}_{split_name}_alpaca.jsonl")
                cls.write_jsonl(al_recs, al_path)
                report["files"].append(al_path)
                report["counts"][f"{split_name}_alpaca"] = len(al_recs)

            report["total_records"] = len(records)
            report["split"] = {k: len(v) for k, v in splits.items()}
        else:
            # No split - single files
            cls.write_jsonl(records, base_path)
            report["files"].append(base_path)
            report["counts"]["raw"] = len(records)

            sg_recs = cls.to_sharegpt(records, system_msg)
            sg_path = os.path.join(base_dir, f"{base_name}_sharegpt.jsonl")
            cls.write_jsonl(sg_recs, sg_path)
            report["files"].append(sg_path)
            report["counts"]["sharegpt"] = len(sg_recs)

            al_recs = cls.to_alpaca(records)
            al_path = os.path.join(base_dir, f"{base_name}_alpaca.jsonl")
            cls.write_jsonl(al_recs, al_path)
            report["files"].append(al_path)
            report["counts"]["alpaca"] = len(al_recs)

            report["total_records"] = len(records)

        return report


# ===========================================================================
# 11. QUALITY TRACKER - histogram, filtering, reporting
# ===========================================================================

class QualityTracker:
    """
    Tracks quality scores across the dataset and produces a report.
    Composite score range: 0-6 (sum of 3 rubric dimensions × 0-2).
    """

    def __init__(self, min_composite_score=3):
        self.scores = []  # list of composite scores
        self.rubric_details = []  # list of {grounded, complete, style} dicts
        self.rejected = []  # list of {score, reason, task}
        self.min_score = min_composite_score
        self.task_scores = defaultdict(list)

    def add(self, composite_score, rubric=None, task="unknown", passed=True, reject_reason=None):
        entry = {
            "composite": composite_score,
            "rubric": rubric or {},
            "task": task,
            "passed": passed,
            "reject_reason": reject_reason,
        }
        self.scores.append(composite_score)
        if rubric:
            self.rubric_details.append(rubric)
        self.task_scores[task].append(composite_score)
        if not passed:
            self.rejected.append(entry)

    def passes_threshold(self, composite_score):
        return composite_score >= self.min_score

    def get_histogram(self):
        """Return score distribution (0-6)."""
        hist = Counter(self.scores)
        return {i: hist.get(i, 0) for i in range(7)}

    def get_task_averages(self):
        """Return average score per task type."""
        return {task: sum(scores) / len(scores)
                for task, scores in self.task_scores.items() if scores}

    def get_report(self):
        if not self.scores:
            return {"error": "No quality data collected"}

        hist = self.get_histogram()
        total = len(self.scores)
        passing = sum(1 for s in self.scores if s >= self.min_score)

        report = {
            "total_evaluated": total,
            "passing": passing,
            "rejected": len(self.rejected),
            "pass_rate": round(passing / total, 3) if total else 0,
            "score_histogram": hist,
            "average_composite": round(sum(self.scores) / total, 2),
            "median_composite": sorted(self.scores)[len(self.scores) // 2],
            "min_score": min(self.scores),
            "max_score": max(self.scores),
            "per_task_average": self.get_task_averages(),
            "rejection_reasons": Counter(r["reject_reason"] for r in self.rejected).most_common(),
            "quality_threshold": self.min_score,
        }

        # Rubric dimension averages
        if self.rubric_details:
            for dim in ["grounded_score", "complete_score", "style_score"]:
                vals = [r.get(dim) for r in self.rubric_details if r.get(dim) is not None]
                if vals:
                    report[f"avg_{dim}"] = round(sum(vals) / len(vals), 2)

        return report

    def write_report(self, filepath):
        report = self.get_report()
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        return report


# ===========================================================================
# 12. SOURCE GATHERING - multi-document with provenance tracking
# ===========================================================================

def gather_chunks(cfg, log_fn):
    """
    Extract and chunk all input files + pasted text.
    Returns list of (source_name, chunk_text, chunk_hash, pages, source_meta) tuples.
    """
    combined = []
    for path in cfg["input_files"]:
        try:
            raw, source_meta = extract_text(path)

            # 25% chance for oversized chunks (multi-section continuity)
            if random.random() < 0.25:
                t_words = cfg["target_words"] * 2
                m_words = cfg["max_words"] * 2
            else:
                t_words = cfg["target_words"]
                m_words = cfg["max_words"]

            chunks = smart_chunk(raw, t_words, m_words)
            log_fn(f"Extracted {len(chunks)} chunk(s) from {source_meta['filename']} "
                   f"({source_meta.get('pages', 'N/A')} pages, target={t_words}w)")

            for chunk_text, pages in chunks:
                clean_text = re.sub(r"\[PAGE_\d+\]\n?", "", chunk_text).strip()
                chunk_hash = hashlib.sha256(clean_text.encode("utf-8")).hexdigest()[:16]
                combined.append((
                    source_meta["filename"],
                    chunk_text,
                    chunk_hash,
                    pages,
                    source_meta,
                ))
        except Exception as e:
            log_fn(f"[ERROR] Failed to read {os.path.basename(path)}: {e}")

    # Pasted text
    pasted = cfg.get("pasted_text", "").strip()
    if pasted:
        chunks = smart_chunk(pasted, cfg["target_words"], cfg["max_words"])
        log_fn(f"Extracted {len(chunks)} chunk(s) from pasted text")
        for chunk_text, pages in chunks:
            clean_text = re.sub(r"\[PAGE_\d+\]\n?", "", chunk_text).strip()
            chunk_hash = hashlib.sha256(clean_text.encode("utf-8")).hexdigest()[:16]
            combined.append(("Pasted text", chunk_text, chunk_hash, pages, {"type": "text"}))

    # Optional: curriculum ordering (group by source)
    if cfg.get("curriculum_order", False):
        log_fn("Curriculum ordering enabled")
        combined.sort(key=lambda x: x[0])

    return combined


# ===========================================================================
# 13. ENHANCED PIPELINE - integrates all v2 components
# ===========================================================================

def run_pipeline(app, cfg):
    log = app.log
    log("=" * 60)
    log("AI Dataset Forge v2 - Starting Pipeline")
    log("=" * 60)
    log(f"Domain: {cfg['domain']}")
    log(f"Router: {cfg['router_model']}  |  Generators: {cfg['model_a']} <-> {cfg['model_b']}")
    log(f"Validation: {'ON' if cfg['validate'] else 'OFF'}  |  "
        f"Min quality score: {cfg.get('min_quality_score', 3)}/6")
    log(f"Export: {cfg.get('export_format', 'all').upper()} format  |  "
        f"Split: {'80/10/10' if cfg.get('do_split', True) else 'none'}")

    if cfg["model_a"] == cfg["model_b"]:
        log("Note: Generator A and B are the same - cross-validation is self-validation.")

    # Initialize v2 components
    if cfg.get("semantic_dedup", False):
        _ensure_simhash()
        log("Semantic deduplication enabled (simhash)")

    balancer = TaskBalancer(cfg["task_types"], min_per_task=cfg.get("min_per_task", 50))
    dedup = DedupTracker(max_recent=50, semantic=cfg.get("semantic_dedup", False))
    quality = QualityTracker(min_composite_score=cfg.get("min_quality_score", 3))

    # Collection of all accepted records (for final export)
    accepted_records = []

    chunks = gather_chunks(cfg, log)
    if not chunks:
        log("[ERROR] No usable chunks found. Check input files.")
        app.finish(written=0)
        return

    app.set_total(len(chunks))
    task_list_str = ", ".join(cfg["task_types"])
    verbose = cfg.get("verbose", False)

    def vlog(msg):
        if verbose:
            log(msg)

    written = rejected = duplicates = skipped = truncated = 0

    for i, item in enumerate(chunks):
        source_name, chunk_text, chunk_hash, pages, source_meta = item
        clean_chunk = re.sub(r"\[PAGE_\d+\]\n?", "", chunk_text).strip()

        if app.stop_event.is_set():
            break
        while app.pause_event.is_set() and not app.stop_event.is_set():
            time.sleep(0.2)
        if app.stop_event.is_set():
            break

        app.set_progress(i + 1, f"routing ({source_name})")

        # --- CHUNK ANALYSIS (v2: extract anchors) ---
        anchors = ChunkAnalyzer.analyze(clean_chunk)
        anchor_text = ChunkAnalyzer.format_anchors(anchors)
        vlog(f"[{i+1}] Anchors: {list(anchors.keys())}")

        # --- ROUTER ---
        router_sys = ROUTER_SYSTEM_TMPL.format(
            domain=cfg["domain"],
            max_tasks=cfg["max_tasks_per_chunk"],
            task_list=task_list_str,
        )
        raw = ""
        try:
            raw = ollama_generate(
                cfg["router_model"],
                f"SOURCE TEXT:\n{clean_chunk}\n\nRespond with JSON only.",
                router_sys,
                temperature=0.0,
            )
            vlog(f"[{i+1}] Router raw: {raw[:150]!r}")
            router_data = json.loads(raw)
        except Exception as e:
            log(f"[{i+1}] Router error: {e} -> skip (raw: {raw[:150]!r})")
            skipped += 1
            continue

        if not router_data.get("usable", False):
            vlog(f"[{i+1}] Router marked unusable")
            skipped += 1
            continue

        difficulty = router_data.get("difficulty", 2)
        router_tasks = [t for t in router_data.get("tasks", []) if t in cfg["task_types"]]

        # --- TASK BALANCER (v2: enforce distribution) ---
        tasks = balancer.select_tasks(router_tasks, cfg["max_tasks_per_chunk"])
        if not tasks:
            tasks = [cfg["task_types"][0]]
        vlog(f"[{i+1}] Tasks after balancing: {tasks}")

        gen_model = cfg["model_a"] if i % 2 == 0 else cfg["model_b"]
        val_model = cfg["model_b"] if i % 2 == 0 else cfg["model_a"]

        for task in tasks:
            if app.stop_event.is_set():
                break

            # Vary prompt length style
            length_style = random.choice(PROMPT_LENGTH_STYLES)

            app.set_progress(i + 1, f"generating [{task}] via {gen_model}")
            task_instr = TASK_INSTRUCTIONS[task]
            gen_sys = GENERATOR_SYSTEM_TMPL.format(
                domain=cfg["domain"],
                anchors=anchor_text,
                length_style=length_style,
                task_instruction=task_instr,
            )

            # --- GENERATOR ---
            raw = ""
            try:
                raw = ollama_generate(
                    gen_model,
                    f"SOURCE TEXT:\n{clean_chunk}\n\nGenerate the JSON training pair now.",
                    gen_sys,
                    temperature=cfg["temperature"],
                    top_p=cfg["top_p"],
                )
                vlog(f"[{i+1}] Gen raw [{task}]: {raw[:150]!r}")
                if not raw or raw.strip() == "{}":
                    log(f"[{i+1}] Empty generation for [{task}]")
                    continue
                gen_data = json.loads(raw)
                pair_text = gen_data.get("text", "").strip()
                if not pair_text or "User:" not in pair_text or "Assistant:" not in pair_text:
                    log(f"[{i+1}] Malformed output for [{task}]")
                    continue
            except Exception as e:
                log(f"[{i+1}] Generation error [{task}]: {e}")
                continue

            # --- RESPONSE CLEANER (v2: strip boilerplate, check truncation) ---
            prompt, response = ExportManager.parse_pair(pair_text)
            cleaned_response, is_truncated, trunc_reason = ResponseCleaner.clean(response)

            if is_truncated:
                truncated += 1
                vlog(f"[{i+1}] Truncated response [{task}]: {trunc_reason}")
                # Optionally reject truncated responses
                if cfg.get("reject_truncated", True):
                    rejected += 1
                    continue

            # Rebuild pair with cleaned response
            if cleaned_response != response:
                pair_text = f"User: {prompt}\nAssistant: {cleaned_response}"

            # --- VALIDATOR (v2: rubric-based 0-6 scoring) ---
            composite_score = 4  # default assumption (mid-range)
            rubric = None
            grounded = True

            if cfg["validate"]:
                app.set_progress(i + 1, f"validating [{task}] via {val_model}")
                val_sys = VALIDATOR_SYSTEM_TMPL.format(domain=cfg["domain"])
                raw = ""
                try:
                    raw = ollama_generate(
                        val_model,
                        f"SOURCE TEXT:\n{clean_chunk}\n\nCANDIDATE PAIR:\n{pair_text}\n\nRespond with JSON only.",
                        val_sys,
                        temperature=0.0,
                    )
                    vlog(f"[{i+1}] Val raw [{task}]: {raw[:150]!r}")
                    val_data = json.loads(raw)
                    grounded = val_data.get("grounded", True)
                    composite_score = val_data.get("composite_score", 4)
                    rubric = {
                        "grounded_score": val_data.get("grounded_score", 2),
                        "complete_score": val_data.get("complete_score", 2),
                        "style_score": val_data.get("style_score", 2),
                    }
                except Exception as e:
                    vlog(f"[{i+1}] Validation error: {e} - using default score")

            # Track quality regardless of pass/fail
            quality.add(
                composite_score=composite_score,
                rubric=rubric,
                task=task,
                passed=grounded and quality.passes_threshold(composite_score),
                reject_reason=None if grounded else "ungrounded",
            )

            if not grounded:
                reason = "ungrounded"
                log(f"[{i+1}] Rejected [{task}] by {val_model}: {reason} (score={composite_score})")
                rejected += 1
                continue

            if not quality.passes_threshold(composite_score):
                vlog(f"[{i+1}] Filtered [{task}]: score {composite_score} < threshold {quality.min_score}")
                rejected += 1
                continue

            # --- DEDUPLICATION (v2: exact + semantic + prompt-level) ---
            is_dup, dup_reason = dedup.is_duplicate(pair_text)
            if is_dup:
                duplicates += 1
                vlog(f"[{i+1}] Duplicate [{task}]: {dup_reason}")
                continue

            dedup.add(pair_text)

            # --- ACCEPT & STORE ---
            balancer.update(task)
            written += 1

            record = {
                "text": pair_text,
                "meta": {
                    "task": task,
                    "difficulty": difficulty,
                    "source": source_name,
                    "chunk_hash": chunk_hash,
                    "pages": pages,
                    "quality_score": composite_score,
                    "rubric": rubric or {},
                    "generator": gen_model,
                    "length_style": length_style,
                }
            }
            accepted_records.append(record)

            # Write to running output (raw format)
            app.write_running(record)
            app.preview(task, pair_text)
            app.set_stats(written, rejected, duplicates, skipped, truncated)

    # --- FINAL EXPORT (v2: multiple formats + split) ---
    log("\n" + "=" * 60)
    log("EXPORT PHASE")
    log("=" * 60)

    if accepted_records:
        export_mgr = ExportManager()
        report = export_mgr.export_all(
            accepted_records,
            cfg["output_path"],
            system_msg=f"You are a helpful assistant specializing in {cfg['domain']}.",
            do_split=cfg.get("do_split", True),
        )

        for fpath in report["files"]:
            log(f"  -> {fpath}")
        log(f"\nExport counts: {report['counts']}")

        # Quality report
        qual_report = quality.write_report(
            os.path.join(os.path.dirname(os.path.abspath(cfg["output_path"])), "quality_report.json")
        )
        log(f"\nQuality Report:")
        log(f"  Average score: {qual_report.get('average_composite', 'N/A')}/6")
        log(f"  Pass rate: {qual_report.get('pass_rate', 'N/A')}")
        log(f"  Histogram: {qual_report.get('score_histogram', {})}")
        log(f"  Per-task avg: {qual_report.get('per_task_average', {})}")

        # Task distribution
        task_dist = balancer.get_stats()
        log(f"\nTask Distribution: {task_dist}")
    else:
        log("\nNo records passed quality filters - nothing to export.")

    log("\n" + "=" * 60)
    log(f"Done. Written: {written} | Rejected: {rejected} | Duplicates: {duplicates} | "
        f"Skipped: {skipped} | Truncated: {truncated}")
    log("=" * 60)

    app.finish(written=written)


# ===========================================================================
# 14. GUI - Enhanced with v2 controls
# ===========================================================================

TASK_TYPES = [
    "summary", "qa", "explanation", "application", "compare",
    "critique", "classify", "edge_case", "step_by_step", "creative_rewrite"
]


class App:
    def __init__(self, root):
        self.root = root
        root.title("AI Dataset Forge v2 - SFT Dataset Generator")
        root.geometry("900x780")
        root.minsize(820, 680)

        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("Start.TButton", foreground="white", background="#2e7d32")
        style.configure("Stop.TButton", foreground="white", background="#c62828")
        style.configure("Header.TLabel", font=("Segoe UI", 10, "bold"))
        style.configure("Note.TLabel", foreground="gray", wraplength=650)

        self.input_files = []
        self.queue = queue.Queue()
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self.worker = None
        self._running_fh = None  # file handle for running output

        self._build_ui()
        self.root.after(100, self._poll_queue)
        threading.Thread(target=self._refresh_models, daemon=True).start()

    # -- UI Construction -----------------------------------------------------

    def _build_ui(self):
        nb = ttk.Notebook(self.root)
        nb.pack(fill="both", expand=True, padx=8, pady=8)

        self.tab_input = ttk.Frame(nb, padding=10)
        self.tab_pipeline = ttk.Frame(nb, padding=10)
        self.tab_quality = ttk.Frame(nb, padding=10) 
        self.tab_run = ttk.Frame(nb, padding=10)

        nb.add(self.tab_input, text="1. Input / Output")
        nb.add(self.tab_pipeline, text="2. Models & Pipeline")
        nb.add(self.tab_quality, text="3. Quality & Export")
        nb.add(self.tab_run, text="4. Run")

        self._build_input_tab()
        self._build_pipeline_tab()
        self._build_quality_tab()
        self._build_run_tab()

    def _build_input_tab(self):
        f = self.tab_input
        f.columnconfigure(0, weight=1)

        ttk.Label(f, text="Source files (txt / pdf / docx)", style="Header.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 4))

        list_frame = ttk.Frame(f)
        list_frame.grid(row=1, column=0, columnspan=2, sticky="nsew")
        f.rowconfigure(1, weight=1)

        self.file_listbox = tk.Listbox(list_frame, height=6, selectmode="extended")
        self.file_listbox.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(list_frame, command=self.file_listbox.yview)
        sb.pack(side="right", fill="y")
        self.file_listbox.config(yscrollcommand=sb.set)

        btns = ttk.Frame(f)
        btns.grid(row=1, column=2, sticky="n", padx=(8, 0))
        ttk.Button(btns, text="Add Files...", command=self._add_files).pack(fill="x", pady=2)
        ttk.Button(btns, text="Remove", command=self._remove_files).pack(fill="x", pady=2)
        ttk.Button(btns, text="Clear", command=self._clear_files).pack(fill="x", pady=2)

        ttk.Label(f, text="Or paste raw text (optional - treated as extra source)",
                  style="Header.TLabel").grid(row=2, column=0, columnspan=3, sticky="w", pady=(12, 4))
        self.paste_text = scrolledtext.ScrolledText(f, height=5, wrap="word")
        self.paste_text.grid(row=3, column=0, columnspan=3, sticky="nsew")
        f.rowconfigure(3, weight=1)

        # Output section
        ttk.Label(f, text="Output dataset (.jsonl)", style="Header.TLabel").grid(
            row=4, column=0, columnspan=3, sticky="w", pady=(12, 4))
        out_frame = ttk.Frame(f)
        out_frame.grid(row=5, column=0, columnspan=3, sticky="ew")
        out_frame.columnconfigure(0, weight=1)
        self.output_var = tk.StringVar()
        ttk.Entry(out_frame, textvariable=self.output_var).grid(row=0, column=0, sticky="ew")
        ttk.Button(out_frame, text="Browse...", command=self._select_output).grid(row=0, column=1, padx=4)
        ttk.Button(out_frame, text="Open Folder", command=self._open_output_folder).grid(row=0, column=2)

        # Domain
        ttk.Label(f, text="Dataset domain / topic", style="Header.TLabel").grid(
            row=6, column=0, columnspan=3, sticky="w", pady=(12, 4))
        self.domain_var = tk.StringVar(value="Design")
        domain_combo = ttk.Combobox(
            f, textvariable=self.domain_var,
            values=["Design", "Industrial Design", "Product Design", "UX/UI Design",
                    "Graphic Design", "Architecture", "Interior Design", "Fashion Design"])
        domain_combo.grid(row=7, column=0, columnspan=3, sticky="ew")

    def _build_pipeline_tab(self):
        f = self.tab_pipeline
        f.columnconfigure(1, weight=1)

        # Models
        ttk.Label(f, text="Ollama Models", style="Header.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 4))
        ttk.Button(f, text="Refresh", command=lambda: threading.Thread(
            target=self._refresh_models, daemon=True).start()).grid(row=0, column=2, sticky="e")
        self.ollama_status_var = tk.StringVar(value="Checking Ollama...")
        ttk.Label(f, textvariable=self.ollama_status_var, foreground="gray").grid(
            row=0, column=1, sticky="w", padx=8)

        ttk.Label(f, text="Router (fast):").grid(row=1, column=0, sticky="w", pady=4)
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

        ttk.Label(f, text="A and B alternate per chunk (generator <-> validator).",
                  style="Note.TLabel").grid(row=4, column=0, columnspan=3, sticky="w", pady=(0, 10))

        # Toggles
        self.validate_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(f, text="Enable cross-model validation (rubric scoring)",
                        variable=self.validate_var).grid(row=5, column=0, columnspan=3, sticky="w")

        self.semantic_dedup_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(f, text="Use semantic deduplication (simhash)",
                        variable=self.semantic_dedup_var).grid(row=6, column=0, columnspan=3, sticky="w")

        self.curriculum_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(f, text="Curriculum ordering (group by source)",
                        variable=self.curriculum_var).grid(row=7, column=0, columnspan=3, sticky="w")

        ttk.Separator(f, orient="horizontal").grid(row=8, column=0, columnspan=3, sticky="ew", pady=10)

        # Task types
        ttk.Label(f, text="Task types to generate", style="Header.TLabel").grid(
            row=9, column=0, columnspan=3, sticky="w", pady=(0, 4))
        self.task_vars = {}
        # More balanced defaults than v1
        defaults = {
            "summary": True, "qa": True, "explanation": True, "application": True,
            "compare": True, "critique": True, "classify": True,
            "edge_case": True, "step_by_step": True, "creative_rewrite": True,
        }
        for idx, t in enumerate(TASK_TYPES):
            row = 10 + (idx // 5)
            col = idx % 5
            var = tk.BooleanVar(value=defaults[t])
            self.task_vars[t] = var
            ttk.Checkbutton(f, text=t, variable=var).grid(row=row, column=col, sticky="w", padx=(0, 8))

        ttk.Label(f, text="Max tasks per chunk:").grid(row=12, column=0, sticky="w", pady=(10, 4))
        self.max_tasks_var = tk.IntVar(value=2)
        ttk.Spinbox(f, from_=1, to=4, textvariable=self.max_tasks_var, width=6).grid(
            row=12, column=1, sticky="w", pady=(10, 4))

        ttk.Separator(f, orient="horizontal").grid(row=13, column=0, columnspan=3, sticky="ew", pady=10)

        # Chunking + generation params
        ttk.Label(f, text="Chunking & Generation", style="Header.TLabel").grid(
            row=14, column=0, columnspan=3, sticky="w", pady=(0, 4))

        ttk.Label(f, text="Target words:").grid(row=15, column=0, sticky="w", pady=2)
        self.target_words_var = tk.IntVar(value=250)
        ttk.Spinbox(f, from_=80, to=800, increment=10, textvariable=self.target_words_var, width=8).grid(
            row=15, column=1, sticky="w", pady=2)

        ttk.Label(f, text="Max words:").grid(row=16, column=0, sticky="w", pady=2)
        self.max_words_var = tk.IntVar(value=450)
        ttk.Spinbox(f, from_=120, to=1200, increment=10, textvariable=self.max_words_var, width=8).grid(
            row=16, column=1, sticky="w", pady=2)

        ttk.Label(f, text="Temperature:").grid(row=17, column=0, sticky="w", pady=2)
        self.temperature_var = tk.DoubleVar(value=0.15)
        ttk.Spinbox(f, from_=0.0, to=1.0, increment=0.05, textvariable=self.temperature_var, width=8).grid(
            row=17, column=1, sticky="w", pady=2)

        ttk.Label(f, text="Top-p:").grid(row=18, column=0, sticky="w", pady=2)
        self.top_p_var = tk.DoubleVar(value=0.9)
        ttk.Spinbox(f, from_=0.1, to=1.0, increment=0.05, textvariable=self.top_p_var, width=8).grid(
            row=18, column=1, sticky="w", pady=2)

    def _build_quality_tab(self):
        """NEW tab - Quality filtering and export controls."""
        f = self.tab_quality
        f.columnconfigure(1, weight=1)

        # Quality threshold
        ttk.Label(f, text="Quality Filtering", style="Header.TLabel").grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 4))

        ttk.Label(f, text="Min composite score (0-6):").grid(row=1, column=0, sticky="w", pady=4)
        self.min_quality_var = tk.IntVar(value=3)
        ttk.Spinbox(f, from_=0, to=6, textvariable=self.min_quality_var, width=6).grid(
            row=1, column=1, sticky="w", pady=4)
        ttk.Label(f, text="3=recommended | 4=strict | 2=permissive", style="Note.TLabel").grid(
            row=1, column=2, sticky="w", padx=8)

        self.reject_truncated_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(f, text="Reject truncated/incomplete responses",
                        variable=self.reject_truncated_var).grid(row=2, column=0, columnspan=3, sticky="w", pady=4)

        ttk.Label(f, text="Min examples per task type:").grid(row=3, column=0, sticky="w", pady=4)
        self.min_per_task_var = tk.IntVar(value=50)
        ttk.Spinbox(f, from_=10, to=500, increment=10, textvariable=self.min_per_task_var, width=8).grid(
            row=3, column=1, sticky="w", pady=4)

        ttk.Separator(f, orient="horizontal").grid(row=4, column=0, columnspan=3, sticky="ew", pady=12)

        # Export settings
        ttk.Label(f, text="Export Settings", style="Header.TLabel").grid(
            row=5, column=0, columnspan=3, sticky="w", pady=(0, 4))

        self.do_split_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(f, text="Create train / validation / test split (80/10/10, stratified)",
                        variable=self.do_split_var).grid(row=6, column=0, columnspan=3, sticky="w", pady=4)

        ttk.Label(f, text="Export format(s):").grid(row=7, column=0, sticky="w", pady=4)
        self.export_format_var = tk.StringVar(value="all")
        formats = ttk.Combobox(f, textvariable=self.export_format_var,
                               values=["all", "sharegpt", "alpaca", "raw"], state="readonly", width=12)
        formats.grid(row=7, column=1, sticky="w", pady=4)
        ttk.Label(f, text="ShareGPT = recommended for most training frameworks",
                  style="Note.TLabel").grid(row=7, column=2, sticky="w", padx=8)

        # Info box
        ttk.Separator(f, orient="horizontal").grid(row=8, column=0, columnspan=3, sticky="ew", pady=12)
        info = (
            "QUALITY SCORING (0-6 composite):\n"
            "  - grounded (0-2): Are all facts traceable to source?\n"
            "  - complete (0-2): Does it fully answer the prompt?\n"
            "  - style (0-2): Does it follow the requested format?\n\n"
            "A score of 3+ means acceptable for training. 5-6 is excellent.\n\n"
            "The quality_report.json file shows histograms and per-task breakdowns."
        )
        ttk.Label(f, text=info, style="Note.TLabel", justify="left").grid(
            row=9, column=0, columnspan=3, sticky="w", pady=4)

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
        self.stats_var = tk.StringVar(value="Written: 0 | Rejected: 0 | Duplicates: 0 | Skipped: 0 | Truncated: 0")
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

    # -- Input tab actions ---------------------------------------------------

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

    # -- Model refresh -------------------------------------------------------

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

    # -- Run controls --------------------------------------------------------

    def _collect_config(self):
        if not self.input_files and not self.paste_text.get("1.0", "end").strip():
            messagebox.showerror("Error", "Add at least one source file or paste text.")
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
            # v2 quality controls
            "min_quality_score": self.min_quality_var.get(),
            "reject_truncated": self.reject_truncated_var.get(),
            "min_per_task": self.min_per_task_var.get(),
            # v2 export controls
            "do_split": self.do_split_var.get(),
            "export_format": self.export_format_var.get(),
        }

    def _start(self):
        cfg = self._collect_config()
        if cfg is None:
            return

        # Clear UI
        self.log_box.config(state="normal")
        self.log_box.delete("1.0", "end")
        self.log_box.config(state="disabled")
        self.preview_box.config(state="normal")
        self.preview_box.delete("1.0", "end")
        self.preview_box.config(state="disabled")

        # Open running output file
        try:
            self._running_fh = open(cfg["output_path"], "w", encoding="utf-8")
            meta = {
                "metadata": {
                    "version": "v2",
                    "domain": cfg["domain"],
                    "task_types": cfg["task_types"],
                    "models": {
                        "router": cfg["router_model"],
                        "generator_a": cfg["model_a"],
                        "generator_b": cfg["model_b"],
                    },
                    "quality_threshold": cfg["min_quality_score"],
                    "validation": cfg["validate"],
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                }
            }
            self._running_fh.write(json.dumps(meta, ensure_ascii=False) + "\n")
        except Exception as e:
            messagebox.showerror("Error", f"Cannot write to output: {e}")
            return

        self.stop_event.clear()
        self.pause_event.clear()
        self.pause_btn.config(text="Pause", state="normal")
        self.stop_btn.config(state="normal")
        self.start_btn.config(state="disabled")
        self.progress["value"] = 0
        self.status_var.set("Starting...")
        self.stats_var.set("Written: 0 | Rejected: 0 | Duplicates: 0 | Skipped: 0 | Truncated: 0")

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

    # -- Running file writer (thread-safe via main thread queue) ------------

    def write_running(self, record):
        """Write a record to the running output file (called from pipeline thread)."""
        self.queue.put(("write", record))

    def _do_write(self, record):
        if self._running_fh and not self._running_fh.closed:
            self._running_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._running_fh.flush()

    # -- Thread-safe callbacks -----------------------------------------------

    def log(self, msg):
        self.queue.put(("log", msg))

    def preview(self, task, text):
        self.queue.put(("preview", (task, text)))

    def set_total(self, total):
        self.queue.put(("total", total))

    def set_progress(self, i, stage):
        self.queue.put(("progress", (i, stage)))

    def set_stats(self, written, rejected, duplicates, skipped, truncated=0):
        self.queue.put(("stats", (written, rejected, duplicates, skipped, truncated)))

    def finish(self, written):
        self.queue.put(("finish", written))

    # -- Queue processing (main thread) --------------------------------------

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
                    w, rj, d, s, t = payload
                    self.stats_var.set(
                        f"Written: {w}  |  Rejected: {rj}  |  Duplicates: {d}  |  "
                        f"Skipped: {s}  |  Truncated: {t}")
                elif kind == "models":
                    self._apply_models(payload)
                elif kind == "finish":
                    if self._running_fh and not self._running_fh.closed:
                        self._running_fh.close()
                        self._running_fh = None
                    self.start_btn.config(state="normal")
                    self.pause_btn.config(state="disabled", text="Pause")
                    self.stop_btn.config(state="disabled")
                    self.status_var.set(f"Finished - {payload} pairs written.")
                    if payload:
                        messagebox.showinfo("Done", f"Generated {payload} training pairs.\n"
                                            "Check the output folder for ShareGPT/Alpaca exports.")
                elif kind == "write":
                    self._do_write(payload)
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
