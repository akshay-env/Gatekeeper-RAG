"""
evaluation/metrics.py

RAGAS-style evaluation metrics for the RAG pipeline.
All scoring is done via Gemini (LLM-as-judge) — no external eval frameworks needed.

Four metrics implemented:

  1. Faithfulness       — Are all answer claims supported by the retrieved context?
  2. Answer Relevance   — Does the answer actually address the question?
  3. Context Recall     — Does the context contain the info needed for the ground truth?
  4. Context Precision  — Are the retrieved chunks actually relevant (low noise)?

Each metric returns a float in [0, 1]. Higher is better.

Run a full evaluation with:
  python -m evaluation.metrics
"""

import os
import json
import time
import re
from google import genai
from dotenv import load_dotenv

from evaluation.testset import load_testset
from generation.generator import ask

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

GENERATION_MODEL = "models/gemini-2.5-flash"
SLEEP_BETWEEN_CALLS = 3     # seconds between LLM judge calls
RATE_LIMIT_WAIT = 60


# ---------------------------------------------------------------------------
# LLM judge helper
# ---------------------------------------------------------------------------

def _judge(prompt: str) -> str:
    """Call Gemini and return the raw text response. Retries on 429."""
    for attempt in range(6):
        try:
            response = client.models.generate_content(
                model=GENERATION_MODEL,
                contents=prompt
            )
            return response.text.strip()
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                print(f"    Rate limit hit. Waiting {RATE_LIMIT_WAIT}s...")
                time.sleep(RATE_LIMIT_WAIT)
            else:
                print(f"    Judge call failed: {e}")
                return ""
    return ""


def _parse_json_list(text: str) -> list:
    """Extract a JSON array from an LLM response, stripping markdown fences."""
    text = re.sub(r"^```[a-z]*\n?", "", text.strip())
    text = re.sub(r"\n?```$", "", text)
    try:
        result = json.loads(text)
        return result if isinstance(result, list) else []
    except Exception:
        return []


def _parse_float(text: str) -> float | None:
    """Extract a single float/int from an LLM response."""
    match = re.search(r"\b(\d+(?:\.\d+)?)\b", text)
    if match:
        return min(1.0, float(match.group(1)) / 10.0)  # normalise 0-10 → 0-1
    return None


# ---------------------------------------------------------------------------
# Metric 1: Faithfulness
# ---------------------------------------------------------------------------

_FAITHFULNESS_PROMPT = """\
You are evaluating whether an AI-generated answer is faithful to the retrieved context.

Context:
{context}

Answer:
{answer}

Task: Extract each factual claim from the answer. For each claim, judge whether it is \
directly supported by the context above.

Return ONLY a valid JSON array (no explanation, no markdown):
[
  {{"claim": "...", "supported": true}},
  {{"claim": "...", "supported": false}},
  ...
]"""


def faithfulness(answer: str, context_chunks: list[dict]) -> float:
    """
    Measures: what fraction of claims in the answer are supported by the context?
    Score: supported_claims / total_claims  (1.0 = fully grounded, 0.0 = hallucinated)
    """
    if not answer or not context_chunks:
        return 0.0

    context_text = "\n\n".join(c.get("text", "") for c in context_chunks)
    prompt = _FAITHFULNESS_PROMPT.format(context=context_text[:3000], answer=answer)

    raw = _judge(prompt)
    claims = _parse_json_list(raw)

    if not claims:
        return 0.0

    supported = sum(1 for c in claims if c.get("supported", False))
    return round(supported / len(claims), 3)


# ---------------------------------------------------------------------------
# Metric 2: Answer Relevance
# ---------------------------------------------------------------------------

_RELEVANCE_PROMPT = """\
You are evaluating how well an AI answer addresses a user's question.

Question: {question}
Answer: {answer}

Score the answer from 0 to 10:
  10 = directly and completely answers the question
   7 = mostly answers the question, minor gaps
   4 = partially answers, misses key aspects
   1 = barely relevant
   0 = does not answer at all

Return ONLY the numeric score (e.g. "8"). No explanation."""


def answer_relevance(question: str, answer: str) -> float:
    """
    Measures: how well does the answer address the question?
    Score: 0–1 (normalised from 0–10 Gemini rating)
    """
    if not answer or not question:
        return 0.0

    prompt = _RELEVANCE_PROMPT.format(question=question, answer=answer)
    raw = _judge(prompt)
    score = _parse_float(raw)
    return score if score is not None else 0.0


# ---------------------------------------------------------------------------
# Metric 3: Context Recall
# ---------------------------------------------------------------------------

_RECALL_PROMPT = """\
You are evaluating whether retrieved context contains the information needed \
to produce a reference answer.

Reference answer (ground truth):
{ground_truth}

Retrieved context:
{context}

Task: For each sentence in the ground truth answer, determine whether the \
retrieved context contains the information needed to produce that sentence.

Return ONLY a valid JSON array (no explanation, no markdown):
[
  {{"sentence": "...", "attributable": true}},
  {{"sentence": "...", "attributable": false}},
  ...
]"""


def context_recall(ground_truth: str, context_chunks: list[dict]) -> float:
    """
    Measures: what fraction of the ground truth can be attributed to the retrieved context?
    Score: attributable_sentences / total_sentences  (1.0 = context covers everything)
    """
    if not ground_truth or not context_chunks:
        return 0.0

    context_text = "\n\n".join(c.get("text", "") for c in context_chunks)
    prompt = _RECALL_PROMPT.format(
        ground_truth=ground_truth,
        context=context_text[:3000]
    )

    raw = _judge(prompt)
    sentences = _parse_json_list(raw)

    if not sentences:
        return 0.0

    attributable = sum(1 for s in sentences if s.get("attributable", False))
    return round(attributable / len(sentences), 3)


# ---------------------------------------------------------------------------
# Metric 4: Context Precision
# ---------------------------------------------------------------------------

_PRECISION_PROMPT = """\
You are evaluating the precision of retrieved context for answering a question.

Question: {question}
Ground truth answer: {ground_truth}

Retrieved chunks:
{chunks}

Task: For each chunk, determine whether it was useful for answering the question \
(i.e. contains relevant information). A chunk is NOT useful if it is off-topic \
or contains no information relevant to the question.

Return ONLY a valid JSON array of booleans, one per chunk, in the same order:
[true, false, true, ...]"""


def context_precision(
    question: str,
    ground_truth: str,
    context_chunks: list[dict]
) -> float:
    """
    Measures: what fraction of retrieved chunks were actually relevant?
    Score: useful_chunks / total_chunks  (1.0 = no noise in retrieval)
    """
    if not context_chunks:
        return 0.0

    chunks_text = "\n\n".join(
        f"[{i + 1}] {c.get('text', '')[:400]}"
        for i, c in enumerate(context_chunks)
    )
    prompt = _PRECISION_PROMPT.format(
        question=question,
        ground_truth=ground_truth,
        chunks=chunks_text
    )

    raw = _judge(prompt)
    usefulness = _parse_json_list(raw)

    if not usefulness or len(usefulness) != len(context_chunks):
        return 0.0

    useful = sum(1 for u in usefulness if u is True)
    return round(useful / len(context_chunks), 3)


# ---------------------------------------------------------------------------
# Full evaluation runner
# ---------------------------------------------------------------------------

def evaluate(num_samples: int | None = None) -> dict:
    """
    Run the full evaluation suite over the saved testset.

    Args:
        num_samples: Number of test cases to evaluate. None = all of them.

    Returns:
        {
          "num_evaluated": int,
          "faithfulness":      float,   # mean across all cases
          "answer_relevance":  float,
          "context_recall":    float,
          "context_precision": float,
          "per_case":          list[dict],  # scores per individual test case
        }
    """
    testset = load_testset()

    if num_samples is not None:
        testset = testset[:num_samples]

    print(f"\nEvaluating {len(testset)} test cases...\n")
    print(f"{'#':<4} {'Question':<50} {'Faith':>6} {'Relev':>6} {'Recall':>7} {'Prec':>6}")
    print("-" * 85)

    per_case = []
    totals = {"faithfulness": 0.0, "answer_relevance": 0.0,
              "context_recall": 0.0, "context_precision": 0.0}

    for i, case in enumerate(testset):
        question     = case["question"]
        ground_truth = case["ground_truth"]

        # Run the full RAG pipeline
        result = ask(query=question, top_k=20, top_n=5,
                     use_hyde=True, use_rerank=True)
        answer = result["answer"]
        chunks = result["chunks_used"]

        time.sleep(SLEEP_BETWEEN_CALLS)

        # Score each metric
        f  = faithfulness(answer, chunks);        time.sleep(SLEEP_BETWEEN_CALLS)
        ar = answer_relevance(question, answer);  time.sleep(SLEEP_BETWEEN_CALLS)
        cr = context_recall(ground_truth, chunks);time.sleep(SLEEP_BETWEEN_CALLS)
        cp = context_precision(question, ground_truth, chunks)

        scores = {
            "question":         question,
            "answer":           answer,
            "ground_truth":     ground_truth,
            "faithfulness":     f,
            "answer_relevance": ar,
            "context_recall":   cr,
            "context_precision":cp,
        }
        per_case.append(scores)

        for k in totals:
            totals[k] += scores[k]

        print(f"{i+1:<4} {question[:48]:<50} {f:>6.2f} {ar:>6.2f} {cr:>7.2f} {cp:>6.2f}")

        if i < len(testset) - 1:
            time.sleep(SLEEP_BETWEEN_CALLS)

    n = len(testset)
    means = {k: round(v / n, 3) for k, v in totals.items()}

    print("\n" + "=" * 85)
    print(f"{'MEAN':<54} {means['faithfulness']:>6.3f} "
          f"{means['answer_relevance']:>6.3f} "
          f"{means['context_recall']:>7.3f} "
          f"{means['context_precision']:>6.3f}")
    print("=" * 85)

    return {
        "num_evaluated":    n,
        "faithfulness":     means["faithfulness"],
        "answer_relevance": means["answer_relevance"],
        "context_recall":   means["context_recall"],
        "context_precision":means["context_precision"],
        "per_case":         per_case,
    }


if __name__ == "__main__":
    import json as _json
    results = evaluate(num_samples=5)
    print("\nFull results saved below:")
    print(_json.dumps({k: v for k, v in results.items() if k != "per_case"}, indent=2))
