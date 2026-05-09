import json
import os
import re
from typing import Any, cast

from llama_index.llms.google_genai import GoogleGenAI

from zettlr_rag.consts import MODEL_NAME

EVAL_PROMPT = """You are evaluating a RAG system's answer against a reference answer derived
from peer-reviewed scientific literature. Score each axis independently using the rubrics
below. A given shortcoming should affect only the axis it pertains to.

For each axis, output a number between 0.0 and 1.0. Use the anchor points as a guide.

1. factual_precision (0-1) -- PRECISION of the RAG answer.
   Of the substantive claims the RAG answer makes, what fraction are supported by the
   reference answer or its supporting papers? Do NOT penalize for missing content here --
   only for incorrect, unsupported, or misleading claims that ARE made.
     - 1.0: Every substantive claim is directly supported.
     - 0.7: Most claims are supported; minor unsupported asides or imprecise wording.
     - 0.5: Roughly half of substantive claims are supported; the rest are unsupported
            but not contradictory.
     - 0.2: Most claims are unsupported or only loosely related.
     - 0.0: Claims are largely fabricated or off-topic.

2. content_recall (0-1) -- RECALL of the key points in the reference.
   Of the key scientific points present in the reference (findings, methodologies,
   limitations, key numbers, important caveats), what fraction does the RAG answer
   cover? Do NOT penalize the RAG for being correct about things it does say --
   only for missing material that the reference covers.
     - 1.0: All key points are covered with comparable specificity.
     - 0.7: Most key points covered; minor omissions or low specificity on a few.
     - 0.5: About half of key points covered; major omissions remain.
     - 0.2: Only a small fraction of key points covered.
     - 0.0: Almost none of the reference's key points appear.

3. fluency_and_structure (0-1) -- WRITING QUALITY only.
   Is the answer well-written, logically structured, and appropriately formatted
   (clear paragraphs, headers/bold/italics where they help, LaTeX for equations,
   code blocks for code)? Judge writing quality only; do not let factual issues
   or coverage influence this score.
     - 1.0: Polished, well-structured, formatting aids comprehension.
     - 0.7: Clear and readable; minor structural or formatting issues.
     - 0.5: Understandable but disorganized or poorly formatted.
     - 0.2: Hard to follow.
     - 0.0: Incoherent.

4. contradiction_detected (0 or 1) -- HARD ERRORS.
   Set to 1 if and only if the RAG answer contains a claim that DIRECTLY
   contradicts a finding in the reference or its citations (not merely omits
   it, not merely unsupported, not merely vague). Otherwise 0.

Notes for the judge:
   - Tables in the reference may be rendered as flat text due to copy/paste
     formatting loss. Evaluate based on factual content only, not visual layout.
   - Treat factual_precision and content_recall as INDEPENDENT axes. A short,
     correct answer should score high on precision and low on recall; a long
     comprehensive answer with some unsupported claims should score the opposite.
   - Reasoning effort should focus on identifying specific claims (for precision)
     and specific reference key points (for recall) before assigning scores.

Respond with strict JSON in this exact shape:
{"factual_precision": <float 0-1>,
 "content_recall": <float 0-1>,
 "fluency_and_structure": <float 0-1>,
 "contradiction_detected": <0 or 1>}
"""


def llm_judge(
    question: str, rag_answer: str, reference_answer: str, reference_citations: str
) -> dict[str, Any]:
    """
    Evaluates a RAG response against a reference using Gemini.

    Notes:
    1. This implementation uses Gemini for both evaluation and as the underlying LLM for the
    RAG system, which may suffer from self-preference bias ("egocentric bias"). In practice, it is
    advisable to use a different model for evaluation.
    2. For this library, we used Claude Sonnet 4.6 for LLM-as-a-judge evaluation. That evaluation
    is not shown as it used a proprietary API and the code is not public.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set.")

    llm = GoogleGenAI(model=f"models/{MODEL_NAME}", api_key=api_key)

    prompt = f"""
    You are evaluating a RAG system's answer against a reference answer
    derived from peer-reviewed scientific literature.

    Question: {question}
    RAG Answer: {rag_answer}
    Reference Answer (from peer-reviewed literature): {reference_answer}
    Supporting papers: {reference_citations}

    {EVAL_PROMPT}
    """

    response = llm.complete(prompt)
    text = response.text

    # Robust JSON extraction
    try:
        # Try to find JSON block in case model added markdown or preamble
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return cast(dict[str, Any], json.loads(match.group(0)))
        return cast(dict[str, Any], json.loads(text))
    except (json.JSONDecodeError, ValueError) as e:
        raise RuntimeError(f"Failed to parse LLM judge response as JSON: {text}") from e
