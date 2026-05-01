EVAL_PROMPT = """
Score the RAG answer on the following. Respond as JSON.

   1. factual_alignment (0-1):
      Does the RAG answer agree with the peer-reviewed reference? Consider the claims in the 
      reference answer and the supporting papers. Are the claims in the RAG answer supported by 
      the reference and its citations?

   2. fluency_coherence_brevity (0-1):
      Is the RAG answer well-written and logically structured including clear headers, appropriate 
      use of formatting (eg. bold, italics) for emphasis?
      Brevity: Does the RAG answer avoid unnecessary verbosity and include only relevant information?

   3. completeness_and_relevance (0-1):
      Does the RAG answer cover the key scientific details present in the reference, including:
         - key findings, methodologies, and limitations?
         - Use of LaTeX for equations where appropriate?
         - Python code snippets for clarity where appropriate?
      Note: the reference answer may contain tables rendered as flat text due to copy/paste 
      formatting loss. Evaluate based on factual content only.

   4. contradiction_detected (Binary: [0, 1]):
      Does the RAG answer contradict any cited finding? 

   5. verdict: "better" | "equivalent" | "worse"
      Overall RAG quality vs. the reference answer.

    {{"factual_alignment": X, "fluency_coherence_brevity": X,
      "completeness_and_relevance": X, "contradiction_detected": X,
      "verdict": "..."}}
"""


def llm_judge(question, rag_answer, reference_answer, reference_citations):
    prompt = f"""
    You are evaluating a RAG system's answer against a reference answer
    derived from peer-reviewed scientific literature.

    Question: {question}
    RAG Answer: {rag_answer}
    Reference Answer (from peer-reviewed literature): {reference_answer}
    Supporting papers: {reference_citations}

    {EVAL_PROMPT}
    """
    return your_llm_client.judge(prompt)
