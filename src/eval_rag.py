import argparse
import os
import re
from itertools import combinations
from typing import List, Dict, Tuple

import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from langchain_groq import ChatGroq
from src.vectorstore import FaissVectorStore


def load_queries(path: str) -> List[str]:
    queries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            queries.append(line)
    return queries


def simple_paraphrases(query: str, n: int) -> List[str]:
    templates = [
        "Explain: {q}",
        "Summarize: {q}",
        "What does this mean: {q}",
        "Give a short answer for: {q}",
        "Answer concisely: {q}",
    ]
    out = [query]
    for t in templates:
        if len(out) >= n:
            break
        out.append(t.format(q=query))
    return out[:n]


def llm_paraphrases(llm, query: str, n: int) -> List[str]:
    prompt = (
        "Generate {n} concise paraphrases of the query below. "
        "Return one paraphrase per line, no numbering.\n\n"
        "Query: {q}"
    ).format(n=n, q=query)
    resp = llm.invoke([prompt]).content
    lines = [l.strip("- ").strip() for l in resp.splitlines() if l.strip()]
    lines = [l for l in lines if l and l.lower() != query.lower()]
    # Ensure original query first
    result = [query] + lines
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for l in result:
        key = l.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(l)
    return deduped[:n]


def jaccard(a: List[int], b: List[int]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / max(1, len(sa | sb))


def split_sentences(text: str) -> List[str]:
    # Naive sentence split
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def answer_with_context(llm, query: str, context: str) -> str:
    prompt = (
        "Answer the query using ONLY the context. "
        "If the context does not contain enough information, reply: Not enough info.\n\n"
        "Query: {q}\n\nContext:\n{ctx}\n\nAnswer:"
    ).format(q=query, ctx=context)
    return llm.invoke([prompt]).content.strip()


def judge_groundedness_llm(llm, context: str, sentences: List[str]) -> List[bool]:
    if not sentences:
        return []
    joined = "\n".join([f"{i+1}. {s}" for i, s in enumerate(sentences)])
    prompt = (
        "Given the context and the sentences, label each sentence as SUPPORTED "
        "only if it is directly supported by the context. Otherwise label UNSUPPORTED.\n\n"
        "Context:\n{ctx}\n\nSentences:\n{sent}\n\n"
        "Return one label per line in the same order, using only SUPPORTED or UNSUPPORTED."
    ).format(ctx=context, sent=joined)
    resp = llm.invoke([prompt]).content
    labels = [l.strip().upper() for l in resp.splitlines() if l.strip()]
    results = []
    for l in labels:
        if "SUPPORTED" in l and "UNSUPPORTED" not in l:
            results.append(True)
        elif "UNSUPPORTED" in l:
            results.append(False)
    # Fallback: if parsing failed, mark all as unsupported
    if len(results) != len(sentences):
        return [False] * len(sentences)
    return results


def judge_groundedness_embed(model, context: str, sentences: List[str], threshold: float = 0.3) -> List[bool]:
    if not sentences:
        return []
    ctx_emb = model.encode([context], normalize_embeddings=True)[0]
    sent_embs = model.encode(sentences, normalize_embeddings=True)
    sims = np.dot(sent_embs, ctx_emb)
    return [s >= threshold for s in sims]


def credibility_for_answer(
    query: str,
    answer: str,
    context: str,
    embed_model: SentenceTransformer,
    llm=None,
) -> Dict[str, float]:
    sentences = split_sentences(answer)
    abstained = answer.strip().lower().startswith("not enough info")
    if llm is not None:
        grounded = judge_groundedness_llm(llm, context, sentences)
    else:
        grounded = judge_groundedness_embed(embed_model, context, sentences)
    grounded_rate = float(np.mean(grounded)) if grounded else 0.0
    return {
        "groundedness_rate": grounded_rate,
        "abstained": float(abstained),
        "sentences": len(sentences),
    }


def average_pairwise_cosine(vecs: np.ndarray) -> float:
    if len(vecs) < 2:
        return 1.0
    sims = []
    for i, j in combinations(range(len(vecs)), 2):
        sims.append(float(np.dot(vecs[i], vecs[j])))
    return float(np.mean(sims)) if sims else 1.0


def main():
    parser = argparse.ArgumentParser(description="RAG credibility evaluation (label-free).")
    parser.add_argument("--queries", required=True, help="Path to queries .txt (one per line).")
    parser.add_argument("--top-k", type=int, default=5, help="Top-k docs to retrieve.")
    parser.add_argument("--paraphrases", type=int, default=3, help="Paraphrases per query (including original).")
    parser.add_argument("--persist-dir", default="faiss_store", help="FAISS store directory.")
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2", help="Embedding model.")
    parser.add_argument("--llm-model", default="llama-3.1-8b-instant", help="Groq LLM model.")
    parser.add_argument("--use-llm", action="store_true", help="Use LLM for paraphrases/groundedness/answers.")
    args = parser.parse_args()

    load_dotenv()
    groq_api_key = os.getenv("GROQ_API_KEY", "")

    llm = None
    if args.use_llm:
        if not groq_api_key:
            raise RuntimeError("GROQ_API_KEY is not set. Export it or run without --use-llm.")
        llm = ChatGroq(groq_api_key=groq_api_key, model_name=args.llm_model)

    store = FaissVectorStore(args.persist_dir, args.embedding_model)
    store.load()
    try:
        embed_model = SentenceTransformer(args.embedding_model)
    except Exception as e:
        print(f"[WARN] Online model load failed ({e}). Falling back to local cache only.")
        embed_model = SentenceTransformer(args.embedding_model, local_files_only=True)

    queries = load_queries(args.queries)
    if not queries:
        raise RuntimeError("No queries found.")

    retrieval_overlaps = []
    grounded_rates = []
    abstain_flags = []
    consistency_scores = []

    for q in queries:
        if llm:
            variants = llm_paraphrases(llm, q, args.paraphrases)
        else:
            variants = simple_paraphrases(q, args.paraphrases)

        # Retrieval stability
        retrieved_indices = []
        retrieved_texts = []
        for v in variants:
            results = store.query(v, top_k=args.top_k)
            retrieved_indices.append([r["index"] for r in results])
            retrieved_texts.append([r["metadata"].get("text", "") for r in results if r["metadata"]])

        pairwise = []
        for i in range(len(retrieved_indices)):
            for j in range(i + 1, len(retrieved_indices)):
                pairwise.append(jaccard(retrieved_indices[i], retrieved_indices[j]))
        retrieval_overlaps.append(float(np.mean(pairwise)) if pairwise else 1.0)

        # Build context from the original query retrieval
        context = "\n\n".join(retrieved_texts[0])

        # Answer + groundedness
        if llm:
            answer = answer_with_context(llm, q, context)
            abstain_flags.append(answer.strip().lower().startswith("not enough info"))
            sentences = split_sentences(answer)
            grounded = judge_groundedness_llm(llm, context, sentences)
            grounded_rates.append(float(np.mean(grounded)) if grounded else 0.0)
        else:
            # No LLM: use an embedding-based proxy
            abstain_flags.append(False)
            sentences = split_sentences(" ".join(variants))
            grounded = judge_groundedness_embed(embed_model, context, sentences)
            grounded_rates.append(float(np.mean(grounded)) if grounded else 0.0)

        # Answer consistency across paraphrases
        if llm:
            answers = []
            for v in variants:
                ctx = "\n\n".join(retrieved_texts[variants.index(v)])
                answers.append(answer_with_context(llm, v, ctx))
            ans_embs = embed_model.encode(answers, normalize_embeddings=True)
            consistency_scores.append(average_pairwise_cosine(ans_embs))
        else:
            # Proxy using retrieved text overlap if LLM is not used
            consistency_scores.append(retrieval_overlaps[-1])

    report = {
        "queries": len(queries),
        "retrieval_stability": float(np.mean(retrieval_overlaps)),
        "groundedness_rate": float(np.mean(grounded_rates)),
        "abstention_rate": float(np.mean(abstain_flags)) if abstain_flags else 0.0,
        "answer_consistency": float(np.mean(consistency_scores)),
    }

    print("\n=== RAG Credibility Report (Label-Free) ===")
    for k, v in report.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")


if __name__ == "__main__":
    main()
