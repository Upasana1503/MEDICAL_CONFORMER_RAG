import os
from pathlib import Path

from src.search import RAGSearch
from src.eval_rag import credibility_for_answer
from src.audio_to_json_chunks import chunk_audio_to_json

if __name__ == "__main__":
    print("🎤 Speech → RAG Application (Conformer-CTC)")

    audio_path = "audio/patient_diagnosis.wav"
    json_output_path = "data/audio_chunks/patient_diagnosis_chunks.json"
    json_data_dir = str(Path(json_output_path).parent)

    # 1) Convert input audio into chunked JSONL for ingestion
    rows = chunk_audio_to_json(
        audio_path=audio_path,
        output_path=json_output_path,
        chunk_sec=30.0,
        overlap_sec=2.0,
    )
    print(f"[INFO] JSON chunks written: {len(rows)} -> {json_output_path}")

    # 2) Keep existing RAG flow, but index the generated JSON chunks
    rag_search = RAGSearch(data_dir=json_data_dir, force_rebuild=True)

    print("\n✅ Ready. Ask questions about the uploaded audio transcription.")
    print("Type 'exit' to stop.")

    while True:
        query = input("\nYour question: ").strip()
        if not query:
            print("Please enter a question.")
            continue
        if query.lower() in {"exit", "quit"}:
            print("Exiting.")
            break

        answer, context = rag_search.search_and_summarize(query, top_k=3, return_context=True)

        print("\n🤖 Answer:")
        print(answer)

        # Credibility metrics (lightweight by default)
        use_llm = os.getenv("RAG_CREDIBILITY_USE_LLM", "0") == "1"
        metrics = credibility_for_answer(
            query=query,
            answer=answer,
            context=context,
            embed_model=rag_search.vectorstore.model,
            llm=rag_search.llm if use_llm else None,
        )

        print("\n✅ Credibility:")
        print(f"- Groundedness rate: {metrics['groundedness_rate']:.2f}")
        print(f"- Abstained: {bool(metrics['abstained'])}")
        print(f"- Sentences evaluated: {metrics['sentences']}")
