import os
from dotenv import load_dotenv
from src.vectorstore import FaissVectorStore
from langchain_groq import ChatGroq

load_dotenv()

CLINICAL_RAG_PROMPT_TEMPLATE = (
    "ROLE:\n"
    "You are a clinical medical assistant.\n\n"
    "CONTEXT:\n"
    "You are provided excerpts from a patient's consultation history.\n\n"
    "OBJECTIVE:\n"
    "Answer the doctor's question using only the provided patient records.\n\n"
    "STRICT RULES:\n"
    "- Use only information explicitly present in the retrieved text.\n"
    "- You may logically compare or contrast the retrieved information with the question.\n"
    "- Do NOT add new medical knowledge beyond what is in the records.\n"
    "- If the topic is not mentioned at all in the retrieved text, respond exactly with:\n"
    "  \"No relevant information found in consultation history.\"\n"
    "- Be concise and clinically precise.\n\n"
    "Retrieved Consultation Excerpts:\n"
    "{retrieved_context}\n\n"
    "Doctor's Question:\n"
    "{question}\n\n"
    "Answer:"
)

class RAGSearch:
    def __init__(
        self,
        persist_dir: str = "faiss_store",
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_model: str = "llama-3.1-8b-instant",
        data_dir: str = "data",
        force_rebuild: bool = False,
    ):
        self.vectorstore = FaissVectorStore(persist_dir, embedding_model)
        # Load or build vectorstore
        faiss_path = os.path.join(persist_dir, "faiss.index")
        meta_path = os.path.join(persist_dir, "metadata.pkl")
        if force_rebuild or not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            from src.data_loader import load_all_documents
            docs = load_all_documents(data_dir)
            self.vectorstore.build_from_documents(docs)
        else:
            self.vectorstore.load()
        groq_api_key = ""
        self.llm = ChatGroq(groq_api_key=groq_api_key, model_name=llm_model)
        print(f"[INFO] Groq LLM initialized: {llm_model}")

    def search_and_summarize(self, query: str, top_k: int = 5, return_context: bool = False):
        results = self.vectorstore.query(query, top_k=top_k)
        texts = []
        for r in results:
            meta = r.get("metadata") or {}
            text = meta.get("text") or meta.get("summary") or ""
            if text:
                texts.append(text)
        context = "\n\n".join(texts)
        if not context:
            fallback = "No relevant information found in consultation history."
            return (fallback, "") if return_context else fallback
        prompt = CLINICAL_RAG_PROMPT_TEMPLATE.format(
            retrieved_context=context,
            question=query,
        )
        response = self.llm.invoke([prompt])
        return (response.content, context) if return_context else response.content
