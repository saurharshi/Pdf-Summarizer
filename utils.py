#import libraries
import os
from dotenv import load_dotenv
load_dotenv()

from pypdf import PdfReader
from langchain_core.documents import Document

try:
    from langchain_text_splitters import CharacterTextSplitter
except ImportError:
    from langchain.text_splitter import CharacterTextSplitter

try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
except ImportError:
    from langchain.embeddings import HuggingFaceEmbeddings

try:
    from langchain_community.vectorstores import FAISS
except ImportError:
    from langchain.vectorstores import FAISS

try:
    from langchain_community.vectorstores.utils import DistanceStrategy
except ImportError:
    try:
        from langchain.vectorstores.utils import DistanceStrategy
    except ImportError:
        DistanceStrategy = None

try:
    from langchain.chains.question_answering import load_qa_chain
except ImportError:
    from langchain_classic.chains.question_answering import load_qa_chain

from langchain_google_genai import ChatGoogleGenerativeAI


def process_pdf(pdf):
    """Extracts text page by page with metadata, splits into chunks, and builds FAISS vector store with Cosine Similarity."""
    pdf_reader = PdfReader(pdf)
    documents = []

    # Extract text with page numbers as metadata
    for i, page in enumerate(pdf_reader.pages):
        page_text = page.extract_text() or ""
        if page_text.strip():
            documents.append(Document(page_content=page_text, metadata={"page": f"Page {i + 1}"}))

    if not documents:
        return None

    # Split the text into smaller chunks
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)

    # Load HuggingFace embeddings (unit-normalized)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create FAISS vector store using MAX_INNER_PRODUCT (exact Cosine Similarity for normalized vectors)
    kwargs = {}
    if DistanceStrategy is not None:
        kwargs["distance_strategy"] = DistanceStrategy.MAX_INNER_PRODUCT

    knowledgebase = FAISS.from_documents(chunks, embeddings, **kwargs)
    return knowledgebase


def ask_question(knowledgebase, query, api_key=None):
    """Retrieves top-k relevant chunks, computes similarity scores, and generates answer with citations."""
    results = knowledgebase.similarity_search_with_score(query, k=3)

    docs = [doc for doc, _ in results]

    # RAG Evaluation: calculate true Cosine Similarity scores
    eval_metrics = []
    sources = []
    for idx, (doc, raw_score) in enumerate(results):
        page = doc.metadata.get("page", "Unknown")
        raw_val = float(raw_score)

        # For normalized embeddings with MAX_INNER_PRODUCT, raw_score is already cosine similarity (-1.0 to 1.0)
        # If legacy L2 distance (where dist = 2 - 2*cos), cos = 1 - dist/2
        if raw_val > 1.0:
            cos_sim = max(0.0, min(1.0, 1.0 - raw_val / 2.0))
        else:
            cos_sim = max(0.0, min(1.0, raw_val))

        # Cite source page if relevance meets threshold (or top-ranked chunk)
        if (cos_sim >= 0.25 or idx == 0) and page not in sources:
            sources.append(page)

        eval_metrics.append({
            "chunk_num": idx + 1,
            "page": page,
            "score": round(cos_sim, 2),
            "snippet": doc.page_content[:120].strip()
        })

    # LLM Initialization with high-quota models and fallback
    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise ValueError("GEMINI_API_KEY is missing. Please set it in your environment or enter it in the app.")

    candidate_models = ["gemini-flash-lite-latest", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-3.8-flash"]
    answer = None
    last_err = None

    for model_name in candidate_models:
        try:
            llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=key, temperature=0.3)
            chain = load_qa_chain(llm, chain_type="stuff")
            try:
                answer = chain.run(input_documents=docs, question=query)
            except AttributeError:
                result = chain.invoke({"input_documents": docs, "question": query})
                answer = result.get("output_text", str(result))
            if answer:
                break
        except Exception as e:
            last_err = e
            continue

    if answer is None:
        raise last_err or RuntimeError("Failed to generate response from available Gemini models.")

    return {
        "answer": answer,
        "sources": sources,
        "metrics": eval_metrics
    }


def summarizer(pdf, api_key=None):
    """Backward compatible summarizer helper."""
    knowledgebase = process_pdf(pdf)
    if not knowledgebase:
        return "No readable text found in the PDF. Please ensure the PDF is not an image-only scan."
    result = ask_question(knowledgebase, "Summarize the content of the PDF in appropriately 3-5 sentences.", api_key=api_key)
    return result["answer"]
