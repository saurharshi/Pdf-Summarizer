#import libraries
import os
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
    from langchain.chains.question_answering import load_qa_chain
except ImportError:
    from langchain_classic.chains.question_answering import load_qa_chain

from langchain_google_genai import ChatGoogleGenerativeAI


def process_pdf(pdf):
    """Extracts text page by page with metadata, splits into chunks, and builds FAISS vector store."""
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

    # Load HuggingFace embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create FAISS vector store from document chunks
    knowledgebase = FAISS.from_documents(chunks, embeddings)
    return knowledgebase


def ask_question(knowledgebase, query, api_key=None):
    """Retrieves top-k relevant chunks, computes similarity scores, and generates answer with citations."""
    # Perform similarity search with score (FAISS returns L2 distance)
    results = knowledgebase.similarity_search_with_score(query, k=3)

    docs = [doc for doc, _ in results]

    # Simple RAG evaluation: convert L2 distance to normalized similarity score (0 to 1)
    eval_metrics = []
    sources = []
    for idx, (doc, dist) in enumerate(results):
        page = doc.metadata.get("page", "Unknown")
        if page not in sources:
            sources.append(page)

        # Convert distance to similarity score
        sim_score = max(0.0, min(1.0, 1.0 - (float(dist) ** 2) / 2.0))
        eval_metrics.append({
            "chunk_num": idx + 1,
            "page": page,
            "score": round(sim_score, 2),
            "snippet": doc.page_content[:120].strip()
        })

    # LLM Initialization
    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise ValueError("GEMINI_API_KEY is missing. Please set it in your environment or enter it in the app.")

    llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", google_api_key=key, temperature=0.3)

    # QA chain
    chain = load_qa_chain(llm, chain_type="stuff")

    try:
        answer = chain.run(input_documents=docs, question=query)
    except AttributeError:
        result = chain.invoke({"input_documents": docs, "question": query})
        answer = result.get("output_text", str(result))

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
