#import libraries
import os
from pypdf import PdfReader

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


def process_text(text):

    # Split the text into smaller chunks for processing
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200, #overlap of 200 characters between chunks to maintain context
        length_function=len
    )
    chunks = text_splitter.split_text(text)  #split the text into chunks

    #load a model for generating embeddings from huggingface
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    #create a FAISS vector store from the text chunks and their embeddings
    knowledgebase = FAISS.from_texts(chunks, embeddings)
    return  knowledgebase

def summarizer(pdf):
    # if a pdf file is provided

    #read the pdf file 
    pdf_reader = PdfReader(pdf)
    text = ""
    #extract text from each page of the pdf
    for page in pdf_reader.pages:
        text += page.extract_text() or ""  #handle cases where extract_text() returns None

    if not text.strip():
        return "No readable text found in the PDF. Please ensure the PDF is not an image-only scan."

    #proccess the extracted text to create a knowledge base
    knowledgebase = process_text(text)

    #define the query for summarization
    query = "Summarize the content of the PDF in appropriately 3-5 sentences"

    if query :
        #perform a similarity search in the knowledge base using the query to retrieve relevant chunks of text
        docs = knowledgebase.similarity_search(query)

        #specify the model to use for generating the summary
        api_key = os.environ.get("GEMINI_API_KEY")
        llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", google_api_key=api_key, temperature=0.5)

        #load a question-answering chain with the specified model
        chain = load_qa_chain(llm, chain_type="stuff")

        #run the chain on the retrieved documents and the query to generate a summary
        try:
            response = chain.run(input_documents=docs, question=query)
        except AttributeError:
            result = chain.invoke({"input_documents": docs, "question": query})
            response = result.get("output_text", str(result))

        return response #return the generated summary
