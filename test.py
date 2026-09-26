import streamlit as st
import os
from utils import process_pdf, ask_question, summarizer

# main function to run the Streamlit app
def main():
    # Set page configurations
    st.set_page_config(page_title="PDF Q&A & Summarizer", page_icon="📄")

    # Title of the app
    st.title("📄 PDF Q&A and Summarizing App")
    st.write("Upload a PDF to ask questions with source citations or generate a summary.")
    st.divider()

    # File uploader for PDF files
    pdf = st.file_uploader("Choose a PDF file", type="pdf")

    # Retrieve the Gemini API key safely from environment variable or Streamlit secrets
    if "GEMINI_API_KEY" not in os.environ:
        try:
            os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]
        except Exception:
            pass

    if pdf is not None:
        # Cache the processed PDF vector store in session state so it doesn't re-index on every query
        if "current_pdf" not in st.session_state or st.session_state.current_pdf != pdf.name:
            with st.spinner("Processing PDF and building vector knowledge base..."):
                st.session_state.knowledgebase = process_pdf(pdf)
                st.session_state.current_pdf = pdf.name
            st.success("✅ PDF processed successfully!")

        st.divider()

        # Feature 1 & 2 & 3: Document Q&A with Sources & Simple RAG Evaluation
        st.subheader("💬 Ask Questions from the Document")
        user_question = st.text_input("Enter your question:", placeholder="e.g. Explain PCA in simple terms")
        ask_btn = st.button("Get Answer")

        if ask_btn and user_question:
            if not os.environ.get("GEMINI_API_KEY"):
                st.error("Gemini API key is missing. Please set it in your environment or secrets.")
            else:
                with st.spinner("Retrieving relevant chunks and generating answer..."):
                    result = ask_question(st.session_state.knowledgebase, user_question)

                    # 1. Answer
                    st.markdown("### Answer:")
                    st.write(result["answer"])

                    # 2. Source / Citation Display
                    st.markdown("### 📌 Sources:")
                    if result["sources"]:
                        st.write(", ".join(result["sources"]))
                    else:
                        st.write("No specific page identified.")

                    # 3. Simple RAG Evaluation
                    st.markdown("### 📊 RAG Evaluation:")
                    st.write(f"**Retrieved Chunks:** {len(result['metrics'])}")
                    st.write("**Similarity Scores:**")
                    for m in result["metrics"]:
                        st.write(f"- Chunk {m['chunk_num']} ({m['page']}) → **{m['score']}**")

        st.divider()

        # Summarize option
        st.subheader("📝 Or Generate a Summary")
        if st.button("Generate Summary"):
            if not os.environ.get("GEMINI_API_KEY"):
                st.error("Gemini API key is missing. Please set it in your environment or secrets.")
            else:
                with st.spinner("Generating summary..."):
                    summary_result = ask_question(
                        st.session_state.knowledgebase,
                        "Summarize the key points of the PDF in appropriately 3-5 sentences."
                    )
                    st.markdown("### Summary:")
                    st.write(summary_result["answer"])
                    st.markdown("### 📌 Sources:")
                    st.write(", ".join(summary_result["sources"]))


# python script execution starts here    
if __name__ == "__main__":
    main()
