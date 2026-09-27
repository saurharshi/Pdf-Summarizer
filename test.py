import streamlit as st
import os
from dotenv import load_dotenv
from utils import process_pdf, ask_question, summarizer

load_dotenv()

# main function to run the Streamlit app
def main():
    # Set page configurations
    st.set_page_config(page_title="PDF Q&A & Summarizer", page_icon="📄")

    # Title of the app
    st.title("📄 PDF Q&A and Summarizing App")
    st.write("Upload a PDF to ask questions with source citations or generate a summary.")
    st.divider()

    # Sidebar for API Key setting
    with st.sidebar:
        st.header("⚙️ Settings")
        env_key = os.environ.get("GEMINI_API_KEY", "")
        if not env_key:
            try:
                env_key = st.secrets.get("GEMINI_API_KEY", "")
            except Exception:
                pass

        user_api_key = st.text_input(
            "Gemini API Key",
            type="password",
            value=env_key,
            placeholder="Paste your key here",
            help="Get your free key from https://aistudio.google.com/"
        )
        if user_api_key:
            os.environ["GEMINI_API_KEY"] = user_api_key

    # File uploader for PDF files
    pdf = st.file_uploader("Choose a PDF file", type="pdf")

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
            active_key = os.environ.get("GEMINI_API_KEY")
            if not active_key:
                st.error("🔑 Gemini API key is missing. Please enter your API key in the left sidebar.")
            else:
                with st.spinner("Retrieving relevant chunks and generating answer..."):
                    try:
                        result = ask_question(st.session_state.knowledgebase, user_question, api_key=active_key)

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
                    except Exception as e:
                        err_str = str(e)
                        if "401" in err_str or "UNAUTHENTICATED" in err_str:
                            st.error("❌ Authentication Error (401): Your Gemini API Key has been revoked or is invalid. Please create a new key at https://aistudio.google.com/ and paste it in the sidebar on the left.")
                        else:
                            st.error(f"❌ Error: {err_str}")

        st.divider()

        # Summarize option
        st.subheader("📝 Or Generate a Summary")
        if st.button("Generate Summary"):
            active_key = os.environ.get("GEMINI_API_KEY")
            if not active_key:
                st.error("🔑 Gemini API key is missing. Please enter your API key in the left sidebar.")
            else:
                with st.spinner("Generating summary..."):
                    try:
                        summary_result = ask_question(
                            st.session_state.knowledgebase,
                            "Summarize the key points of the PDF in appropriately 3-5 sentences.",
                            api_key=active_key
                        )
                        st.markdown("### Summary:")
                        st.write(summary_result["answer"])
                        st.markdown("### 📌 Sources:")
                        st.write(", ".join(summary_result["sources"]))
                    except Exception as e:
                        err_str = str(e)
                        if "401" in err_str or "UNAUTHENTICATED" in err_str:
                            st.error("❌ Authentication Error (401): Your Gemini API Key has been revoked or is invalid. Please create a new key at https://aistudio.google.com/ and paste it in the sidebar on the left.")
                        else:
                            st.error(f"❌ Error: {err_str}")


# python script execution starts here    
if __name__ == "__main__":
    main()
