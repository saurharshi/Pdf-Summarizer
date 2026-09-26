import streamlit as st 
## This is a simple Streamlit app that demonstrates the use of various Streamlit components.frontend
import os
from utils import summarizer

# main function to run the Streamlit app
def main():
    #set page configurations
    st.set_page_config(page_title="PDF Summarizer")

    # Title of the app
    st.title("PDF Summarizing App")
    st.write("Upload a PDF file and get a summary of its content.")
    st.divider()  #insert a divider for better visual separation

    # File uploader for PDF files
    pdf = st.file_uploader("Choose a PDF file", type="pdf")

    #creating a button for user to submit their pdf for summarization
    submit = st.button(" Generate Summary")

    #setting the Gemini API key from the environment variable
    os.environ["GEMINI_API_KEY"] = "AQ.Ab8RN6IccbFfBMe932prEBF5bnxmC1plxE4T_hfU31dGayKw9g"

    #if the submit button is pressed 
    if submit:
        if pdf is not None:
            with st.spinner("Generating summary..."):
                #calling the summarizer function from utils.py to process the uploaded pdf and generate a summary
                response = summarizer(pdf)

                #display the generated summary in a text area for user to read
                st.subheader("Summary of the PDF")
                st.write(response)  #display the summary in the Streamlit app
        else:
            st.warning("Please upload a PDF file first.")


#python script execution starts here    
if __name__ == "__main__":
    main() #calling the main function to run the app
