import streamlit as st
import requests

# Set page config for a cleaner look
st.set_page_config(page_title="Document Analyzer", page_icon="📄", layout="centered")

st.title("📄 AI Document Analyzer")
st.markdown("Extract summaries, action items, and key decisions from your meeting notes or documents.")

# Define the API URL
API_URL = "http://127.0.0.1:8000/api/analyze"

# Input methods: Text area or File upload
input_method = st.radio("Choose input method:", ["Paste Text", "Upload .txt File"])

document_text = ""

if input_method == "Paste Text":
    document_text = st.text_area("Enter your text here:", height=200)
else:
    uploaded_file = st.file_uploader("Choose a .txt file", type="txt")
    if uploaded_file is not None:
        document_text = uploaded_file.getvalue().decode("utf-8")
        st.info("File loaded successfully.")

# Submit button
if st.button("Analyze Document", type="primary"):
    if not document_text.strip():
        st.warning("Please provide some text to analyze.")
    else:
        with st.spinner("Analyzing document with Hugging Face models..."):
            try:
                # Send POST request to the FastAPI backend
                response = requests.post(API_URL, json={"text": document_text})
                
                if response.status_code == 200:
                    st.success("Analysis Complete!")
                    result = response.json().get("analysis", "")
                    
                    # Display the result in a nice container
                    with st.container(border=True):
                        st.markdown(result)
                else:
                    error_detail = response.json().get("detail", "Unknown error occurred.")
                    st.error(f"API Error: {error_detail}")
                    
            except requests.exceptions.ConnectionError:
                st.error("Failed to connect to the API. Is the FastAPI server running?")