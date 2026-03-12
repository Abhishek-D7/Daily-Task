import streamlit as st
import requests
import json

# Set page config for a cleaner look
st.set_page_config(page_title="Document Analyzer", page_icon="📄", layout="centered")

st.title("📄 AI Document Analyzer")
st.markdown("Extract summaries, action items, and key decisions from your meeting notes or documents.")

# Define the API URL
API_URL = "http://127.0.0.1:8000/api/analyze"

# --- Tabs for Features ---
tab1, tab2 = st.tabs(["📊 Extract Summaries & Data", "❓ Ask Document Questions"])

# --- TAB 1: Data Extraction ---
with tab1:
    st.header("1. Input Document")
    input_method = st.radio("Choose input method:", ["Paste Text", "Upload .txt File"], key="extract_radio")

    document_text = ""

    if input_method == "Paste Text":
        document_text = st.text_area("Enter your text here:", height=200, key="extract_text")
    else:
        uploaded_file = st.file_uploader("Choose a .txt file", type="txt", key="extract_file")
        if uploaded_file is not None:
            document_text = uploaded_file.getvalue().decode("utf-8")
            st.info("File loaded successfully.")

    st.header("2. Prompt Testing (Optional)")
    st.markdown("Provide the expected JSON output to validate if the AI extraction is correct.")

    with st.expander("Show Expected Output Input"):
        expected_json_str = st.text_area(
            "Expected JSON Output", 
            height=200, 
            help="Paste the exact JSON you expect the AI to return. Must be valid JSON.",
            key="extract_expected"
        )

    st.header("3. Analysis")

    if st.button("Analyze Document", type="primary", key="extract_btn"):
        if not document_text.strip():
            st.warning("Please provide some text to analyze.")
        else:
            with st.spinner("Analyzing document with Hugging Face models..."):
                try:
                    response = requests.post(API_URL, json={"text": document_text})
                    
                    if response.status_code == 200:
                        st.success("Analysis Complete!")
                        result_data = response.json().get("analysis", "")
                        
                        st.subheader("AI Output")
                        with st.container(border=True):
                            if isinstance(result_data, dict):
                                st.json(result_data)
                            else:
                                st.markdown(result_data)
                        
                        if expected_json_str.strip():
                            st.subheader("Semantic Validation Result")
                            with st.spinner("Evaluating meaning with AI Judge..."):
                                eval_payload = {
                                    "expected_output": expected_json_str,
                                    "actual_output": json.dumps(result_data) if isinstance(result_data, dict) else result_data
                                }
                                eval_response = requests.post(API_URL.replace("/analyze", "/evaluate"), json=eval_payload)
                                
                                if eval_response.status_code == 200:
                                    eval_result = eval_response.json().get("evaluation", {})
                                    is_match = eval_result.get("is_match", False)
                                    reasoning = eval_result.get("reasoning", "No reasoning provided.")
                                    
                                    if is_match:
                                        st.success(f"✅ **Validation Passed!** {reasoning}")
                                    else:
                                        st.error(f"❌ **Validation Failed!** {reasoning}")
                                        
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.markdown("**Expected:**")
                                        try:
                                            st.json(json.loads(expected_json_str))
                                        except json.JSONDecodeError:
                                            st.markdown(expected_json_str)
                                    with col2:
                                        st.markdown("**Actual:**")
                                        if isinstance(result_data, dict):
                                            st.json(result_data)
                                        else:
                                            st.markdown(result_data)
                                else:
                                    st.error(f"Evaluation API Error: {eval_response.text}")
                        
                    else:
                        error_detail = response.json().get("detail", "Unknown error occurred.")
                        st.error(f"API Error: {error_detail}")
                        
                except requests.exceptions.ConnectionError:
                    st.error("Failed to connect to the API. Is the FastAPI server running?")

# --- TAB 2: Document Q&A ---
with tab2:
    st.header("1. Input Document Context")
    qa_input_method = st.radio("Choose input method:", ["Paste Text", "Upload .txt File"], key="qa_radio")

    qa_document_text = ""

    if qa_input_method == "Paste Text":
        qa_document_text = st.text_area("Enter your context text here:", height=200, key="qa_text")
    else:
        qa_uploaded_file = st.file_uploader("Choose a .txt file", type="txt", key="qa_file")
        if qa_uploaded_file is not None:
            qa_document_text = qa_uploaded_file.getvalue().decode("utf-8")
            st.info("Context file loaded successfully.")

    st.header("2. Ask a Question")
    user_query = st.text_input("What do you want to know from the document?", placeholder="e.g., How many sick days are allowed?")

    if st.button("Ask Question", type="primary", key="qa_btn"):
        if not qa_document_text.strip():
            st.warning("Please provide context text first.")
        elif not user_query.strip():
            st.warning("Please enter a question to ask.")
        else:
            with st.spinner("Searching document with AI..."):
                try:
                    qa_url = API_URL.replace("/analyze", "/qa")
                    response = requests.post(qa_url, json={"text": qa_document_text, "query": user_query})
                    
                    if response.status_code == 200:
                        qa_result_data = response.json().get("qa_result", {})
                        
                        answer = qa_result_data.get("answer", "Unknown")
                        reasoning = qa_result_data.get("confidence_reasoning", "")
                        
                        if answer == "Not in context":
                            st.warning(f"⚠️ **Not found explicitly in the text.**")
                            st.caption(f"*AI Note:* {reasoning}")
                        else:
                            st.success(f"**Answer:** {answer}")
                            st.info(f"**Source check:** {reasoning}")
                            
                        with st.expander("View Raw JSON Output"):
                            st.json(qa_result_data)

                    else:
                        error_detail = response.json().get("detail", "Unknown error occurred.")
                        st.error(f"API Error: {error_detail}")
                        
                except requests.exceptions.ConnectionError:
                    st.error("Failed to connect to the API. Is the FastAPI server running?")
