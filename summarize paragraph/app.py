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
tab1, tab2, tab3 = st.tabs([
    "📊 Extract Summaries & Data", 
    "❓ Ask Document Questions",
    "🎙️ Meeting Intelligence"
])

# --- TAB 1: Data Extraction ---
with tab1:
    st.header("1. Input Document")
    input_method = st.radio("Choose input method:", ["Paste Text", "Upload .txt File"], key="extract_radio")

    document_text = ""

    if input_method == "Paste Text":
        document_text = st.text_area("Enter your text here:", height=200, key="extract_text")
    else:
        st.info("ℹ️ Note: If you upload a document containing images, the images will be ignored. Only text data is extracted and analyzed.")
        uploaded_file = st.file_uploader("Choose a .txt file (Max 2MB)", type=["txt", "doc", "docx"], key="extract_file")
        if uploaded_file is not None:
            if uploaded_file.size > 2 * 1024 * 1024:
                st.error("❌ File size limit exceeded! Maximum allowed size is 2MB. Please upload a smaller file.")
            else:
                try:
                    document_text = uploaded_file.getvalue().decode("utf-8")
                    st.success("File loaded successfully.")
                except UnicodeDecodeError:
                    st.error("Error decoding file. Please ensure it's a valid text document.")

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
        st.info("ℹ️ Note: If you upload a document containing images, the images will be ignored. Only text data is extracted and analyzed.")
        qa_uploaded_file = st.file_uploader("Choose a .txt file (Max 2MB)", type=["txt", "doc", "docx"], key="qa_file")
        if qa_uploaded_file is not None:
            if qa_uploaded_file.size > 2 * 1024 * 1024:
                st.error("❌ File size limit exceeded! Maximum allowed size is 2MB. Please upload a smaller file.")
            else:
                try:
                    qa_document_text = qa_uploaded_file.getvalue().decode("utf-8")
                    st.success("Context file loaded successfully.")
                except UnicodeDecodeError:
                    st.error("Error decoding file. Please ensure it's a valid text document.")

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

# --- TAB 3: Meeting Intelligence ---
with tab3:
    st.header("1. Input Meeting Transcript")
    st.markdown("Paste your meeting notes or upload a transcript file to extract structured insights.")
    
    meeting_input_method = st.radio("Choose input method:", ["Paste Text", "Upload .txt File"], key="meeting_radio", horizontal=True)

    meeting_text = ""

    if meeting_input_method == "Paste Text":
        meeting_text = st.text_area("Enter your meeting transcript here:", height=250, key="meeting_text")
    else:
        st.info("ℹ️ Note: If you upload a document containing images, the images will be ignored. Only text data is extracted and analyzed.")
        meeting_uploaded_file = st.file_uploader("Choose a .txt file (Max 2MB)", type=["txt", "doc", "docx"], key="meeting_file")
        if meeting_uploaded_file is not None:
            if meeting_uploaded_file.size > 2 * 1024 * 1024:
                st.error("❌ File size limit exceeded! Maximum allowed size is 2MB. Please upload a smaller file.")
            else:
                try:
                    meeting_text = meeting_uploaded_file.getvalue().decode("utf-8")
                    st.success("Transcript file loaded successfully.")
                except UnicodeDecodeError:
                    st.error("Error decoding file. Please ensure it's a valid text document.")

    st.header("2. AI Analysis")
    st.markdown("Extract Summary, Action Items, Risks, and Decision Points automatically.")

    if st.button("Generate Meeting Insights", type="primary", key="meeting_btn", use_container_width=True):
        if not meeting_text.strip():
            st.warning("Please provide a meeting transcript to analyze.")
        else:
            with st.spinner("Analyzing meeting transcript with AI..."):
                try:
                    meeting_url = API_URL.replace("/analyze", "/meeting")
                    response = requests.post(meeting_url, json={"transcript": meeting_text})
                    
                    if response.status_code == 200:
                        st.success("Analysis Complete!")
                        result_data = response.json().get("meeting_result", {})
                        
                        # Aesthetic layout using columns and expanders
                        st.subheader("Meeting Overview")
                        st.info(result_data.get("summary", "No summary available."))
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("✅ Action Items")
                            tasks = result_data.get("tasks", [])
                            if tasks:
                                for task in tasks:
                                    st.markdown(f"- {task}")
                            else:
                                st.markdown("*No action items found.*")
                                
                            st.subheader("🤝 Decision Points")
                            decisions = result_data.get("decision_points", [])
                            if decisions:
                                for decision in decisions:
                                    st.markdown(f"- {decision}")
                            else:
                                st.markdown("*No decisions found.*")

                        with col2:
                            st.subheader("⚠️ Risks & Blockers")
                            risks = result_data.get("risks", [])
                            if risks:
                                for risk in risks:
                                    st.error(f"- {risk}")
                            else:
                                st.success("*No risks identified!*")
                                
                        with st.expander("View Raw JSON Output"):
                            st.json(result_data)

                    else:
                        error_detail = response.json().get("detail", "Unknown error occurred.")
                        st.error(f"API Error: {error_detail}")
                        
                except requests.exceptions.ConnectionError:
                    st.error("Failed to connect to the API. Make sure the FastAPI backend is running.")
