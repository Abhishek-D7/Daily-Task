import streamlit as st
import requests
import json
from deepdiff import DeepDiff

# Set page config for a cleaner look
st.set_page_config(page_title="Document Analyzer", page_icon="📄", layout="centered")

st.title("📄 AI Document Analyzer")
st.markdown("Extract summaries, action items, and key decisions from your meeting notes or documents.")

# Define the API URL
API_URL = "http://127.0.0.1:8000/api/analyze"

# --- Main Input Section ---
st.header("1. Input Document")
input_method = st.radio("Choose input method:", ["Paste Text", "Upload .txt File"])

document_text = ""

if input_method == "Paste Text":
    document_text = st.text_area("Enter your text here:", height=200)
else:
    uploaded_file = st.file_uploader("Choose a .txt file", type="txt")
    if uploaded_file is not None:
        document_text = uploaded_file.getvalue().decode("utf-8")
        st.info("File loaded successfully.")

# --- Prompt Testing Section ---
st.header("2. Prompt Testing (Optional)")
st.markdown("Provide the expected JSON output to validate if the AI extraction is correct.")

with st.expander("Show Expected Output Input"):
    expected_json_str = st.text_area(
        "Expected JSON Output", 
        height=200, 
        help="Paste the exact JSON you expect the AI to return. Must be valid JSON."
    )

# --- Analysis & Validation ---
st.header("3. Analysis")

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
                    result_data = response.json().get("analysis", "")
                    
                    # Display the actual result
                    st.subheader("AI Output")
                    with st.container(border=True):
                        if isinstance(result_data, dict):
                            st.json(result_data)
                        else:
                            st.markdown(result_data)
                    
                    # Perform Comparison if Expected JSON is provided
                    if expected_json_str.strip():
                        st.subheader("Validation Result")
                        try:
                            # Parse expected JSON
                            expected_data = json.loads(expected_json_str)
                            
                            # Ensure result is also a dictionary for comparison
                            if isinstance(result_data, str):
                                try:
                                    actual_data = json.loads(result_data)
                                except json.JSONDecodeError:
                                    st.error("❌ Validation Failed: The AI output is not valid JSON, so it cannot be compared.")
                                    actual_data = None
                            else:
                                actual_data = result_data

                            if actual_data is not None:
                                # Compare JSON using DeepDiff
                                diff = DeepDiff(expected_data, actual_data, ignore_order=True)
                                
                                if not diff:
                                    st.success("✅ **Validation Passed!** The AI output matches the expected output perfectly.")
                                else:
                                    st.error("❌ **Validation Failed!** The AI output differs from the expected output.")
                                    with st.expander("View Differences"):
                                        st.write(diff)
                                        
                                    # Show side-by-side comparison
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.markdown("**Expected:**")
                                        st.json(expected_data)
                                    with col2:
                                        st.markdown("**Actual:**")
                                        st.json(actual_data)
                                        
                        except json.JSONDecodeError as e:
                            st.warning(f"⚠️ Could not validate: The Expected Output you provided is not valid JSON. Error: {e}")
                    
                else:
                    error_detail = response.json().get("detail", "Unknown error occurred.")
                    st.error(f"API Error: {error_detail}")
                    
            except requests.exceptions.ConnectionError:
                st.error("Failed to connect to the API. Is the FastAPI server running?")
