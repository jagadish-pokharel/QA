import streamlit as st
import requests
import os
# 👇 FIXED: Added the missing import for load_dotenv
from dotenv import load_dotenv

# Initialize environment variables from .env file
load_dotenv()

# --- Configuration ---
API_URL = "https://api-inference.huggingface.co/models/jaagguu/my-bert-qa-model"

# Fetch the token from the environment variable
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    st.error("Missing 'HF_TOKEN'. Please ensure it is set in your .env file or server environment variables.")
    st.stop()

headers = {"Authorization": f"Bearer {HF_TOKEN}"}

# --- Streamlit App Title and Description ---
st.set_page_config(page_title="Extractive QA with BERT", page_icon="📚")
st.title("📚 Extractive Question Answering with BERT")
st.markdown("""
This application uses a fine-tuned BERT model hosted via Hugging Face Inference API to answer questions based on a given context.
The model will extract the most relevant span of text from the context as the answer.
""")

# --- User Input ---
st.header("Provide Context and Question")

context_input = st.text_area(
    "**Context:** Paste the text you want to ask questions about here.",
    "France is a country located in Western Europe. Its capital and largest city is Paris. Paris is known for its art, fashion, gastronomy, and culture. The Eiffel Tower is a famous landmark in Paris.",
    height=200
)

question_input = st.text_input(
    "**Question:** Enter your question based on the context.",
    "What is the capital of France?"
)

# --- Prediction Button ---
if st.button("Get Answer"):
    if not context_input.strip() or not question_input.strip():
        st.warning("Please provide both context and a question.")
    else:
        with st.spinner("Querying Hugging Face Inference API..."):
            try:
                # Structure the JSON payload as expected by Hugging Face's Question-Answering pipeline
                payload = {
                    "inputs": {
                        "question": question_input,
                        "context": context_input
                    }
                }

                response = requests.post(API_URL, headers=headers, json=payload)
                
                # Check if the model is currently loading (Cold start)
                if response.status_code == 503:
                    error_data = response.json()
                    estimated_time = error_data.get("estimated_time", 20)
                    st.info(f"Model is waking up on Hugging Face servers. Estimated load time: {estimated_time:.1f}s. Please wait a moment and try again.")
                
                elif response.status_code == 200:
                    result = response.json()
                    
                    # Safety check for expected API output keys
                    if "answer" in result:
                        # Display results
                        st.subheader("Answer:")
                        st.success(f"**{result['answer']}**")
                        st.info(f"Confidence Score: {result['score']:.4f}")

                        # Highlight answer in context
                        st.subheader("Context with Answer Highlighted:")
                        answer_start = context_input.find(result['answer'])
                        answer_end = answer_start + len(result['answer'])

                        if answer_start != -1:
                            highlighted_context = (
                                context_input[:answer_start]
                                + "<mark>"
                                + context_input[answer_start:answer_end]
                                + "</mark>"
                                + context_input[answer_end:]
                            )
                            st.markdown(highlighted_context, unsafe_allow_html=True)
                        else:
                            st.markdown(context_input)
                    else:
                        st.error(f"Unexpected response format from API: {result}")
                        
                else:
                    st.error(f"API Error ({response.status_code}): {response.text}")

            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")

# --- Footer ---
st.markdown("---")