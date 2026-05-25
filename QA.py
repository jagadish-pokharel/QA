import os
import streamlit as st
from dotenv import load_dotenv
# 👇 FIXED: Uses the native Hugging Face Client instead of manual requests URLs
from huggingface_hub import InferenceClient

# --- Environment Setup ---
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    st.error("Missing 'HF_TOKEN'. Please ensure it is set in your .env file.")
    st.stop()

# 👇 FIXED: Initialize the verified client
client = InferenceClient(token=HF_TOKEN)
MODEL_ID = "jaagguu/my-bert-qa-model"

# --- Streamlit App Title and Description ---
st.set_page_config(page_title="Extractive QA with BERT", page_icon="📚")
st.title("📚 Extractive Question Answering with BERT")
st.markdown("""
This application uses your fine-tuned BERT model hosted via Hugging Face to extract answers based on a given context.
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
        with st.spinner("Querying Hugging Face Infrastructure..."):
            try:
                # 👇 FIXED: Using the SDK's built-in task handler.
                # This automatically bypasses dead URLs and hits their modern routing layer perfectly.
                result = client.question_answering(
                    question=question_input,
                    context=context_input,
                    model=MODEL_ID
                )
                
                # Check for standard pipeline dictionary outputs safely
                if result and hasattr(result, "answer"):
                    st.subheader("Answer:")
                    st.success(f"**{result.answer}**")
                    st.info(f"Confidence Score: {result.score:.4f}")

                    # Highlight answer in context
                    st.subheader("Context with Answer Highlighted:")
                    answer_start = context_input.find(result.answer)
                    answer_end = answer_start + len(result.answer)

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
                    st.error(f"Unexpected response structure: {result}")

            except Exception as e:
                # Handle model wake-up cold starts gracefully
                if "503" in str(e) or "loading" in str(e).lower():
                    st.info("Model is waking up on Hugging Face servers. Please wait a moment and try again.")
                else:
                    st.error(f"An error occurred during prediction: {e}")

# --- Footer ---
st.markdown("---")