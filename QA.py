import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline

# --- Configuration ---
MODEL_SAVE_PATH = "./my_bert_qa_model" # Ensure this matches where you saved your model
# Determine device (CPU or GPU) for inference
# -1 for CPU, 0 for GPU (if multiple GPUs, specify the index)
PIPELINE_DEVICE = 0 if torch.cuda.is_available() else -1

# --- Streamlit App Title and Description ---
st.set_page_config(page_title="Extractive QA with BERT", page_icon="📚")
st.title("📚 Extractive Question Answering with BERT")
st.markdown("""
This application uses a fine-tuned BERT model to answer questions based on a given context.
The model will extract the most relevant span of text from the context as the answer.
""")

# --- Model Loading (Cached for Performance) ---
# Use st.cache_resource to load the model only once when the app starts
@st.cache_resource
def load_qa_model():
    """Loads the fine-tuned BERT QA model and tokenizer."""
    try:
        st.spinner("Loading model and tokenizer... This might take a moment.")
        qa_model = AutoModelForQuestionAnswering.from_pretrained(MODEL_SAVE_PATH)
        qa_tokenizer = AutoTokenizer.from_pretrained(MODEL_SAVE_PATH)
        qa_pipeline = pipeline(
            "question-answering",
            model=qa_model,
            tokenizer=qa_tokenizer,
            device=PIPELINE_DEVICE
        )
        st.success("Model loaded successfully!")
        return qa_pipeline
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop() # Stop the app if model loading fails

qa_pipeline = load_qa_model()

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
        with st.spinner("Finding answer..."):
            try:
                # Perform prediction
                result = qa_pipeline(question=question_input, context=context_input)

                # Display results
                st.subheader("Answer:")
                st.success(f"**{result['answer']}**")
                st.info(f"Confidence Score: {result['score']:.4f}")

                # Optional: Highlight answer in context
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
                    st.markdown(context_input) # Fallback if highlight fails

            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")

# --- Footer ---
st.markdown("---")

