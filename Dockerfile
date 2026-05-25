FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application code
COPY QA.py .

# OPTIONAL: If your model is on Hugging Face, pre-download it during build
# RUN python -c "from transformers import AutoTokenizer, AutoModelForQuestionAnswering; AutoTokenizer.from_pretrained('your-username/my-bert-qa-model'); AutoModelForQuestionAnswering.from_pretrained('your-username/my-bert-qa-model')"

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "QA.py", "--server.port=8501", "--server.address=0.0.0.0"]

