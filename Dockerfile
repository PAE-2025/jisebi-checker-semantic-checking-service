# Use official Python image
FROM python:3.11

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model
RUN python -m spacy download en_core_web_sm

ENV TRANSFORMERS_CACHE=/app/model_cache
ENV HF_HOME=/app/model_cache

# Download ML Models
RUN python -c "from transformers import AutoModelForTokenClassification; AutoModelForTokenClassification.from_pretrained('dslim/bert-base-NER', cache_dir='/app/model_cache')"
RUN python -c "from transformers import T5ForConditionalGeneration; T5ForConditionalGeneration.from_pretrained('grammarly/coedit-large', cache_dir='/app/model_cache')"
RUN python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt'); nltk.download('punkt_tab')"


# Copy the FastAPI app files
COPY . .

# Expose port
EXPOSE 8000

# Start the FastAPI application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
