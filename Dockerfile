FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -m nltk.downloader punkt stopwords

# Copy application code
COPY . .

EXPOSE 8000

# Command to run the application
CMD ["python", "main.py"]