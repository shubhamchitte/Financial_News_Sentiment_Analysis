# Financial News Sentiment Analysis API

<div align="center">
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Hugging_Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="Hugging Face">
  <img src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker">
  <img src="https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=git&logoColor=white" alt="Git">
  <img src="https://img.shields.io/badge/NLTK-154F5B?style=for-the-badge&logo=python&logoColor=white" alt="NLTK">
</div>

## Project Overview

This project provides an end-to-end solution for analyzing the sentiment of financial news for specific companies. It retrieves recent financial news headlines via the Finnhub API, processes the text, analyzes sentiment using a pre-trained model specifically fine-tuned for financial news, and exposes the results through a REST API endpoint.

### Key Features

- ✅ Retrieves financial news for any ticker symbol for specified number of days
- ✅ Pre-processes text data for optimal analysis
- ✅ Analyzes sentiment using a LLM finetuned on financial news-specific data for better performance. Model:"mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
- ✅ Aggregates sentiment across multiple news items
- ✅ Provides results via a simple REST API
- ✅ Handles API rate limiting with intelligent backoff strategy

## How It Works

The application follows a modular architecture with specialized components for each part of the pipeline:

### Components and Workflow

1. **News Retrieval (`NewsRetriever` class)**
   - Fetches company news from Finnhub API
   - Handles rate limiting with exponential backoff
   - Converts raw API data to pandas DataFrame for further processing

2. **Text Preprocessing (`SentimentAnalyzer.preprocess_text` method)**
   - Removes URLs, special characters, and stop words
   - Converts text to lowercase
   - Tokenizes text for better analysis
   - Prepares clean text for sentiment model input

3. **Sentiment Analysis (`SentimentAnalyzer` class)**
   - Uses `mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis` model
   - Applies sentiment analysis to preprocessed headlines
   - Returns sentiment label (POSITIVE, NEGATIVE, NEUTRAL) and confidence score

4. **Sentiment Aggregation (`SentimentAggregator` class)**
   - Compiles sentiment results from multiple news items
   - Determines overall sentiment based on frequency distribution
   - Handles ties by reporting mixed sentiment

5. **API Endpoint (`app` FastAPI instance)**
   - Provides `/sentiment` endpoint that accepts ticker symbol and time range
   - Returns JSON with sentiment analysis results and metadata

### Design Rationale

- **Financial-Specific Sentiment Model**: We chose a model specifically fine-tuned on financial news rather than general-purpose sentiment models. Financial language has domain-specific terminology and context that general models might misinterpret (e.g., "shares dropped" is negative in finance but might be neutral in other contexts).

- **Modular Architecture**: Each component is encapsulated in its own class to enhance maintainability, testability, and separation of concerns.

- **Rate Limit Handling**: The application implements an exponential backoff strategy to handle API rate limits, ensuring robustness when dealing with external services.

- **FastAPI Framework**: Chosen for its performance, automatic documentation generation, and type checking capabilities.

## Setup and Installation

### Option 1: Using Docker (Recommended)
##### Note: Please have atleast 15 GBs free space in our DockerData root folder for huge deep learning modules like Torch.
1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/financial-news-sentiment-api.git
   cd financial-news-sentiment-api
   ```

2. **Create a `.env` file with your Finnhub API key**
   ```
   FINNHUB_API_KEY=your_finnhub_api_key_here
   ```

4. **Build the Docker image and Run the container through compose.yml**
   ```bash
   docker-compose up --build > app_logs.log 2>&1
   ```
This also save build logs.
This process will take quite some time based on your internet speed as we are doinloading large modules like torch. It is recommended to have around 15 GBs of space in you DockerData root folder.
Once, auto-generated app_logs.log file shows following logs, our endpoint will be ready.
financial_sentiment_api  | INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)

5. **Access the API**
   - API documentation: http://localhost:8000/docs
   - Sentiment endpoint: http://localhost:8000/sentiment?ticker=AAPL&days=7

### Option 2: Running Locally

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/financial-news-sentiment-api.git
   cd financial-news-sentiment-api
   ```

2. **Create a virtual environment and install dependencies**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Create a `.env` file with your Finnhub API key**
   ```
   FINNHUB_API_KEY=your_finnhub_api_key_here
   ```
4. **Uncomment following part in main.py at line**
   ```python
   if __name__ == "__main__":
   uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
   ```
5. **Run the application**
   ```bash
   python main.py
   ```

6. **Access the API**
   - API documentation: http://localhost:8000/docs
   - Sentiment endpoint: http://localhost:8000/sentiment?ticker=AAPL&days=7

## API Usage

### Sentiment Analysis Endpoint

**Endpoint**: `/sentiment`

**Method**: GET

**Query Parameters**:
- `ticker` (required): Company ticker symbol (e.g., AAPL, MSFT, GOOGL)
- `days` (optional, default=7): Number of days to look back for news

**Example Request**:
```
GET http://localhost:8000/sentiment?ticker=AAPL&days=5
```

**Example Response**:
```json
{
  "ticker": "AAPL",
  "period": "2025-04-05 to 2025-04-10",
  "news_count": 12,
  "sentiment": "POSITIVE",
  "sentiment_distribution": {
    "POSITIVE": 7,
    "NEUTRAL": 4,
    "NEGATIVE": 1
  }
}
```

## Project Structure

```
financial-news-sentiment-api/
├── main.py               # Main application file with all components and API
├── Dockerfile            # Docker configuration
├── requirements.txt      # Python dependencies
├── .env                  # Environment variables (API keys) - Ideally shouldn't be here but I kept it for faster testing for you.
├── AAPL_news.csv         # Apple's sample news data
└── README.md             # Project documentation
```

## Future Improvements

For a production environment, the following improvements could be considered:

1. **Component Separation**: Split the monolithic `main.py` into separate modules
2. **Caching**: Implement Redis caching to reduce API calls and improve response time
3. **Database Storage**: Store historical sentiment data for trend analysis
4. **Testing**: Add unit and integration tests for each component
5. **Authentication**: Add API authentication for secure access
6. **Monitoring**: Implement logging and monitoring for production use
7. **CI/CD Pipeline**: Set up automated testing and deployment

## Dependencies

The project relies on the following key libraries:
- **FastAPI**: Web framework for building APIs
- **Finnhub-python**: Client for accessing Finnhub financial data
- **Hugging Face Transformers**: For accessing pre-trained sentiment models
- **Torch**: For running finetuned LLM. We are using CPU support only
- **NLTK**: For text preprocessing and tokenization
- **Pandas**: For data manipulation and analysis
- **Numpy**: Required for Tranformers and Torch
- **Uvicorn**: ASGI server to run the FastAPI application

