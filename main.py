"""
Financial News Sentiment Analysis API

This script provides an end-to-end solution for retrieving financial news,
analyzing sentiment, and exposing the results via a REST API.
"""

import os
import re
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Any

import finnhub
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from transformers import pipeline
import nltk
import uvicorn
import requests


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")

# Download NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)


# Initialize FastAPI
app = FastAPI(
    title="Financial News Sentiment Analysis API",
    description="API for analyzing sentiment from financial news",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RateLimitHandler:
    """Handles Finnhub API rate limiting with backoff strategy."""
    
    def __init__(self, max_retries: int = 3, initial_backoff: float = 61.0):
        """
        Initialize the rate limit handler.
        
        Args:
            max_retries: Maximum number of retry attempts
            initial_backoff: Initial backoff time in seconds (just over a minute for Finnhub)
        """
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff
        
    def execute_with_backoff(self, func, *args, **kwargs):
        """
        Execute a function with exponential backoff for rate limiting.
        
        Args:
            func: Function to execute
            *args, **kwargs: Arguments to pass to the function
            
        Returns:
            Result of the function call
            
        Raises:
            Exception: If max retries are exhausted
        """
        retries = 0
        backoff_time = self.initial_backoff
        
        while retries <= self.max_retries:
            try:
                return func(*args, **kwargs)
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:  # Too Many Requests
                    retries += 1
                    if retries > self.max_retries:
                        logger.error(f"Rate limit exceeded after {self.max_retries} retries")
                        raise
                    
                    logger.warning(f"Rate limit hit. Backing off for {backoff_time} seconds")
                    time.sleep(backoff_time)
                    backoff_time *= 2  # Exponential backoff
                else:
                    # Not a rate limiting issue
                    raise
            except Exception as e:
                # Not a rate limiting issue
                raise


class SentimentAnalyzer:
    """Class for performing sentiment analysis on financial news."""
    
    def __init__(self):
        """Initialize the sentiment analysis pipeline."""
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis", 
            model="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
        )
        logger.info("Sentiment analysis model loaded successfully")
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for sentiment analysis.
        
        Args:
            text: Raw text string to preprocess
            
        Returns:
            Preprocessed text string
        """
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and replace with space
        text = re.sub(r'\W', ' ', text)
        # Tokenize text
        tokens = word_tokenize(text)
        # Remove stop words
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
        # Join tokens back into a string
        preprocessed_text = ' '.join(tokens)
        return preprocessed_text
    
    def analyze_sentiment(self, text: str) -> Tuple[str, float]:
        """
        Analyze sentiment of a text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Tuple of (sentiment_label, confidence_score)
        """
        preprocessed_text = self.preprocess_text(text)
        result = self.sentiment_pipeline(preprocessed_text)
        return result[0]['label'], result[0]['score']


class NewsRetriever:
    """Class for retrieving financial news data."""
    
    def __init__(self, api_key: str):
        """
        Initialize the news retriever with API credentials.
        
        Args:
            api_key: Finnhub API key
        """
        self.client = finnhub.Client(api_key=api_key)
        self.rate_limiter = RateLimitHandler()
        logger.info("Finnhub client initialized")
    
    def _fetch_company_news(self, symbol: str, from_date: str, to_date: str):
        """
        Internal method to fetch news that can be retried on rate limiting.
        
        Args:
            symbol: Company ticker symbol
            from_date: Start date in format YYYY-MM-DD
            to_date: End date in format YYYY-MM-DD
            
        Returns:
            Raw news data from Finnhub
        """
        return self.client.company_news(symbol=symbol, _from=from_date, to=to_date)
    
    def get_company_news(
        self, 
        symbol: str, 
        from_date: str, 
        to_date: str
    ) -> pd.DataFrame:
        """
        Retrieve news for a specific company and date range.
        
        Args:
            symbol: Company ticker symbol
            from_date: Start date in format YYYY-MM-DD
            to_date: End date in format YYYY-MM-DD
            
        Returns:
            DataFrame containing news data
        """
        try:
            # Use rate limiter to handle potential 429 responses
            news = self.rate_limiter.execute_with_backoff(
                self._fetch_company_news,
                symbol,
                from_date,
                to_date
            )
            
            if not news:
                logger.warning(f"No news found for {symbol} between {from_date} and {to_date}")
                return pd.DataFrame(columns=['headline', 'source', 'url', 'datetime'])
            
            news_df = pd.DataFrame(news)
            news_df = news_df[['headline', 'source', 'url', 'datetime']]
            news_df['datetime'] = pd.to_datetime(news_df['datetime'], unit='s')
            
            logger.info(f"Retrieved {len(news_df)} news items for {symbol}")
            return news_df
        
        except Exception as e:
            logger.error(f"Error retrieving news: {str(e)}")
            raise


class SentimentAggregator:
    """Class for aggregating sentiment results from multiple news items."""
    
    @staticmethod
    def aggregate_sentiments(df: pd.DataFrame) -> str:
        """
        Aggregate sentiment labels to determine overall sentiment.
        
        Args:
            df: DataFrame with 'sentiment_label' column
            
        Returns:
            String indicating the most frequent sentiment or a tie
        """
        # Get sentiment counts
        counts = df['sentiment_label'].value_counts().to_dict()
        
        # Ensure all three keys are present in the result
        default_categories = {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0}
        for key, value in counts.items():
            default_categories[key.upper()] = value
        
        # Find max count
        max_count = max(default_categories.values())
        
        # Find all sentiments with the max count
        top_sentiments = [
            sentiment for sentiment, count in default_categories.items() 
            if count == max_count
        ]
        
        if len(top_sentiments) > 1:
            return f"MIXED (tie between {' and '.join(top_sentiments)})"
        else:
            return top_sentiments[0]


class FinancialSentimentService:
    """Service class that orchestrates the end-to-end sentiment analysis process."""
    
    def __init__(self, api_key: str):
        """
        Initialize the service with all required components.
        
        Args:
            api_key: Finnhub API key
        """
        self.news_retriever = NewsRetriever(api_key)
        self.sentiment_analyzer = SentimentAnalyzer()
        self.aggregator = SentimentAggregator()
    
    def analyze_company_sentiment(
        self, 
        symbol: str, 
        days: int = 7
    ) -> Dict[str, Any]:
        """
        Perform end-to-end sentiment analysis for a company.
        
        Args:
            symbol: Company ticker symbol
            days: Number of days to look back for news (default: 7)
            
        Returns:
            Dictionary with analysis results
        """
        # Calculate date range
        to_date = datetime.now().strftime('%Y-%m-%d')
        from_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        # Get news
        news_df = self.news_retriever.get_company_news(
            symbol=symbol,
            from_date=from_date,
            to_date=to_date
        )
        
        if news_df.empty:
            return {
                "ticker": symbol,
                "period": f"{from_date} to {to_date}",
                "news_count": 0,
                "sentiment": "UNKNOWN",
                "message": "No news found for the specified period"
            }
        
        # Analyze sentiment for each headline
        results = []
        for headline in news_df['headline']:
            label, score = self.sentiment_analyzer.analyze_sentiment(headline)
            results.append((label, score))
        
        news_df[['sentiment_label', 'sentiment_score']] = pd.DataFrame(results)
        
        # Aggregate sentiments
        overall_sentiment = self.aggregator.aggregate_sentiments(news_df)
        
        # Get sentiment distribution
        sentiment_counts = news_df['sentiment_label'].value_counts().to_dict()
        
        return {
            "ticker": symbol,
            "period": f"{from_date} to {to_date}",
            "news_count": len(news_df),
            "sentiment": overall_sentiment,
            "sentiment_distribution": sentiment_counts
        }


# Initialize the service
service = FinancialSentimentService(FINNHUB_API_KEY)


@app.get("/")
def root():
    """Root endpoint with API information."""
    return {
        "name": "Financial News Sentiment Analysis API",
        "version": "1.0.0",
        "documentation": "/docs"
    }


@app.get("/sentiment")
def get_sentiment(
    ticker: str = Query(..., description="Company ticker symbol"),
    days: int = Query(7, description="Number of days to look back for news")
):
    """
    Get sentiment analysis for a specific company.
    
    Args:
        ticker: Company ticker symbol
        days: Number of days to look back for news (default: 7)
        
    Returns:
        JSON with sentiment analysis results
    """
    try:
        return service.analyze_company_sentiment(ticker, days)
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )


# if __name__ == "__main__":
#     uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)