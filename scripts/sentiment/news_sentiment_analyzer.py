import os
import torch
from transformers import pipeline
import pandas as pd
import requests
from datetime import datetime, timedelta
import numpy as np
from newsapi import NewsApiClient
from dotenv import load_dotenv
import time
import re

# Load environment variables
load_dotenv()

class IndianStockSentimentAnalyzer:
    def __init__(self):
        """
        Initialize the sentiment analyzer with:
        1. NewsAPI client for fetching news
        2. FinBERT model for financial sentiment analysis
        """
        # Initialize NewsAPI client
        api_key = os.getenv('NEWSAPI_KEY')
        if not api_key:
            raise ValueError("NEWSAPI_KEY not found in environment variables")
        
        self.newsapi = NewsApiClient(api_key=api_key)
        
        # Initialize FinBERT for financial sentiment
        print("Loading FinBERT model for sentiment analysis...")
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
            device=0 if torch.cuda.is_available() else -1  # Use GPU if available
        )
        print("FinBERT model loaded successfully!")
        
        # Mapping of stock symbols to company names for better news search
        self.symbol_to_company = {
            'RELIANCE.NS': 'Reliance Industries',
            'TCS.NS': 'Tata Consultancy Services',
            'INFY.NS': 'Infosys',
            'HDFCBANK.NS': 'HDFC Bank',
            'ICICIBANK.NS': 'ICICI Bank',
            'ITC.NS': 'ITC Limited',
            'LT.NS': 'Larsen Toubro',
            'KOTAKBANK.NS': 'Kotak Mahindra Bank',
            'SBIN.NS': 'State Bank India',
            'BHARTIARTL.NS': 'Bharti Airtel',
            'AXISBANK.NS': 'Axis Bank',
            'HCLTECH.NS': 'HCL Technologies',
            'MARUTI.NS': 'Maruti Suzuki',
            'ASIANPAINT.NS': 'Asian Paints',
            'WIPRO.NS': 'Wipro',
            'ULTRACEMCO.NS': 'UltraTech Cement',
            'HINDUNILVR.NS': 'Hindustan Unilever',
            'BAJFINANCE.NS': 'Bajaj Finance',
            'NTPC.NS': 'NTPC',
            'TITAN.NS': 'Titan Company',
            'SUNPHARMA.NS': 'Sun Pharmaceutical',
            'TECHM.NS': 'Tech Mahindra',
            'INDUSINDBK.NS': 'IndusInd Bank',
            'POWERGRID.NS': 'Power Grid Corporation',
            'JSWSTEEL.NS': 'JSW Steel',
            'ADANIENT.NS': 'Adani Enterprises',
            'BAJAJFINSV.NS': 'Bajaj Finserv',
            'CIPLA.NS': 'Cipla',
            'COALINDIA.NS': 'Coal India',
            'EICHERMOT.NS': 'Eicher Motors',
            'HINDALCO.NS': 'Hindalco Industries',
            'ONGC.NS': 'Oil Natural Gas Corporation',
            'TATACONSUM.NS': 'Tata Consumer Products',
            'NESTLEIND.NS': 'Nestle India',
            'DRREDDY.NS': 'Dr Reddys Laboratories',
            'M&M.NS': 'Mahindra Mahindra',
            'GRASIM.NS': 'Grasim Industries',
            'HDFCLIFE.NS': 'HDFC Life Insurance',
            'BRITANNIA.NS': 'Britannia Industries',
            'APOLLOHOSP.NS': 'Apollo Hospitals',
            'DIVISLAB.NS': 'Divis Laboratories',
            'BPCL.NS': 'Bharat Petroleum',
            'SBILIFE.NS': 'SBI Life Insurance',
            'TATAMOTORS.NS': 'Tata Motors',
            'HEROMOTOCO.NS': 'Hero MotoCorp',
            'BAJAJ-AUTO.NS': 'Bajaj Auto',
            'UPL.NS': 'UPL Limited',
            'TATASTEEL.NS': 'Tata Steel',
            'ADANIPORTS.NS': 'Adani Ports',
            'IOC.NS': 'Indian Oil Corporation'
        }
    
    def clean_text(self, text):
        """Clean and preprocess news text"""
        if not text:
            return ""
        
        # Remove HTML tags
        text = re.sub('<.*?>', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        # Truncate to BERT's max length
        if len(text) > 512:
            text = text[:512]
        
        return text
    
    def get_stock_news(self, symbol, days=7):
        """
        Fetch news articles for a specific stock
        
        Args:
            symbol: Stock symbol (e.g., 'RELIANCE.NS')
            days: Number of days to look back for news
            
        Returns:
            DataFrame with news articles
        """
        company_name = self.symbol_to_company.get(symbol, symbol.replace('.NS', ''))
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        print(f"Fetching news for {company_name} ({symbol})...")
        
        try:
            # Search for news articles
            articles = self.newsapi.get_everything(
                q=f'"{company_name}" OR "{symbol.replace(".NS", "")}"',
                domains='economictimes.indiatimes.com,moneycontrol.com,business-standard.com,livemint.com',
                from_param=start_date.strftime('%Y-%m-%d'),
                to=end_date.strftime('%Y-%m-%d'),
                language='en',
                sort_by='publishedAt',
                page_size=50  # Max articles per request
            )
            
            if not articles['articles']:
                print(f"No news found for {company_name}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            news_data = []
            for article in articles['articles']:
                news_data.append({
                    'symbol': symbol,
                    'company': company_name,
                    'title': article['title'],
                    'description': article['description'],
                    'content': article['content'],
                    'url': article['url'],
                    'published_at': article['publishedAt'],
                    'source': article['source']['name']
                })
            
            df = pd.DataFrame(news_data)
            df['published_at'] = pd.to_datetime(df['published_at'])
            
            print(f"✅ Found {len(df)} articles for {company_name}")
            return df
            
        except Exception as e:
            print(f"Error fetching news for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def analyze_sentiment_batch(self, texts):
        """
        Analyze sentiment for multiple news texts
        
        Args:
            texts: List of text strings
            
        Returns:
            List of sentiment dictionaries
        """
        if not texts:
            return []
        
        print(f"Analyzing sentiment for {len(texts)} articles...")
        
        sentiments = []
        for i, text in enumerate(texts):
            try:
                cleaned_text = self.clean_text(text)
                if not cleaned_text:
                    sentiments.append({'label': 'neutral', 'score': 0.5})
                    continue
                
                result = self.sentiment_pipeline(cleaned_text)[0]
                sentiments.append(result)
                
                # Add small delay to avoid overwhelming the model
                if i % 10 == 0:
                    time.sleep(0.1)
                    
            except Exception as e:
                print(f"Error analyzing text {i}: {str(e)}")
                sentiments.append({'label': 'neutral', 'score': 0.5})
        
        return sentiments
    
    def generate_sentiment_features(self, symbol, news_df):
        """
        Generate numerical sentiment features from news data
        
        Args:
            symbol: Stock symbol
            news_df: DataFrame with news articles
            
        Returns:
            Dictionary with sentiment features
        """
        if news_df.empty:
            return {
                f'{symbol}_sentiment_mean': 0.0,
                f'{symbol}_sentiment_std': 0.0,
                f'{symbol}_sentiment_trend': 0.0,
                f'{symbol}_news_volume': 0,
                f'{symbol}_positive_ratio': 0.0,
                f'{symbol}_negative_ratio': 0.0
            }
        
        # Combine title and description for sentiment analysis
        texts = []
        for _, row in news_df.iterrows():
            combined_text = f"{row['title']} {row['description'] or ''}"
            texts.append(combined_text)
        
        # Analyze sentiment
        sentiments = self.analyze_sentiment_batch(texts)
        
        # Convert sentiment labels to numerical scores
        sentiment_scores = []
        positive_count = 0
        negative_count = 0
        
        for sentiment in sentiments:
            label = sentiment['label'].lower()
            score = sentiment['score']
            
            if label == 'positive':
                sentiment_scores.append(score)
                positive_count += 1
            elif label == 'negative':
                sentiment_scores.append(-score)
                negative_count += 1
            else:  # neutral
                sentiment_scores.append(0.0)
        
        # Calculate features
        sentiment_array = np.array(sentiment_scores)
        total_articles = len(sentiment_scores)
        
        # Time-weighted sentiment (recent news gets higher weight)
        news_df_sorted = news_df.sort_values('published_at')
        weights = np.linspace(0.5, 1.0, len(sentiment_scores))  # Recent news weighted higher
        weighted_sentiment = np.average(sentiment_scores, weights=weights)
        
        # Trend calculation (recent 3 days vs older)
        if len(sentiment_scores) >= 6:
            recent_sentiment = np.mean(sentiment_scores[-3:])
            older_sentiment = np.mean(sentiment_scores[:-3])
            sentiment_trend = recent_sentiment - older_sentiment
        else:
            sentiment_trend = 0.0
        
        features = {
            f'{symbol}_sentiment_mean': float(np.mean(sentiment_scores)) if sentiment_scores else 0.0,
            f'{symbol}_sentiment_std': float(np.std(sentiment_scores)) if len(sentiment_scores) > 1 else 0.0,
            f'{symbol}_sentiment_weighted': float(weighted_sentiment),
            f'{symbol}_sentiment_trend': float(sentiment_trend),
            f'{symbol}_news_volume': total_articles,
            f'{symbol}_positive_ratio': positive_count / total_articles if total_articles > 0 else 0.0,
            f'{symbol}_negative_ratio': negative_count / total_articles if total_articles > 0 else 0.0
        }
        
        return features
    
    def process_all_stocks(self, symbols, days=7):
        """
        Process sentiment analysis for all stocks
        
        Args:
            symbols: List of stock symbols
            days: Number of days to look back
            
        Returns:
            DataFrame with sentiment features for all stocks
        """
        all_features = {}
        all_news_data = []
        
        print(f"Starting sentiment analysis for {len(symbols)} stocks...")
        
        for i, symbol in enumerate(symbols, 1):
            print(f"\nProcessing {symbol} ({i}/{len(symbols)})")
            
            # Get news data
            news_df = self.get_stock_news(symbol, days)
            
            # Generate sentiment features
            features = self.generate_sentiment_features(symbol, news_df)
            all_features.update(features)
            
            # Store news data for potential later analysis
            if not news_df.empty:
                all_news_data.append(news_df)
            
            # Rate limiting - NewsAPI has limits
            time.sleep(1)
        
        # Convert features to DataFrame
        features_df = pd.DataFrame([all_features])
        features_df.index = [datetime.now().strftime('%Y-%m-%d')]
        
        # Save all news data
        if all_news_data:
            combined_news = pd.concat(all_news_data, ignore_index=True)
            os.makedirs("data", exist_ok=True)
            combined_news.to_csv("data/news_data.csv", index=False)
            print(f"Saved news data to data/news_data.csv")
        
        return features_df