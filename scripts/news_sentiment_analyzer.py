#!/usr/bin/env python3
"""
Comprehensive sentiment analysis system for NIFTY 50 stocks.

This module implements multi-model sentiment analysis using:
- VADER sentiment analyzer (rule-based)
- TextBlob (polarity and subjectivity)
- FinBERT (financial domain-specific BERT)
- Custom keyword-based scoring

Features generated:
- Daily sentiment scores (positive, negative, neutral percentages)
- Rolling sentiment averages (7d, 30d)
- Sentiment momentum and volatility
- Market-wide sentiment aggregation
"""

import pandas as pd
import numpy as np
import os
import re
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Sentiment analysis libraries
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """
    Multi-model sentiment analysis for financial news and text.
    """
    
    def __init__(self):
        """Initialize sentiment analysis models."""
        logger.info("Initializing sentiment analysis models...")
        
        # VADER analyzer (rule-based)
        self.vader = SentimentIntensityAnalyzer()
        
        # FinBERT model for financial sentiment
        self.finbert_model = None
        self.finbert_tokenizer = None
        self._load_finbert()
        
        # Financial keywords for custom scoring
        self.positive_keywords = [
            'profit', 'growth', 'bullish', 'positive', 'gains', 'increase', 'rise',
            'strong', 'outperform', 'buy', 'upgrade', 'beat', 'exceed', 'record',
            'surge', 'rally', 'boom', 'expansion', 'revenue', 'earnings'
        ]
        
        self.negative_keywords = [
            'loss', 'decline', 'bearish', 'negative', 'fall', 'decrease', 'drop',
            'weak', 'underperform', 'sell', 'downgrade', 'miss', 'below', 'crash',
            'plunge', 'recession', 'contraction', 'debt', 'bankruptcy'
        ]
        
        # NIFTY 50 company name mapping for news filtering
        self.company_mapping = {
            'RELIANCE.NS': ['reliance', 'ril', 'mukesh ambani'],
            'TCS.NS': ['tcs', 'tata consultancy', 'tata consultancy services'],
            'INFY.NS': ['infosys', 'infy', 'narayana murthy'],
            'HDFCBANK.NS': ['hdfc bank', 'hdfc', 'housing development finance'],
            'ICICIBANK.NS': ['icici bank', 'icici'],
            'ITC.NS': ['itc', 'indian tobacco'],
            'LT.NS': ['larsen toubro', 'l&t', 'larsen & toubro'],
            'KOTAKBANK.NS': ['kotak mahindra', 'kotak bank', 'kotak'],
            'SBIN.NS': ['sbi', 'state bank', 'state bank of india'],
            'BHARTIARTL.NS': ['bharti airtel', 'airtel', 'bharti'],
            'AXISBANK.NS': ['axis bank', 'axis'],
            'HCLTECH.NS': ['hcl technologies', 'hcl tech', 'hcl'],
            'MARUTI.NS': ['maruti suzuki', 'maruti', 'suzuki'],
            'ASIANPAINT.NS': ['asian paints', 'asian paint'],
            'WIPRO.NS': ['wipro', 'azim premji'],
            'ULTRACEMCO.NS': ['ultratech cement', 'ultratech'],
            'HINDUNILVR.NS': ['hindustan unilever', 'hul', 'unilever'],
            'BAJFINANCE.NS': ['bajaj finance', 'bajaj fin'],
            'NTPC.NS': ['ntpc', 'national thermal power'],
            'TITAN.NS': ['titan', 'titan company'],
            'SUNPHARMA.NS': ['sun pharma', 'sun pharmaceutical'],
            'TECHM.NS': ['tech mahindra', 'tech m'],
            'INDUSINDBK.NS': ['indusind bank', 'indusind'],
            'POWERGRID.NS': ['power grid', 'powergrid'],
            'JSWSTEEL.NS': ['jsw steel', 'jsw'],
            'ADANIENT.NS': ['adani enterprises', 'adani', 'gautam adani'],
            'BAJAJFINSV.NS': ['bajaj finserv', 'bajaj financial'],
            'CIPLA.NS': ['cipla', 'cipla pharma'],
            'COALINDIA.NS': ['coal india', 'cil'],
            'EICHERMOT.NS': ['eicher motors', 'royal enfield'],
            'HINDALCO.NS': ['hindalco', 'aditya birla'],
            'ONGC.NS': ['ongc', 'oil natural gas'],
            'TATACONSUM.NS': ['tata consumer', 'tata tea'],
            'NESTLEIND.NS': ['nestle india', 'nestle'],
            'DRREDDY.NS': ['dr reddy', 'dr reddys'],
            'M&M.NS': ['mahindra', 'mahindra & mahindra'],
            'GRASIM.NS': ['grasim', 'grasim industries'],
            'HDFCLIFE.NS': ['hdfc life', 'hdfc life insurance'],
            'BRITANNIA.NS': ['britannia', 'britannia industries'],
            'APOLLOHOSP.NS': ['apollo hospitals', 'apollo'],
            'DIVISLAB.NS': ['divis laboratories', 'divis lab'],
            'BPCL.NS': ['bharat petroleum', 'bpcl'],
            'SBILIFE.NS': ['sbi life', 'sbi life insurance'],
            'TATAMOTORS.NS': ['tata motors', 'tata motor'],
            'HEROMOTOCO.NS': ['hero motocorp', 'hero moto'],
            'BAJAJ-AUTO.NS': ['bajaj auto', 'bajaj'],
            'UPL.NS': ['upl', 'united phosphorus'],
            'TATASTEEL.NS': ['tata steel', 'tata steel'],
            'ADANIPORTS.NS': ['adani ports', 'adani port'],
            'IOC.NS': ['indian oil', 'ioc', 'indian oil corporation']
        }
        
        logger.info("Sentiment analysis models initialized successfully")
    
    def _load_finbert(self):
        """Load FinBERT model for financial sentiment analysis."""
        try:
            model_name = "ProsusAI/finbert"
            self.finbert_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.finbert_model = AutoModelForSequenceClassification.from_pretrained(model_name)
            
            # Create pipeline
            self.finbert_pipeline = pipeline(
                "sentiment-analysis",
                model=self.finbert_model,
                tokenizer=self.finbert_tokenizer,
                return_all_scores=True
            )
            logger.info("FinBERT model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load FinBERT model: {e}")
            self.finbert_pipeline = None
    
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text for sentiment analysis."""
        if not text or not isinstance(text, str):
            return ""
        
        # Remove URLs, mentions, and special characters
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'@\w+|#\w+', '', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text.lower()
    
    def vader_sentiment(self, text: str) -> Dict[str, float]:
        """Get VADER sentiment scores."""
        scores = self.vader.polarity_scores(text)
        return {
            'vader_positive': scores['pos'],
            'vader_negative': scores['neg'],
            'vader_neutral': scores['neu'],
            'vader_compound': scores['compound']
        }
    
    def textblob_sentiment(self, text: str) -> Dict[str, float]:
        """Get TextBlob sentiment scores."""
        blob = TextBlob(text)
        return {
            'textblob_polarity': blob.sentiment.polarity,
            'textblob_subjectivity': blob.sentiment.subjectivity
        }
    
    def finbert_sentiment(self, text: str) -> Dict[str, float]:
        """Get FinBERT sentiment scores."""
        if not self.finbert_pipeline:
            return {
                'finbert_positive': 0.0,
                'finbert_negative': 0.0,
                'finbert_neutral': 0.0
            }
        
        try:
            # Truncate text to avoid token limit
            if len(text) > 512:
                text = text[:512]
            
            results = self.finbert_pipeline(text)
            
            scores = {
                'finbert_positive': 0.0,
                'finbert_negative': 0.0,
                'finbert_neutral': 0.0
            }
            
            for result in results:
                label = result['label'].lower()
                score = result['score']
                
                if 'positive' in label:
                    scores['finbert_positive'] = score
                elif 'negative' in label:
                    scores['finbert_negative'] = score
                elif 'neutral' in label:
                    scores['finbert_neutral'] = score
            
            return scores
        except Exception as e:
            logger.warning(f"FinBERT analysis failed: {e}")
            return {
                'finbert_positive': 0.0,
                'finbert_negative': 0.0,
                'finbert_neutral': 0.0
            }
    
    def keyword_sentiment(self, text: str) -> Dict[str, float]:
        """Custom keyword-based sentiment scoring."""
        words = text.lower().split()
        
        positive_count = sum(1 for word in words if any(kw in word for kw in self.positive_keywords))
        negative_count = sum(1 for word in words if any(kw in word for kw in self.negative_keywords))
        total_count = len(words)
        
        if total_count == 0:
            return {
                'keyword_positive': 0.0,
                'keyword_negative': 0.0,
                'keyword_net': 0.0
            }
        
        positive_ratio = positive_count / total_count
        negative_ratio = negative_count / total_count
        net_sentiment = positive_ratio - negative_ratio
        
        return {
            'keyword_positive': positive_ratio,
            'keyword_negative': negative_ratio,
            'keyword_net': net_sentiment
        }
    
    def analyze_text(self, text: str) -> Dict[str, float]:
        """Comprehensive sentiment analysis using all models."""
        if not text:
            return self._get_empty_sentiment()
        
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        if not processed_text:
            return self._get_empty_sentiment()
        
        # Get sentiment from all models
        sentiment_scores = {}
        
        # VADER
        sentiment_scores.update(self.vader_sentiment(processed_text))
        
        # TextBlob
        sentiment_scores.update(self.textblob_sentiment(processed_text))
        
        # FinBERT
        sentiment_scores.update(self.finbert_sentiment(processed_text))
        
        # Custom keywords
        sentiment_scores.update(self.keyword_sentiment(processed_text))
        
        # Composite sentiment score
        sentiment_scores['composite_sentiment'] = self._calculate_composite_sentiment(sentiment_scores)
        
        return sentiment_scores
    
    def _get_empty_sentiment(self) -> Dict[str, float]:
        """Return neutral sentiment scores for empty text."""
        return {
            'vader_positive': 0.0,
            'vader_negative': 0.0,
            'vader_neutral': 1.0,
            'vader_compound': 0.0,
            'textblob_polarity': 0.0,
            'textblob_subjectivity': 0.0,
            'finbert_positive': 0.0,
            'finbert_negative': 0.0,
            'finbert_neutral': 1.0,
            'keyword_positive': 0.0,
            'keyword_negative': 0.0,
            'keyword_net': 0.0,
            'composite_sentiment': 0.0
        }
    
    def _calculate_composite_sentiment(self, scores: Dict[str, float]) -> float:
        """Calculate weighted composite sentiment score."""
        # Weight different models based on their reliability for financial text
        weights = {
            'vader_compound': 0.2,
            'textblob_polarity': 0.15,
            'finbert_net': 0.4,  # Highest weight for financial-specific model
            'keyword_net': 0.25
        }
        
        # Calculate FinBERT net sentiment
        finbert_net = scores.get('finbert_positive', 0) - scores.get('finbert_negative', 0)
        
        # Composite score
        composite = (
            weights['vader_compound'] * scores.get('vader_compound', 0) +
            weights['textblob_polarity'] * scores.get('textblob_polarity', 0) +
            weights['finbert_net'] * finbert_net +
            weights['keyword_net'] * scores.get('keyword_net', 0)
        )
        
        # Normalize to [-1, 1] range
        return np.clip(composite, -1.0, 1.0)
    
    def find_company_mentions(self, text: str) -> List[str]:
        """Find which NIFTY 50 companies are mentioned in the text."""
        text_lower = text.lower()
        mentioned_companies = []
        
        for symbol, keywords in self.company_mapping.items():
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    mentioned_companies.append(symbol)
                    break
        
        return mentioned_companies


class SentimentFeatureGenerator:
    """Generate sentiment features for portfolio optimization."""
    
    def __init__(self):
        self.analyzer = SentimentAnalyzer()
        self.nifty50_symbols = list(self.analyzer.company_mapping.keys())
    
    def create_mock_news_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Create mock news data for testing (since we don't have real news API access)."""
        logger.info("Creating mock news data for testing...")
        
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        business_days = [d for d in date_range if d.weekday() < 5]
        
        # Sample news headlines with different sentiments
        positive_headlines = [
            "{company} reports strong quarterly earnings, beats estimates",
            "{company} announces major expansion plans",
            "{company} stock upgraded by analysts",
            "{company} sees strong demand for products",
            "{company} launches innovative new service",
        ]
        
        negative_headlines = [
            "{company} faces regulatory challenges",
            "{company} reports lower than expected revenue",
            "{company} stock downgraded due to market concerns",
            "{company} dealing with supply chain issues",
            "{company} announces job cuts amid restructuring",
        ]
        
        neutral_headlines = [
            "{company} announces quarterly board meeting",
            "{company} updates on business operations",
            "{company} files regular compliance report",
            "{company} conducts investor conference call",
            "{company} provides business update",
        ]
        
        news_data = []
        np.random.seed(42)  # For reproducible mock data
        
        for date in business_days:
            # Generate 1-5 news items per day randomly
            num_news = np.random.randint(1, 6)
            
            for _ in range(num_news):
                # Randomly select company and sentiment
                company_symbol = np.random.choice(self.nifty50_symbols)
                company_name = company_symbol.replace('.NS', '').replace('&', 'and')
                
                sentiment_type = np.random.choice(['positive', 'negative', 'neutral'], 
                                                p=[0.3, 0.2, 0.5])  # More neutral news
                
                if sentiment_type == 'positive':
                    headline = np.random.choice(positive_headlines).format(company=company_name)
                elif sentiment_type == 'negative':
                    headline = np.random.choice(negative_headlines).format(company=company_name)
                else:
                    headline = np.random.choice(neutral_headlines).format(company=company_name)
                
                news_data.append({
                    'date': date,
                    'headline': headline,
                    'company': company_symbol,
                    'text': headline,  # In real scenario, this would be full article text
                    'source': 'MockNews'
                })
        
        return pd.DataFrame(news_data)
    
    def generate_sentiment_features(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Generate comprehensive sentiment features for all NIFTY 50 stocks."""
        logger.info(f"Generating sentiment features from {start_date} to {end_date}")
        
        # Create mock news data (in production, this would fetch from news APIs)
        news_df = self.create_mock_news_data(start_date, end_date)
        
        # Analyze sentiment for each news item
        logger.info("Analyzing sentiment for news articles...")
        sentiment_results = []
        
        for _, news_item in news_df.iterrows():
            sentiment_scores = self.analyzer.analyze_text(news_item['text'])
            sentiment_scores['date'] = news_item['date']
            sentiment_scores['company'] = news_item['company']
            sentiment_scores['headline'] = news_item['headline']
            sentiment_results.append(sentiment_scores)
        
        sentiment_df = pd.DataFrame(sentiment_results)
        
        # Aggregate daily sentiment by company
        daily_sentiment = self._aggregate_daily_sentiment(sentiment_df)
        
        # Generate rolling features
        sentiment_features = self._generate_rolling_features(daily_sentiment)
        
        # Add market-wide sentiment
        sentiment_features = self._add_market_sentiment(sentiment_features)
        
        logger.info(f"Generated sentiment features with shape: {sentiment_features.shape}")
        return sentiment_features
    
    def _aggregate_daily_sentiment(self, sentiment_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate news sentiment by company and date."""
        logger.info("Aggregating daily sentiment scores...")
        
        # Group by date and company, then aggregate
        daily_agg = sentiment_df.groupby(['date', 'company']).agg({
            'vader_compound': 'mean',
            'textblob_polarity': 'mean',
            'textblob_subjectivity': 'mean',
            'finbert_positive': 'mean',
            'finbert_negative': 'mean',
            'finbert_neutral': 'mean',
            'keyword_positive': 'mean',
            'keyword_negative': 'mean',
            'keyword_net': 'mean',
            'composite_sentiment': 'mean',
            'headline': 'count'  # Number of news articles
        }).reset_index()
        
        # Rename count column
        daily_agg.rename(columns={'headline': 'news_count'}, inplace=True)
        
        # Create full date range for all companies
        date_range = pd.date_range(
            start=sentiment_df['date'].min(),
            end=sentiment_df['date'].max(),
            freq='D'
        )
        business_days = [d for d in date_range if d.weekday() < 5]
        
        # Create complete DataFrame with all company-date combinations
        full_index = pd.MultiIndex.from_product(
            [business_days, self.nifty50_symbols],
            names=['date', 'company']
        ).to_frame(index=False)
        
        # Merge with aggregated data
        daily_sentiment = full_index.merge(daily_agg, on=['date', 'company'], how='left')
        
        # Fill missing values with neutral sentiment
        sentiment_columns = [
            'vader_compound', 'textblob_polarity', 'textblob_subjectivity',
            'finbert_positive', 'finbert_negative', 'finbert_neutral',
            'keyword_positive', 'keyword_negative', 'keyword_net',
            'composite_sentiment'
        ]
        
        for col in sentiment_columns:
            if col in ['finbert_neutral']:
                daily_sentiment[col] = daily_sentiment[col].fillna(1.0)  # Neutral for probabilities
            else:
                daily_sentiment[col] = daily_sentiment[col].fillna(0.0)  # Neutral for scores
        
        daily_sentiment['news_count'] = daily_sentiment['news_count'].fillna(0)
        
        return daily_sentiment
    
    def _generate_rolling_features(self, daily_sentiment: pd.DataFrame) -> pd.DataFrame:
        """Generate rolling sentiment features."""
        logger.info("Generating rolling sentiment features...")
        
        # Sort by company and date
        daily_sentiment = daily_sentiment.sort_values(['company', 'date'])
        
        rolling_features = []
        
        for company in self.nifty50_symbols:
            company_data = daily_sentiment[daily_sentiment['company'] == company].copy()
            company_data = company_data.set_index('date').sort_index()
            
            # Generate rolling features for different windows
            for window in [7, 30]:  # 1 week and 1 month
                # Rolling averages
                company_data[f'sentiment_ma_{window}d'] = \
                    company_data['composite_sentiment'].rolling(window=window, min_periods=1).mean()
                
                # Rolling volatility
                company_data[f'sentiment_vol_{window}d'] = \
                    company_data['composite_sentiment'].rolling(window=window, min_periods=1).std()
                
                # Rolling news volume
                company_data[f'news_volume_{window}d'] = \
                    company_data['news_count'].rolling(window=window, min_periods=1).sum()
                
                # Sentiment momentum (current vs. rolling average)
                company_data[f'sentiment_momentum_{window}d'] = \
                    company_data['composite_sentiment'] - company_data[f'sentiment_ma_{window}d']
            
            # Reset index and add company column back
            company_data = company_data.reset_index()
            company_data['company'] = company
            rolling_features.append(company_data)
        
        return pd.concat(rolling_features, ignore_index=True)
    
    def _add_market_sentiment(self, sentiment_features: pd.DataFrame) -> pd.DataFrame:
        """Add market-wide sentiment features."""
        logger.info("Adding market-wide sentiment features...")
        
        # Calculate daily market sentiment (average across all companies)
        market_sentiment = sentiment_features.groupby('date').agg({
            'composite_sentiment': 'mean',
            'news_count': 'sum',
            'vader_compound': 'mean',
            'textblob_polarity': 'mean'
        }).reset_index()
        
        market_sentiment.rename(columns={
            'composite_sentiment': 'market_sentiment',
            'news_count': 'market_news_count',
            'vader_compound': 'market_vader',
            'textblob_polarity': 'market_textblob'
        }, inplace=True)
        
        # Add rolling market features
        market_sentiment = market_sentiment.set_index('date').sort_index()
        for window in [7, 30]:
            market_sentiment[f'market_sentiment_ma_{window}d'] = \
                market_sentiment['market_sentiment'].rolling(window=window, min_periods=1).mean()
            market_sentiment[f'market_sentiment_vol_{window}d'] = \
                market_sentiment['market_sentiment'].rolling(window=window, min_periods=1).std()
        
        market_sentiment = market_sentiment.reset_index()
        
        # Merge with company-level features
        final_features = sentiment_features.merge(market_sentiment, on='date', how='left')
        
        return final_features


def main():
    """Main function to generate sentiment features."""
    logger.info("Starting sentiment feature generation...")
    
    # Ensure output directory exists
    os.makedirs("data", exist_ok=True)
    
    # Generate features for the last 6 months
    end_date = datetime.today()
    start_date = end_date - timedelta(days=180)
    
    # Initialize feature generator
    generator = SentimentFeatureGenerator()
    
    # Generate sentiment features
    sentiment_features = generator.generate_sentiment_features(start_date, end_date)
    
    # Pivot data to create features for each company
    logger.info("Reshaping data for portfolio optimization...")
    
    # Select key features for each company
    feature_columns = [
        'composite_sentiment', 'sentiment_ma_7d', 'sentiment_ma_30d',
        'sentiment_vol_7d', 'sentiment_vol_30d', 'sentiment_momentum_7d',
        'sentiment_momentum_30d', 'news_volume_7d', 'news_volume_30d'
    ]
    
    # Pivot to get company features as columns
    pivoted_features = []
    
    for feature in feature_columns:
        pivot_df = sentiment_features.pivot(index='date', columns='company', values=feature)
        pivot_df.columns = [f"{col}_{feature}" for col in pivot_df.columns]
        pivoted_features.append(pivot_df)
    
    # Combine all features
    final_sentiment_df = pd.concat(pivoted_features, axis=1)
    
    # Add market-wide features (these are same for all companies)
    market_features = ['market_sentiment', 'market_sentiment_ma_7d', 'market_sentiment_ma_30d', 
                      'market_sentiment_vol_7d', 'market_sentiment_vol_30d']
    
    for feature in market_features:
        market_data = sentiment_features[['date', feature]].drop_duplicates()
        market_data = market_data.set_index('date')
        final_sentiment_df = final_sentiment_df.merge(market_data, left_index=True, right_index=True, how='left')
    
    # Drop rows with too many NaN values
    final_sentiment_df = final_sentiment_df.dropna(thresh=len(final_sentiment_df.columns) * 0.7)
    
    # Save sentiment features
    output_path = "data/sentiment_features.csv"
    final_sentiment_df.to_csv(output_path)
    
    logger.info(f"Sentiment features saved to {output_path}")
    logger.info(f"Final shape: {final_sentiment_df.shape}")
    logger.info(f"Date range: {final_sentiment_df.index.min()} to {final_sentiment_df.index.max()}")
    
    # Preview
    print("\nSentiment features preview:")
    print(final_sentiment_df.head())
    print(f"\nFeature columns: {len(final_sentiment_df.columns)}")
    print(f"Sample features: {list(final_sentiment_df.columns[:10])}")


if __name__ == "__main__":
    main()