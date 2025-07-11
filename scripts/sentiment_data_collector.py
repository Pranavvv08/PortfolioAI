#!/usr/bin/env python3
"""
News data collection system for NIFTY 50 stocks.

This module handles:
- Multi-source news collection (News API, RSS feeds, web scraping)
- Company name mapping and filtering
- Rate limiting and API management
- Text preprocessing and cleaning
- Date alignment with stock price data
"""

import pandas as pd
import numpy as np
import os
import time
import logging
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
import re
from requests_ratelimiter import LimiterSession
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NewsDataCollector:
    """
    Multi-source news data collector for NIFTY 50 stocks.
    """
    
    def __init__(self, news_api_key: Optional[str] = None):
        """Initialize news data collector."""
        self.news_api_key = news_api_key
        
        # Rate limiting session
        self.session = LimiterSession(per_minute=60)  # 60 requests per minute
        
        # NIFTY 50 company mappings
        self.company_mapping = {
            'RELIANCE.NS': {
                'names': ['Reliance Industries', 'RIL', 'Mukesh Ambani', 'Jio'],
                'keywords': ['reliance', 'ril', 'jio', 'mukesh ambani', 'petrochemicals']
            },
            'TCS.NS': {
                'names': ['Tata Consultancy Services', 'TCS'],
                'keywords': ['tcs', 'tata consultancy', 'it services', 'software']
            },
            'INFY.NS': {
                'names': ['Infosys', 'INFY'],
                'keywords': ['infosys', 'infy', 'it services', 'narayana murthy']
            },
            'HDFCBANK.NS': {
                'names': ['HDFC Bank', 'Housing Development Finance Corporation'],
                'keywords': ['hdfc bank', 'hdfc', 'banking', 'private bank']
            },
            'ICICIBANK.NS': {
                'names': ['ICICI Bank'],
                'keywords': ['icici bank', 'icici', 'banking', 'private bank']
            },
            'ITC.NS': {
                'names': ['ITC', 'Indian Tobacco Company'],
                'keywords': ['itc', 'tobacco', 'fmcg', 'cigarettes']
            },
            'LT.NS': {
                'names': ['Larsen & Toubro', 'L&T'],
                'keywords': ['larsen toubro', 'l&t', 'engineering', 'construction']
            },
            'KOTAKBANK.NS': {
                'names': ['Kotak Mahindra Bank', 'Kotak Bank'],
                'keywords': ['kotak', 'kotak mahindra', 'banking', 'uday kotak']
            },
            'SBIN.NS': {
                'names': ['State Bank of India', 'SBI'],
                'keywords': ['sbi', 'state bank', 'public sector bank', 'government bank']
            },
            'BHARTIARTL.NS': {
                'names': ['Bharti Airtel', 'Airtel'],
                'keywords': ['bharti airtel', 'airtel', 'telecom', 'mobile network']
            }
            # Add more mappings as needed...
        }
        
        # RSS feeds for financial news
        self.rss_feeds = [
            'https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms',
            'https://www.business-standard.com/rss/markets-106.rss',
            'https://feeds.feedburner.com/NDTV-Business',
            'https://www.moneycontrol.com/rss/results.xml',
            'https://www.livemint.com/rss/markets',
        ]
        
        # Company-specific search terms
        self.search_terms = self._generate_search_terms()
        
        logger.info("News data collector initialized")
    
    def _generate_search_terms(self) -> Dict[str, List[str]]:
        """Generate search terms for each company."""
        search_terms = {}
        
        for symbol, info in self.company_mapping.items():
            terms = []
            terms.extend(info['names'])
            terms.extend(info['keywords'])
            search_terms[symbol] = terms
        
        return search_terms
    
    def fetch_news_api_data(self, company_symbol: str, days: int = 30) -> List[Dict]:
        """Fetch news data from News API."""
        if not self.news_api_key:
            logger.warning("News API key not provided, skipping News API data collection")
            return []
        
        try:
            from newsapi import NewsApiClient
            newsapi = NewsApiClient(api_key=self.news_api_key)
            
            # Get search terms for the company
            search_terms = self.search_terms.get(company_symbol, [company_symbol.replace('.NS', '')])
            
            news_articles = []
            
            for term in search_terms[:3]:  # Limit to top 3 terms to avoid rate limits
                try:
                    # Calculate date range
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=days)
                    
                    # Search for news
                    response = newsapi.get_everything(
                        q=term,
                        language='en',
                        from_param=start_date.strftime('%Y-%m-%d'),
                        to=end_date.strftime('%Y-%m-%d'),
                        sort_by='publishedAt',
                        page_size=20  # Limit articles per term
                    )
                    
                    if response['status'] == 'ok':
                        for article in response['articles']:
                            news_articles.append({
                                'company': company_symbol,
                                'date': datetime.strptime(article['publishedAt'][:10], '%Y-%m-%d'),
                                'headline': article['title'],
                                'content': article['content'] or article['description'] or '',
                                'source': article['source']['name'],
                                'url': article['url']
                            })
                    
                    # Rate limiting
                    time.sleep(1)
                    
                except Exception as e:
                    logger.warning(f"Error fetching News API data for {term}: {e}")
                    continue
            
            logger.info(f"Fetched {len(news_articles)} articles for {company_symbol} from News API")
            return news_articles
            
        except ImportError:
            logger.warning("newsapi-python not installed, skipping News API collection")
            return []
        except Exception as e:
            logger.error(f"Error with News API: {e}")
            return []
    
    def fetch_rss_feeds(self, company_symbol: str) -> List[Dict]:
        """Fetch news from RSS feeds."""
        news_articles = []
        
        # Get search terms for filtering
        search_terms = self.search_terms.get(company_symbol, [company_symbol.replace('.NS', '')])
        search_pattern = '|'.join(search_terms)
        
        for feed_url in self.rss_feeds:
            try:
                logger.info(f"Fetching RSS feed: {feed_url}")
                
                response = self.session.get(feed_url, timeout=30)
                response.raise_for_status()
                
                # Parse RSS feed
                root = ET.fromstring(response.content)
                
                # Find all items
                items = root.findall('.//item')
                
                for item in items:
                    title = item.find('title')
                    description = item.find('description')
                    pub_date = item.find('pubDate')
                    link = item.find('link')
                    
                    if title is not None:
                        title_text = title.text or ''
                        desc_text = description.text if description is not None else ''
                        
                        # Check if article mentions the company
                        combined_text = (title_text + ' ' + desc_text).lower()
                        
                        if re.search(search_pattern, combined_text, re.IGNORECASE):
                            # Parse publication date
                            try:
                                if pub_date is not None and pub_date.text:
                                    # Handle different date formats
                                    pub_date_text = pub_date.text
                                    # Remove day of week and timezone info
                                    pub_date_text = re.sub(r'^[A-Za-z]{3},?\s*', '', pub_date_text)
                                    pub_date_text = re.sub(r'\s*[+-]\d{4}$', '', pub_date_text)
                                    
                                    article_date = datetime.strptime(pub_date_text.strip(), '%d %b %Y %H:%M:%S')
                                else:
                                    article_date = datetime.now()
                            except:
                                article_date = datetime.now()
                            
                            news_articles.append({
                                'company': company_symbol,
                                'date': article_date,
                                'headline': title_text,
                                'content': desc_text,
                                'source': feed_url.split('/')[2],  # Extract domain
                                'url': link.text if link is not None else ''
                            })
                
                # Rate limiting
                time.sleep(2)
                
            except Exception as e:
                logger.warning(f"Error fetching RSS feed {feed_url}: {e}")
                continue
        
        logger.info(f"Fetched {len(news_articles)} relevant articles for {company_symbol} from RSS feeds")
        return news_articles
    
    def scrape_financial_websites(self, company_symbol: str) -> List[Dict]:
        """Scrape news from financial websites (basic implementation)."""
        # This is a simplified example - in production, you'd need more sophisticated scraping
        # and should respect robots.txt and rate limits
        
        news_articles = []
        
        try:
            # Example: scraping from a hypothetical financial news site
            # In practice, you'd implement specific scrapers for different sites
            
            search_terms = self.search_terms.get(company_symbol, [company_symbol.replace('.NS', '')])
            
            # This is just a placeholder - implement actual scraping logic
            logger.info(f"Web scraping placeholder for {company_symbol}")
            
            # For now, return empty list to avoid actual scraping without permission
            return news_articles
            
        except Exception as e:
            logger.warning(f"Error in web scraping for {company_symbol}: {e}")
            return news_articles
    
    def preprocess_news_text(self, text: str) -> str:
        """Clean and preprocess news text."""
        if not text or not isinstance(text, str):
            return ""
        
        # Remove HTML tags
        text = BeautifulSoup(text, 'html.parser').get_text()
        
        # Remove special characters and extra whitespace
        text = re.sub(r'[^\w\s.,!?-]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common news artifacts
        text = re.sub(r'(REUTERS|PTI|ANI|ET Bureau|Bloomberg)[\s:-]*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'(Read more|Continue reading|Click here).*', '', text, flags=re.IGNORECASE)
        
        return text.strip()
    
    def collect_company_news(self, company_symbol: str, days: int = 30) -> pd.DataFrame:
        """Collect news for a specific company from all sources."""
        logger.info(f"Collecting news for {company_symbol} for last {days} days")
        
        all_articles = []
        
        # Collect from News API
        try:
            news_api_articles = self.fetch_news_api_data(company_symbol, days)
            all_articles.extend(news_api_articles)
        except Exception as e:
            logger.warning(f"News API collection failed for {company_symbol}: {e}")
        
        # Collect from RSS feeds
        try:
            rss_articles = self.fetch_rss_feeds(company_symbol)
            all_articles.extend(rss_articles)
        except Exception as e:
            logger.warning(f"RSS collection failed for {company_symbol}: {e}")
        
        # Collect from web scraping (disabled for safety)
        # scraping_articles = self.scrape_financial_websites(company_symbol)
        # all_articles.extend(scraping_articles)
        
        if not all_articles:
            logger.warning(f"No articles found for {company_symbol}")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(all_articles)
        
        # Preprocess text
        df['headline'] = df['headline'].apply(self.preprocess_news_text)
        df['content'] = df['content'].apply(self.preprocess_news_text)
        
        # Combine headline and content
        df['full_text'] = df['headline'] + ' ' + df['content']
        
        # Remove duplicates based on headline similarity
        df = self._remove_duplicate_articles(df)
        
        # Filter by date range
        cutoff_date = datetime.now() - timedelta(days=days)
        df = df[df['date'] >= cutoff_date]
        
        # Sort by date
        df = df.sort_values('date', ascending=False)
        
        logger.info(f"Collected {len(df)} unique articles for {company_symbol}")
        return df
    
    def _remove_duplicate_articles(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate articles based on headline similarity."""
        if df.empty:
            return df
        
        # Simple deduplication based on headline
        df = df.drop_duplicates(subset=['headline', 'date'], keep='first')
        
        # More sophisticated deduplication could use text similarity
        # For now, keep it simple
        
        return df
    
    def collect_all_companies_news(self, days: int = 30) -> pd.DataFrame:
        """Collect news for all NIFTY 50 companies."""
        logger.info(f"Collecting news for all companies for last {days} days")
        
        all_company_news = []
        
        for company_symbol in self.company_mapping.keys():
            try:
                company_news = self.collect_company_news(company_symbol, days)
                if not company_news.empty:
                    all_company_news.append(company_news)
                
                # Rate limiting between companies
                time.sleep(3)
                
            except Exception as e:
                logger.error(f"Error collecting news for {company_symbol}: {e}")
                continue
        
        if not all_company_news:
            logger.warning("No news articles collected for any company")
            return pd.DataFrame()
        
        # Combine all company news
        combined_df = pd.concat(all_company_news, ignore_index=True)
        
        # Final deduplication across all companies
        combined_df = self._remove_duplicate_articles(combined_df)
        
        logger.info(f"Total articles collected: {len(combined_df)}")
        return combined_df
    
    def save_news_data(self, news_df: pd.DataFrame, filename: str = "collected_news.csv"):
        """Save collected news data to CSV."""
        if news_df.empty:
            logger.warning("No news data to save")
            return
        
        # Ensure data directory exists
        os.makedirs("data", exist_ok=True)
        
        filepath = os.path.join("data", filename)
        news_df.to_csv(filepath, index=False)
        
        logger.info(f"News data saved to {filepath}")
        logger.info(f"Saved {len(news_df)} articles")
    
    def create_mock_news_data(self, days: int = 30) -> pd.DataFrame:
        """Create realistic mock news data for testing."""
        logger.info(f"Creating mock news data for {days} days")
        
        # Generate realistic headlines and content
        positive_templates = [
            "{company} reports {metric}% increase in quarterly revenue",
            "{company} beats analyst estimates, shares surge {metric}%",
            "{company} announces major expansion into {sector}",
            "{company} launches innovative {product} service",
            "{company} receives upgrade from {analyst} to buy",
            "{company} signs deal worth ₹{value} crores",
            "{company} posts strong {period} results, outlook positive"
        ]
        
        negative_templates = [
            "{company} faces regulatory investigation over {issue}",
            "{company} reports {metric}% decline in profits",
            "{company} shares fall {metric}% on weak outlook",
            "{company} delays {project} due to {reason}",
            "{company} downgrades guidance amid {challenge}",
            "{company} CEO steps down after {issue}",
            "{company} struggles with {challenge} in {market}"
        ]
        
        neutral_templates = [
            "{company} announces board meeting on {date}",
            "{company} files quarterly compliance report",
            "{company} updates on business operations",
            "{company} holds investor conference call",
            "{company} provides {period} business update",
            "{company} announces dividend payment schedule",
            "{company} participates in industry conference"
        ]
        
        # Generate mock data
        np.random.seed(42)
        mock_articles = []
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        for company_symbol in list(self.company_mapping.keys())[:10]:  # Limit for testing
            company_name = company_symbol.replace('.NS', '').replace('&', 'and')
            
            # Generate 5-15 articles per company
            num_articles = np.random.randint(5, 16)
            
            for _ in range(num_articles):
                # Random date within range
                random_date = start_date + timedelta(
                    days=np.random.randint(0, days),
                    hours=np.random.randint(0, 24),
                    minutes=np.random.randint(0, 60)
                )
                
                # Random sentiment
                sentiment_type = np.random.choice(['positive', 'negative', 'neutral'], 
                                                p=[0.3, 0.2, 0.5])
                
                if sentiment_type == 'positive':
                    template = np.random.choice(positive_templates)
                    headline = template.format(
                        company=company_name,
                        metric=np.random.randint(5, 25),
                        sector=np.random.choice(['digital', 'renewable energy', 'healthcare']),
                        product=np.random.choice(['fintech', 'AI-powered', 'mobile']),
                        analyst=np.random.choice(['Goldman Sachs', 'Morgan Stanley', 'CLSA']),
                        value=np.random.randint(100, 5000),
                        period=np.random.choice(['Q1', 'Q2', 'Q3', 'Q4', 'FY23'])
                    )
                elif sentiment_type == 'negative':
                    template = np.random.choice(negative_templates)
                    headline = template.format(
                        company=company_name,
                        metric=np.random.randint(5, 20),
                        issue=np.random.choice(['compliance', 'accounting', 'governance']),
                        project=np.random.choice(['expansion', 'acquisition', 'IPO']),
                        reason=np.random.choice(['market conditions', 'regulatory issues']),
                        challenge=np.random.choice(['supply chain', 'inflation', 'competition']),
                        market=np.random.choice(['domestic', 'international', 'emerging'])
                    )
                else:
                    template = np.random.choice(neutral_templates)
                    headline = template.format(
                        company=company_name,
                        date=random_date.strftime('%B %d'),
                        period=np.random.choice(['quarterly', 'annual', 'monthly'])
                    )
                
                # Generate content based on headline
                content = f"According to latest reports, {headline.lower()}. " + \
                         f"Industry experts are closely monitoring the developments. " + \
                         f"More details are expected to be announced soon."
                
                mock_articles.append({
                    'company': company_symbol,
                    'date': random_date,
                    'headline': headline,
                    'content': content,
                    'full_text': headline + ' ' + content,
                    'source': np.random.choice(['ET', 'BS', 'Mint', 'MC', 'NDTV']),
                    'url': f"https://example.com/news/{np.random.randint(1000, 9999)}"
                })
        
        mock_df = pd.DataFrame(mock_articles)
        mock_df = mock_df.sort_values('date', ascending=False)
        
        logger.info(f"Created {len(mock_df)} mock news articles")
        return mock_df


def main():
    """Main function to demonstrate news data collection."""
    logger.info("Starting news data collection...")
    
    # Initialize collector
    # In production, pass actual News API key: collector = NewsDataCollector(news_api_key="your_key_here")
    collector = NewsDataCollector()
    
    # For testing, create mock data
    news_df = collector.create_mock_news_data(days=30)
    
    # Save the data
    collector.save_news_data(news_df, "mock_news_data.csv")
    
    # Display summary
    print("\nNews collection summary:")
    print(f"Total articles: {len(news_df)}")
    print(f"Companies covered: {news_df['company'].nunique()}")
    print(f"Date range: {news_df['date'].min()} to {news_df['date'].max()}")
    print(f"Sources: {news_df['source'].unique()}")
    
    # Show sample articles
    print("\nSample articles:")
    for _, article in news_df.head(3).iterrows():
        print(f"\n{article['company']} - {article['date'].strftime('%Y-%m-%d')}")
        print(f"Headline: {article['headline']}")
        print(f"Source: {article['source']}")


if __name__ == "__main__":
    main()