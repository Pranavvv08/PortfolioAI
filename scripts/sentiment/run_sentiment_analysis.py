import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from scripts.sentiment.news_sentiment_analyzer import IndianStockSentimentAnalyzer
import pandas as pd

def main():
    """Run sentiment analysis for all NIFTY 50 stocks"""
    
    # Load your existing stock symbols
    try:
        price_df = pd.read_csv("data/price_matrix.csv", index_col=0)
        symbols = price_df.columns.tolist()
    except FileNotFoundError:
        # Fallback to hardcoded list
        symbols = [
            "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
            "ITC.NS", "LT.NS", "KOTAKBANK.NS", "SBIN.NS", "BHARTIARTL.NS"
        ]  # Add more symbols as needed
    
    print(f"Running sentiment analysis for {len(symbols)} stocks")
    
    # Initialize sentiment analyzer
    analyzer = IndianStockSentimentAnalyzer()
    
    # Process all stocks
    sentiment_features = analyzer.process_all_stocks(symbols, days=7)
    
    # Save sentiment features
    os.makedirs("data", exist_ok=True)
    sentiment_features.to_csv("data/sentiment_features.csv")
    
    print("\nSentiment analysis complete!")
    print("Files created:")
    print("   - data/sentiment_features.csv (sentiment features)")
    print("   - data/news_data.csv (raw news data)")
    
    # Display summary
    print("\nSentiment Summary:")
    for col in sentiment_features.columns:
        if '_sentiment_mean' in col:
            symbol = col.replace('_sentiment_mean', '')
            sentiment_mean = sentiment_features[col].iloc[0]
            news_volume = sentiment_features.get(f'{symbol}_news_volume', [0]).iloc[0]
            print(f"   {symbol}: Sentiment={sentiment_mean:.3f}, News Volume={news_volume}")

if __name__ == "__main__":
    main()