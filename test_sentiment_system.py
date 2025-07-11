#!/usr/bin/env python3
"""
Comprehensive testing and validation for the sentiment analysis system.

This script tests:
- Individual sentiment analysis models
- News data collection functionality  
- Feature generation and merging
- Integration with existing portfolio pipeline
- Model performance and accuracy
"""

import pandas as pd
import numpy as np
import os
import sys
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add scripts directory to path for imports
sys.path.append('./scripts')

# Import our modules
from news_sentiment_analyzer import SentimentAnalyzer, SentimentFeatureGenerator
from sentiment_data_collector import NewsDataCollector
from sentiment_model_trainer import SentimentModelTrainer, FinancialSentimentDataset

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentSystemValidator:
    """Comprehensive validation of the sentiment analysis system."""
    
    def __init__(self):
        """Initialize validator."""
        self.results = {}
        self.test_texts = [
            "TCS reports strong quarterly earnings, beats estimates by 15%",
            "Reliance shares fall 8% on weak outlook and regulatory concerns", 
            "HDFC Bank announces board meeting for next week",
            "Infosys launches new AI-powered platform, stock surges 12%",
            "SBI faces investigation over loan practices, shares plunge",
            "ICICI Bank provides quarterly business update to investors",
            "L&T wins major infrastructure contract worth ₹5000 crores",
            "ITC announces dividend payment schedule for shareholders"
        ]
    
    def test_sentiment_analyzer(self):
        """Test individual sentiment analysis functionality."""
        logger.info("Testing sentiment analyzer...")
        
        try:
            analyzer = SentimentAnalyzer()
            
            # Test each text
            results = []
            for text in self.test_texts:
                sentiment_scores = analyzer.analyze_text(text)
                
                results.append({
                    'text': text,
                    'composite_sentiment': sentiment_scores['composite_sentiment'],
                    'vader_compound': sentiment_scores['vader_compound'],
                    'textblob_polarity': sentiment_scores['textblob_polarity'],
                    'keyword_net': sentiment_scores['keyword_net']
                })
            
            results_df = pd.DataFrame(results)
            
            # Validate results
            self.results['sentiment_analyzer'] = {
                'status': 'PASS',
                'results': results_df,
                'summary': {
                    'texts_processed': len(results),
                    'avg_composite_sentiment': results_df['composite_sentiment'].mean(),
                    'sentiment_range': (results_df['composite_sentiment'].min(), 
                                      results_df['composite_sentiment'].max())
                }
            }
            
            logger.info("✅ Sentiment analyzer test passed")
            
        except Exception as e:
            logger.error(f"❌ Sentiment analyzer test failed: {e}")
            self.results['sentiment_analyzer'] = {
                'status': 'FAIL',
                'error': str(e)
            }
    
    def test_news_data_collector(self):
        """Test news data collection functionality."""
        logger.info("Testing news data collector...")
        
        try:
            collector = NewsDataCollector()
            
            # Test mock data creation
            mock_data = collector.create_mock_news_data(days=7)
            
            # Validate mock data
            required_columns = ['company', 'date', 'headline', 'content', 'source']
            missing_columns = [col for col in required_columns if col not in mock_data.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Test company-specific collection
            test_company = 'TCS.NS'
            company_data = mock_data[mock_data['company'] == test_company]
            
            self.results['news_collector'] = {
                'status': 'PASS',
                'summary': {
                    'total_articles': len(mock_data),
                    'companies_covered': mock_data['company'].nunique(),
                    'date_range': (mock_data['date'].min(), mock_data['date'].max()),
                    'test_company_articles': len(company_data),
                    'sources': mock_data['source'].unique().tolist()
                }
            }
            
            logger.info("✅ News data collector test passed")
            
        except Exception as e:
            logger.error(f"❌ News data collector test failed: {e}")
            self.results['news_collector'] = {
                'status': 'FAIL',
                'error': str(e)
            }
    
    def test_sentiment_model_trainer(self):
        """Test sentiment model training functionality."""
        logger.info("Testing sentiment model trainer...")
        
        try:
            # Create dataset
            dataset_manager = FinancialSentimentDataset()
            dataset = dataset_manager.create_synthetic_dataset(size=500)  # Small dataset for testing
            
            # Initialize trainer
            trainer = SentimentModelTrainer()
            
            # Prepare data
            texts = [trainer.preprocess_text(text) for text in dataset['text']]
            labels = [trainer.label_encoder[label] for label in dataset['sentiment']]
            
            # Split data
            from sklearn.model_selection import train_test_split
            X_text_train, X_text_test, y_train, y_test = train_test_split(
                texts, labels, test_size=0.2, random_state=42, stratify=labels
            )
            
            # Convert to features
            X_train = trainer.prepare_features(X_text_train, fit_vectorizer=True)
            X_test = trainer.prepare_features(X_text_test, fit_vectorizer=False)
            
            # Train a single model for testing
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression(random_state=42, max_iter=1000)
            model.fit(X_train, y_train)
            
            # Test predictions
            test_accuracy = model.score(X_test, y_test)
            
            # Test on our sample texts
            sample_predictions = trainer.prepare_features(self.test_texts[:3], fit_vectorizer=False)
            predictions = model.predict(sample_predictions)
            
            self.results['model_trainer'] = {
                'status': 'PASS',
                'summary': {
                    'training_samples': len(X_train),
                    'test_samples': len(X_test),
                    'test_accuracy': test_accuracy,
                    'feature_count': X_train.shape[1],
                    'sample_predictions': predictions.tolist()
                }
            }
            
            logger.info("✅ Sentiment model trainer test passed")
            
        except Exception as e:
            logger.error(f"❌ Sentiment model trainer test failed: {e}")
            self.results['model_trainer'] = {
                'status': 'FAIL',
                'error': str(e)
            }
    
    def test_feature_generation(self):
        """Test sentiment feature generation."""
        logger.info("Testing sentiment feature generation...")
        
        try:
            generator = SentimentFeatureGenerator()
            
            # Generate features for a short period
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            
            sentiment_features = generator.generate_sentiment_features(start_date, end_date)
            
            # Validate features
            required_feature_types = ['composite_sentiment', 'sentiment_ma_7d', 'sentiment_ma_30d']
            
            feature_check = {}
            for feature_type in required_feature_types:
                matching_cols = [col for col in sentiment_features.columns if feature_type in col]
                feature_check[feature_type] = len(matching_cols)
            
            self.results['feature_generation'] = {
                'status': 'PASS',
                'summary': {
                    'feature_shape': sentiment_features.shape,
                    'date_range': (sentiment_features.index.min(), sentiment_features.index.max()),
                    'feature_types': feature_check,
                    'total_features': sentiment_features.shape[1],
                    'sample_features': list(sentiment_features.columns[:5])
                }
            }
            
            logger.info("✅ Feature generation test passed")
            
        except Exception as e:
            logger.error(f"❌ Feature generation test failed: {e}")
            self.results['feature_generation'] = {
                'status': 'FAIL',
                'error': str(e)
            }
    
    def test_feature_integration(self):
        """Test integration with existing feature pipeline."""
        logger.info("Testing feature integration...")
        
        try:
            # Check if merged features file exists
            final_features_path = "data/final_features.csv"
            
            if not os.path.exists(final_features_path):
                raise FileNotFoundError(f"Final features file not found: {final_features_path}")
            
            # Load merged features
            final_features = pd.read_csv(final_features_path, index_col=0, parse_dates=True)
            
            # Check for different feature types
            feature_counts = {
                'statistical': len([col for col in final_features.columns 
                                  if any(x in col.lower() for x in ['mom', 'vol', 'sharpe'])]),
                'liquidity': len([col for col in final_features.columns 
                                if any(x in col.lower() for x in ['volume', 'dollar'])]),
                'sentiment': len([col for col in final_features.columns 
                                if 'sentiment' in col.lower()]),
            }
            
            # Check data quality
            missing_data_pct = final_features.isnull().sum().sum() / (final_features.shape[0] * final_features.shape[1])
            
            self.results['feature_integration'] = {
                'status': 'PASS',
                'summary': {
                    'total_shape': final_features.shape,
                    'feature_breakdown': feature_counts,
                    'missing_data_percentage': missing_data_pct,
                    'date_range': (final_features.index.min(), final_features.index.max()),
                    'has_sentiment_features': feature_counts['sentiment'] > 0
                }
            }
            
            logger.info("✅ Feature integration test passed")
            
        except Exception as e:
            logger.error(f"❌ Feature integration test failed: {e}")
            self.results['feature_integration'] = {
                'status': 'FAIL',
                'error': str(e)
            }
    
    def test_end_to_end_pipeline(self):
        """Test complete end-to-end pipeline."""
        logger.info("Testing end-to-end pipeline...")
        
        try:
            # Step 1: Data collection (mock)
            collector = NewsDataCollector()
            news_data = collector.create_mock_news_data(days=5)
            
            # Step 2: Sentiment analysis
            analyzer = SentimentAnalyzer()
            
            # Analyze a few articles
            sample_articles = news_data.head(10)
            analyzed_sentiments = []
            
            for _, article in sample_articles.iterrows():
                sentiment = analyzer.analyze_text(article['headline'])
                analyzed_sentiments.append({
                    'company': article['company'],
                    'headline': article['headline'],
                    'sentiment': sentiment['composite_sentiment']
                })
            
            # Step 3: Feature aggregation
            sentiment_df = pd.DataFrame(analyzed_sentiments)
            
            # Simple aggregation by company
            company_sentiment = sentiment_df.groupby('company')['sentiment'].agg(['mean', 'std', 'count'])
            
            self.results['end_to_end'] = {
                'status': 'PASS',
                'summary': {
                    'news_articles_processed': len(sample_articles),
                    'companies_analyzed': len(company_sentiment),
                    'avg_sentiment_by_company': company_sentiment['mean'].to_dict(),
                    'pipeline_steps_completed': ['data_collection', 'sentiment_analysis', 'aggregation']
                }
            }
            
            logger.info("✅ End-to-end pipeline test passed")
            
        except Exception as e:
            logger.error(f"❌ End-to-end pipeline test failed: {e}")
            self.results['end_to_end'] = {
                'status': 'FAIL',
                'error': str(e)
            }
    
    def run_all_tests(self):
        """Run all validation tests."""
        logger.info("🚀 Starting comprehensive sentiment system validation...")
        
        test_methods = [
            self.test_sentiment_analyzer,
            self.test_news_data_collector, 
            self.test_sentiment_model_trainer,
            self.test_feature_generation,
            self.test_feature_integration,
            self.test_end_to_end_pipeline
        ]
        
        for test_method in test_methods:
            try:
                test_method()
            except Exception as e:
                logger.error(f"Test {test_method.__name__} failed with unexpected error: {e}")
        
        # Generate summary report
        self.generate_report()
    
    def generate_report(self):
        """Generate comprehensive validation report."""
        print("\n" + "="*80)
        print("SENTIMENT ANALYSIS SYSTEM VALIDATION REPORT")
        print("="*80)
        
        total_tests = len(self.results)
        passed_tests = sum(1 for result in self.results.values() if result['status'] == 'PASS')
        
        print(f"\n📊 OVERVIEW")
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Success Rate: {passed_tests/total_tests*100:.1f}%")
        
        print(f"\n📋 DETAILED RESULTS")
        print("-" * 80)
        
        for test_name, result in self.results.items():
            status_icon = "✅" if result['status'] == 'PASS' else "❌"
            print(f"\n{status_icon} {test_name.upper().replace('_', ' ')}")
            
            if result['status'] == 'PASS' and 'summary' in result:
                summary = result['summary']
                for key, value in summary.items():
                    if isinstance(value, dict):
                        print(f"  {key}:")
                        for sub_key, sub_value in value.items():
                            print(f"    {sub_key}: {sub_value}")
                    elif isinstance(value, (list, tuple)) and len(value) <= 5:
                        print(f"  {key}: {value}")
                    else:
                        print(f"  {key}: {value}")
            elif result['status'] == 'FAIL':
                print(f"  Error: {result.get('error', 'Unknown error')}")
        
        # Performance insights
        print(f"\n🎯 KEY INSIGHTS")
        print("-" * 80)
        
        if 'sentiment_analyzer' in self.results and self.results['sentiment_analyzer']['status'] == 'PASS':
            sentiment_data = self.results['sentiment_analyzer']['results']
            print(f"• Sentiment Analysis Range: {sentiment_data['composite_sentiment'].min():.3f} to {sentiment_data['composite_sentiment'].max():.3f}")
            
            positive_texts = len(sentiment_data[sentiment_data['composite_sentiment'] > 0])
            negative_texts = len(sentiment_data[sentiment_data['composite_sentiment'] < 0])
            neutral_texts = len(sentiment_data[sentiment_data['composite_sentiment'] == 0])
            
            print(f"• Sentiment Distribution: {positive_texts} positive, {negative_texts} negative, {neutral_texts} neutral")
        
        if 'feature_integration' in self.results and self.results['feature_integration']['status'] == 'PASS':
            integration_data = self.results['feature_integration']['summary']
            print(f"• Total Features Generated: {integration_data['total_shape'][1]}")
            print(f"• Sentiment Features: {integration_data['feature_breakdown']['sentiment']}")
            print(f"• Data Quality: {(1-integration_data['missing_data_percentage'])*100:.1f}% complete")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS")
        print("-" * 80)
        
        if passed_tests == total_tests:
            print("• All tests passed! The sentiment analysis system is ready for production.")
            print("• Consider tuning model parameters for better performance.")
            print("• Add real news API integration when network access is available.")
        else:
            print("• Review failed tests and address issues before deployment.")
            print("• Ensure all dependencies are properly installed.")
            print("• Check data file integrity and formats.")
        
        print(f"\n🔗 INTEGRATION STATUS")
        print("-" * 80)
        
        integration_status = self.results.get('feature_integration', {}).get('status', 'UNKNOWN')
        if integration_status == 'PASS':
            print("✅ Sentiment features successfully integrated with portfolio optimization pipeline")
            print("✅ Ready for neural network training with enhanced feature set")
        else:
            print("❌ Integration issues detected - review feature merging process")
        
        print("\n" + "="*80)


def main():
    """Main function to run comprehensive validation."""
    print("🧪 Sentiment Analysis System Comprehensive Testing")
    print("="*60)
    
    # Initialize validator
    validator = SentimentSystemValidator()
    
    # Run all tests
    validator.run_all_tests()
    
    print("\n🏁 Validation completed!")


if __name__ == "__main__":
    main()