#!/usr/bin/env python3
"""
Custom sentiment model trainer for financial text analysis.

This module implements:
- Financial text dataset preparation
- Fine-tuning BERT/FinBERT for Indian market context
- Model evaluation and validation
- Export trained model for inference
"""

import pandas as pd
import numpy as np
import os
import logging
import pickle
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Machine learning libraries
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Text processing
import re
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Deep learning (optional, fallback to sklearn if torch not available)
try:
    import torch
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        TrainingArguments, Trainer, AutoConfig
    )
    from datasets import Dataset
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    Dataset = None  # Define as None for type hints
    logging.warning("Transformers not available, using sklearn models only")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FinancialSentimentDataset:
    """Dataset creation and management for financial sentiment analysis."""
    
    def __init__(self):
        """Initialize dataset manager."""
        self.vader = SentimentIntensityAnalyzer()
        
        # Financial sentiment keywords
        self.positive_indicators = [
            'profit', 'growth', 'bullish', 'buy', 'upgrade', 'beat', 'outperform',
            'strong', 'surge', 'rally', 'boom', 'record', 'increase', 'rise',
            'gain', 'revenue', 'earnings', 'expansion', 'acquisition', 'merger'
        ]
        
        self.negative_indicators = [
            'loss', 'decline', 'bearish', 'sell', 'downgrade', 'miss', 'underperform',
            'weak', 'crash', 'plunge', 'recession', 'bankruptcy', 'decrease', 'fall',
            'drop', 'debt', 'layoffs', 'losses', 'contraction', 'investigation'
        ]
        
        self.neutral_indicators = [
            'meeting', 'report', 'announcement', 'conference', 'update', 'schedule',
            'compliance', 'filing', 'board', 'AGM', 'disclosure', 'notification'
        ]
    
    def create_synthetic_dataset(self, size: int = 1000) -> pd.DataFrame:
        """Create synthetic financial sentiment dataset for training."""
        logger.info(f"Creating synthetic dataset with {size} samples")
        
        # Templates for different sentiment categories
        positive_templates = [
            "{company} reports strong {period} earnings, beats estimates by {percent}%",
            "{company} stock surges {percent}% on positive outlook",
            "{company} announces major acquisition worth ₹{amount} crores",
            "{company} receives buy rating from {analyst}",
            "{company} quarterly profit jumps {percent}% year-on-year",
            "{company} expands operations with new {facility}",
            "{company} launches innovative {product} in market",
            "{company} board approves {percent}% dividend increase"
        ]
        
        negative_templates = [
            "{company} shares fall {percent}% on weak earnings",
            "{company} faces regulatory probe over {issue}",
            "{company} quarterly losses widen to ₹{amount} crores",
            "{company} downgrades guidance amid {challenge}",
            "{company} stock hits {period} low on concerns",
            "{company} announces {number} job cuts in restructuring",
            "{company} delays {project} due to market conditions",
            "{company} rating cut to sell by {analyst}"
        ]
        
        neutral_templates = [
            "{company} announces board meeting on {date}",
            "{company} files quarterly compliance report",
            "{company} updates on business operations",
            "{company} holds investor conference call",
            "{company} provides {period} business update",
            "{company} announces ex-dividend date",
            "{company} participates in industry event",
            "{company} submits regulatory filing"
        ]
        
        # Company names (simplified)
        companies = ['TCS', 'Reliance', 'HDFC Bank', 'Infosys', 'ICICI Bank', 
                    'ITC', 'Kotak Bank', 'SBI', 'Airtel', 'L&T']
        
        np.random.seed(42)
        data = []
        
        # Generate balanced dataset
        samples_per_class = size // 3
        
        for sentiment, templates in [
            ('positive', positive_templates),
            ('negative', negative_templates), 
            ('neutral', neutral_templates)
        ]:
            for _ in range(samples_per_class):
                template = np.random.choice(templates)
                company = np.random.choice(companies)
                
                # Fill template with random values
                text = template.format(
                    company=company,
                    period=np.random.choice(['Q1', 'Q2', 'Q3', 'Q4', 'FY23', '6-month']),
                    percent=np.random.randint(5, 30),
                    amount=np.random.randint(100, 5000),
                    analyst=np.random.choice(['Goldman Sachs', 'Morgan Stanley', 'CLSA']),
                    facility=np.random.choice(['manufacturing unit', 'R&D center', 'office']),
                    product=np.random.choice(['fintech solution', 'mobile app', 'service']),
                    issue=np.random.choice(['compliance', 'accounting', 'governance']),
                    challenge=np.random.choice(['supply chain', 'inflation', 'competition']),
                    number=np.random.randint(100, 1000),
                    project=np.random.choice(['expansion', 'IPO', 'acquisition']),
                    date=f"March {np.random.randint(1, 30)}"
                )
                
                data.append({
                    'text': text,
                    'sentiment': sentiment,
                    'company': company
                })
        
        df = pd.DataFrame(data)
        
        # Add some realistic noise and variations
        df = self._add_text_variations(df)
        
        logger.info(f"Created dataset with {len(df)} samples")
        logger.info(f"Sentiment distribution: {df['sentiment'].value_counts().to_dict()}")
        
        return df
    
    def _add_text_variations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add realistic variations to synthetic text."""
        
        def add_noise(text):
            # Add some realistic financial text patterns
            variations = [
                f"Breaking: {text}",
                f"Market Update: {text}",
                f"According to sources, {text.lower()}",
                f"{text}. Industry experts believe this will impact market sentiment.",
                f"{text}. More details expected soon.",
            ]
            
            if np.random.random() < 0.3:  # 30% chance of variation
                return np.random.choice(variations)
            return text
        
        df['text'] = df['text'].apply(add_noise)
        return df
    
    def prepare_real_world_dataset(self, news_file: str = "data/mock_news_data.csv") -> pd.DataFrame:
        """Prepare dataset from real news data with auto-labeling."""
        logger.info("Preparing real-world dataset from news data")
        
        if not os.path.exists(news_file):
            logger.warning(f"News file {news_file} not found, creating synthetic dataset instead")
            return self.create_synthetic_dataset()
        
        # Load news data
        news_df = pd.read_csv(news_file)
        news_df['date'] = pd.to_datetime(news_df['date'])
        
        # Auto-label sentiment using rule-based approach
        labeled_data = []
        
        for _, row in news_df.iterrows():
            text = f"{row['headline']} {row.get('content', '')}"
            
            # Use VADER for initial labeling
            vader_scores = self.vader.polarity_scores(text)
            compound = vader_scores['compound']
            
            # Use keyword-based refinement
            text_lower = text.lower()
            positive_count = sum(1 for word in self.positive_indicators if word in text_lower)
            negative_count = sum(1 for word in self.negative_indicators if word in text_lower)
            neutral_count = sum(1 for word in self.neutral_indicators if word in text_lower)
            
            # Determine label
            if compound > 0.1 or positive_count > negative_count:
                sentiment = 'positive'
            elif compound < -0.1 or negative_count > positive_count:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'
            
            labeled_data.append({
                'text': text,
                'headline': row['headline'],
                'sentiment': sentiment,
                'company': row['company'],
                'vader_compound': compound,
                'positive_keywords': positive_count,
                'negative_keywords': negative_count
            })
        
        labeled_df = pd.DataFrame(labeled_data)
        
        logger.info(f"Labeled {len(labeled_df)} real news articles")
        logger.info(f"Sentiment distribution: {labeled_df['sentiment'].value_counts().to_dict()}")
        
        return labeled_df


class SentimentModelTrainer:
    """Train custom sentiment analysis models for financial text."""
    
    def __init__(self):
        """Initialize model trainer."""
        self.models = {}
        self.vectorizer = None
        self.label_encoder = {'positive': 2, 'neutral': 1, 'negative': 0}
        self.reverse_label_encoder = {v: k for k, v in self.label_encoder.items()}
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for model training."""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep financial terms
        text = re.sub(r'[^\w\s%₹$.-]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def prepare_features(self, texts: List[str], fit_vectorizer: bool = False) -> np.ndarray:
        """Convert texts to feature vectors."""
        
        if fit_vectorizer or self.vectorizer is None:
            self.vectorizer = TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                stop_words='english',
                min_df=2,
                max_df=0.95
            )
            features = self.vectorizer.fit_transform(texts)
        else:
            features = self.vectorizer.transform(texts)
        
        return features.toarray()
    
    def train_sklearn_models(self, X_train: np.ndarray, y_train: np.ndarray, 
                           X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Train multiple sklearn models."""
        logger.info("Training sklearn models...")
        
        models_to_train = {
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'svm': SVC(kernel='rbf', random_state=42, probability=True)
        }
        
        results = {}
        
        for name, model in models_to_train.items():
            logger.info(f"Training {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Evaluate
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)
            
            y_pred = model.predict(X_test)
            
            results[name] = {
                'model': model,
                'train_accuracy': train_score,
                'test_accuracy': test_score,
                'classification_report': classification_report(y_test, y_pred),
                'predictions': y_pred
            }
            
            # Store best model
            self.models[name] = model
            
            logger.info(f"{name} - Train: {train_score:.3f}, Test: {test_score:.3f}")
        
        return results
    
    def train_transformer_model(self, train_dataset, eval_dataset) -> Optional[Dict]:
        """Train a transformer-based model (if available)."""
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("Transformers not available, skipping transformer training")
            return None
        
        logger.info("Training transformer model...")
        
        try:
            # Use a smaller model for faster training
            model_name = "distilbert-base-uncased"
            
            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            config = AutoConfig.from_pretrained(model_name, num_labels=3)
            model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)
            
            # Training arguments
            training_args = TrainingArguments(
                output_dir='./sentiment_model_output',
                num_train_epochs=3,
                per_device_train_batch_size=16,
                per_device_eval_batch_size=16,
                warmup_steps=500,
                weight_decay=0.01,
                logging_dir='./logs',
                logging_steps=10,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
            )
            
            # Create trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=tokenizer,
            )
            
            # Train model
            trainer.train()
            
            # Evaluate
            eval_results = trainer.evaluate()
            
            # Save model
            model.save_pretrained('./trained_sentiment_model')
            tokenizer.save_pretrained('./trained_sentiment_model')
            
            self.models['transformer'] = {
                'model': model,
                'tokenizer': tokenizer,
                'trainer': trainer
            }
            
            logger.info(f"Transformer training completed. Eval loss: {eval_results.get('eval_loss', 'N/A')}")
            
            return {
                'model': model,
                'tokenizer': tokenizer,
                'eval_results': eval_results
            }
            
        except Exception as e:
            logger.error(f"Error training transformer model: {e}")
            return None
    
    def evaluate_models(self, X_test: np.ndarray, y_test: np.ndarray, 
                       test_texts: List[str]) -> Dict:
        """Comprehensive model evaluation."""
        logger.info("Evaluating all trained models...")
        
        evaluation_results = {}
        
        for name, model in self.models.items():
            if name == 'transformer':
                continue  # Skip transformer for now
            
            logger.info(f"Evaluating {name}...")
            
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
            
            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, 
                                         target_names=['negative', 'neutral', 'positive'],
                                         output_dict=True)
            
            evaluation_results[name] = {
                'accuracy': accuracy,
                'classification_report': report,
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
                'predictions': y_pred.tolist(),
                'probabilities': y_pred_proba.tolist() if y_pred_proba is not None else None
            }
        
        return evaluation_results
    
    def save_models(self, output_dir: str = "trained_models"):
        """Save trained models to disk."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save sklearn models
        for name, model in self.models.items():
            if name != 'transformer':
                model_path = os.path.join(output_dir, f"{name}_model.pkl")
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                logger.info(f"Saved {name} model to {model_path}")
        
        # Save vectorizer
        if self.vectorizer:
            vectorizer_path = os.path.join(output_dir, "vectorizer.pkl")
            with open(vectorizer_path, 'wb') as f:
                pickle.dump(self.vectorizer, f)
            logger.info(f"Saved vectorizer to {vectorizer_path}")
        
        # Save label encoder
        encoder_path = os.path.join(output_dir, "label_encoder.pkl")
        with open(encoder_path, 'wb') as f:
            pickle.dump(self.label_encoder, f)
        logger.info(f"Saved label encoder to {encoder_path}")
    
    def load_models(self, model_dir: str = "trained_models"):
        """Load trained models from disk."""
        logger.info(f"Loading models from {model_dir}")
        
        # Load sklearn models
        model_files = ['logistic_regression_model.pkl', 'random_forest_model.pkl', 'svm_model.pkl']
        
        for model_file in model_files:
            model_path = os.path.join(model_dir, model_file)
            if os.path.exists(model_path):
                model_name = model_file.replace('_model.pkl', '')
                with open(model_path, 'rb') as f:
                    self.models[model_name] = pickle.load(f)
                logger.info(f"Loaded {model_name} model")
        
        # Load vectorizer
        vectorizer_path = os.path.join(model_dir, "vectorizer.pkl")
        if os.path.exists(vectorizer_path):
            with open(vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
            logger.info("Loaded vectorizer")
        
        # Load label encoder
        encoder_path = os.path.join(model_dir, "label_encoder.pkl")
        if os.path.exists(encoder_path):
            with open(encoder_path, 'rb') as f:
                self.label_encoder = pickle.load(f)
                self.reverse_label_encoder = {v: k for k, v in self.label_encoder.items()}
            logger.info("Loaded label encoder")
    
    def predict_sentiment(self, texts: List[str], model_name: str = 'logistic_regression') -> List[Dict]:
        """Predict sentiment for new texts."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available models: {list(self.models.keys())}")
        
        if not self.vectorizer:
            raise ValueError("Vectorizer not loaded. Train a model or load from disk first.")
        
        # Preprocess texts
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # Vectorize
        X = self.prepare_features(processed_texts, fit_vectorizer=False)
        
        # Predict
        model = self.models[model_name]
        predictions = model.predict(X)
        probabilities = model.predict_proba(X) if hasattr(model, 'predict_proba') else None
        
        results = []
        for i, text in enumerate(texts):
            result = {
                'text': text,
                'predicted_sentiment': self.reverse_label_encoder[predictions[i]],
                'confidence': None
            }
            
            if probabilities is not None:
                max_prob = max(probabilities[i])
                result['confidence'] = float(max_prob)
                result['probabilities'] = {
                    'negative': float(probabilities[i][0]),
                    'neutral': float(probabilities[i][1]),
                    'positive': float(probabilities[i][2])
                }
            
            results.append(result)
        
        return results


def main():
    """Main function to train sentiment models."""
    logger.info("Starting sentiment model training...")
    
    # Initialize dataset and trainer
    dataset_manager = FinancialSentimentDataset()
    trainer = SentimentModelTrainer()
    
    # Create or load dataset
    try:
        # Try to use real news data
        dataset = dataset_manager.prepare_real_world_dataset()
    except:
        # Fallback to synthetic data
        logger.info("Using synthetic dataset for training")
        dataset = dataset_manager.create_synthetic_dataset(size=2000)
    
    # Prepare data
    logger.info("Preparing training data...")
    
    # Preprocess texts
    texts = [trainer.preprocess_text(text) for text in dataset['text']]
    labels = [trainer.label_encoder[label] for label in dataset['sentiment']]
    
    # Split data
    X_text_train, X_text_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Convert to features
    X_train = trainer.prepare_features(X_text_train, fit_vectorizer=True)
    X_test = trainer.prepare_features(X_text_test, fit_vectorizer=False)
    
    logger.info(f"Training set size: {len(X_train)}")
    logger.info(f"Test set size: {len(X_test)}")
    
    # Train sklearn models
    sklearn_results = trainer.train_sklearn_models(X_train, y_train, X_test, y_test)
    
    # Evaluate models
    evaluation_results = trainer.evaluate_models(X_test, y_test, X_text_test)
    
    # Print results
    print("\n" + "="*50)
    print("MODEL TRAINING RESULTS")
    print("="*50)
    
    for model_name, results in evaluation_results.items():
        print(f"\n{model_name.upper()} MODEL:")
        print(f"Accuracy: {results['accuracy']:.3f}")
        print(f"Precision (avg): {results['classification_report']['macro avg']['precision']:.3f}")
        print(f"Recall (avg): {results['classification_report']['macro avg']['recall']:.3f}")
        print(f"F1-score (avg): {results['classification_report']['macro avg']['f1-score']:.3f}")
    
    # Save models
    trainer.save_models()
    
    # Test prediction on sample texts
    sample_texts = [
        "TCS reports strong quarterly earnings, beats estimates",
        "Reliance shares fall on weak outlook",
        "HDFC Bank announces board meeting next week"
    ]
    
    print("\n" + "="*50)
    print("SAMPLE PREDICTIONS")
    print("="*50)
    
    for model_name in ['logistic_regression', 'random_forest']:
        if model_name in trainer.models:
            print(f"\n{model_name.upper()} predictions:")
            predictions = trainer.predict_sentiment(sample_texts, model_name)
            
            for pred in predictions:
                print(f"Text: {pred['text']}")
                print(f"Sentiment: {pred['predicted_sentiment']} (confidence: {pred.get('confidence', 'N/A'):.3f})")
                print()
    
    logger.info("Model training completed successfully!")


if __name__ == "__main__":
    main()