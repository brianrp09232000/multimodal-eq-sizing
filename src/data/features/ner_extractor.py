import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import pandas as pd
from typing import List, Set

class FinanceEntityExtractor:
    """
    Wraps a BERT-NER model to extract named entities (ORG, PER, LOC) from text
    Computes the novelty feature required by the news tower so we first 
    identify which companies or people are mentioned in the headlines
    """
    def __init__(self, model_name="dslim/bert-base-NER", device=None):
        """NER extractor
        model_name: HuggingFace model ID
        device: 'cuda' or 'cpu'. If None, detects automatically
        """
        # Autodetect GPU for faster extraction
        if device is None:
            device = 0 if torch.cuda.is_available() else -1
        
        print(f"Loading NER Model: {model_name} on {device}")
        # Load pretrained tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        
        # Initialize pipeline with simple aggregation to merge subtokens
        self.nlp = pipeline(
            "ner", 
            model=self.model, 
            tokenizer=self.tokenizer, 
            aggregation_strategy="simple",
            device=device
        )

    def extract_unique_entities(self, sentences: List[str], confidence_threshold=0.8) -> Set[str]:
        """
        Runs NER on a list of sentences in one document and returns unique entity names
        Iterates through sentences, filters by confidence score, and lowercases for consistent comparison
        """
        unique_entities = set()
        
        # Loop through sentences to avoid passing massive lists that might OOM the pipeline
        for sent in sentences:
            if not sent.strip(): 
                continue
                
            results = self.nlp(sent)
            for r in results:
                # Filter noise: only keep high confidence entities
                if r['score'] > confidence_threshold:
                    unique_entities.add(r['word'].lower())
        
        return unique_entities

def compute_novelty_feature(df: pd.DataFrame, ticker_col='ticker', date_col='Date', text_col='sentences') -> pd.DataFrame:
    """
    Computes the novelty score for Leg 2
    Novelty (% new entities vs 3 days)
    
    1. Extracts entities for all rows
    2. Sorts by time to simulate a causal stream
    3. Maintains a rolling 3-day history buffer to compare current entities against
    """
    extractor = FinanceEntityExtractor()
    
    # 1. Extract Entities pre-computation step
    print("Extracting entities from headlines...")
    # Applies the extractor to every document in the dataframe
    df['entities'] = df[text_col].apply(lambda x: extractor.extract_unique_entities(x))
    
    # 2. Compute novelty per ticker
    # Sort by ticker and date to ensure we process history sequentially
    df = df.sort_values([ticker_col, date_col])
    
    novelty_scores = []
    
    # Group by ticker to handle history independently for each stock
    for ticker, group in df.groupby(ticker_col):
        # Sliding window buffer to store entities from the past 3 days
        history_window = [] 
        
        for i, row in group.iterrows():
            current_entities = row['entities']
            
            # Handle empty news days
            if not current_entities:
                novelty_scores.append(0.0)
                history_window.append(set()) # Push empty to history to maintain time window
                if len(history_window) > 3: history_window.pop(0)
                continue
                
            # Flatten 3-day history into one set of known entities
            past_entities = set().union(*history_window)
            
            # Identify strictly new entities appearing today
            new_entities = current_entities - past_entities
            
            # Calculate ratio: new / total
            score = len(new_entities) / len(current_entities)
            novelty_scores.append(score)
            
            # Update rolling window: Add today, remove oldest day
            history_window.append(current_entities)
            if len(history_window) > 3: history_window.pop(0)

    # Assign calculated scores back to the dataframe
    # Logic works because we iterate in the same sort order as the dataframe
    df['novelty'] = novelty_scores
    
    return df