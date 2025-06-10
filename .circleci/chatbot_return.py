import pandas as pd
import re
from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import fuzz


class InsurancePlanMatcher:
    def __init__(self, df: pd.DataFrame, description_column: str):
        self.df = df
        self.column = description_column
        self.unique_values = df[description_column].dropna().unique()
        
        # Setup TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            token_pattern=r'\b[A-Za-z0-9%\/&-]+\b',
            ngram_range=(1, 3),
            max_features=3000
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.unique_values)
        
    def _normalize_text(self, text: str) -> str:
        """Normalize text by removing extra whitespace."""
        return re.sub(r'\s+', ' ', text.strip())
        
    def extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from text."""
        text = self._normalize_text(text)
        text_upper = text.upper()
        keywords = []
        
        # Extract acronyms, percentages, and ordinals
        keywords.extend(re.findall(r'\b[A-Z]{2,}\b', text_upper))
        keywords.extend(re.findall(r'\d+%?', text))
        keywords.extend(re.findall(r'\d+(?:ST|ND|RD|TH)', text_upper))
        
        return keywords
    
    def _calculate_scores(self, query: str) -> List[Tuple[str, float]]:
        """Calculate combined scores for all unique values."""
        query = self._normalize_text(query)
        query_lower = query.lower()
        
        # Get TF-IDF scores
        query_vector = self.vectorizer.transform([query_lower])
        tfidf_scores = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Get fuzzy matching scores
        fuzzy_scores = [
            fuzz.token_sort_ratio(query_lower, desc.lower()) / 100.0 
            for desc in self.unique_values
        ]
        
        # Get keyword scores
        keywords = self.extract_keywords(query)
        keyword_scores = []
        for desc in self.unique_values:
            if keywords:
                desc_upper = desc.upper()
                score = sum(1 for kw in keywords if kw in desc_upper) / len(keywords)
            else:
                score = 0.0
            keyword_scores.append(score)
        
        # Calculate combined scores
        combined_scores = []
        query_no_space = query_lower.replace(' ', '')
        
        for idx, desc in enumerate(self.unique_values):
            # Check if query is substring of description
            substring_bonus = 0.1 if query_no_space in desc.lower().replace(' ', '') else 0.0
            
            # Weighted combination: 35% TF-IDF, 35% fuzzy, 30% keyword + substring bonus
            combined_score = (
                0.35 * tfidf_scores[idx] + 
                0.35 * fuzzy_scores[idx] + 
                0.3 * keyword_scores[idx] + 
                substring_bonus
            )
            
            if combined_score > 0:
                combined_scores.append((desc, combined_score))
        
        # Sort by score descending
        combined_scores.sort(key=lambda x: x[1], reverse=True)
        return combined_scores
    
    def match(self, query: str, threshold: float = 0.3) -> pd.DataFrame:
        """Find the best matching insurance plan description."""
        # Get all scores
        scores = self._calculate_scores(query)
        
        if not scores:
            return pd.DataFrame()
        
        # Check if best match meets threshold
        best_match, best_score = scores[0]
        
        # First check if we have a strong keyword match
        keywords = self.extract_keywords(query)
        if keywords:
            keyword_matches = [
                (desc, score) for desc, score in scores 
                if any(kw in desc.upper() for kw in keywords)
            ]
            if keyword_matches and keyword_matches[0][1] >= 0.5:
                return self.df[self.df[self.column] == keyword_matches[0][0]]
        
        # Otherwise use the combined score
        if best_score >= threshold:
            return self.df[self.df[self.column] == best_match]
        
        return pd.DataFrame()  # No match found
    
    def get_suggestions(self, query: str, top_n: int = 10) -> List[Tuple[str, float]]:
        """Get top N suggestions for a query."""
        # Get all scores and return top N
        scores = self._calculate_scores(query)
        return scores[:top_n]


from dateutil import parser
from datetime import datetime

def parse_any_date(date_string):
    """Parse any common date format"""
    try:
        return parser.parse(date_string)
    except ValueError:
        return None

# Examples
dates = [
    "2023-12-25",
    "December 25, 2023",
    "25/12/2023",
    "12-25-23",
    "2023/12/25 14:30:00"
]

for date_str in dates:
    parsed = parse_any_date(date_str)
    if parsed:
        print(f"{date_str} -> {parsed}")


import pandas as pd
from dateutil import parser

# Your DataFrame
df = pd.DataFrame({
    'date_strings': [
        "2023-12-25",
        "December 25, 2023", 
        "25/12/2023",
        "12-25-23",
        "2023/12/25 14:30:00",
        "invalid_date",
        None
    ]
})

df['parsed_dates'] = pd.to_datetime(df['date_strings'], errors='coerce')

# Single date to compare against
comparison_date = pd.to_datetime("2023-12-20")

# Compare entire column - NaN values automatically return False
df['after_dec_20'] = df['parsed_dates'] > comparison_date
df['before_dec_20'] = df['parsed_dates'] < comparison_date
df['equal_dec_20'] = df['parsed_dates'] == comparison_date

print("Basic Comparison:")
print(df[['date_strings', 'parsed_dates', 'after_dec_20', 'before_dec_20']])
