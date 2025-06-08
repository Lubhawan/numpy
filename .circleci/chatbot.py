import pandas as pd
import re
from fuzzywuzzy import fuzz, process
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class MixerSearchEngine:
    def __init__(self, df, column_name):
        """
        Initialize search engine with DataFrame and column name
        
        Args:
            df: pandas DataFrame
            column_name: name of column containing mixer descriptions
        """
        self.df = df
        self.column_name = column_name
        self.unique_descriptions = df[column_name].dropna().unique()
        
        # Auto-extract abbreviations and patterns
        self.abbreviations = self._extract_abbreviations()
        
        # Initialize TF-IDF vectorizer
        self._build_tfidf_index()
    
    def _extract_abbreviations(self):
        """Auto-extract abbreviations from data"""
        abbreviations = set()
        for desc in self.unique_descriptions:
            if pd.isna(desc):
                continue
            # Find 2-5 letter abbreviations
            abbrevs = re.findall(r'\b[A-Z]{2,5}\b', str(desc))
            abbreviations.update(abbrevs)
        return abbreviations
    
    def _build_tfidf_index(self):
        """Build TF-IDF index for semantic search"""
        def tokenizer(text):
            # Extract words and preserve abbreviations
            tokens = re.findall(r'\b\w+\b', str(text).upper())
            return tokens
        
        self.vectorizer = TfidfVectorizer(
            tokenizer=tokenizer,
            lowercase=False,
            ngram_range=(1, 2),
            max_features=3000
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.unique_descriptions)
    
    def _exact_search(self, query):
        """Find exact and partial matches"""
        query_upper = query.upper()
        
        # Exact substring match
        mask = self.df[self.column_name].str.contains(
            re.escape(query_upper), case=False, na=False, regex=True
        )
        results = self.df[mask].copy()
        
        if not results.empty:
            results['match_type'] = 'exact'
            results['score'] = 1.0
            return results
        
        return pd.DataFrame()
    
    def _fuzzy_search(self, query, threshold=60):
        """Find fuzzy matches for typos and variations"""
        matches = process.extractBests(
            query.upper(),
            self.unique_descriptions,
            scorer=fuzz.partial_ratio,
            score_cutoff=threshold,
            limit=20
        )
        
        if matches:
            matched_descriptions = [match[0] for match in matches]
            scores = {match[0]: match[1]/100 for match in matches}
            
            mask = self.df[self.column_name].isin(matched_descriptions)
            results = self.df[mask].copy()
            results['match_type'] = 'fuzzy'
            results['score'] = results[self.column_name].map(scores)
            
            return results.sort_values('score', ascending=False)
        
        return pd.DataFrame()
    
    def _semantic_search(self, query, threshold=0.1, top_k=15):
        """Find semantically similar matches"""
        try:
            # Transform query
            query_vector = self.vectorizer.transform([query.upper()])
            
            # Calculate similarities
            similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
            
            # Get top matches
            top_indices = np.where(similarities >= threshold)[0]
            top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]][:top_k]
            
            if len(top_indices) > 0:
                matched_descriptions = self.unique_descriptions[top_indices]
                scores = dict(zip(matched_descriptions, similarities[top_indices]))
                
                mask = self.df[self.column_name].isin(matched_descriptions)
                results = self.df[mask].copy()
                results['match_type'] = 'semantic'
                results['score'] = results[self.column_name].map(scores)
                
                return results.sort_values('score', ascending=False)
        
        except Exception:
            pass
        
        return pd.DataFrame()
    
    def search(self, query, max_results=20):
        """
        Search for mixer descriptions matching the query
        
        Args:
            query: user search string
            max_results: maximum number of results to return
            
        Returns:
            pandas DataFrame with matching rows, match_type, and score columns
        """
        if not query or len(query.strip()) < 2:
            return pd.DataFrame()
        
        query = query.strip()
        all_results = []
        seen_descriptions = set()
        
        # 1. Exact matches (highest priority)
        exact_results = self._exact_search(query)
        if not exact_results.empty:
            exact_results['final_score'] = exact_results['score'] * 1.0
            all_results.append(exact_results)
            seen_descriptions.update(exact_results[self.column_name])
        
        # 2. Fuzzy matches (medium priority)
        fuzzy_results = self._fuzzy_search(query)
        if not fuzzy_results.empty:
            new_fuzzy = fuzzy_results[
                ~fuzzy_results[self.column_name].isin(seen_descriptions)
            ]
            if not new_fuzzy.empty:
                new_fuzzy['final_score'] = new_fuzzy['score'] * 0.8
                all_results.append(new_fuzzy)
                seen_descriptions.update(new_fuzzy[self.column_name])
        
        # 3. Semantic matches (lower priority)
        semantic_results = self._semantic_search(query)
        if not semantic_results.empty:
            new_semantic = semantic_results[
                ~semantic_results[self.column_name].isin(seen_descriptions)
            ]
            if not new_semantic.empty:
                new_semantic['final_score'] = new_semantic['score'] * 0.6
                all_results.append(new_semantic)
        
        # Combine and return top results
        if all_results:
            final_results = pd.concat(all_results, ignore_index=True)
            final_results = final_results.sort_values('final_score', ascending=False)
            return final_results.head(max_results)
        
        return pd.DataFrame()

# Usage Example
def load_and_search(excel_file, column_name, user_query):
    """
    Complete workflow: load Excel, initialize search, and return results
    
    Args:
        excel_file: path to Excel file or DataFrame
        column_name: name of column with mixer descriptions  
        user_query: user search string
        
    Returns:
        DataFrame with matching rows
    """
    # Load data
    if isinstance(excel_file, str):
        df = pd.read_excel(excel_file)
    else:
        df = excel_file
    
    # Initialize search engine
    search_engine = MixerSearchEngine(df, column_name)
    
    # Search and return results
    results = search_engine.search(user_query)
    
    return results

# Simple usage examples:
if __name__ == "__main__":
    # Example 1: Direct usage
    # df = pd.read_excel('mixer_data.xlsx')
    # search_engine = MixerSearchEngine(df, 'mixer_description')
    # results = search_engine.search('HSA lumenos')
    # print(results[['mixer_description', 'match_type', 'final_score']])
    
    # Example 2: One-liner usage
    # results = load_and_search('mixer_data.xlsx', 'mixer_description', 'AFG 90th percentile')
    # print(f"Found {len(results)} matches")
    
    pass
