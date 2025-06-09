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
        
    def extract_keywords(self, text: str) -> List[str]:
        text = re.sub(r'\s+', ' ', text.strip())
        text_upper = text.upper()
        keywords = []
        
        keywords.extend(re.findall(r'\b[A-Z]{2,}\b', text_upper))
        keywords.extend(re.findall(r'\d+%?', text))
        keywords.extend(re.findall(r'\d+(?:ST|ND|RD|TH)', text_upper))
        
        return keywords
    
    def match(self, query: str, threshold: float = 0.3) -> pd.DataFrame:

        query = re.sub(r'\s+', ' ', query.strip())
        
        keywords = self.extract_keywords(query)
        matches = []
        
        if keywords:
            for desc in self.unique_values:
                desc_upper = desc.upper()
                keyword_score = sum(1 for kw in keywords if kw in desc_upper) / max(len(keywords), 1)
                if keyword_score > 0:
                    matches.append((desc, keyword_score))
        
        if matches:
            matches.sort(key=lambda x: x[1], reverse=True)
            best_match, keyword_score = matches[0]
            if keyword_score >= 0.5:  
                return self.df[self.df[self.column] == best_match]
        
        # Compute TF-IDF similarity
        query_vector = self.vectorizer.transform([query.lower()])
        tfidf_scores = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Compute fuzzy matching scores
        fuzzy_scores = [fuzz.token_sort_ratio(query.lower(), desc.lower()) / 100.0 for desc in self.unique_values]
        
        # Combine scores with substring bonus
        combined_scores = []
        for idx, desc in enumerate(self.unique_values):
            keyword_score = next((score for match, score in matches if match == desc), 0.0)
            # Substring bonus: +0.1 if query is a substring of description
            substring_bonus = 0.1 if query.lower().replace(' ', '') in desc.lower().replace(' ', '') else 0.0
            # Weighted combination: 35% TF-IDF, 35% fuzzy, 30% keyword + substring bonus
            combined_score = 0.35 * tfidf_scores[idx] + 0.35 * fuzzy_scores[idx] + 0.3 * keyword_score + substring_bonus
            combined_scores.append((desc, combined_score))
        
        # Get best match
        if combined_scores:
            combined_scores.sort(key=lambda x: x[1], reverse=True)
            best_match, best_score = combined_scores[0]
            if best_score >= threshold:
                return self.df[self.df[self.column] == best_match]
        
        return pd.DataFrame()  # No match found
    
    def get_suggestions(self, query: str, top_n: int = 10) -> List[Tuple[str, float]]:

        query = re.sub(r'\s+', ' ', query.strip())
        
        query_vector = self.vectorizer.transform([query.lower()])
        tfidf_scores = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        fuzzy_scores = [fuzz.token_sort_ratio(query.lower(), desc.lower()) / 100.0 for desc in self.unique_values]
        
        keywords = self.extract_keywords(query)
        keyword_scores = []
        for desc in self.unique_values:
            desc_upper = desc.upper()
            score = sum(1 for kw in keywords if kw in desc_upper) / max(len(keywords), 1) if keywords else 0.0
            keyword_scores.append(score)
        
        # Combine scores with substring bonus
        suggestions = []
        for idx, desc in enumerate(self.unique_values):
            substring_bonus = 0.1 if query.lower().replace(' ', '') in desc.lower().replace(' ', '') else 0.0
            # Weighted combination: 35% TF-IDF, 35% fuzzy, 30% keyword + substring bonus
            combined_score = 0.35 * tfidf_scores[idx] + 0.35 * fuzzy_scores[idx] + 0.3 * keyword_scores[idx] + substring_bonus
            if combined_score > 0:
                suggestions.append((desc, combined_score))
        
        # Sort and return top N
        suggestions.sort(key=lambda x: x[1], reverse=True)
        return suggestions[:top_n]
