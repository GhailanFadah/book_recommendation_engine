"""
Book Recommendation Engine
CS 5130 - Lab 6

This module demonstrates concepts from Weeks 6-12:
- Week 6: Python for scientific computing with NumPy/pandas
- Week 7: Functional programming patterns
- Week 8: Object-oriented design
- Week 9: Clean, modular, maintainable code
- Week 10: Algorithmic design and optimization
- Week 12: Efficient data handling
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Callable
from functools import reduce
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer


class RecommendationStrategy:
    """
    Base class for recommendation strategies (Week 8: OOP Design)

    This demonstrates the Strategy pattern - different algorithms
    can be swapped in and out easily.
    """

    def recommend(self, book_id: int, n: int = 5) -> List[Tuple[int, str, float]]:
        """
        Generate recommendations for a given book.

        Args:
            book_id: The book to base recommendations on
            n: Number of recommendations to return

        Returns:
            List of tuples: (book_id, title, score)
        """
        raise NotImplementedError("Subclasses must implement recommend()")


class ContentBasedRecommender(RecommendationStrategy):
    """
    Content-based recommendation using book features.

    TODO: Implement content-based filtering using:
    - Genre similarity
    - Author matching
    - Other book features

    Week 10: Use vectorized operations for efficiency!
    """

    def __init__(self, books_df: pd.DataFrame):
        """
        Initialize the recommender with book data.

        Args:
            books_df: DataFrame with book information
        """
        self.books_df = books_df.copy()
        self.sim_mat = None
        self._prepare_features()

    def _prepare_features(self):
        """
        Prepare feature vectors for similarity computation.

        TODO: Create feature vectors from book metadata
        - One-hot encode genres (use MultiLabelBinarizer)
        - Consider author, page count, ratings

        Week 10 concept: Vectorization for efficiency
        """
        # TODO: Create genre feature matrix
        # Hint: Use MultiLabelBinarizer to convert list of genres to binary matrix
        label = MultiLabelBinarizer()
        genre_feat = label.fit_transform(self.books_df['genres'])

        # TODO: Add normalized numerical features
        # Normalize page count and ratings to 0-1 scale
        pg_count_norm = self.books_df['num_pages'] / self.books_df['num_pages'].max()

        rating_norm = self.books_df['average_rating'] / self.books_df['average_rating'].max()

        # Combine features
        # Shape: (n_books, n_genre_features + 2)

        self.feat_mat = np.hstack([genre_feat, pg_count_norm.values.reshape(-1, 1), rating_norm.values.reshape(-1, 1)])
        
        # TODO: Compute similarity matrix
        # Week 10: This is computationally expensive - how can we optimize?
        self._compute_similarity_matrix()
        

    def _compute_similarity_matrix(self):
        """
        Compute pairwise similarity between all books.

        TODO: Use cosine similarity efficiently

        Week 10 concept: This operation is O(n²) - consider:
        - Using scipy/sklearn optimized implementations
        - For very large datasets, computing on-demand might be better

        Week 11 bonus: Could this be parallelized?
        """
        # TODO: Compute cosine similarity matrix
        # Hint: sklearn.metrics.pairwise.cosine_similarity is optimized

        self.sim_mat = cosine_similarity(self.feat_mat)
        

    def recommend(self, book_id: int, n: int = 5) -> List[Tuple[int, str, float]]:
        """
        Recommend books similar to the given book.

        TODO: Implement the recommendation logic
        1. Find the book's index in the dataframe
        2. Get similarity scores for all books
        3. Sort and return top N (excluding the input book)

        Args:
            book_id: ID of the book to base recommendations on
            n: Number of recommendations

        Returns:
            List of (book_id, title, similarity_score)
        """
        # TODO: Find book index
        idx = self.books_df[self.books_df['book_id'] == book_id].index[0]

        # TODO: Get similarity scores
        scores = self.sim_mat[idx]

        # TODO: Get indices of top N similar books (excluding self)
        # Week 10: Use NumPy's argsort for efficiency
        sim_indx = np.argsort(scores)[::-1][1:n + 1]

        # TODO: Build result list
        rec_list = []

        for book_idx in sim_indx:
            book_row = self.books_df.iloc[book_idx]
            rec_list.append((book_row['book_id'], book_row['book_title'],float(scores[idx])))

        return rec_list


       

class PopularityRecommender(RecommendationStrategy):
    """
    Recommend popular books, optionally filtered by genre.

    This is simpler but serves as a good baseline.
    """

    def __init__(self, books_df: pd.DataFrame):
        self.books_df = books_df.copy()

    def recommend(self,genre: str = None, n: int = 5) -> List[Tuple[int, str, float]]:
        """
        Recommend top books by popularity.

        TODO: Implement popularity-based recommendations

        Args:
            n: Number of recommendations

        Returns:
            List of (book_id, title, popularity_score)
        """
        df = self.books_df.copy()

        # TODO: Filter by genre if specified
        if genre:
            df = df[df['genres'].apply(lambda x: genre in x)]
      
       
        # TODO: Sort by popularity and return top N
        # Week 6/10: Use pandas efficiently
        top_n_books = df.nlargest(n, 'popularity_score')
      
        rec_list = []
        for x, row in top_n_books.iterrows():
            rec_list.append((row['book_id'],row['book_title'],row['popularity_score']))
        
        return rec_list


class HybridRecommender(RecommendationStrategy):
    """
    Combines multiple recommendation strategies.

    TODO: Implement a hybrid approach that combines:
    - Content-based similarity
    - Popularity

    Week 7: Use functional programming concepts to combine strategies
    """

    def __init__(self, books_df: pd.DataFrame,
                 content_weight: float = 0.7,
                 popularity_weight: float = 0.3):
        """
        Initialize hybrid recommender.

        Args:
            books_df: Book data
            content_weight: Weight for content-based score
            popularity_weight: Weight for popularity score
        """
        self.content_recommender = ContentBasedRecommender(books_df)
        self.popularity_recommender = PopularityRecommender(books_df)
        self.books_df = books_df
        self.content_weight = content_weight
        self.popularity_weight = popularity_weight

    def recommend(self, book_id: int, n: int = 5) -> List[Tuple[int, str, float]]:
        """
        Generate hybrid recommendations.

        TODO: Combine content-based and popularity scores

        Week 7 concept: Think about functional composition
        Week 10 concept: Efficient score combination

        Args:
            book_id: Book to base recommendations on
            n: Number of recommendations

        Returns:
            List of (book_id, title, combined_score)
        """
        # TODO: Get content-based recommendations
        content_recs = self.content_recommender.recommend(book_id, n * 3)

        # TODO: Create a score dictionary for efficient lookup

        con_scores = {}
        id_list = []
        for rec in content_recs:
            book_id, title, score = rec
            con_scores[book_id] = score
            id_list.append(book_id)
        

        # TODO: Normalize popularity scores for the candidate books
        
        candidate_books = self.books_df[self.books_df['book_id'].isin(id_list)]

        max_pop = candidate_books['popularity_score'].max()

        # TODO: Combine scores
        combined_scores = []
        for _, row in candidate_books.iterrows():
            book_id = row['book_id']
            content_score = con_scores.get(book_id, 0)

            if max_pop > 0 :
                popularity_score = row['popularity_score'] / max_pop
            else:
                popularity_score = 0 
            

            # Weighted combination
            combined_wt = (self.content_weight * content_score + self.popularity_weight * popularity_score)

            combined_scores.append((book_id,row['book_title'],combined_wt))

        # TODO: Sort by combined score and return top N
        combined_scores.sort(key=lambda x: x[2], reverse=True)
        return combined_scores[:n]


class BookRecommendationEngine:
    """
    Main interface for the recommendation system.

    Week 9: Clean, maintainable design with clear responsibilities
    """

    def __init__(self, books_df: pd.DataFrame):
        """
        Initialize the recommendation engine.

        Args:
            books_df: Preprocessed book data
        """
        self.books_df = books_df
        self.strategies = {'content': ContentBasedRecommender(books_df),'popularity': 
                           PopularityRecommender(books_df),'hybrid': HybridRecommender(books_df)}

    def get_recommendations(self,book_title: str,strategy: str = 'hybrid',n: int = 5, genre: str = None) -> List[Tuple[int, str, float]]:
        """
        Get book recommendations.

        Args:
            book_title: Title of book to base recommendations on
            strategy: 'content', 'popularity', or 'hybrid'
            n: Number of recommendations

        Returns:
            List of recommended books with scores
        """
        # TODO: Find book by title
        matches = self.books_df[self.books_df['book_title'].str.contains(book_title, case=False, na=False)]

        if matches.empty:
            print(f"Book '{book_title}' not found")
            return []

        book_id = matches.iloc[0]['book_id']
       
        # TODO: Get recommendations using selected strategy
        if strategy not in self.strategies:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        
        if strategy == 'popularity':
            return self.strategies[strategy].recommend(genre, n)

        return self.strategies[strategy].recommend(book_id, n = n)

    def display_recommendations(self, recommendations: List[Tuple[int, str, float]]):
        """
        Pretty print recommendations.

        Args:
            recommendations: List of (book_id, title, score) tuples
        """
      
        print("RECOMMENDATIONS")
       

        for i, (book_id, title, score) in enumerate(recommendations, 1):
            # Get book details
            book = self.books_df[self.books_df['book_id'] == book_id].iloc[0]

            print(f"\n{i}. {title}")
            print(f"   Author: {book['author']}")
            print(f"   Rating: {book['average_rating']:.2f}  ({book['num_ratings']:,} ratings)")
            print(f"   Genres: {', '.join(book['genres'][:3])}")
            print(f"   Match Score: {score:.3f}")

        print("\n" + "=" * 80)


# Example usage
if __name__ == "__main__":
    from data_loader import BookDataLoader

    # Load data
    print("Loading data...")
    loader = BookDataLoader("book_data/Book_Details.csv")
    books_df = loader.load_and_preprocess(True)
   

    # Create recommendation engine
    print("Building recommendation engine...")
    engine = BookRecommendationEngine(books_df)

    # Test recommendations
    test_book = "Dune"
    print(f"\nGetting recommendations for books like '{test_book}'...")

    for strategy in ['content', 'popularity', 'hybrid']:
       
        print(f"Strategy: {strategy.upper()}")
        

        recommendations = engine.get_recommendations(test_book, strategy=strategy, n=5, genre='Fiction')
        engine.display_recommendations(recommendations)

