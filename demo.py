"""
Demo Script for Book Recommendation Engine
CS 5130 - Lab 6

This script demonstrates all components of your recommendation system.
TODO: Complete the TODOs to create a comprehensive demonstration.
"""

import time
import pandas as pd
from data_loader import BookDataLoader
from book_recommender import BookRecommendationEngine


def demonstrate_data_loading(file: str) -> pd.DataFrame:
    """
    Demonstrate efficient data loading with memory optimization.

    TODO:
    1. Load data twice - once with all columns, once with optimized columns
    2. Compare memory usage
    3. Show the impact of dtype optimization
    """

    # TODO: Load data with optimization
    print("\n Loading data with optimized columns and dtypes...")
    start_time = time.time()

    loader_opt = BookDataLoader("book_data/Book_Details.csv")
    data_opt = loader_opt.load_and_preprocess(True)

    load_time = time.time() - start_time

    print(f" Data loaded in {load_time:.2f} seconds")
    print(f" Dataset shape: {data_opt.shape}")

    # TODO: Show memory usage
    print("\n Memory Usage Analysis:")
    memory_info = loader_opt.get_memory_usage()
    print(f"Total memory: {memory_info['total_memory_mb']:.2f} MB")


    # TODO: Show data types
    print("\n Optimized Data Types:")
    print(data_opt.dtypes)

    # TODO: Show sample data
    print("\n Sample Data (first 3 rows):")
    print(data_opt.head(3))

    ################ NO OPT ##############
    print("\n Loading data without optimizing columns and dtypes...")
    start_time = time.time()

    loader_no_opt = BookDataLoader("book_data/Book_Details.csv")
    data_no_opt = loader_no_opt.load_and_preprocess(False)

    load_time = time.time() - start_time

    print(f" Data loaded in {load_time:.2f} seconds")
    print(f" Dataset shape: {data_no_opt.shape}")

    # TODO: Show memory usage
    print("\n Memory Usage Analysis:")
    memory_info = loader_no_opt.get_memory_usage()
    print(f"Total memory: {memory_info['total_memory_mb']:.2f} MB")


    # TODO: Show data types
    print("\n Non-optimized Data Types:")
    print(data_no_opt.dtypes)

    # TODO: Show sample data
    print("\n Sample Data (first 3 rows):")
    print(data_no_opt.head(3))

    return data_opt


def demonstrate_engine(books: list[str], engine: BookRecommendationEngine):
     # Test recommendations
    
    for book in books:
        print(f"\nGetting recommendations for books like '{book}'...")
        for strategy in ['content', 'popularity', 'hybrid']:
        
            print(f"Strategy: {strategy.upper()}")
            
            start = time.time() 
            recommendations = engine.get_recommendations(book, strategy=strategy, n=5)
            elapsed = time.time() - start
            print(f"\n  Recommendation time: {elapsed:.4f} seconds\n")
            engine.display_recommendations(recommendations)

def demonstrate_n_scaled(test_book: str, nums_n: list[int], engine: BookRecommendationEngine):
    times_for_stragies = {}
    
        
    for strategy in ['content', 'popularity', 'hybrid']:
        strategy_time = []
        for n in nums_n:
        
            start = time.time() 
            recommendations = engine.get_recommendations(test_book, strategy=strategy, n=n)
            elapsed = time.time() - start
            strategy_time.append(elapsed)

        times_for_stragies[strategy] = strategy_time

    return times_for_stragies
        
            


            
    


def main():
    """
    Main demonstration script.

    TODO: Complete all demonstration sections
    """
    # data loading
    print("BOOK RECOMMENDATION ENGINE DEMONSTRATION")
    print("PART 1: EFFICIENT VS NON-EFFICIENT DATA LOADING")
    books_df = demonstrate_data_loading("book_data/Book_Details.csv")
   
    # testing recommendations
    print("PART 2: COMPARING RECOMMENDATIONS")
    start_time = time.time()
    engine = BookRecommendationEngine(books_df)
    build_time = time.time() - start_time
    print(f" Engine ready! (built in {build_time:.2f} seconds)")

    book_test = ['The Hobbit', 'Harry Potter', 'The Great Gatsby']
    demonstrate_engine(book_test, engine)

    # testing how n scales
    print("PART 3: Time Analysis")
    n_list = [2, 4, 16, 32, 100, 1000, 10000]
    times = demonstrate_n_scaled('Harry Potter', n_list, engine)
    print(times)

   

  


if __name__ == "__main__":
    main()