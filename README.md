# 📚 Book Recommendation Engine

A sophisticated Python-based book recommendation system that leverages machine learning, linear algebra, and efficient data processing techniques to provide personalized book suggestions. Built with a focus on computational efficiency, clean architecture, and scalability.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![pandas](https://img.shields.io/badge/pandas-2.0+-green.svg)](https://pandas.pydata.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/license-Educational-green.svg)]()

A CS5130 Pattern Recognition & Computer Vision project demonstrating advanced concepts in data science, algorithm design, and software engineering.

---

## 📋 Table of Contents
- [Features](#features)
- [Key Functionalities](#key-functionalities)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Recommendation Strategies](#recommendation-strategies)
- [Performance Analysis](#performance-analysis)
- [Technical Implementation](#technical-implementation)
- [Project Structure](#project-structure)
- [Results](#results)
- [Requirements](#requirements)
- [Future Enhancements](#future-enhancements)
- [Author](#author)

---

## ✨ Features

### 🎯 Core Capabilities
- **Three recommendation strategies**: Content-based, Popularity-based, and Hybrid
- **Memory-optimized data loading**: 86.7% reduction in memory usage
- **Vectorized operations**: Fast matrix computations using NumPy and scikit-learn
- **Scalable architecture**: Strategy pattern for easy algorithm swapping
- **Feature engineering**: Smart preprocessing with one-hot encoding and normalization
- **Performance benchmarking**: Built-in timing and comparison tools
- **Genre-based filtering**: Optional genre constraints for recommendations

### 🚀 Technical Highlights
- **Cosine similarity matrix** for content-based filtering
- **Efficient data types** (int32, float32) for memory optimization
- **Strategy design pattern** for clean OOP architecture
- **Vectorized NumPy operations** for computational efficiency
- **Comprehensive error handling** and data validation

---

## 🔑 Key Functionalities

### 1. **Memory-Efficient Data Loading**
```python
# Optimizes memory usage by 86.7%
# Original: 34.53 MB → Optimized: 4.58 MB
```
- **Selective column loading**: Only loads necessary features
- **Dtype optimization**: Uses int32/float32 instead of int64/float64
- **Impact**: Critical for scaling to billions of samples

**Key Implementation:**
- Specifies optimal dtypes: `{'book_id': 'int32', 'average_rating': 'float32'}`
- Uses `usecols` parameter in pandas to load only required columns
- Parses complex string representations (genres, page counts)

### 2. **Feature Engineering Pipeline**
Transforms raw book data into ML-ready features:

- **One-Hot Encoding**: Converts genre lists into binary feature vectors using `MultiLabelBinarizer`
- **Normalization**: Scales page counts and ratings to [0, 1] range
- **Popularity Score**: 
  ```
  popularity = (num_ratings / max_ratings) × (average_rating / 5.0)
  ```
- **Missing Value Handling**: Median imputation for numerical features

### 3. **Content-Based Recommendation**
Uses cosine similarity to find books with similar features:

```python
# Computes N×N similarity matrix (4.69 seconds for full dataset)
similarity_matrix = cosine_similarity(feature_matrix)
```

**Features Used:**
- Genre overlap (one-hot encoded)
- Normalized page count
- Normalized average rating

**Process:**
1. Find book's index in dataset
2. Retrieve similarity scores for all books
3. Sort and return top N matches (excluding self)

**Complexity**: O(N²) for matrix computation, O(N log N) for recommendation

### 4. **Popularity-Based Recommendation**
Simple but effective baseline approach:

- Ranks books by popularity score
- Optional genre filtering
- Fast execution (~milliseconds)
- Great for discovering trending books

### 5. **Hybrid Recommendation System**
Combines content-based and popularity approaches:

```python
hybrid_score = (content_weight × content_score) + 
               (popularity_weight × popularity_score)
```

**Default Weights:**
- Content: 70%
- Popularity: 30%

**Advantages:**
- Balances personalization with popular trends
- Handles cold-start problem better than pure content-based
- More robust recommendations

**Example Result (Harry Potter):**
- Content-based: 4/5 Harry Potter books
- Popularity-based: 2/5 Harry Potter books
- **Hybrid: 5/5 Harry Potter books** ✨

### 6. **Performance Benchmarking**
Built-in tools to analyze recommendation speed:

- Tests across different values of N (2 to 10,000)
- Compares all three strategies
- Measures wall-clock time for each recommendation
- Generates performance curves (see Results section)

---

## 🏗️ Architecture

### Design Patterns

**Strategy Pattern**: Enables easy swapping of recommendation algorithms
```
RecommendationStrategy (Abstract Base Class)
    ├── ContentBasedRecommender
    ├── PopularityRecommender
    └── HybridRecommender
```

**Benefits:**
- Open/Closed Principle: Easy to add new strategies
- Single Responsibility: Each recommender focuses on one algorithm
- Dependency Inversion: Engine depends on abstractions

### Class Hierarchy

```
BookDataLoader
    └── Handles data loading, preprocessing, memory optimization

RecommendationStrategy (ABC)
    ├── ContentBasedRecommender
    │   ├── Feature engineering
    │   ├── Similarity matrix computation
    │   └── Nearest neighbor search
    │
    ├── PopularityRecommender
    │   └── Ranking by popularity score
    │
    └── HybridRecommender
        ├── Uses ContentBasedRecommender
        ├── Uses PopularityRecommender
        └── Weighted score combination

BookRecommendationEngine
    └── Main interface, manages all strategies
```

---

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Quick Start

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd book-recommendation-engine
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare data**
   - Place `Book_Details.csv` in `book_data/` directory
   - Ensure CSV has required columns: `book_id`, `book_title`, `author`, `genres`, etc.

4. **Run the demo**
   ```bash
   python demo.py
   ```

---

## 🚀 Usage

### Basic Usage

```python
from data_loader import BookDataLoader
from book_recommender import BookRecommendationEngine

# Load and preprocess data
loader = BookDataLoader("book_data/Book_Details.csv")
books_df = loader.load_and_preprocess(optimized=True)

# Create recommendation engine
engine = BookRecommendationEngine(books_df)

# Get recommendations
recommendations = engine.get_recommendations(
    book_title="Harry Potter",
    strategy="hybrid",
    n=5
)

# Display results
engine.display_recommendations(recommendations)
```

### Advanced Usage

#### Content-Based Recommendations
```python
recommendations = engine.get_recommendations(
    book_title="The Hobbit",
    strategy="content",
    n=10
)
```

#### Popularity with Genre Filter
```python
recommendations = engine.get_recommendations(
    book_title="Any Book",
    strategy="popularity",
    n=5,
    genre="Fiction"
)
```

#### Custom Hybrid Weights
```python
from book_recommender import HybridRecommender

custom_hybrid = HybridRecommender(
    books_df,
    content_weight=0.8,
    popularity_weight=0.2
)

recommendations = custom_hybrid.recommend(book_id=123, n=5)
```

---

## 🎯 Recommendation Strategies

### 1. Content-Based Filtering

**How it works:**
- Analyzes book features (genres, ratings, pages)
- Computes similarity between books
- Recommends books with similar characteristics

**Best for:**
- Users who liked specific books
- Finding niche, similar content
- Personalized recommendations

**Limitations:**
- Can't recommend outside user's historical preferences
- May create "filter bubble"

---

### 2. Popularity-Based Filtering

**How it works:**
- Ranks books by popularity score
- Optionally filters by genre
- Returns top-rated, widely-read books

**Best for:**
- New users (cold-start problem)
- Discovering trending books
- General recommendations

**Limitations:**
- Not personalized
- May miss niche interests

---

### 3. Hybrid Approach

**How it works:**
- Combines content similarity with popularity
- Weighted average of both scores
- Balances personalization and trends

**Best for:**
- Balanced recommendations
- Overcoming limitations of single approaches
- Production systems

**Advantages:**
- More robust than individual methods
- Better coverage and diversity
- Handles cold-start better than pure content-based

---

## ⚡ Performance Analysis

### Memory Optimization Results

| Metric | Unoptimized | Optimized | Improvement |
|--------|-------------|-----------|-------------|
| **Memory Usage** | 34.53 MB | 4.58 MB | **86.7% reduction** |
| **Columns Loaded** | All (~20+) | 8 essential | 60% fewer |
| **Data Types** | int64, float64 | int32, float32 | 50% smaller |

### Computational Performance

**Similarity Matrix Computation:**
- Time: 4.69 seconds (full dataset)
- Complexity: O(N²) space and time
- Critical operation: Uses optimized scikit-learn implementation

### Recommendation Speed (by N)

From performance benchmarking on test dataset:

| N | Content (ms) | Popularity (ms) | Hybrid (ms) |
|---|--------------|-----------------|-------------|
| 2 | 12 | 8 | 15 |
| 100 | 45 | 25 | 90 |
| 1,000 | 120 | 85 | 280 |
| 10,000 | 165 | 125 | 450 |

**Key Insights:**
- **Popularity**: Fastest (simple sorting)
- **Content**: Moderate (array indexing + sorting)
- **Hybrid**: Slowest (combines both + score normalization)
- All scale sub-linearly due to efficient indexing

### Scalability Characteristics

```
Time Complexity:
├── Data Loading: O(N)
├── Similarity Matrix: O(N²) - one-time cost
├── Content Recommendation: O(N log N) per query
├── Popularity Recommendation: O(N log N) per query
└── Hybrid Recommendation: O(N log N) per query
```

**Memory Complexity:**
- Feature Matrix: O(N × F) where F = number of features
- Similarity Matrix: O(N²) - largest memory consumer
- Recommendation Buffer: O(N) per query

---

## 🔬 Technical Implementation

### Feature Engineering

**1. Genre Encoding**
```python
from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer()
genre_features = mlb.fit_transform(books_df['genres'])
# Output: Binary matrix (n_books × n_unique_genres)
```

**2. Numerical Normalization**
```python
page_count_norm = books_df['num_pages'] / books_df['num_pages'].max()
rating_norm = books_df['average_rating'] / 5.0
```

**3. Feature Matrix Construction**
```python
feature_matrix = np.hstack([
    genre_features,                      # One-hot encoded genres
    page_count_norm.values.reshape(-1, 1),  # Normalized pages
    rating_norm.values.reshape(-1, 1)       # Normalized ratings
])
```

### Similarity Computation

**Cosine Similarity Formula:**
```
similarity(A, B) = (A · B) / (||A|| × ||B||)
```

**Implementation:**
```python
from sklearn.metrics.pairwise import cosine_similarity

# Vectorized computation for all pairs
similarity_matrix = cosine_similarity(feature_matrix)
# Result: N×N matrix where sim_matrix[i][j] = similarity(book_i, book_j)
```

**Why Cosine Similarity?**
- Measures angle between feature vectors (not distance)
- Normalized to [0, 1] or [-1, 1]
- Robust to magnitude differences
- Efficient implementation in scikit-learn

### Recommendation Retrieval

**Efficient Top-N Selection:**
```python
# Get similarity scores for target book
scores = similarity_matrix[book_index]

# Use NumPy's argsort for O(N log N) sorting
sorted_indices = np.argsort(scores)[::-1]

# Exclude the book itself and take top N
top_n_indices = sorted_indices[1:n+1]
```

### Data Preprocessing

**Parsing Complex Types:**
```python
import ast

# Safely parse string representations of lists
genres = ast.literal_eval("['Fiction', 'Fantasy']")
# Result: ['Fiction', 'Fantasy']

pages = ast.literal_eval("['652']")[0]
# Result: 652
```

**Missing Value Strategy:**
- **Ratings/Author/Title**: Drop rows (critical fields)
- **Genres**: Fill with empty list `[]`
- **Pages**: Fill with median value

---

## 📁 Project Structure

```
book-recommendation-engine/
├── book_recommender.py        # Main recommendation algorithms
│   ├── RecommendationStrategy (ABC)
│   ├── ContentBasedRecommender
│   ├── PopularityRecommender
│   ├── HybridRecommender
│   └── BookRecommendationEngine
│
├── data_loader.py             # Data loading and preprocessing
│   └── BookDataLoader
│       ├── Memory optimization
│       ├── Feature parsing
│       └── Popularity score calculation
│
├── demo.py                    # Demonstration script
│   ├── Data loading comparison
│   ├── Strategy demonstrations
│   └── Performance analysis
│
├── requirements.txt           # Python dependencies
├── README.md                  # This file
│
└── book_data/
    └── Book_Details.csv       # Goodreads dataset (not included)
```

### File Descriptions

| File | Purpose | Key Classes/Functions |
|------|---------|----------------------|
| `book_recommender.py` | Core recommendation logic | All recommender classes + engine |
| `data_loader.py` | ETL pipeline | `BookDataLoader` |
| `demo.py` | Usage examples and testing | Demonstration functions |
| `requirements.txt` | Dependencies | N/A |

---

## 📊 Results

### Example Recommendations

#### Test Case: "Harry Potter"

**Content-Based (4/5 Harry Potter books):**
1. Harry Potter and the Prisoner of Azkaban
2. Harry Potter and the Chamber of Secrets
3. Harry Potter and the Goblet of Fire
4. Harry Potter and the Order of the Phoenix
5. The Hunger Games (similar fantasy genre)

**Popularity-Based (2/5 Harry Potter books):**
1. Harry Potter and the Sorcerer's Stone
2. The Hunger Games
3. To Kill a Mockingbird
4. The Great Gatsby
5. Harry Potter and the Deathly Hallows

**Hybrid (5/5 Harry Potter books - Best Performance!):**
1. Harry Potter and the Chamber of Secrets
2. Harry Potter and the Prisoner of Azkaban
3. Harry Potter and the Goblet of Fire
4. Harry Potter and the Order of the Phoenix
5. Harry Potter and the Deathly Hallows

### Performance Curves

Based on Figure 1 from project report:

**Time vs Number of Recommendations:**
- **Popularity**: Linear growth, fastest (~0.125 sec for 10K recommendations)
- **Content**: Linear growth, moderate speed (~0.165 sec for 10K)
- **Hybrid**: Linear growth, slowest (~0.45 sec for 10K)

**Key Observation**: All strategies scale sub-linearly due to efficient NumPy operations and optimized data structures.

---

## 📋 Requirements

### Python Packages

```txt
pandas>=2.0.0          # DataFrame operations
numpy>=1.24.0          # Numerical computing
scikit-learn>=1.3.0    # Machine learning utilities
```

### Optional Packages
```txt
jupyter>=1.0.0         # Interactive development
matplotlib>=3.7.0      # Visualizations
```

### System Requirements
- **RAM**: 4GB minimum (8GB recommended)
- **CPU**: Modern processor (vectorization support)
- **Storage**: ~50MB for dependencies + dataset

### Data Requirements

**CSV Format:** `Book_Details.csv`

**Required Columns:**
- `book_id` (int)
- `book_title` (str)
- `author` (str)
- `genres` (str - list representation)
- `num_ratings` (int)
- `num_reviews` (int)
- `average_rating` (float)
- `num_pages` (str - list representation)

---

## 🐛 Troubleshooting

### Common Issues

**Issue**: "FileNotFoundError: Book_Details.csv"
```bash
# Solution: Ensure CSV is in correct location
mkdir book_data
mv Book_Details.csv book_data/
```

**Issue**: Memory error during similarity matrix computation
```bash
# Solution: Reduce dataset size or use chunking
# Or increase available RAM
```

**Issue**: "ValueError: could not convert string to float"
```bash
# Solution: Check data quality
# Ensure proper parsing of list-formatted columns
loader.preprocess_data()  # Handles parsing
```

**Issue**: Slow performance on large datasets
```bash
# Solution: 
# 1. Ensure dtype optimization is enabled
loader.load_and_preprocess(optimized=True)

# 2. Consider computing similarity matrix once and caching
# 3. Use fewer features in similarity computation
```


## 📚 References

### Academic
- **Cosine Similarity**: [Wikipedia](https://en.wikipedia.org/wiki/Cosine_similarity)
- **Recommendation Systems**: [Stanford CS246](http://web.stanford.edu/class/cs246/)
- **Matrix Factorization**: Koren et al. (2009) "Matrix Factorization Techniques for Recommender Systems"

### Libraries
- **pandas**: [Documentation](https://pandas.pydata.org/docs/)
- **NumPy**: [User Guide](https://numpy.org/doc/stable/user/)
- **scikit-learn**: [API Reference](https://scikit-learn.org/stable/modules/classes.html)

### Datasets
- **Goodreads Books**: Public dataset from Kaggle/Goodreads API

---

## 📄 License

This project is created for educational purposes as part of CS5130 coursework at Northeastern University.

---

## 👤 Author

**Ghailan Fadah**  
Northeastern University  
CS5130 - Pattern Recognition & Computer Vision  
December 2025

---

## 🙏 Acknowledgments

- **CS5130 Course Staff** for project guidance and feedback
- **scikit-learn Community** for excellent ML utilities
- **pandas Development Team** for powerful data manipulation tools
- **Goodreads** for making book data accessible

---

## 📈 Project Statistics

- **Total Lines of Code**: ~1,500
- **Classes**: 5 (Strategy pattern)
- **Functions**: 30+
- **Memory Optimization**: 86.7% reduction
- **Similarity Matrix Computation**: 4.69 seconds
- **Recommendation Strategies**: 3 (Content, Popularity, Hybrid)
- **Performance Benchmark**: Tested up to N=10,000

---

<div align="center">

**Made with 🐍 Python, 🧮 NumPy, and 📊 pandas**

**Demonstrating the power of linear algebra in real-world applications**

⭐ Star this repository if you found it helpful!

</div>
