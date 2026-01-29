# MiniProject 1: Semantic Similarity with Embeddings

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://llmminiproject1-demo-yjzzqsasqrmuvvcyyw5pys.streamlit.app/)

A comparative analysis of text embedding models for semantic similarity tasks, implemented with an interactive Streamlit web application.

##  Project Overview

This project implements and compares **four different text embedding approaches** for semantic similarity:
- **GloVe** (25d, 50d, 100d) - Word-level embeddings
- **Sentence Transformers** (384d) - Context-aware sentence embeddings
- **OpenAI Small** (1536d) - API-based embeddings
- **OpenAI Large** (3072d) - Advanced API-based embeddings

### Key Finding
Context-aware models (Transformers, OpenAI) achieved **100% accuracy** on semantic tasks, while word-averaging models (GloVe) achieved only **25% accuracy**, demonstrating that **architecture matters more than dimensionality**.

##  Live Demo

Try the app: [Streamlit Demo](https://llmminiproject1-demo-yjzzqsasqrmuvvcyyw5pys.streamlit.app/)

##  Features

- Real-time semantic similarity comparison across 4 embedding models
- Interactive category selection
- Visual results with pie charts
- Detailed confidence scores
- Support for custom sentences and categories

##  Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/PriyaBharathiArul/LLM_miniproject1-demo.git
cd LLM_miniproject1-demo
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set OpenAI API Key** (Optional but recommended)
```bash
# Linux/Mac
export OPENAI_API_KEY="your-api-key-here"

# Windows
set OPENAI_API_KEY=your-api-key-here
```

4. **Run the application**
```bash
streamlit run miniproject_1_priya.py
```

The app will open in your browser at `http://localhost:8501`

## 📖 Usage

### Basic Usage

1. **Enter Categories**: Space-separated categories (e.g., `Flowers Colors Cars Weather Food`)
2. **Enter Sentence**: Any sentence you want to classify (e.g., `"Roses are red, trucks are blue"`)
3. **View Results**: See which category each model predicts with confidence scores
4. **Compare Models**: View pie charts and comparison tables

### Example Tests

**Test 1: Multi-Category**
```
Categories: Flowers Colors Cars Weather Food
Input: "Roses are red, trucks are blue"
Expected: Flowers (context-aware models) vs Colors (GloVe)
```

**Test 2: Sentiment**
```
Categories: Positive Negative
Input: "The movie was upsetting"
Expected: Negative (all context-aware models) vs Positive (GloVe)
```

**Test 3: Word Order**
```
Categories: Beverages Candy Desserts Food
Input 1: "chocolate milk"
Input 2: "milk chocolate"
Expected: Different results (Transformers/OpenAI) vs Same results (GloVe)
```

## 📊 Project Structure

```
LLM_miniproject1-demo/
├── miniproject_1_priya.py       # Main Streamlit application
├── requirements.txt              # Python dependencies
├── MininiProjectPart1_Analysis.pdf  # Detailed analysis document
└── README.md                     # This file
```

## 🔑 Key Results

### Accuracy Comparison

| Model | Test 1 (Roses) | Test 2 (Sentiment) | Overall |
|-------|---------------|-------------------|---------|
| GloVe 25d/50d/100d | Partial (Colors) | ❌ Wrong | 25% |
| Sentence Transformer | ✓ Correct | ✓ Correct | 100% |
| OpenAI Small/Large | ✓ Correct | ✓ Correct | 100% |

### Model Characteristics

| Model | Speed | Cost | Accuracy | Best For |
|-------|-------|------|----------|----------|
| **GloVe** |  Very Fast (~10ms) | Free | Low | Keyword matching |
| **Sentence Transformers** | ⚡ Fast (~150ms) | Free | High | Most applications |
| **OpenAI** |  Slower (~1s) | $$ Paid | Highest | Quality-critical tasks |

## 💡 Key Insights

1. **Architecture > Dimensionality**: GloVe 50d ≈ GloVe 100d in performance, but Sentence Transformer (384d) significantly outperforms both.

2. **Word Order Matters**: GloVe is order-blind due to averaging. "Chocolate milk" = "Milk chocolate" for GloVe, but not for Transformers.

3. **Context is Critical**: Even sentiment analysis benefits from contextual understanding. GloVe failed on "The movie was upsetting."

4. **Best Practical Choice**: Sentence Transformers offer the best balance of speed, cost, and accuracy for most applications.

## 🔧 Dependencies

- `streamlit` - Web application framework
- `numpy` - Numerical operations
- `pandas` - Data handling
- `sentence-transformers` - Transformer-based embeddings
- `openai` - OpenAI API client
- `gdown` - Google Drive file downloads
- `matplotlib` - Visualization

## 📄 Analysis Document

For detailed analysis including:
- Complete test results and interpretations
- Dimensionality vs performance analysis
- Real-world application examples
- Speed/cost trade-off comparisons

See: [MininiProjectPart1_Analysis.pdf](./MininiProjectPart1_Analysis.pdf)

##  Academic Context

This project was completed as part of an LLM (Large Language Models) fundamentals course, focusing on:
- Understanding semantic similarity measures
- Comparing embedding techniques
- Analyzing model architectures vs dimensionality
- Practical implementation of NLP concepts

##  Author

**Priya Bharathi Arul**
- GitHub: [@PriyaBharathiArul](https://github.com/PriyaBharathiArul)
- LinkedIn: [Priya Bharathi Arul](https://www.linkedin.com/in/priyabharathiarul/)