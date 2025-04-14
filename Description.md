
## 💬 Sentiment Analysis Using NLP & Machine Learning

This project presents a machine learning pipeline for performing **sentiment analysis on textual data**, particularly focusing on **movie reviews**. The model is trained to classify reviews as either **positive** or **negative**, making it useful for applications like feedback analysis, social media monitoring, product reviews, and more.

---

### 🚀 Project Highlights

- **Dataset Used**: The [IMDb Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/) – a large collection of 50,000+ movie reviews labeled by sentiment.
- **Tech Stack**:
  - Language: Python
  - Environment: Jupyter Notebook
  - Libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `nltk`, `sklearn`

---

### 🧠 Machine Learning Pipeline

#### 1. **Data Preprocessing**
- Checked for missing/null values and class distributions.
- Converted sentiment labels into binary numerical format.
- Performed Exploratory Data Analysis (EDA) with review length distributions and word cloud visualizations.

#### 2. **Text Cleaning & NLP**
- Lowercasing
- Removal of HTML tags, special characters, and stopwords
- Tokenization
- Lemmatization (using NLTK’s `WordNetLemmatizer`)

#### 3. **Feature Engineering**
- Converted textual data into numeric using:
  - **TF-IDF Vectorization**
  - **CountVectorizer** (for comparison)

#### 4. **Model Training & Evaluation**
- Tested various ML models:
  - Logistic Regression
  - Support Vector Machine (SVM)
  - Random Forest
  - Naive Bayes
- Evaluation Metrics:
  - Accuracy
  - Precision, Recall, F1-Score
  - ROC-AUC Score
  - Confusion Matrix

---

### 📊 Results

- **Best Performing Model**: Logistic Regression with TF-IDF
- **Accuracy Achieved**: ~89-91%
- Lightweight and fast inference speed for real-time analysis.

---

### 📁 Repository Structure

```bash
Sentiment-Analysis/
│
├── sentiment analysis.ipynb           # Main notebook
├── README.md                          # Project description and guide
├── requirements.txt                   # All Python dependencies
└── dataset/                           # (optional) For local dataset storage
```

---

### ✅ How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/sentiment-analysis.git
   cd sentiment-analysis
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Launch Jupyter Notebook:
   ```bash
   jupyter notebook "sentiment analysis.ipynb"
   ```

---

### 📌 Future Improvements

- Deploy as a Streamlit app for interactive demo
- Fine-tune BERT or RoBERTa for more accurate predictions
- Add language detection and multilingual support
- Collect and analyze tweets, YouTube comments, or product reviews
