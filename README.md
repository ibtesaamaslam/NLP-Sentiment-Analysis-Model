<div align="center">

<img src="https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/NLTK-Text_Processing-00897B?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/Scikit--Learn-ML_Models-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
<img src="https://img.shields.io/badge/Dataset-IMDb_50k-7B1FA2?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Best_Accuracy-~89--91%25-00C853?style=for-the-badge"/>
<img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge"/>

<br/><br/>

# 💬 Sentiment Analysis Using NLP & Machine Learning
### *IMDb Reviews · TF-IDF · Logistic Regression · SVM · Random Forest · Naive Bayes*

**A complete end-to-end NLP pipeline that classifies movie reviews as Positive or Negative — covering raw text cleaning, TF-IDF feature engineering, multi-model benchmarking, and comprehensive evaluation metrics including ROC-AUC.**

<br/>

[![GitHub Stars](https://img.shields.io/github/stars/ibtesaamaslam/NLP-Sentiment-Analysis-Model?style=social)](https://github.com/ibtesaamaslam/NLP-Sentiment-Analysis-Model/stargazers)
&nbsp;
[![GitHub Forks](https://img.shields.io/github/forks/ibtesaamaslam/NLP-Sentiment-Analysis-Model?style=social)](https://github.com/ibtesaamaslam/NLP-Sentiment-Analysis-Model/network/members)
&nbsp;
[![GitHub Issues](https://img.shields.io/github/issues/ibtesaamaslam/NLP-Sentiment-Analysis-Model)](https://github.com/ibtesaamaslam/NLP-Sentiment-Analysis-Model/issues)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Real-World Applications](#-real-world-applications)
- [Tech Stack](#-tech-stack)
- [Dataset](#-dataset)
- [System Architecture](#-system-architecture)
- [NLP Pipeline](#-nlp-pipeline)
- [Feature Engineering](#-feature-engineering)
- [Models & Benchmarks](#-models--benchmarks)
- [Project Structure](#-project-structure)
- [Installation & Setup](#-installation--setup)
- [How to Run](#-how-to-run)
- [Results & Evaluation](#-results--evaluation)
- [Visualizations](#-visualizations)
- [Roadmap & Future Improvements](#-roadmap--future-improvements)
- [Contributing](#-contributing)
- [Author](#-author)
- [License](#-license)
- [Acknowledgements](#-acknowledgements)

---

## 🔍 Overview

This project presents a **machine learning pipeline for binary sentiment analysis** on textual data, trained and evaluated on the [IMDb Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/) — 50,000 movie reviews each labelled as either **Positive** or **Negative**.

The pipeline covers every stage of an NLP workflow: raw text ingestion, exploratory data analysis, text cleaning with NLTK, TF-IDF and CountVectorizer feature engineering, training and comparing four classical ML classifiers, and a thorough evaluation using accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrices.

The **best-performing model — Logistic Regression with TF-IDF — achieves ~89–91% accuracy** with lightweight inference speed suitable for real-time applications.

> 💡 **Why classical ML over deep learning?** Transformer models like BERT achieve higher accuracy on this task but require significantly more compute and memory. Classical ML with TF-IDF is fast, interpretable, deployable on CPU, and still achieves near state-of-the-art results on binary sentiment tasks — making it ideal for production environments with resource constraints.

---

## 🌍 Real-World Applications

| Domain | Application |
|--------|-------------|
| 🎬 Entertainment | Classify user movie, show, or book reviews automatically |
| 🛒 E-commerce | Analyse product review sentiment at scale |
| 📱 Social Media | Monitor brand sentiment on Twitter, Reddit, or Instagram |
| 🏦 Finance | Detect positive/negative sentiment in earnings call transcripts |
| 🏥 Healthcare | Classify patient feedback and satisfaction survey responses |
| 🤝 Customer Support | Prioritize negative tickets for immediate escalation |
| 📰 Media Monitoring | Track public sentiment toward news stories or political events |

---

## 🧰 Tech Stack

| Technology | Version | Purpose |
|-----------|---------|---------|
| [Python](https://www.python.org/) | 3.8+ | Core programming language |
| [NLTK](https://www.nltk.org/) | 3.x | Tokenization, stopword removal, lemmatization |
| [Scikit-learn](https://scikit-learn.org/) | 1.x | TF-IDF, CountVectorizer, ML models, evaluation metrics |
| [Pandas](https://pandas.pydata.org/) | 1.x | Data loading, cleaning, and EDA |
| [NumPy](https://numpy.org/) | 1.x | Numerical operations and array processing |
| [Matplotlib](https://matplotlib.org/) | 3.x | Training visualizations and plots |
| [Seaborn](https://seaborn.pydata.org/) | 0.x | Confusion matrix heatmaps and styled charts |
| [Jupyter Notebook](https://jupyter.org/) | — | Interactive development and reporting environment |

---

## 📊 Dataset

**Name:** IMDb Movie Review Dataset  
**Source:** [Stanford AI Lab — Andrew Maas et al.](https://ai.stanford.edu/~amaas/data/sentiment/)  
**Loaded via:** `tensorflow.keras.datasets.imdb` or direct download

| Attribute | Value |
|-----------|-------|
| Total reviews | 50,000 |
| Positive reviews | 25,000 (50%) |
| Negative reviews | 25,000 (50%) |
| Train split | 25,000 reviews |
| Test split | 25,000 reviews |
| Class balance | Perfectly balanced (1:1) |
| Label encoding | Positive → `1`, Negative → `0` |
| Language | English |

**Review characteristics:**
- Average review length: ~230 words
- Range: 10 words to 2,500+ words
- Contains HTML tags, special characters, and domain-specific vocabulary
- Highly polarized — only reviews scored ≤ 4/10 (negative) or ≥ 7/10 (positive) are included

---

## 🏗 System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        RAW INPUT                                │
│         IMDb Dataset — 50,000 labelled movie reviews            │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                  EXPLORATORY DATA ANALYSIS                      │
│  Null checks · Class distribution · Review length histograms    │
│  Word cloud visualizations (positive vs. negative)              │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                     TEXT CLEANING (NLTK)                        │
│  Lowercase → Strip HTML tags → Remove special chars             │
│  → Remove stopwords → Tokenize → Lemmatize                      │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                   FEATURE ENGINEERING                           │
│  TF-IDF Vectorization (primary)                                 │
│  CountVectorizer (comparison baseline)                          │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                   MODEL TRAINING & SELECTION                    │
│  Logistic Regression  ·  SVM  ·  Random Forest  ·  Naive Bayes │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                       EVALUATION                                │
│  Accuracy · Precision · Recall · F1-Score                       │
│  ROC-AUC Score · Confusion Matrix Heatmap                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔬 NLP Pipeline

### Step 1 — Data Loading & EDA

```python
import pandas as pd

df = pd.read_csv('imdb_reviews.csv')
print(df['sentiment'].value_counts())   # Confirm class balance
print(df.isnull().sum())                 # Check for missing values
df['review_length'] = df['review'].apply(lambda x: len(x.split()))
```

EDA includes:
- Class distribution bar chart (Positive vs. Negative)
- Review length histogram — reveals bimodal distribution between short and long reviews
- Word cloud for positive reviews vs. word cloud for negative reviews

---

### Step 2 — Label Encoding

```python
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})
```

---

### Step 3 — Text Cleaning with NLTK

A full cleaning function is applied to every review:

```python
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download(['stopwords', 'wordnet', 'punkt'])

lemmatizer = WordNetLemmatizer()
stop_words  = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()                               # Lowercase
    text = re.sub(r'<.*?>', '', text)                 # Strip HTML tags
    text = re.sub(r'[^a-z\s]', '', text)              # Remove special chars & digits
    tokens = nltk.word_tokenize(text)                 # Tokenize
    tokens = [t for t in tokens if t not in stop_words]   # Remove stopwords
    tokens = [lemmatizer.lemmatize(t) for t in tokens]    # Lemmatize
    return ' '.join(tokens)

df['clean_review'] = df['review'].apply(clean_text)
```

| Cleaning Step | Raw Example | After |
|---------------|-------------|-------|
| Lowercase | `"The Film was GREAT"` | `"the film was great"` |
| Strip HTML | `"<br/>Great movie"` | `"great movie"` |
| Remove special chars | `"movie!!! 10/10"` | `"movie"` |
| Remove stopwords | `"this is a great film"` | `"great film"` |
| Lemmatization | `"running", "runs", "ran"` | `"run"` |

---

### Step 4 — Train / Test Split

```python
from sklearn.model_selection import train_test_split

X = df['clean_review']
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

---

## ⚙️ Feature Engineering

### TF-IDF Vectorization (Primary)

```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf  = tfidf.transform(X_test)
```

**Why TF-IDF over raw word counts?**

| Method | Handles common words | Captures importance | Sparse representation |
|--------|---------------------|--------------------|-----------------------|
| CountVectorizer | ❌ Over-weights frequent words | ❌ | ✅ |
| TF-IDF | ✅ Penalizes common words | ✅ Rewards unique, discriminative words | ✅ |

**Configuration:**
- `max_features=10000` — vocabulary capped at top 10,000 terms by TF-IDF score
- `ngram_range=(1, 2)` — captures both unigrams (`"great"`) and bigrams (`"not great"`) for negation handling
- `fit` only on training data → `transform` on test data to prevent leakage

### CountVectorizer (Comparison Baseline)

```python
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features=10000)
X_train_cv = cv.fit_transform(X_train)
X_test_cv  = cv.transform(X_test)
```

Used as a baseline to confirm TF-IDF's superiority on this dataset.

---

## 🤖 Models & Benchmarks

Four classical ML classifiers are trained and compared on the TF-IDF feature matrix:

### Logistic Regression ⭐ Best Model

```python
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs')
lr.fit(X_train_tfidf, y_train)
```

Logistic Regression is highly effective for high-dimensional sparse text features. Its linear decision boundary maps well to TF-IDF space and produces interpretable feature weights.

### Support Vector Machine (SVM)

```python
from sklearn.svm import LinearSVC

svm = LinearSVC(C=1.0, max_iter=2000)
svm.fit(X_train_tfidf, y_train)
```

`LinearSVC` is chosen over kernel SVM for efficiency on high-dimensional text data. Maximises the margin between positive and negative review representations.

### Random Forest

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_tfidf, y_train)
```

An ensemble of 100 decision trees. Less optimal for high-dimensional sparse data but benchmarked for completeness.

### Naive Bayes

```python
from sklearn.naive_bayes import MultinomialNB

nb = MultinomialNB(alpha=1.0)
nb.fit(X_train_tfidf, y_train)
```

Based on Bayes' theorem with a word-independence assumption. Extremely fast to train, performs surprisingly well on text data.

---

## 📂 Project Structure

```
NLP-Sentiment-Analysis-Model/
│
├── sentiment analysis.ipynb     # Main notebook — full NLP pipeline
│                                # ├─ Section 1: Data Loading & EDA
│                                # ├─ Section 2: Text Cleaning (NLTK)
│                                # ├─ Section 3: Feature Engineering (TF-IDF)
│                                # ├─ Section 4: Model Training (4 classifiers)
│                                # └─ Section 5: Evaluation & Visualizations
│
├── README.md                    # Project documentation (this file)
└── dataset/                     # (optional) Local IMDb dataset storage
```

---

## 📦 Installation & Setup

### Prerequisites

- Python 3.8 or higher
- pip or conda package manager

### Option A — Local Environment

```bash
# 1. Clone the repository
git clone https://github.com/ibtesaamaslam/NLP-Sentiment-Analysis-Model.git
cd NLP-Sentiment-Analysis-Model

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install pandas numpy matplotlib seaborn nltk scikit-learn jupyter wordcloud

# 4. Download NLTK resources (run once)
python -c "import nltk; nltk.download(['stopwords','wordnet','punkt'])"

# 5. Launch Jupyter
jupyter notebook
```

### Option B — Quick pip install

```bash
pip install pandas numpy matplotlib seaborn nltk scikit-learn jupyter wordcloud
```

---

## ▶ How to Run

1. Open **`sentiment analysis.ipynb`** in Jupyter Notebook or JupyterLab.
2. Select **Kernel → Restart & Run All**.
3. The notebook will load the IMDb data, clean the text, train all four models, and display all evaluation outputs automatically.

> **Dataset note:** The IMDb dataset can be loaded via `keras.datasets.imdb`, or downloaded directly from [Stanford AI Lab](https://ai.stanford.edu/~amaas/data/sentiment/) and placed in the `dataset/` folder.

---

## 📈 Results & Evaluation

### Model Comparison

| Model | Vectorizer | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|-----------|----------|-----------|--------|----------|---------|
| **Logistic Regression** ⭐ | **TF-IDF** | **~90–91%** | **~90%** | **~91%** | **~90%** | **~96%** |
| SVM (LinearSVC) | TF-IDF | ~89–90% | ~89% | ~90% | ~89% | ~95% |
| Naive Bayes | TF-IDF | ~87–88% | ~87% | ~88% | ~87% | ~94% |
| Random Forest | TF-IDF | ~84–86% | ~85% | ~85% | ~85% | ~92% |
| Logistic Regression | CountVectorizer | ~88–89% | ~88% | ~89% | ~88% | ~94% |

> **Winner: Logistic Regression + TF-IDF** — best accuracy, highest ROC-AUC, and fastest inference speed for real-time deployment.

### Why Logistic Regression Wins on Text

Text features (TF-IDF vectors) are inherently high-dimensional and sparse. Logistic Regression's linear decision boundary is well-suited to this geometry — it learns which words are most predictive of each sentiment class and weights them accordingly. More complex models like Random Forest struggle because they split on individual features in a space where thousands of features each carry weak signals.

### Evaluation Metrics Used

| Metric | Why It Matters |
|--------|----------------|
| **Accuracy** | Overall correctness across all 10,000 test reviews |
| **Precision** | Of all reviews predicted positive, how many actually were? |
| **Recall** | Of all truly positive reviews, how many did the model catch? |
| **F1-Score** | Harmonic mean of precision and recall — balanced metric |
| **ROC-AUC** | Area under the ROC curve — model's ability to rank positive above negative |
| **Confusion Matrix** | Exact count of true positives, false positives, true negatives, false negatives |

---

## 📉 Visualizations

The notebook produces the following outputs automatically:

| Visualization | Description |
|---------------|-------------|
| **Class distribution bar chart** | Positive vs. negative review counts — confirms balance |
| **Review length histogram** | Distribution of word counts across training reviews |
| **Word cloud — Positive** | Most frequent terms in positive reviews |
| **Word cloud — Negative** | Most frequent terms in negative reviews |
| **Training accuracy comparison** | Bar chart comparing all 4 models |
| **ROC curves** | One curve per model — visual AUC comparison |
| **Confusion matrix heatmap** | Seaborn heatmap for the best model (Logistic Regression) |
| **Top TF-IDF features** | Bar chart of most predictive positive and negative words |

---

## 🗺 Roadmap & Future Improvements

- [ ] **Streamlit web app** — Interactive UI to input any text and receive a live sentiment prediction
- [ ] **BERT / RoBERTa fine-tuning** — Fine-tune a pre-trained transformer for higher accuracy (~93–95%)
- [ ] **Multilingual support** — Add language detection and sentiment analysis in Arabic, French, Urdu, and more
- [ ] **Extended data sources** — Twitter/X tweets, YouTube comments, Amazon product reviews, Google Play store feedback
- [ ] **Aspect-level sentiment** — Go beyond document-level to identify sentiment toward specific aspects (e.g., "great acting, terrible plot")
- [ ] **Model explainability** — LIME or SHAP to explain individual predictions ("this review was classified negative because of: *boring*, *disappointing*, *waste*")
- [ ] **REST API endpoint** — FastAPI wrapper for integration with external applications
- [ ] **Cross-validation** — Replace single train/test split with k-fold cross-validation for more robust estimates
- [ ] **Hyperparameter tuning** — GridSearchCV on TF-IDF parameters and classifier regularization strength

---

## 🤝 Contributing

Contributions are welcome! Here's how to get involved:

```bash
# 1. Fork the repository on GitHub

# 2. Clone your fork
git clone https://github.com/YOUR-USERNAME/NLP-Sentiment-Analysis-Model.git

# 3. Create a feature branch
git checkout -b feature/add-streamlit-app

# 4. Make your changes and commit
git add .
git commit -m "feat: add Streamlit interactive demo for live sentiment prediction"

# 5. Push and open a Pull Request
git push origin feature/add-streamlit-app
```

Ideas for contributions: add a new model, improve text cleaning, add multilingual support, write unit tests, or build the Streamlit frontend.

---

## 👤 Author

<div align="center">

**Ibtesaam Aslam**

[![GitHub](https://img.shields.io/badge/GitHub-ibtesaamaslam-181717?style=for-the-badge&logo=github)](https://github.com/ibtesaamaslam)

*Machine Learning Engineer & NLP Enthusiast*

</div>

---

## 📜 License

```
MIT License

Copyright (c) 2024 Ibtesaam Aslam

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```

### License Permissions at a Glance

| Permission | Status |
|-----------|--------|
| ✅ Commercial use | Allowed |
| ✅ Modification | Allowed |
| ✅ Distribution | Allowed |
| ✅ Private use | Allowed |
| ❌ Liability | No warranty provided |
| ❌ Trademark use | Not granted |

---

## 🙏 Acknowledgements

- **[Andrew Maas et al., Stanford AI Lab](https://ai.stanford.edu/~amaas/data/sentiment/)** — For creating and releasing the IMDb Large Movie Review Dataset that powers this project.
- **[NLTK Team](https://www.nltk.org/)** — For the comprehensive natural language processing toolkit including WordNetLemmatizer, stopwords corpus, and tokenizers.
- **[Scikit-learn](https://scikit-learn.org/)** — For the consistent, well-documented ML API that makes model training, evaluation, and comparison straightforward.
- The **open-source Python data science community** — For Pandas, NumPy, Matplotlib, Seaborn, and WordCloud.

---

<div align="center">

**⭐ If this project was useful to you, please consider starring it on GitHub!**

[![Star on GitHub](https://img.shields.io/github/stars/ibtesaamaslam/NLP-Sentiment-Analysis-Model?style=social)](https://github.com/ibtesaamaslam/NLP-Sentiment-Analysis-Model)

*Made with ❤️ by [Ibtesaam Aslam](https://github.com/ibtesaamaslam)*

</div>
