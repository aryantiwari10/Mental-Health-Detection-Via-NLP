# 🧠 Mental Health Detection via NLP on Social Media

This project aims to detect potential **suicidal tendencies** in users based on their **social media posts**, using **Natural Language Processing (NLP)** techniques and both traditional & deep learning models.

We compare performance across:
- ✅ Logistic Regression
- ✅ Naive Bayes (MultinomialNB)
- ✅ LSTM (Long Short-Term Memory Neural Network)

---

## 🔍 Problem Statement

With mental health becoming a global concern, early detection of suicidal ideation through language can save lives. This project performs **binary classification** of text posts as either:

- `suicide` (1)
- `non-suicide` (0)

---

## 🗃️ Dataset

- **File**: `Suicide_Detection.csv`
- **Source**: Reddit `SuicideWatch` posts
- **Columns**:
  - `text`: the Reddit post content
  - `class`: label (`suicide` or `non-suicide`)

---

## 📊 Workflow Summary

### 1. 📥 Data Loading & Inspection
- Loaded CSV
- Checked shape, data types, nulls

### 2. 🧼 Text Preprocessing
- Lowercased text
- Removed stopwords, punctuation
- Tokenization, Lemmatization using spaCy
- Converted labels to binary (1/0)

### 3. 🔍 Exploratory Data Analysis (EDA)
- Class distribution plot
- Word clouds for each class
- Frequent words per class using bar plots

### 4. 🔢 Feature Extraction
- Applied **TF-IDF Vectorizer** (for traditional ML)
- Tokenizer + padding for **LSTM**

---

## 🧠 Models Used

| Model                | Description |
|---------------------|-------------|
| Logistic Regression | Simple linear classifier using TF-IDF features |
| Multinomial Naive Bayes | Probabilistic model for text data |
| LSTM Neural Network | Deep learning model capturing sequential dependencies |

---

## 🧪 Evaluation Metrics

- **Confusion Matrix**
- **Classification Report**
  - Precision
  - Recall (important in suicide detection)
  - F1-Score
- Plotted model performances for comparison

---

## 🏆 Results Summary

| Model      | Precision | Recall | F1-Score |
|------------|-----------|--------|----------|
| Logistic Regression | High     | High   | Good     |
| Naive Bayes         | Moderate | Lower  | Avg      |
| LSTM                | Very High| High   | Best     |

> 🥇 **LSTM outperformed traditional models** in most metrics.

---

## ⚙️ Tech Stack

- Python
- NLP: NLTK, spaCy
- ML: scikit-learn
- DL: Keras (TensorFlow backend)
- EDA: Matplotlib, Seaborn, WordCloud

---

## ▶️ How to Run

1. Clone this repo
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Open the notebook:
   ```bash
   jupyter notebook mental-health-detection-via-nlp-on-social-media.ipynb
   ```

---

## 💡 Key Insights

- **Preprocessing** & **text cleaning** significantly impacted model performance
- **LSTM** was able to learn deeper contextual patterns compared to TF-IDF-based models
- **Class imbalance** was addressed using stratified splitting and careful model evaluation

---

## 🚀 Future Work

- Deploy LSTM model with a front-end (Streamlit / Flask)
- Use BERT-based models for improved performance
- Monitor false negatives closely in production

---

## 📚 References

- Reddit SuicideWatch Dataset
- scikit-learn documentation
- Keras LSTM tutorials
- NLTK and spaCy for preprocessing