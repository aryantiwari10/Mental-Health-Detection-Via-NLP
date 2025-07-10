# 🧠 Mental Health Detection via NLP on Social Media

This project aims to detect potential **suicidal tendencies** in users based on their **social media posts**, using **Natural Language Processing (NLP)** and machine learning techniques.

We train and evaluate multiple models including **Logistic Regression, Random Forest, and XGBoost**, leveraging **TF-IDF** for feature extraction and **SMOTE** for handling class imbalance.

---

## 🔍 Problem Statement

With the rise in mental health issues globally, early detection of suicidal behavior can save lives. This project focuses on **binary classification**:  
> 🔹 `suicide` vs. `non-suicide` based on the textual content of social media posts.

---

## 🗃️ Dataset

- **Source**: Reddit `SuicideWatch` posts  
- **File**: `Suicide_Detection.csv`  
- **Columns**:
  - `text` – User's post content
  - `class` – Label (`suicide` or `non-suicide`)

---

## 📊 Project Workflow

### 1. 📥 Data Loading
- Read CSV using pandas
- Checked for nulls and basic info

### 2. 🧼 Preprocessing
- Lowercasing, punctuation & stopword removal
- Tokenization and lemmatization
- Label encoding (`suicide` = 1, `non-suicide` = 0)

### 3. 📈 EDA (Exploratory Data Analysis)
- Class distribution visualization
- Word cloud generation
- Term frequency analysis

### 4. 🧪 Feature Engineering
- Applied **TF-IDF Vectorization**
- Used unigrams & bigrams with max features

### 5. 🧠 Model Building
- **Logistic Regression**
- **Random Forest**
- **XGBoost Classifier**

All models were trained and tested on **80-20 train-test split**.

### 6. ⚖️ Handling Class Imbalance
- Applied **SMOTE (Synthetic Minority Oversampling Technique)**
- Balanced minority class (`suicide`) during training

### 7. 🧾 Evaluation Metrics
- **Confusion Matrix**
- **Classification Report**
  - Precision
  - Recall (critical for suicide detection)
  - F1-Score
- Visualizations of results (heatmaps, bar charts)

---

## 🏆 Results

| Model               | Precision | Recall | F1-Score |
|--------------------|-----------|--------|----------|
| Logistic Regression| 0.92      | 0.88   | 0.90     |
| Random Forest       | 0.95      | 0.91   | 0.93     |
| XGBoost             | 0.96      | 0.94   | 0.95     |

> 📌 **XGBoost** performed best in terms of overall precision, recall, and F1-score.

---

## ⚙️ Tech Stack

- 🐍 Python (Pandas, NumPy)
- 🧠 Scikit-learn, XGBoost
- 📊 Matplotlib, Seaborn, WordCloud
- 🗣️ NLTK & spaCy (for NLP)
- 🧪 SMOTE (via imbalanced-learn)

---

## ▶️ How to Run

1. Clone this repo
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the notebook:
   ```bash
   jupyter notebook mental-health-detection-via-nlp-on-social-media.ipynb
   ```

---

## 💡 Key Insights

- **Text-based features** (TF-IDF) work well with traditional ML models.
- **SMOTE** significantly improved model recall, especially for the suicidal class.
- **XGBoost** yielded the best balance of precision and recall.

---

## 🚀 Future Improvements

- Deploy as a web app (e.g., using Streamlit)
- Use BERT or other transformer models for improved accuracy
- Fine-tune threshold for maximizing recall on the `suicide` class
- Build real-time monitoring dashboard (for mental health professionals)

---

## 📚 References

- Reddit SuicideWatch Dataset
- [imbalanced-learn](https://imbalanced-learn.org/)
- [scikit-learn documentation](https://scikit-learn.org/stable/)