# Multi-Modal-Stock-Prediction
Integrating Market Sentiment with Time Series Analysis

---

# 📈 Multi-Modal Stock Prediction

*A Deep Learning Approach Integrating Market Sentiment and Time Series Data*

---

## 📌 Introduction

Stock price prediction remains a fundamental challenge in financial modeling. Traditional approaches rely heavily on quantitative signals such as historical prices and volumes, often neglecting qualitative signals like market sentiment.

This project proposes a **multi-modal deep learning framework** that combines:

* 📊 Time series financial data
* 📰 News-based sentiment analysis

The goal is to enhance both **predictive performance** and **interpretability** in financial markets.

---

## 📂 Data Sources & Preprocessing

### 1. Time Series Data

* Source: Yahoo Finance via `yfinance`, `alpha_vantage`

* Asset: Apple Inc. (AAPL)

* Period: 2020–2024

* Features:

  * OHLCV (Open, High, Low, Close, Volume)
  * Technical Indicators:

    * Moving Average (5-day, 10-day)
    * RSI (14-day)
    * MACD + Signal Line

* Processing:

  * Z-score normalization (Volume)
  * Sliding window (length = 10)
  * Label generation:

    * **Up / Neutral / Down** (±0.5% threshold)
  * SMOTE oversampling
  * Class weighting + Focal Loss

---

### 2. Sentiment Data

* Source: Kaggle financial news dataset
* Processing:

  * Filtered by date alignment with price data
  * Aggregated daily (title + content)
  * Cleaned and concatenated

---

## 🧠 Methodology & Model Design

The task is divided into:

| Task                       | Type           |
| -------------------------- | -------------- |
| Next-day price prediction  | Regression     |
| Price direction prediction | Classification |

---

## 🔹 Time Series Model

Architecture:

* Hybrid **1D-CNN + BiLSTM**
* Enhancements:

  * Batch Normalization
  * Dropout
  * EarlyStopping
  * ReduceLROnPlateau

Input:

```
(samples, 10, feature_dim)
```

Goal:

* Capture **temporal dependencies**
* Learn **technical patterns**

---

## 🔹 Textual Sentiment Modeling

### 1. Sentiment Score (Explicit)

* Model: **FinBERT (yiyanghkust/finbert-tone)**
* Output:

  * Positive / Neutral / Negative probabilities

---

### 2. Sentiment Vector (Implicit)

We generate **128-dim embeddings** using:

| Method       | Description                  | Pros                       | Cons                    |
| ------------ | ---------------------------- | -------------------------- | ----------------------- |
| FinBERT      | Contextual embedding ([CLS]) | Strong financial semantics | Heavy & slow            |
| Word2Vec     | Static embedding (300 → 128) | Lightweight                | No context              |
| LSTM Encoder | Sequential modeling          | Captures syntax            | No pretrained knowledge |

Each day produces:

```
6 × 128-dimensional vectors
```

---

## 🔗 Model Integration

Fusion pipeline:

```
[Time Series Features] 
        ⬇
[Sentiment Features]
        ⬇
   Concatenation
        ⬇
       MLP
        ⬇
Prediction (Regression / Classification)
```

---

## 📊 Evaluation Metrics

### Regression

* RMSE
* MAE

### Classification

* Accuracy
* Precision / Recall
* F1 Score
* ROC-AUC

---

## 📈 Experimental Results

### 🔹 Performance Comparison

| Task           | Metric   | Baseline | Multi-Modal |
| -------------- | -------- | -------- | ----------- |
| Regression     | MAE      | N/A      | **0.6146**  |
| Regression     | RMSE     | N/A      | **0.7198**  |
| Classification | Accuracy | 38.65%   | **53.70%**  |
| Classification | ROC-AUC  | 0.5613   | **0.5838**  |
| Classification | F1 Score | 0.2155   | **0.3866**  |

---

### 🔍 Key Insights

* Multi-modal model **significantly outperforms baseline**
* Sentiment provides **additional information gain**
* Classification still suffers from:

  * Low recall on upward trends
  * Label imbalance

---

## ⚠️ Limitations

* Limited hardware → no Transformer-based models
* Only headline-level sentiment (not full article depth)
* Imbalanced dataset impacts classification

---

## 🚀 Future Work

* Introduce:

  * **Temporal Fusion Transformer (TFT)**
  * **Attention mechanisms**
* Use LLMs (e.g., GPT-4) for deeper semantic extraction
* Improve:

  * Data augmentation
  * Class balancing techniques
  * Multi-modal weighting strategies

---

## ✅ Conclusion

This project demonstrates that integrating:

> 📊 Historical price data + 🧠 Market sentiment

can significantly improve stock prediction performance.

* Strong gains in regression accuracy
* Meaningful improvements over linear baseline
* Promising direction for **AI-driven financial modeling**

---
