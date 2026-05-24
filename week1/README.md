# Statistics & Probability for Data Science & ML

A comprehensive Jupyter notebook covering fundamental statistics, probability theory, hypothesis testing, and statistical monitoring concepts essential for machine learning practitioners.

## 📋 Overview

This project provides hands-on implementations of core statistical concepts with practical ML applications. All code is written **from scratch** (minimal library dependencies) to build intuition and understanding.

**Topics Covered:**
- Part 5: Descriptive Statistics, Hypothesis Testing, Error Metrics, Distribution Testing, Model Monitoring
- Part 6: Probability Fundamentals, Distributions, Bayes' Theorem, Central Limit Theorem

## 📁 Project Structure

```
.
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── statistics_probability.ipynb       # Main Jupyter notebook
└── data/                              # (Optional) Sample datasets
```

## 🎯 Part Breakdown

### Part 5: Statistical Foundations for ML

#### 5.1 Descriptive Statistics
- **Concepts:** Mean, median, standard deviation, IQR, range
- **Implementation:** Manual computation + histogram with KDE overlay
- **Use Case:** Understanding data distributions before modeling

#### 5.2 Hypothesis Testing
- **Concepts:** Null/alternative hypotheses, t-tests, p-values, Pearson correlation
- **Implementation:** One-sample t-test, correlation analysis
- **Use Case:** Testing if Engineering salaries differ from company average

#### 5.3 Error Metrics (from scratch)
- **Implemented:** MAE, MSE, RMSE, R², Adjusted R²
- **No sklearn used** — pure NumPy implementations
- **Use Case:** Model evaluation without relying on black-box libraries

#### 5.4 Distribution Testing & Stationarity
- **Tests:** Kolmogorov-Smirnov (KS test), Augmented Dickey-Fuller (ADF)
- **Concept:** Detecting non-stationary time series, differencing
- **Use Case:** Pre-processing time series data for forecasting

#### 5.5 Model Monitoring
- **Implementation:** Population Stability Index (PSI) from scratch
- **Concepts:** Concept drift, covariate drift, distribution shift detection
- **Use Case:** Triggering model retraining when data shifts

### Part 6: Probability Theory

#### 6.1 Core Probability Concepts
- **Topics:** Sample space, basic probabilities, joint & conditional probability, independence
- **Example:** Marble drawing without replacement
- **Use Case:** Foundation for Bayesian reasoning

#### 6.2 Distributions in the Wild
- **Distributions:** Normal, Binomial, Poisson
- **Visualization:** PDF/PMF plots with multiple parameter sets
- **ML Applications:** Feature scaling, A/B testing, event count prediction

#### 6.3 Bayes' Theorem
- **Implementation:** Spam filter example with posterior computation
- **Function:** `naive_bayes_predict()` — computing P(Spam | word)
- **Terms:** Prior, likelihood, evidence, posterior with real-world mapping

#### 6.4 Central Limit Theorem (CLT)
- **Experiment:** Draw 5,000 samples from exponential distribution, plot sample means
- **Validation:** KS test confirms sample means are approximately normal
- **Insight:** Why classical statistical tests work even when raw data is non-normal

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/statistics-probability-ml.git
cd statistics-probability-ml
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch Jupyter Notebook
```bash
jupyter notebook statistics_probability.ipynb
```

### 4. Run Cells
- Execute cells sequentially from top to bottom
- Each section is self-contained with sample data generation

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `numpy` | ≥1.21.0 | Array operations, mathematical functions |
| `pandas` | ≥1.3.0 | Data manipulation (cleaning, filling NaN) |
| `scipy` | ≥1.7.0 | Statistical tests (ttest, kstest, pearsonr) |
| `statsmodels` | ≥0.13.0 | Augmented Dickey-Fuller test (ADF) |
| `matplotlib` | ≥3.4.0 | Plotting distributions, histograms, overlays |
| `jupyter` | ≥1.0.0 | Interactive notebook environment |

See `requirements.txt` for exact pinned versions.

## 🔧 Key Implementations (From Scratch)

### MAE, MSE, RMSE, R², Adjusted R²
```python
residuals = y_true - y_pred
mae = np.mean(np.abs(residuals))
mse = np.mean(residuals ** 2)
rmse = np.sqrt(mse)
ss_res = np.sum(residuals ** 2)
ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
r2 = 1 - (ss_res / ss_tot)
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
```

### Population Stability Index (PSI)
```python
def compute_psi(expected, actual, bins=10):
    breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
    expected_pct = np.histogram(expected, bins=breakpoints)[0] / len(expected) + 1e-10
    actual_pct = np.histogram(actual, bins=breakpoints)[0] / len(actual) + 1e-10
    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return psi
```

### Naive Bayes Posterior
```python
def naive_bayes_predict(prior_spam, p_word_given_spam, p_word_given_ham):
    prior_ham = 1 - prior_spam
    p_word = (p_word_given_spam * prior_spam) + (p_word_given_ham * prior_ham)
    posterior_spam = (p_word_given_spam * prior_spam) / p_word
    return posterior_spam
```

## 📊 Example Outputs

### Part 5.1 - Descriptive Statistics
```
Mean:   73981
Median: 74895
Std:    23275
Range:  9506 – 143004
IQR:    30150
```

### Part 5.2 - Hypothesis Testing
```
Overall mean salary: $75447
Engineering mean salary: $83293
t-statistic: 2.0521
p-value: 0.0460
Reject H₀ at α=0.05? True
✓ Engineering salaries ARE significantly higher
```

### Part 5.3 - Error Metrics
```
MAE:        0.3500
MSE:        0.1400
RMSE:       0.3742
R²:         0.9689
Adj. R²:    0.9564
```

### Part 5.5 - Model Monitoring
```
PSI: 0.5234
Shift severity: Major
→ Immediate retraining recommended
```

### Part 6.4 - Central Limit Theorem
```
Population mean: 0.9990, Population std: 1.0010
Sample means mean: 0.9990, std: 0.1829
CLT predicted std: 0.1828
KS test p-value: 0.7341 → Approximately normal? True
```

## 💡 Learning Outcomes

After completing this notebook, you will:

✅ **Understand core statistical concepts** — mean, variance, distributions, hypothesis testing  
✅ **Implement metrics from scratch** — build intuition by coding MAE, MSE, R² manually  
✅ **Apply statistical tests** — t-tests, KS tests, ADF tests with proper interpretation  
✅ **Reason about probability** — joint, conditional, Bayesian thinking  
✅ **Monitor production models** — detect distribution shift with PSI, trigger retraining  
✅ **Validate statistical assumptions** — normality, stationarity, independence  

## 🔗 Real-World Applications

| Concept | Use Case |
|---------|----------|
| Descriptive Statistics | EDA before modeling, feature engineering |
| Hypothesis Testing | A/B testing, model comparison, statistical validation |
| Error Metrics | Model evaluation, hyperparameter tuning, benchmarking |
| Distribution Testing | Time series preprocessing, anomaly detection setup |
| PSI & Drift Detection | Production monitoring, automated retraining triggers |
| Bayes' Theorem | Spam filtering, fraud detection, recommendation systems |
| CLT | Justifies t-tests on real (non-normal) data, confidence intervals |

## 📚 Further Reading

- **Probability & Statistics:**
  - *Introduction to Statistical Learning* — James, Witten, Hastie, Tibshirani
  - *Bayesian Methods for Hackers* — Davidson-Pilon
  - *StatQuest with Josh Starmer* (YouTube) — Excellent visual explanations

- **ML Monitoring:**
  - *Monitoring Machine Learning Models in Production* — Shreya Shankar et al.
  - *Evidently AI* — Open-source tool for drift detection

- **Statistical Testing:**
  - *The Book of Why* — Judea Pearl (causal inference beyond correlation)
  - SciPy Documentation — Detailed test descriptions and assumptions

## ⚠️ Assumptions & Limitations

- **Sample sizes:** Examples use n=200–1000 for computational efficiency; real datasets often larger
- **Data generation:** Synthetic data used for demonstration; real data has more complexity
- **Statistical tests:** Assumptions (normality, homogeneity of variance) not always tested
- **No preprocessing:** Some sections assume clean, pre-filled data (NaN handling skipped)

## 🤝 Contributing

Contributions welcome! To improve this notebook:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/add-bayesian-networks`)
3. Commit changes (`git commit -m 'Add Bayesian network example'`)
4. Push to branch (`git push origin feature/add-bayesian-networks`)
5. Open a Pull Request

## 📝 License

This project is licensed under the **MIT License** — see LICENSE file for details.

## 👤 Author

Created as an educational resource for statistics and probability in machine learning.

**Questions or feedback?**  
Open an issue on GitHub or reach out with suggestions for improvements.

---

## 🎓 How to Use This Notebook

### For Students
- Work through each section sequentially
- Modify code, change parameters, run experiments
- Answer the markdown questions to solidify understanding
- Implement additional test cases

### For Practitioners
- Reference specific sections as needed
- Copy implementations for your own projects
- Use as a checklist for statistical validation steps
- Adapt PSI monitoring for production models

### For Educators
- Use sections as lecture material or problem sets
- Modify examples to match your domain (e.g., medical data, finance)
- Extend with additional distributions or tests

---

**Last Updated:** May 2026  
**Python Version:** 3.8+  
**Status:** Complete & Ready for Use ✓
