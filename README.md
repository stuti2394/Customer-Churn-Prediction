# Customer Churn Prediction

## Project Overview

This project focuses on predicting customer churn — identifying whether a customer is likely to leave a service provider — using classical machine learning algorithms. The dataset used is the **Telco Customer Churn dataset** from Kaggle, which contains demographic, service usage, and billing details of telecom customers.

By applying predictive modeling techniques, the project aims to help businesses improve customer retention strategies and reduce churn rates.

The methodology includes data preprocessing, feature engineering, model training, hyperparameter optimization, and model evaluation across multiple algorithms.

---

## Features

* **Machine Learning Models**:

  * Logistic Regression
  * Random Forest
  * K-Nearest Neighbors (KNN)
  * Naive Bayes
  * XGBoost

* **Class Imbalance Handling**: Applied class weights and evaluation metrics to deal with churn imbalance.

* **Hyperparameter Optimization**: Used `GridSearchCV` for model tuning to achieve better precision, recall, and F1-score.

* **Clustering Analysis**: Applied K-means clustering for unsupervised customer segmentation.

* **Evaluation Metrics**: Accuracy, Precision, Recall, F1 Score, Confusion Matrix.

---

## Objective

* Predict customer churn by identifying customers at risk of leaving.
* Compare multiple machine learning models to determine the most effective approach.
* Enable data-driven decision making to reduce churn and improve customer retention.
* Explore unsupervised clustering to segment customers for targeted strategies.

---

## Requirements

The following Python libraries are required to run this project:

* `numpy` — numerical operations
* `pandas` — data manipulation and preprocessing
* `scikit-learn` — ML models, preprocessing, evaluation
* `matplotlib` — data visualization
* `seaborn` — advanced visualization
* `xgboost` — gradient boosting classifier

---

## Results

The final tuned models achieved the following approximate performance (replace with your actual results):

| Model               | Accuracy | Precision | Recall | F1 Score |
| ------------------- | -------- | --------- | ------ | -------- |
| Logistic Regression | \~0.75   | \~0.52    | \~0.76 | \~0.62   |
| Random Forest       | \~0.78   | \~0.60    | \~0.51 | \~0.55   |
| KNN                 | \~0.76   | \~0.56    | \~0.54 | \~0.55   |
| Naive Bayes         | \~0.28   | \~0.26    | \~0.95 | \~0.41   |
| XGBoost             | \~0.77   | \~0.59    | \~0.51 | \~0.54   |

Key observations:

* XGBoost and Random Forest performed best overall.
* Naive Bayes showed very high recall but very low precision, making it unsuitable in practice but useful for comparison.
* Logistic Regression provided strong baseline performance.

---

## Future Work

* Perform advanced feature engineering, including interaction terms and domain-specific transformations.
* Explore deep learning models such as ANN and LSTM for churn prediction.
* Build a REST API for real-time churn predictions.
* Develop a PowerBI or Streamlit dashboard for visualization and business use.

---

## Contributing

Feel free to fork this repository, submit issues, and create pull requests. Contributions are welcome.

---

## Contact

For any questions or inquiries, please contact **Stuti Srivastava** at [stutisrivastava0923@gmail.com](mailto:stutisrivastava0923@gmail.com).

---

Would you like me to also draft the **requirements.txt file** so your repo is fully ready-to-run for recruiters?
