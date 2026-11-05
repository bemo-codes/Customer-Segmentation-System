# Customer Segmentation System – Streamlit ML Dashboard

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)
![Unsupervised ML](https://img.shields.io/badge/Machine%20Learning-K--Means-orange)
![Scikit-Learn](https://img.shields.io/badge/Library-scikit--learn-F7931E?logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow)

An interactive Unsupervised Machine Learning dashboard built with Streamlit and Scikit-Learn. The system segments mall customers based on Annual Income and Spending Score using K-Means Clustering, StandardScaler feature normalization, and interactive Elbow Method evaluation.

## Features

- **StandardScaler Preprocessing:** Standardizes feature distributions for reliable Euclidean distance calculation.
- **Interactive Elbow Method:** Computes and plots Within-Cluster Sum of Squares (WCSS) to evaluate optimal $K$.
- **Dynamic Cluster Profiling:** Adjust number of clusters live from the sidebar and view updated centroids.
- **Real-Time Customer Profiling:** Input custom income and spending scores to predict segment placement.

## Project Structure

```text
Customer-Segmentation-System/
│
├── CustomerSegmentation.py
├── Mall_Customers.csv
├── requirements.txt
├── .gitignore
├── README.md
├── LICENSE
└── screenshots/