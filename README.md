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

---

## 📐 Understanding the Elbow Method

The **Elbow Method** is a heuristic technique used in unsupervised machine learning to determine the optimal number of clusters ($K$) for a dataset.

### 1. Within-Cluster Sum of Squares (WCSS)
K-Means aims to minimize the variance within each cluster. WCSS measures the sum of squared Euclidean distances between every data point and its assigned cluster centroid:

$$WCSS = \sum_{k=1}^{K} \sum_{i \in C_k} \vert{}\vert{}x_i - \mu_k\vert{}\vert{}^2$$

Where:
- $K$ = Total number of clusters
- $C_k$ = Set of data points assigned to cluster $k$
- $x_i$ = Individual data point vector
- $\mu_k$ = Centroid (mean vector) of cluster $k$

### 2. How the Curve Behaves
- When $K = 1$, WCSS is at its maximum because all data points are assigned to a single centroid.
- As $K$ increases, WCSS **monotonically decreases** because the clusters become smaller and data points sit closer to their respective centroids.
- If $K = N$ (where $N$ is total data points), $WCSS = 0$ because every point becomes its own centroid, creating an overfitted model.

### 3. Identifying the "Elbow" Point
When plotting $K$ (x-axis) against WCSS (y-axis), the graph forms an arm shape:
1. **Steep Drop:** From $K=1$ to $K=4$ or $5$, WCSS drops rapidly as obvious groups are separated.
2. **Diminishing Returns:** After a certain $K$ value, the rate of WCSS reduction slows down drastically.
3. **The Elbow:** The point of inflection (where the steep decline sharply flattens into a gentle slope) indicates the optimal balance between cluster compactness and model simplicity.

```text
  WCSS
   |  *
   |   \
   |    \
   |     \  <- Steep WCSS reduction
   |      \
   |_______* (Elbow Point: Optimal K)
   |        \___________________ <- Diminishing returns
   +---------------------------- K

## Project Structure

Customer-Segmentation-System/
│
├── CustomerSegmentation.py
├── Mall_Customers.csv
├── requirements.txt
├── .gitignore
├── README.md
├── LICENSE
└── screenshots/
