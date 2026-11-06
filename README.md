# 🛍️ Customer Segmentation System - Interactive ML Dashboard

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine-Learning-orange)
![Unsupervised Learning](https://img.shields.io/badge/Learning-Unsupervised-blueviolet)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-K--Means-F7931E?logo=scikitlearn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data_Analysis-150458?logo=pandas&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow?logo=opensourceinitiative&logoColor=white)

**Ever wondered why some customers spend thousands while others barely make a purchase?**

Understanding customer behavior is one of the biggest challenges in business. Not every customer has the same income, shopping habits, or spending preferences.

This is where Machine Learning comes into play!

**Customer Segmentation System** is an interactive Unsupervised Machine Learning application built using Python, Streamlit, and Scikit-Learn. It uses the **K-Means Clustering Algorithm** to group mall customers based on their annual income and spending scores.

Instead of manually categorizing customers, the application discovers patterns in customer data and groups similar customers into different segments.

From exploring customer demographics to finding optimal clusters and profiling new customers, the entire segmentation workflow is available through an interactive dashboard.

---

## 🚀 Key Features

### 1. 📊 Dataset Exploration & EDA

Understand the customer dataset before performing clustering.

- Preview customer records directly through the dashboard.
- View the total number of records and columns.
- Identify missing values in the dataset.
- Explore customer distribution across different clusters.
- Generate statistical summaries using Pandas.

### 2. 📈 Elbow Method for Optimal K

Choosing the right number of clusters is important for effective customer segmentation.

The application implements the **Elbow Method** to evaluate different cluster configurations.

- Calculate Within-Cluster Sum of Squares (WCSS).
- Evaluate cluster counts from K = 1 to the selected maximum K.
- Generate an Elbow Method visualization.
- Display WCSS values in a tabular format.
- Experiment with different maximum K values.

The Elbow Method helps identify a suitable cluster count by observing where the reduction in WCSS starts becoming less significant.

### 3. 🎯 Customer Cluster Visualization

Visualize how customers are grouped based on their income and spending behavior.

- Interactive 2D scatter plot of customer segments.
- Different colors and markers for different clusters.
- Display cluster centroids on the graph.
- Compare customers based on annual income and spending score.
- Understand similarities and differences between customer groups.

### 4. ⚙️ Dynamic Model Configuration

Experiment with K-Means hyperparameters directly from the Streamlit sidebar.

**Available configurations:**

| Parameter | Minimum | Maximum | Default |
|---|---:|---:|---:|
| Number of Clusters (K) | 2 | 10 | 5 |
| Elbow Method Max K | 5 | 15 | 10 |

Changing the number of clusters dynamically retrains the K-Means model and updates the segmentation results.

### 5. 🔮 Interactive Customer Profiling

Wondering which customer segment a new customer belongs to?

Enter custom customer information and let the trained model determine the appropriate cluster.

Users can adjust:

- Annual Income ($k)
- Spending Score (1-100)

Click the **Predict Segment** button to assign the customer to a cluster.

The application immediately displays the predicted cluster.

---

## 🧠 Machine Learning Approach

This project implements **Unsupervised Machine Learning** using the K-Means Clustering algorithm.

Unlike supervised learning, clustering does not require predefined labels. Instead, it discovers patterns and groups similar data points based on their features.

### 1. Dataset Preparation

The project uses the `Mall_Customers.csv` dataset.

The supplied dataset contains 100 customer records and 4 columns.

| Column | Description |
|---|---|
| CustomerID | Unique customer identifier |
| Age | Customer age |
| AnnualIncome | Customer's annual income |
| SpendingScore | Customer spending score |

**Features selected for clustering:**

- AnnualIncome
- SpendingScore

The application uses these two features to identify customer segments.

Age and CustomerID are not included in the current clustering model.

### 2. Feature Scaling

Before applying K-Means, the selected features are standardized using Scikit-Learn's `StandardScaler`.

```python
X = df[["AnnualIncome", "SpendingScore"]]

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
```

Why is scaling important?

K-Means uses distance calculations to group similar data points.

Since annual income and spending scores have different numerical scales, standardization ensures that both features contribute more fairly to the distance calculations.

### 3. Finding Optimal Clusters Using the Elbow Method

The Elbow Method helps evaluate the appropriate number of clusters.

The application calculates **WCSS (Within-Cluster Sum of Squares)** for different K values.

```python
wcss = []

for k in range(1, max_k + 1):

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    model.fit(X_scaled)

    wcss.append(model.inertia_)
```

How does it work?

- Start with one cluster.
- Increase the number of clusters.
- Calculate WCSS for every configuration.
- Plot the WCSS values against the number of clusters.
- Observe the elbow point where additional clusters provide diminishing reductions in WCSS.

The application allows users to inspect this graph and choose the cluster count they want to use.

**Note:** The application calculates the WCSS values and displays the Elbow Method graph. It does not automatically select the optimal K.

### 4. K-Means Clustering

After selecting the number of clusters, the application trains the final K-Means model.

```python
model = KMeans(
    n_clusters=n_clusters,
    random_state=42,
    n_init=10
)

clusters = model.fit_predict(X_scaled)
```

K-Means works by:

1. Initializing cluster centroids.
2. Assigning each customer to the nearest centroid.
3. Recalculating centroid positions.
4. Reassigning customers based on updated distances.
5. Repeating the process until the clustering converges.

Each customer is assigned a cluster label.

### 5. Cluster Visualization

The application visualizes customer segments using a scatter plot.

- X-axis: Annual Income ($k)
- Y-axis: Spending Score (1-100)
- Different colors and markers represent different clusters.
- Black X markers represent cluster centroids.

The centroids are transformed back into their original feature scales for visualization.

### 6. New Customer Profiling

The application accepts custom customer measurements and uses the trained K-Means model to determine their cluster.

```python
user_data = np.array([[income_input, spending_input]])

user_scaled = scaler.transform(user_data)

predicted_cluster = model.predict(user_scaled)[0]
```

The customer is assigned to the nearest learned cluster based on the standardized input features.

---

## 🎮 Application Dashboard

The application contains four interactive dashboard sections.

| Dashboard Tab | Functionality |
|---|---|
| 📊 Dataset & EDA | Dataset preview, missing values, cluster distribution and statistical summary |
| 📈 Elbow Method | WCSS calculation, Elbow Method graph and WCSS table |
| 🎯 Cluster Segments | Customer segmentation visualization, centroids and cluster mean summary |
| 🔮 Real-time Profiling | Assign custom customer measurements to an existing cluster |

### How to Use

1. Launch the Streamlit application.
2. Configure the number of clusters using the sidebar.
3. Select the maximum K value for the Elbow Method.
4. Explore the dataset through the Dataset & EDA tab.
5. Analyze the WCSS graph to understand the optimal cluster count.
6. Visualize customer segments and their centroids.
7. Enter custom income and spending score values in Real-time Profiling.
8. Click **Predict Segment** to view the assigned cluster.

---

## 🧰 Technologies & Libraries Used

| Technology / Library | Purpose |
|---|---|
| Python 3 | Core programming language |
| Streamlit | Interactive web application |
| Pandas | Data processing and manipulation |
| NumPy | Numerical computing |
| Matplotlib | Data visualization |
| Seaborn | Statistical data visualization |
| Scikit-Learn | Machine Learning implementation |
| StandardScaler | Feature standardization |
| KMeans | Customer clustering algorithm |

---

## 📂 Project Structure

```text
Customer-Segmentation-System/
│
├── CustomerSegmentation.py   # Main Streamlit application
│
├── Mall_Customers.csv        # Customer dataset
│
├── requirements.txt          # Project dependencies
│
├── .gitignore                # Git ignored files
│
├── README.md                 # Project documentation
│
└── LICENSE                   # MIT License
```

---

## 💻 Installation & Setup

Want to run the Customer Segmentation Dashboard locally? Follow these steps.

### Prerequisites

- Python 3.x installed on your system.
- pip package manager.
- Basic knowledge of Python and Machine Learning.

### Step 1: Clone the Repository

```bash
git clone https://github.com/bemo-codes/Customer-Segmentation-System
```

Navigate to the project directory:

```bash
cd Customer-Segmentation-System
```

### Step 2: Create a Virtual Environment

```bash
python -m venv venv
```

Activate the virtual environment.

**Windows:**

```bash
venv\Scripts\activate
```

**Linux / macOS:**

```bash
source venv/bin/activate
```

### Step 3: Install Dependencies

Install the required libraries:

```bash
pip install -r requirements.txt
```

The project uses the following dependencies:

```text
streamlit
pandas
numpy
matplotlib
seaborn
scikit-learn
```

### Step 4: Run the Application

Execute the following command:

```bash
streamlit run CustomerSegmentation.py
```

The application will start locally.

Open the following URL in your browser:

```text
http://localhost:8501
```

You're ready to explore customer segments and experiment with clustering! 🛍️

---

## ⚡ Implementation Highlights

Some important implementation details:

- **Streamlit Caching:** Uses `@st.cache_data` to avoid repeatedly loading the CSV dataset.
- **Feature Standardization:** Uses StandardScaler to normalize annual income and spending score.
- **Reproducible Clustering:** Uses `random_state=42` and `n_init=10` for K-Means.
- **Elbow Method:** Computes WCSS using the K-Means inertia attribute.
- **Centroid Visualization:** Converts cluster centroids back to their original feature scales.
- **Dynamic Clustering:** Allows users to change the number of clusters through the sidebar.
- **Interactive Profiling:** Uses the trained model to assign new customer measurements to existing clusters.

---

## ⚠️ Current Limitations

- The model currently uses only annual income and spending score for customer segmentation.
- Age and CustomerID are excluded from the clustering features.
- The Elbow Method provides a visual aid rather than automatically selecting the optimal cluster count.
- Cluster labels are numerical identifiers and do not automatically represent business categories.
- The application does not currently provide predefined customer personas.
- The model is retrained when the Streamlit application reruns.
- No model persistence is currently implemented.
- The application does not currently include supervised prediction or classification accuracy metrics.

---

## 🔮 Future Enhancements

There is plenty of room to expand this project into a more comprehensive customer analytics solution!

Some possible improvements include:

- **Customer Persona Generation:** Automatically identify customer groups such as high-value customers, budget-conscious shoppers, and potential customers.
- **Advanced Visualization:** Introduce PCA-based visualizations for exploring higher-dimensional customer data.
- **Alternative Clustering Algorithms:** Compare K-Means with DBSCAN and Hierarchical Clustering.
- **Silhouette Analysis:** Evaluate clustering quality using silhouette scores.
- **Customer Recommendations:** Suggest targeted marketing strategies for different customer segments.
- **Model Comparison Dashboard:** Compare different clustering configurations and algorithms.
- **Dataset Upload:** Allow users to upload custom customer datasets.
- **Business Intelligence Dashboard:** Introduce advanced customer analytics and segment-level insights.
- **Model Persistence:** Save trained models and scalers for reuse.

---

## 👨‍💻 Author

**Shivam Ramesh Kurlekar**

GitHub: [@bemo-codes](https://github.com/bemo-codes)

---

## 📄 License

This project is licensed under the **MIT License**.

See the [LICENSE](LICENSE) file for more information.

---

<p align="center">
  Built with Python 🐍 | Powered by Machine Learning 🤖 | Driven by Customer Insights 🛍️
</p>

<p align="center">
  ⭐ If you find this project interesting, consider giving the repository a star!
</p>
