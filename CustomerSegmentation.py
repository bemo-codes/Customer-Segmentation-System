#############################################################################
#
#   Importing required libraries
#
#############################################################################
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

#############################################################################
#
#   Function name:      load_data
#   Input:              data_path (string)
#   Description:        Loads the Mall Customers dataset from a CSV file and
#                       caches it for optimal Streamlit performance.
#   Date:               05/11/2026
#   Author:             Shivam Ramesh Kurlekar
#
#############################################################################
@st.cache_data
def load_data(data_path):
    df = pd.read_csv(data_path)
    return df

#############################################################################
#
#   Function name:      compute_elbow
#   Input:              X_scaled (ndarray), max_k (int)
#   Description:        Computes Within-Cluster Sum of Squares (WCSS) across
#                       a range of cluster values to determine the optimal K.
#   Date:               05/11/2026
#   Author:             Shivam Ramesh Kurlekar
#
#############################################################################
def compute_elbow(X_scaled, max_k=10):
    wcss = []
    for k in range(1, max_k + 1):
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        model.fit(X_scaled)
        wcss.append(model.inertia_)
    return wcss

#############################################################################
#
#   Function name:      plot_elbow
#   Input:              wcss (list), max_k (int)
#   Description:        Generates a Matplotlib figure for the Elbow Method plot.
#   Date:               05/11/2026
#   Author:             Shivam Ramesh Kurlekar
#
#############################################################################
def plot_elbow(wcss, max_k=10):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(range(1, max_k + 1), wcss, marker="o", color="#1f77b4", linewidth=2, markersize=6)
    ax.set_xlabel("Number of Clusters (K)")
    ax.set_ylabel("WCSS (Within-Cluster Sum of Squares)")
    ax.set_title("Elbow Method for Optimal K")
    ax.grid(True)
    return fig

#############################################################################
#
#   Function name:      plot_clusters
#   Input:              df (DataFrame), scaler (StandardScaler), model (KMeans)
#   Description:        Plots customer clusters and their corresponding centroids
#                       on a 2D scatter plot.
#   Date:               05/11/2026
#   Author:             Shivam Ramesh Kurlekar
#
#############################################################################
def plot_clusters(df, scaler, model):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    
    sns.scatterplot(
        data=df,
        x="AnnualIncome",
        y="SpendingScore",
        hue="Cluster",
        palette="tab10",
        style="Cluster",
        s=80,
        ax=ax
    )
    
    centroids_scaled = model.cluster_centers_
    centroids = scaler.inverse_transform(centroids_scaled)
    
    ax.scatter(
        centroids[:, 0],
        centroids[:, 1],
        s=250,
        c="black",
        marker="X",
        label="Centroids"
    )
    
    ax.set_title("Customer Segments & Centroids")
    ax.set_xlabel("Annual Income ($k)")
    ax.set_ylabel("Spending Score (1-100)")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True)
    return fig

#############################################################################
#
#   Function name:      main
#   Input:              None
#   Description:        Main entry point managing Streamlit layout, tabs, 
#                       sidebar configurations, and customer segmentation.
#   Date:               05/11/2026
#   Author:             Shivam Ramesh Kurlekar
#
#############################################################################
def main():
    st.set_page_config(page_title="Customer Segmentation Dashboard", page_icon="🛍️", layout="wide")

    st.title("🛍️ Customer Segmentation Dashboard (K-Means Clustering)")
    st.write("Unsupervised Machine Learning application to segment customers based on Annual Income and Spending Score.")

    # Sidebar parameters
    st.sidebar.header("⚙️ Model Configuration")
    n_clusters = st.sidebar.slider("Number of Clusters (K)", min_value=2, max_value=10, value=5, step=1)
    max_k = st.sidebar.slider("Elbow Method Max K", min_value=5, max_value=15, value=10, step=1)

    # Load data
    try:
        df = load_data("Mall_Customers.csv")
    except Exception:
        st.error("Error: `Mall_Customers.csv` not found in the root directory!")
        return

    # Standardize column naming if necessary
    rename_dict = {}
    for col in df.columns:
        if "annual" in col.lower() and "income" in col.lower():
            rename_dict[col] = "AnnualIncome"
        elif "spending" in col.lower() and "score" in col.lower():
            rename_dict[col] = "SpendingScore"
    if rename_dict:
        df = df.rename(columns=rename_dict)

    if "AnnualIncome" not in df.columns or "SpendingScore" not in df.columns:
        st.error("Error: Dataset must contain 'AnnualIncome' and 'SpendingScore' columns.")
        return

    # Extract features and scale
    X = df[["AnnualIncome", "SpendingScore"]]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Compute Elbow WCSS
    wcss = compute_elbow(X_scaled, max_k=max_k)

    # Fit final KMeans model
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = model.fit_predict(X_scaled)
    df["Cluster"] = clusters

    # Tabs navigation
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Dataset & EDA",
        "📈 Elbow Method",
        "🎯 Cluster Segments",
        "🔮 Real-time Profiling"
    ])

    with tab1:
        st.subheader("Exploratory Data Analysis")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Dataset Preview:**")
            st.dataframe(df.head(10))
            st.write(f"**Total Records:** `{df.shape[0]}` rows | `{df.shape[1]}` columns")
        with col2:
            st.write("**Missing Values:**")
            st.dataframe(df.isnull().sum())
            st.write("**Cluster Distribution:**")
            st.dataframe(df["Cluster"].value_counts().sort_index())

        st.write("**Statistical Summary:**")
        st.dataframe(df.describe())

    with tab2:
        st.subheader("Elbow Method for Finding Optimal Clusters")
        st.write("The elbow point indicates the optimal value of $K$ where adding more clusters yields diminishing returns in WCSS reduction.")
        fig_elbow = plot_elbow(wcss, max_k=max_k)
        st.pyplot(fig_elbow)

        st.write("**WCSS Values Table:**")
        wcss_df = pd.DataFrame({"Clusters (K)": range(1, max_k + 1), "WCSS": wcss})
        st.dataframe(wcss_df)

    with tab3:
        st.subheader("Customer Segments Visualization")
        fig_clusters = plot_clusters(df, scaler, model)
        st.pyplot(fig_clusters)

        st.write("**Cluster Mean Summary:**")
        cluster_summary = df.groupby("Cluster")[["AnnualIncome", "SpendingScore"]].mean().reset_index()
        st.dataframe(cluster_summary)

    with tab4:
        st.subheader("Predict Customer Segment")
        st.write("Enter custom customer parameters to assign them to a cluster:")

        col_in1, col_in2 = st.columns(2)
        with col_in1:
            income_input = st.number_input("Annual Income ($k)", min_value=1.0, max_value=200.0, value=50.0, step=1.0)
        with col_in2:
            spending_input = st.number_input("Spending Score (1-100)", min_value=1.0, max_value=100.0, value=50.0, step=1.0)

        if st.button("Predict Segment", type="primary"):
            user_data = np.array([[income_input, spending_input]])
            user_scaled = scaler.transform(user_data)
            predicted_cluster = model.predict(user_scaled)[0]
            st.success(f"Assigned Customer Segment: **Cluster {predicted_cluster}**")

#############################################################################
#
#   Starter of main
#
#############################################################################

if __name__ == "__main__":
    main()