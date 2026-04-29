import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="World Development Clustering Dashboard", layout="wide")

st.title("World Development Clustering Dashboard")
st.write("Analysis of global development indicators using KMeans, Hierarchical and DBSCAN clustering.")

# Sidebar Menu
menu = st.sidebar.selectbox("Select Option", [
    "Dashboard",
    "Visualization",
    "Clustering"
])

# Load Dataset
df_original = pd.read_excel("World_development_mesurement.xlsx")
df = df_original.copy()

# Cleaning special characters
df['GDP'] = df['GDP'].replace(r'[\$,]', '', regex=True).astype(float)
df['Tourism Inbound'] = df['Tourism Inbound'].replace(r'[\$,]', '', regex=True).astype(float)
df['Tourism Outbound'] = df['Tourism Outbound'].replace(r'[\$,]', '', regex=True).astype(float)
df['Health Exp/Capita'] = df['Health Exp/Capita'].replace(r'[\$,]', '', regex=True).astype(float)

# Drop columns with more than 40% missing values
missing_percent = (df.isnull().sum() / len(df)) * 100
cols_to_drop = missing_percent[missing_percent > 40].index
df = df.drop(columns=cols_to_drop)
# Dropping this as it doesnot provide any meaningful info.
df = df.drop(columns=["Number of Records"])

# Fill missing values
df.fillna(df.median(numeric_only=True), inplace=True)

# Aggregate by country
df_country = df.groupby("Country").mean(numeric_only=True)

# Store original country data for display
df_display = df_country.copy()

# Log Transformation
df_country = np.log1p(df_country)

# Outlier Capping using IQR
for col in df_country.columns:
    Q1 = df_country[col].quantile(0.25)
    Q3 = df_country[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    df_country[col] = np.clip(df_country[col], lower, upper)

# Scaling
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_country)

# Elbow Curve Data
wcss = []
for i in range(2, 11):
    km = KMeans(n_clusters=i, random_state=42, n_init=10)
    km.fit(df_scaled)
    wcss.append(km.inertia_)

# KMeans Clustering
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(df_scaled)

# Hierarchical Clustering
hc = AgglomerativeClustering(n_clusters=4)
hc_labels = hc.fit_predict(df_scaled)

# DBSCAN
dbscan = DBSCAN(eps=1.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(df_scaled)

# Add cluster labels
df_display['KMeans_Cluster'] = kmeans_labels
df_display['HC_Cluster'] = hc_labels
df_display['DBSCAN_Cluster'] = dbscan_labels

# PCA for visualization
pca = PCA(n_components=2)
pca_data = pca.fit_transform(df_scaled)

df_display['PCA1'] = pca_data[:, 0]
df_display['PCA2'] = pca_data[:, 1]

# Silhouette Scores
kmeans_score = silhouette_score(df_scaled, kmeans_labels)
hc_score = silhouette_score(df_scaled, hc_labels)

# Cluster Names
cluster_names = {
    0: "Low Development",
    1: "Moderate Development",
    2: "High Development",
    3: "Developing Countries"
}

df_display['Cluster_Name'] = df_display['KMeans_Cluster'].map(cluster_names)

numeric_df = df_display.select_dtypes(include=['number'])

# ---------------- DASHBOARD ----------------
if menu == "Dashboard":

    st.header("Dashboard")


    col1, col2, col3 = st.columns(3)

    col1.metric("Total Records", df_original.shape[0])
    col2.metric("Total Features (Original)", df_original.shape[1])
    col3.metric("Features After Cleaning", df.shape[1])

    st.subheader("Raw Dataset")
    st.dataframe(df.head())

    st.subheader("Original Dataset Shape")
    st.write(df_original.shape)

    st.subheader("Missing Values Before Cleaning")
    st.dataframe(df_original.isnull().sum().reset_index().rename(columns={0:"Missing Values"}))

    st.subheader("Dropped Columns(>40% missing values)")
    st.write(list(cols_to_drop))

    st.subheader("Removed Irrelevant Column")
    st.write("Number of records")

    st.subheader("Cleaned Dataset Shape")
    st.write(df.shape)

    st.success("All missing values have been successfully handled.")

    st.subheader("Country Level Dataset (Before Clustering)")
    st.write(df_country.shape)
    st.dataframe(df_country.head())

    st.subheader("Country Level Dataset (After Clustering)")
    st.write(df_display.shape)
    st.dataframe(df_display.head())
    
    st.subheader("Summary Statistics")
    st.dataframe(df_display.describe().T)

# ---------------- VISUALIZATION ----------------
elif menu == "Visualization":

    st.header("Visualization")

    #---------------Heatmap---------------
    st.subheader("Correlation Heatmap")

    # Removing unwanted columns
    cols_to_remove = ['KMeans_Cluster','HC_Cluster', 'DBSCAN_Cluster', 'PCA1', 'PCA2']
    heatmap_df = df_display.drop(columns=cols_to_remove, errors='ignore')

    corr = heatmap_df.corr(numeric_only=True)

    fig1, ax1 = plt.subplots(figsize=(14, 8))
    sns.heatmap(corr, cmap="coolwarm", linewidths=0.5, ax=ax1)

    st.pyplot(fig1)

    st.write(
        "The heatmap shows strong positive correlations between indicators like GDP, Energy Usage and CO₂ Emissions, indicating that more developed economies tend to consume more energy,"
        "It also shows strong negative correlations between Birth Rate and Life Expectancy, suggesting that countries with higher life expectancy generally have lower birth rates."
    )


    #---------------Histogram---------------
    st.subheader("Histograms of Numeric Features")

    # Select only numeric columns
    numeric_df = df_display.select_dtypes(include='number')
    numeric_df = numeric_df.drop(columns=cols_to_remove, errors='ignore')

    # Plot histograms
    numeric_df.hist(figsize=(20, 15))

    st.pyplot(plt)
    st.write("Most features are highly skewed with a few countries showing extreme values, indicating significant inequality in development indicators across countries.")

    #---------------Boxplot---------------
    st.subheader("📦 Boxplot to Detect Outliers in Development Indicators")

    plot_columns = [
        col for col in numeric_df.columns
        if col not in ['KMeans_Cluster', 'HC_Cluster', 'DBSCAN_Cluster', 'PCA1', 'PCA2']
    ]

    fig_box, ax_box = plt.subplots(figsize=(14, 7))
    sns.boxplot(data=numeric_df[plot_columns], ax=ax_box)

    ax_box.set_xticklabels(ax_box.get_xticklabels(), rotation=90)
    ax_box.set_title("Outlier Detection for Development Indicators")

    plt.tight_layout()
    st.pyplot(fig_box)

    st.write("The boxplots reveal multiple outliers and wide variation in key indicators, highlighting significant differences in development levels across countries.")
     
    #---------------Pairplot---------------
    st.subheader("Pairplot of Important Development Indicators")

    pairplot_fig = sns.pairplot(
        df_display[['GDP', 'Birth Rate', 'Internet Usage', 'Life Expectancy Female', 'CO2 Emissions']]
    )

    st.pyplot(pairplot_fig)

    #---------------Worldmap---------------
    st.subheader("🌍 World Development Map")

    # Reseting index to bring country as column
    map_df = df_display.reset_index()

    # Creating map
    fig = px.choropleth(
        map_df,
        locations="Country",
        locationmode="country names",
        color="Cluster_Name",
        hover_name="Country",
        hover_data=["GDP", "Internet Usage", "Life Expectancy Female"],
        title="World Development Clusters"
    )

    fig.update_layout(height=800)
    st.plotly_chart(fig, use_container_width=True)

# ---------------- CLUSTERING ----------------
elif menu == "Clustering":

    st.header("Clustering")

    st.subheader("Elbow Method")
    fig5, ax5 = plt.subplots(figsize=(8, 5))
    ax5.plot(range(2, 11), wcss, marker='o')
    ax5.set_xlabel("Number of Clusters")
    ax5.set_ylabel("WCSS")
    ax5.set_title("Elbow Method for Optimal Clusters")
    st.pyplot(fig5)
    st.write("From the Elbow Method, the optimal number of clusters is chosen as 4, as the curve starts to flatten after this point.")


    st.subheader("Cluster Scatter Plots using PCA")
    fig6, axes = plt.subplots(1, 3, figsize=(18, 5))

    kmeans_counts = df_display['KMeans_Cluster'].value_counts().sort_index()
    hc_counts = df_display['HC_Cluster'].value_counts().sort_index()
    dbscan_counts = df_display['DBSCAN_Cluster'].value_counts().sort_index()

    axes[0].scatter(df_display['PCA1'], df_display['PCA2'], c=df_display['KMeans_Cluster'], cmap='viridis')
    axes[0].set_title("KMeans Clusters")

    axes[1].scatter(df_display['PCA1'], df_display['PCA2'], c=df_display['HC_Cluster'], cmap='plasma')
    axes[1].set_title("Hierarchical Clusters")

    axes[2].scatter(df_display['PCA1'], df_display['PCA2'], c=df_display['DBSCAN_Cluster'], cmap='coolwarm')
    axes[2].set_title("DBSCAN Clusters")

    plt.tight_layout()
    st.pyplot(fig6)


    # Silhouette Scores
    kmeans_score = silhouette_score(df_scaled, kmeans_labels)
    hc_score = silhouette_score(df_scaled, hc_labels)

    # DBSCAN silhouette score only if more than 1 cluster is formed
    if len(set(dbscan_labels)) > 1 and len(set(dbscan_labels)) < len(df_scaled):
        dbscan_score = silhouette_score(df_scaled, dbscan_labels)
    else:
        dbscan_score = "Failed"

    st.subheader("Silhouette Scores")

    scores_df = pd.DataFrame({
        "Algorithm": ["KMeans", "Hierarchical", "DBSCAN"],
        "Silhouette Score": [kmeans_score, hc_score, dbscan_score]
    })

    st.dataframe(scores_df)

    st.write(
    "KMeans was selected as the best clustering method because it has the highest silhouette score, "
    "which indicates better separated and more compact clusters compared to Hierarchical and DBSCAN."
    )

    cluster_counts = df_display['KMeans_Cluster'].value_counts().sort_index()

    cluster_table = cluster_counts.reset_index()
    cluster_table.columns = ["Cluster", "Count"]

    st.dataframe(cluster_table)

    st.subheader("KMeans Cluster Counts")

    cluster_counts = df_display['KMeans_Cluster'].value_counts().sort_index()

    fig7, ax7 = plt.subplots(figsize=(8, 5))
    cluster_counts.plot(kind='bar', ax=ax7, color='Purple')

    ax7.set_xlabel("Cluster")
    ax7.set_ylabel("Number of Countries")
    ax7.set_title("KMeans Cluster Counts")

    st.pyplot(fig7)

    st.subheader("Countries by Cluster")
    selected_cluster = st.selectbox(
        "Select KMeans Cluster",
        sorted(df_display['KMeans_Cluster'].unique())
    )

    filtered_df = df_display[df_display['KMeans_Cluster'] == selected_cluster]
    st.dataframe(filtered_df)

    st.subheader("Cluster Wise Mean Values")
    cluster_summary = df_display.groupby('KMeans_Cluster').mean(numeric_only=True).T
    st.dataframe(cluster_summary)

    st.subheader("Countries with Cluster Names")
    st.dataframe(df_display[['KMeans_Cluster', 'Cluster_Name']].head(20))

    csv = df_display.to_csv().encode('utf-8')

    st.download_button(
        label="Download Final Clustered Dataset",
        data=csv,
        file_name='final_country_data.csv',
        mime='text/csv'
    )

    st.subheader("Top 10 Countries by GDP")

    top_gdp = df_display.sort_values(by='GDP', ascending=False).head(10)

    fig8, ax8 = plt.subplots(figsize=(10, 6))
    ax8.bar(top_gdp.index, top_gdp['GDP'], color='green')

    ax8.set_xlabel("Country")
    ax8.set_ylabel("GDP")
    ax8.set_title("Top 10 Countries by GDP")

    plt.xticks(rotation=45)
    st.pyplot(fig8)
    st.dataframe(top_gdp[['GDP']])
    