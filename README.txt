WORLD DEVELOPMENT CLUSTERING – Machine Learning Project

This project includes:

Clustering world development.ipynb – Data cleaning, preprocessing, EDA, and clustering model building
kmeans_model.pkl – Saved trained KMeans model
scaler.pkl – Saved StandardScaler used for preprocessing
final_country_data.csv – Final processed dataset with cluster labels
dash.py – Streamlit dashboard for visualization and analysis
app.py – Streamlit web application for cluster prediction

How to Run:

Open terminal
Navigate to project folder
Run: streamlit run dash.py (for dashboard)
Run: streamlit run app.py (for prediction app)

This project groups countries into different development categories using clustering techniques such as K-Means, Hierarchical Clustering, and DBSCAN. Based on evaluation, K-Means performed the best and was used for final clustering.