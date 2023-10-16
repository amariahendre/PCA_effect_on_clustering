import streamlit as st
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

st.set_page_config(page_title='PCA effect on clustering')

st.title('PCA effect on clustering')

uploaded_file = st.sidebar.file_uploader("Upload your data file (.npy, .csv, .txt)")

# If a new file is uploaded, overwrite the default data
if uploaded_file is not None:

    # Load data based on file format
    if uploaded_file.name.endswith('.npy'):
        data = np.load(uploaded_file)
    elif uploaded_file.name.endswith('.csv'):
        data = pd.read_csv(uploaded_file)
        data = data_df.values
    elif uploaded_file.name.endswith('.txt'):
        data = np.loadtxt(uploaded_file)
    
    st.sidebar.write('Data shape:', data.shape)
    st.sidebar.write("---")

    # Add checkbox for data transformation
    apply_transformation = st.sidebar.checkbox('Apply log2(x+1) transformation to data', value=True)
    if apply_transformation:
        data = np.log2(data + 1)

    # Sidebar options
    n_clusters = st.sidebar.slider('Number of Clusters', 1, 10, 3)
    n_components = st.sidebar.slider('Number of PCA components', 10, 500, 50)

    # PCA
    pca = PCA(n_components=n_components)
    z = pca.fit_transform(data)

    # Clustering
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    cluster_labels = kmeans.fit_predict(z)
    
    silhouette_avg = silhouette_score(z, cluster_labels)
    st.write(f'Silhouette score for {n_clusters} clusters and {n_components} PCs: {silhouette_avg:.2f}')

        
    # t-SNE visualization
    tsne = TSNE(n_components=2, random_state=42, perplexity=40)
    z_tsne = tsne.fit_transform(z)

    # Cluster visualization
    fig, ax = plt.subplots(figsize=(8, 6))
    for label in np.unique(cluster_labels):
        idx = cluster_labels == label
        ax.scatter(z_tsne[idx, 0], z_tsne[idx, 1], label=f"Cluster {label}", alpha=0.7)

    # Elbow method plot
    inertia = []
    K = range(1, 11)
    for k in K:
        kmeanModel = KMeans(n_clusters=k, n_init=10)
        kmeanModel.fit(z)
        inertia.append(kmeanModel.inertia_)

    ax.set_title(f"{n_components} Principal Components with {n_clusters} Clusters")
    ax.legend()

    st.pyplot(fig)
    
    fig, ax = plt.subplots()
    ax.plot(K, inertia, 'bx-')
    ax.set_xlabel('Number of clusters')
    ax.set_ylabel('WSS')
    ax.set_title('The Elbow Method showing the optimal k')
    st.pyplot(fig)

else:
    st.write("Please upload a data file to proceed.")


st.sidebar.write("---")
st.sidebar.markdown(
                '<h6>Made in &nbsp<img src="https://streamlit.io/images/brand/streamlit-mark-color.png" alt="Streamlit logo" height="16">&nbsp by <a href="https://www.linkedin.com/in/anamariahendre/">@anamariahendre</a></h6>',
                unsafe_allow_html=True,
            )
