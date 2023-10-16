import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

st.title('PCA effect on clustering')

uploaded_file = st.sidebar.file_uploader("Upload your data file (.npy, .csv, .txt)")

if uploaded_file is not None:

    if uploaded_file.name.endswith('.npy'):
        data = np.load(uploaded_file)
    elif uploaded_file.name.endswith('.csv'):
        data = pd.read_csv(uploaded_file).values
    elif uploaded_file.name.endswith('.txt'):
        data = np.loadtxt(uploaded_file)
    
    st.sidebar.write('Data shape:', data.shape)
    st.sidebar.write("---")

    apply_transformation = st.sidebar.checkbox('Apply log2(x+1) transformation to data', value=True)
    if apply_transformation:
        data = np.log2(data + 1)

    n_clusters = st.sidebar.text_input('Number of Clusters', '2,3,5,8,10').split(',')
    n_clusters = [int(k) for k in n_clusters]

    n_components = st.sidebar.text_input('Number of PCA components', '2,5,10,50,100,500').split(',')
    n_components = [int(pc) for pc in n_components]

    results = {}
    wss_plot_data = {}

    for i, pc in enumerate(n_components):
        pca = PCA(n_components=pc)
        X_pca = pca.fit_transform(data)
        
        pc_results = {}
        wss_values = []
        
        for j, k in enumerate(n_clusters):
            kmeans = KMeans(n_clusters=k, random_state=42)
            cluster_labels = kmeans.fit_predict(X_pca)
            score = silhouette_score(X_pca, cluster_labels)
            pc_results[k] = score
            wss_values.append(kmeans.inertia_)

        results[pc] = pc_results
        wss_plot_data[pc] = wss_values

    df = pd.DataFrame(results).T
    df.columns = [f'k={k}' for k in df.columns]
    plt.figure(figsize=(10, 6))
    sns.heatmap(df, cmap='YlGnBu', annot=True, fmt=".2f", linewidths=.5)
    plt.title('Silhouette Scores for Different PCA Components and k values')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Number of PCA Components')
    st.pyplot(plt)

    plt.figure(figsize=(10, 6))
    for pc, wss_values in wss_plot_data.items():
        plt.plot(n_clusters, wss_values, '-o', label=f'PC={pc}')
    plt.xlabel('Number of clusters')
    plt.ylabel('WSS')
    plt.legend()
    plt.title(f'Elbow Method for Various PCA Components')
    st.pyplot(plt)

    # fig, axs = plt.subplots(len(n_components), len(n_clusters), figsize=(15, 15))
    # for i, pc in enumerate(n_components):
    #     pca = PCA(n_components=pc)
    #     X_pca = pca.fit_transform(data)
    
    #     for j, k in enumerate(n_clusters):
    #         kmeans = KMeans(n_clusters=k, random_state=42)
    #         cluster_labels = kmeans.fit_predict(X_pca)

    #         tsne = TSNE(n_components=2, random_state=42, perplexity=40)
    #         z_tsne = tsne.fit_transform(X_pca)

    #         for label in np.unique(cluster_labels):
    #             idx = cluster_labels == label
    #             axs[i, j].scatter(z_tsne[idx, 0], z_tsne[idx, 1], label=f"Cluster {label}", alpha=0.7)
    #         axs[i, j].set_title(f"PC={pc}, k={k}")

    # plt.tight_layout() 
    # st.pyplot(fig)

else:
    st.write("Please upload a data file to proceed.")

st.sidebar.write("---")
st.sidebar.markdown(
    '<h6>Made in &nbsp<img src="https://streamlit.io/images/brand/streamlit-mark-color.png" alt="Streamlit logo" height="16">&nbsp by <a href="https://www.linkedin.com/in/anamariahendre/">@anamariahendre</a></h6>',
    unsafe_allow_html=True,
)
