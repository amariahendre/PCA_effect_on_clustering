# PCA effect on clustering 

This is a Streamlit app that allows users to visualize the effect of Principal Component Analysis (PCA) on clustering data. Users can upload their data, apply PCA, perform clustering using the KMeans algorithm, and visualize the results using t-SNE.

## Features

- Upload data files in formats: `.npy`, `.csv`, or `.txt`.
- Optional log2(x+1) data transformation.
- Configure the number of PCA components.
- Configure the number of clusters for KMeans.
- Visualize data clusters using t-SNE.
- Determine the optimal number of clusters using the elbow method.

## Getting Started

### Requirements

- Python
- Streamlit
- numpy
- pandas
- scikit-learn
- matplotlib

### Installation

Install the required packages using pip:

```bash
pip install streamlit numpy pandas scikit-learn matplotlib
```

### Running the App

Navigate to the directory containing the Streamlit app script and run:

```bash
streamlit run your_script_name.py
```

Replace `your_script_name.py` with the name of the script if it's different.

### Usage

1. Upload your data file using the sidebar file uploader.
2. Adjust the number of PCA components and clusters using the sliders.
3. View the t-SNE visualization and elbow method plot on the main page.
4. Optional: Apply a log2(x+1) transformation to the data.

## Author

- **Ana Maria Hendre** - [LinkedIn](https://www.linkedin.com/in/anamariahendre/)

