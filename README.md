# PCA Effect on Clustering using Streamlit

This application allows users to visualize the effect of PCA (Principal Component Analysis) on clustering using the KMeans algorithm. It also provides an interface for users to upload datasets and adjust parameters for better insights.

## Features:
1. Upload datasets in `.npy`, `.csv`, or `.txt` format.
2. View the first 5 rows of the uploaded dataset.
3. Option to apply `log2(x+1)` transformation to the data.
4. Adjust the number of clusters for the KMeans algorithm.
5. Adjust the number of PCA components.
6. Visualization of clusters using t-SNE.
7. An elbow method plot to help determine the optimal number of clusters.

## How to use:
1. Run the Streamlit app.
2. Upload your dataset using the sidebar.
3. Adjust parameters using the sidebar sliders.
4. View the results and visualizations on the main pane.

## Requirements:
To run the app, you need to have the following libraries installed:
- streamlit
- numpy
- pandas
- scikit-learn
- matplotlib

You can install these using pip:
```
pip install streamlit numpy pandas scikit-learn matplotlib
```

## Running the app:
Navigate to the directory containing the Streamlit script and execute:
```
streamlit run your_script_name.py
```
Replace `your_script_name.py` with the name you saved the provided script as.

## Credits:
Made with ❤️ in Streamlit by [@anamariahendre](https://www.linkedin.com/in/anamariahendre/)

---

Feel free to customize this README as per your needs. Would you like me to save this as a `.md` file for you?
