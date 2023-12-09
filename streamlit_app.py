import pandas as pd, numpy as np
import matplotlib, klib, bokeh, seaborn as sns
import matplotlib.pyplot as plt
from bokeh.plotting import figure
import streamlit as st
import sklearn, imblearn, torch
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from preprocess_data import DataPreprocessor

def data_preprocess(data, numerical_columns = None, categorical_columns = None):
    # 1. Standardize the range of numerical features
    if numerical_columns is not None:
        std_scaler = StandardScaler(copy = False).fit(data[numerical_columns])
        data[numerical_columns] = std_scaler.transform(data[numerical_columns])
    
    # 2. Encoding categorical features by using one hot encoder
    if categorical_columns is not None:
        onehot_enc = OneHotEncoder(handle_unknown='ignore').fit(data[categorical_columns])
        onehot_result = onehot_enc.transform(data[categorical_columns])

        onehot_df = pd.DataFrame(onehot_result, columns=onehot_enc.get_feature_names_out(categorical_columns))
        data = pd.concat([data.drop(categorical_columns, axis=1), onehot_df], axis=1)

    # 2. Convert 'Result' target feature into binary form like '0' and '1'
    data["Result"] = (data["Result"] == "positive").astype(int)

    return data

st.sidebar.title("Machine Learning in Data Classification")

add_selectbox = st.sidebar.radio(
    "Menu",
    ("Data Description", "Data Preprocessing (optional)", "PCA", "Train & Evaluate Model")
)

data = pd.read_csv("Medicaldataset.csv")

#data = pd.read_csv("https://raw.githubusercontent.com/FuZhangCheng/fyp-project/main/dataset/Medicaldataset.csv")

if add_selectbox == "Data Description":
    st.title("Heart Disease Data Classification")
    st.empty()
    st.write("---")
    st.header("1. Data Description")
    st.markdown("The heart attack datasets were collected at Zheen hospital in Erbil, Iraq, from **January 2019 to May 2019.**")
    st.markdown("Dataset Link: **https://data.mendeley.com/datasets/wmhctcrt5v/1**")
    st.markdown("According to the provided information, the medical dataset **classifies** either heart attack or none.")

    st.table(data.head())

    st.caption('These are the first 5 records which are shown in the table above.')

    st.markdown(f'''
    - Data has **`{len(data)} records`** and **`{len(data.columns)} columns`**.
    - the memory usage of the data is **`{data.memory_usage().sum()}`** bytes.
    - There is no **null** value in dataset.
    ''')

    st.markdown("Download the file: ")

    download_clk = st.download_button(
        label="Download Data as CSV file",
        data=data.to_csv(index=False).encode('utf-8'),
        file_name="Heart_Dataset.csv",
        mime="text/csv"
    )

    if download_clk:
        st.success("Downloading Data completed!")

    st.markdown("---")

    # checkbox for Data exploration and visualisation
    data_vis_check = st.checkbox("Data exploration and visualisation")

    if data_vis_check:
        ### Data exploration and visualisation
        st.header("2. Data Exploration and Visualisation")

        st.markdown("Show the bar plot (frequency) of **'Result'** feature")

        # plot bar chart of "Result" feature
        fig_bar, axes_bar = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
        ax_bar = data['Result'].value_counts().plot(kind='bar', fontsize=13, color='lightgreen', figsize=(7, 4), width = 0.3, ax=axes_bar)
        ax_bar.set_title('Result', size = 20, pad = 30)
        ax_bar.set_ylabel('Counts', fontsize = 14)
        st.pyplot(fig_bar)

        option = st.selectbox("Select features", data.columns)

        st.subheader("Show the histogram of features")

        # plot histogram of features
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
        sns.histplot(data, x = option, bins=50, kde=True, ax=axes, alpha=0.5, color = "red", legend = True, hue = 'Result')
        axes.set_title(f'Histogram with Density Plot for {option}', size = 20, pad = 30)
        axes.set_ylabel('Sample number (frequency)', fontsize = 14)
        plt.tight_layout()
        st.pyplot(fig)

        st.subheader("Draw boxplot of the features")

        # Plot a Boxplot of the features
        fig_box, axes_box = plt.subplots(nrows=1, ncols=1, figsize=(12, 5))
        sns.boxplot(x = option, data = data, ax=axes_box, y = 'Result')
        axes_box.set_title(f'Boxplot for {option}', size = 20, pad = 30)
        plt.tight_layout()
        st.pyplot(fig_box)

        st.subheader("Draw scatterplot of the features")

        data_copy = data.copy()
        data_copy['index'] = data.copy().index
        fig_scatter, axes_scatter = plt.subplots(nrows=1, ncols=1, figsize=(7, 7))
        sns.scatterplot(data=data_copy, x="index", y=option, ax = axes_scatter, hue = 'Result')
        axes_scatter.set_title(f'Scatterplot for {option}', size = 10, pad = 10)
        plt.tight_layout()
        st.pyplot(fig_scatter)

        st.markdown("---")

        st.subheader("Descriptive statistic of Data features")

        desc_data = data.drop(columns="Gender").describe()

        st.table(desc_data)

        st.markdown("Show correlation matrix of the features and **'Result'**.")

        data_copy = data.copy()
        data_copy["Result"] = (data_copy["Result"] == "positive").astype(int)
        data_corr = data_copy.corr()

        # Plot a correlation matrix into heatmap
        fig_corr, axes_corr = plt.subplots(nrows=1, ncols=1, figsize=(12, 5))
        sns.set(rc = {'figure.figsize':(15,5)})
        sns.heatmap(data_corr, annot=True, cmap=sns.color_palette("crest", as_cmap=True), ax = axes_corr)
        st.pyplot(fig_corr)

elif add_selectbox == "Data Preprocessing (optional)":

    st.header("3. Data Preprocessing")

    st.caption("The original data: ")

    st.table(data.head())

    # st.subheader("Standardization")

    num_processed_options = st.multiselect("The numerical features to be preprocessed: ", options=data.columns, default=['Age', 'Heart rate', 'Systolic blood pressure', 'Diastolic blood pressure', 'Blood sugar', 'CK-MB', 'Troponin'])

    cat_processed_options = st.multiselect("The categorical features to be preprocessed: ", options=data.columns)

    # st.caption("In this case, we only use StandardScaler() from sklearn library to preprocess our data.")

    # st.latex(r"z = \frac{x - \mu}{\sigma}")

    st.caption("The related code: ")
    code = '''def data_preprocess(data, numerical_columns, categorical_columns):
      # 1. Standardize the range of numerical features
      std_scaler = StandardScaler(copy = False).fit(data[numerical_columns])
      data[numerical_columns] = std_scaler.transform(data[numerical_columns])
        
      # 2. Encoding categorical features by using one hot encoder
      onehot_enc = OneHotEncoder(handle_unknown='ignore').fit(data[categorical_columns])
      data[categorical_columns] = onehot_enc.transform(data[categorical_columns])

      # 2. Convert 'Result' target feature into binary form like '0' and '1'
      data["Result"] = (data["Result"] == "positive").astype(int)

      return data'''
    st.code(code, language='python')

    prep_clk = st.button("Preprocess Data")

    # Preprocess data by using standard scaler
    if prep_clk:
        data_prep = data_preprocess(data, numerical_columns = num_processed_options)
        st.caption("The preprocessed data: ")
        st.table(data_prep.head())
        st.success("Data preprocessing completed!")

    file_name = st.text_input("Please give a file name to save data:", "data.csv")
    if st.button(" Preprocess & Save Data"):
        data_prep = data_preprocess(data, numerical_columns = num_processed_options)
        data_prep.to_csv(file_name, index=False)
        st.success("Save processed Data completed!")

elif add_selectbox == "PCA":
    st.header("Principal component analysis (PCA)")

    st.markdown("`Principal Component Analysis (PCA)` is a dimensionality reduction technique used in machine learning and data analysis.")

    uploaded_file = st.file_uploader("Choose a CSV file: ", type = ['csv'])

    if uploaded_file is not None:

        data_pca = pd.read_csv(uploaded_file)

        st.markdown("Original Data: ")

        st.table(data_pca.head())

        y_feature = st.selectbox("Choose a target feature: ", options=data_pca.columns)

        X = data_pca.drop(y_feature, axis = 1)
        y = data_pca[y_feature]

        st.markdown("After performing PCA (n_components=2) , ...")

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)

        st.table(pd.DataFrame(X_pca).head())

        pca_checkbox = st.checkbox("Show PCA Plotting")

        if pca_checkbox:
            fig_scatter, axes_scatter = plt.subplots(nrows=1, ncols=1, figsize=(7, 7))
            sns.scatterplot(x = X_pca[:,0], y = X_pca[:,1], ax = axes_scatter, hue = y)
            axes_scatter.set_title(f'Scatterplot for PCA Plotting', size = 10, pad = 10)
            plt.tight_layout()
            st.pyplot(fig_scatter)

else:
    st.header("4. Train Model")

    option = st.selectbox("Choose an algorithm to train model: ", options=["Logistic Regression", "SVM", "KNN", "Naive Bayes", "Decision Tree", "Random Forest", "Deep Learning (ANN)"])

    uploaded_file = st.file_uploader("Choose a CSV file: ", type = ['csv'])

    if uploaded_file is not None:

        data_train = pd.read_csv(uploaded_file)

        st.table(data_train.head())

        y_feature = st.selectbox("Choose a target feature: ", options=data_train.columns)

        X = data_train.drop(y_feature, axis = 1)
        y = data_train[y_feature]

        st.markdown("---")

        st.subheader("Split into Training and Testing Dataset")

        train_size = st.slider('The proportion of training dataset: ', 0.0, 1.0, 0.8)

        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=42)

    st.markdown("---")
