import pandas as pd, numpy as np, pickle, joblib
import matplotlib, klib, bokeh, seaborn as sns
import matplotlib.pyplot as plt
from bokeh.plotting import figure
import streamlit as st
import sklearn, imblearn, torch
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
import urllib.request, requests
from io import BytesIO
from types import ModuleType
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

def save_or_load_model(filename_or_io, model=None, action='load'):
    if action == 'save':
        with open(filename_or_io, 'wb') as file:
            pickle.dump(model, file)
        print(f"Model saved to {filename_or_io}")

    elif action == 'load':
        if isinstance(filename_or_io, str):
            with open(filename_or_io, 'rb') as file:
                loaded_model = pickle.load(file)
        elif isinstance(filename_or_io, BytesIO):
            loaded_model = pickle.load(filename_or_io)
        else:
            raise ValueError("Invalid type for filename_or_io. Use str (file path) or BytesIO.")
        
        print(f"Model loaded from {filename_or_io}")
        return loaded_model

    else:
        raise ValueError("Invalid action. Use 'save' or 'load.'")

st.sidebar.title("Machine Learning in Data Classification")

add_selectbox = st.sidebar.radio(
    "Menu",
    ("Data Description", "Data Preprocessing", "PCA", "Model Training & Evaluation", "Test the System")
)

data = pd.read_csv("https://raw.githubusercontent.com/FuZhangCheng/fyp-project/main/dataset/Medicaldataset.csv")

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

elif add_selectbox == "Data Preprocessing":

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

elif add_selectbox == "Model Training & Evaluation":
    st.header("4. Model Training & Evaluation")

    st.subheader("Traning Evaluation (Evaluate by 80% training dataset)")

    train_table = pd.read_csv("https://raw.githubusercontent.com/FuZhangCheng/fyp-project/main/evaluation/train_evaluation.csv")
    train_eval = train_table.copy()
    columns_to_convert = train_eval.columns.difference(['Model'])
    train_eval[columns_to_convert] = (train_eval[columns_to_convert] * 100).round(2).astype(str) + "%"

    st.dataframe(train_eval)

    st.subheader("Testing Evaluation (Evaluate by 20% testing dataset)")

    test_table = pd.read_csv("https://raw.githubusercontent.com/FuZhangCheng/fyp-project/main/evaluation/test_evaluation.csv")
    test_eval = test_table.copy()
    columns_to_convert = test_eval.columns.difference(['Model'])
    test_eval[columns_to_convert] = (test_eval[columns_to_convert] * 100).round(2).astype(str) + "%"

    st.dataframe(test_eval)

    st.markdown("---")

    st.subheader("Model Accuracy Comparison")

    fig_line, axes_line = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
    sns.lineplot(data=train_table, x="Model", y="Accuracy", ax = axes_line, color = "blue", label = "Training Accuracy")
    sns.lineplot(data=test_table, x="Model", y="Accuracy", ax = axes_line, color = "red", label = "Testing Accuracy")
    plt.tight_layout()
    plt.grid(True)
    st.pyplot(fig_line)

else:
    st.header("5. Test Model")

    option = st.selectbox("Please choose a model to test system: ", options=["Logistic Regression", "SVM", "KNN", "Naive Bayes", "Decision Tree", "Random Forest", "Deep Learning (ANN)"])

    model_dict = {
        "Logistic Regression": "https://raw.githubusercontent.com/FuZhangCheng/fyp-project/main/model/logistic.pkl",
        "SVM": "https://raw.githubusercontent.com/FuZhangCheng/fyp-project/main/model/svm.pkl", 
        "KNN": "https://raw.githubusercontent.com/FuZhangCheng/fyp-project/main/model/knn.pkl",
        "Naive Bayes": "https://raw.githubusercontent.com/FuZhangCheng/fyp-project/main/model/nb.pkl",
        "Decision Tree": "https://raw.githubusercontent.com/FuZhangCheng/fyp-project/main/model/dt.pkl",
        "Random Forest": "https://raw.githubusercontent.com/FuZhangCheng/fyp-project/main/model/rf.pkl",
        "Deep Learning (ANN)": "https://raw.githubusercontent.com/FuZhangCheng/fyp-project/main/model/mlp.pkl"
    }

    age = st.slider('Enter your age(int): ', 0, 120, 20)
    gender = {'Male': 1, 'Female': 0}[st.selectbox('Enter your gender: ', ('Male', 'Female'))]
    heart_rate = int(st.text_input('Enter Heart Rate: ', '80'))
    sys_blood_press = int(st.text_input('Enter Systolic blood pressure: ', '120'))
    dias_blood_press = int(st.text_input('Enter Diastolic blood pressure: ', '80'))
    bld_sgr = int(st.text_input('Enter Blood Sugar: ', '80'))
    ck_mb = float(st.text_input('Enter CK-MB: ', '3.10'))
    trop = float(st.text_input('Enter Troponin: ', '0.10'))

    record = {
        'Age': age,
        "Gender": gender,
        'Heart rate': heart_rate,
        'Systolic blood pressure': sys_blood_press,
        "Diastolic blood pressure": dias_blood_press,
        "Blood sugar": bld_sgr,
        "CK-MB": ck_mb,
        "Troponin": trop
    }

    st.write(record)

    # Load preprocess file
    data_preprocess_file = "https://raw.githubusercontent.com/FuZhangCheng/fyp-project/main/data_preprocessing/preprocessing_1.joblib"
    preprocessor = DataPreprocessor()
    file_content = urllib.request.urlopen(data_preprocess_file).read()
    preprocessor.load((BytesIO(file_content)))

    # Preprocess record
    preprocessed_record = preprocessor.transform(pd.DataFrame([record]))

    st.text("Preprocessed data: ")
    st.table(preprocessed_record)

    predict_btn = st.button("Predict Data")

    if predict_btn:
        model_file_content = urllib.request.urlopen(model_dict[option]).read()
        model = save_or_load_model(BytesIO(model_file_content), action='load')
        result = model.predict(preprocessed_record)
        if result == 1:
            st.success("The Result is positive")
        else:
            st.error("The Result is negative.")
