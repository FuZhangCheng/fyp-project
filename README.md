# FYP Project in Heart Disease Classification
FYP Project (Machine Learning Algorithm in data classification)

Explore the world of machine learning in healthcare with our project focusing on classifying heart disease data. We compare various algorithms to find the most effective model. Interact with the results through a user-friendly web application built using [Streamlit](https://streamlit.io/).

Web Application Link: https://fyp-project-jtqhremwyudrjnwvczc7tv.streamlit.app/  
  
#### 1. Dataset
I am going to do investigations into heart-related data classification which may help researchers in the domain of cardiovascular healthcare. A lot of heart disease dataset can be found in any resources including UCI, PHP, BIDMC CHF dataset, and PTB Diagnostic ECG datasets.  
Dataset: [Mendeley Heart Disease Dataset](https://data.mendeley.com/datasets/wmhctcrt5v/1) (Used in this project)  
  
#### 2. Machine Learning Algorithms
* Logistic Regression
* Support Vector Machine
* K-Nearest Neighbour
* Naiye bayes
* Decision Tree
* Random Forest
* Deep Learning (ANN)  

#### 3. Libraries/Dependencies (requirements.txt)
* Data Manipulation: [NumPy](https://numpy.org/doc/stable/), [Pandas](https://pandas.pydata.org/docs/)
* Data Visualization: [Matplotlib](https://matplotlib.org/stable/index.html), [Seaborn](https://seaborn.pydata.org/api.html), [Bokeh](https://docs.bokeh.org/en/latest/)
* Machine Learning: [Scikit-Learn](https://scikit-learn.org/stable/)
* For others, please refer to the requirements.txt. [Instruction for setting up **requirements.txt**](https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app/app-dependencies)  

#### 4. GitHub Repository Structure
This repository contains the code and resources for my Final Year Project (FYP).
- **data_preprocessing**: Contains joblib file for preprocessing_1.

- **dataset**: Contains datasets used in the project.
  - `Medicaldataset.csv`: Original medical dataset.
  - `preprocessed_Medicaldataset.csv`: Preprocessed version of the dataset.
  - `dataset_desc.txt`: Description of the dataset.

- **evaluation**: Contains evaluation-related files.
  - `eval_desc.txt`: Description of the evaluation process.
  - `test_evaluation.csv`: Evaluation results for the test set.
  - `train_evaluation.csv`: Evaluation results for the training set.

- **model**: Contains saved models and related files.
  - `model.txt`: Description of the models.
  - `logistic.pkl`, `svm.pkl`, `knn.pkl`, `nb.pkl`, `dt.pkl`, `rf.pkl`, `mlp.pkl`: Saved machine learning models.

- **image**: Contains images used in the project.
  - `heart.jpg`, `page-1.jpg`, `page-2.jpg`, `page-3.jpg`, `page-4.jpg`, `page-5.jpg`: Project-related images.

- **streamlit_app.py**: Streamlit application for [brief description].

- **FYP_notebook.ipynb**: Jupyter notebook containing the main project code.

- **preprocess_data.py**: Script for data preprocessing.

- **requirements.txt**: List of Python dependencies.  
