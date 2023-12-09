import pandas as pd, numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
import joblib, os

class DataPreprocessor:
    def __init__(self, data=None, remove_columns=None, numerical_columns=None, categorical_columns=None):
        self.data = data
        self.remove_columns = remove_columns
        self.numerical_columns = numerical_columns
        self.categorical_columns = categorical_columns
        
        # Placeholder for transformers
        self.column_transformer = None

    def fit_transform(self):

        # Drop specified columns
        if self.remove_columns:
            self.data = self.data.drop(columns=self.remove_columns)

        # Store the names of columns before preprocessing
        original_columns = self.data.columns

        # Define transformers for numerical and categorical columns
        numerical_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')

        # Create transformer for preprocessing
        transformers = []

        # Numerical columns
        if self.numerical_columns:
            transformers.append(('numerical', numerical_transformer, self.numerical_columns))
            original_columns = list(set(original_columns).symmetric_difference(set(self.numerical_columns)))

        # Categorical columns
        if self.categorical_columns:
            transformers.append(('categorical', categorical_transformer, self.categorical_columns))
            original_columns = list(set(original_columns).symmetric_difference(set(self.categorical_columns)))

        # Create column transformer
        self.column_transformer = ColumnTransformer(transformers)

        # Fit and transform the data
        transformed_data = self.column_transformer.fit_transform(self.data)

        # Concatenate the preprocessed data with the unprocessed columns
        transformed_data_with_original = pd.concat([pd.DataFrame(transformed_data), self.data[original_columns]], axis=1)

        return transformed_data_with_original

    def transform(self, data=None):
        if data is None:
            data = self.data

        # Drop specified columns
        if self.remove_columns:
            data = data.drop(columns=self.remove_columns)

        # Transform the data using the pre-fitted column transformer
        if self.column_transformer:
            transformed_data = self.column_transformer.transform(data)
            return transformed_data
        else:
            raise ValueError("fit_transform must be called before transform")
    
    def save(self, file_path):
        if self.column_transformer:
            # Save the preprocessor object
            joblib.dump(self, file_path)
            print(f"DataPreprocessor object saved to {file_path}")
        else:
            print("fit_transform must be called before saving.")
    
    def load(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File '{file_path}' not found. Please provide a valid file path.")

        # Load the preprocessor object
        loaded_preprocessor = joblib.load(file_path)

        # Copy the loaded attributes to the current object
        self.__dict__.update(loaded_preprocessor.__dict__)

        print(f"DataPreprocessor object loaded from {file_path}")


# # Sample data
# data = pd.DataFrame({
#     'Age': [25, 30, 35, 40],
#     'Income': [50000, 60000, 75000, 90000],
#     'Gender': ['Male', 'Female', 'Male', 'Female'],
#     'City': ['A', 'B', 'A', 'C']
# })

# # Create DataPreprocessor object
# preprocessor = DataPreprocessor(data=data,
#                                 remove_columns=None,  # No columns to remove in this example
#                                 numerical_columns=['Age', 'Income'],
#                                 categorical_columns=['Gender', 'City'],
#                                 pca_n_components=None)  # No PCA in this example

# # Fit and transform the data
# transformed_data = preprocessor.fit_transform()

# # Display the transformed DataFrame
# transformed_df = pd.DataFrame(transformed_data, columns=preprocessor.column_transformer.get_feature_names_out())
# print(transformed_df)

# # Set the test data
# test_data = pd.DataFrame({
#     'Age': [45],
#     'Income': [100000],
#     'Gender': ['Male'],
#     'City': ['B']
# })

# preprocessor.save("preprocessor_model.joblib")
# preprocessor.load("preprocessor_model.joblib")

# # Use the pre-fitted transformer to transform the test data
# transformed_test_data = preprocessor.transform(data = test_data)

# # Display the transformed test data
# transformed_test_df = pd.DataFrame(transformed_test_data, columns=preprocessor.column_transformer.get_feature_names_out())
# print("Transformed Test Data:")
# print(transformed_test_df)