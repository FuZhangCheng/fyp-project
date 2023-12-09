import pandas as pd, numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
import joblib, os

# Define a class for data preprocessing
class DataPreprocessor:
    def __init__(self, data=None, remove_columns=None, numerical_columns=None, categorical_columns=None):
        """
        Constructor for DataPreprocessor class.

        Parameters:
        - data (pd.DataFrame): Input data in tabular form.
        - remove_columns (list): Columns to be removed from the data.
        - numerical_columns (list): Numerical columns for preprocessing.
        - categorical_columns (list): Categorical columns for preprocessing.
        """
        self.data = data
        self.remove_columns = remove_columns
        self.numerical_columns = numerical_columns
        self.categorical_columns = categorical_columns
        
        # Placeholder for transformers
        self.column_transformer = None

    def fit_transform(self):
         """
        Fit and transform the input data using specified preprocessing steps.

        Returns:
        - pd.DataFrame: Transformed data with original columns.
        """
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
        """
        Transform new data using the pre-fitted column transformer.

        Parameters:
        - data (pd.DataFrame, optional): New data to be transformed. If not provided, the original data is used.

        Returns:
        - np.ndarray: Transformed data.
        """
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
        """
        Save the DataPreprocessor object to a file using joblib.

        Parameters:
        - file_path (str): File path to save the object.

        Outputs:
        - None
        """
        if self.column_transformer:
            # Save the preprocessor object
            joblib.dump(self, file_path)
            print(f"DataPreprocessor object saved to {file_path}")
        else:
            print("fit_transform must be called before saving.")
    
    def load(self, file_path):
        """
        Load a DataPreprocessor object from a file.

        Parameters:
        - file_path (str): File path from which to load the object.

        Outputs:
        - None
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File '{file_path}' not found. Please provide a valid file path.")

        # Load the preprocessor object
        loaded_preprocessor = joblib.load(file_path)

        # Copy the loaded attributes to the current object
        self.__dict__.update(loaded_preprocessor.__dict__)

        print(f"DataPreprocessor object loaded from {file_path}")
