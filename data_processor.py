"""
data_processor.py
Author: Shanmukha Rao Bodala
Description: Data Engineering and Pipeline for CRM Datasets.
Handles loading, cleaning, and transformation of IBM Watson/Kaggle CRM data.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

class CRMDataProcessor:
    def __init__(self):
        self.preprocessor = None
        # Standard features from the IBM Watson Sales-Win-Loss dataset
        self.numeric_features = [
            'Elapsed Days In Sales Stage', 
            'Opportunity Amount USD', 
            'Total Days Identified Through Closing',
            'Ratio to Benchmark'
        ]
        self.categorical_features = [
            'Supplies Subgroup', 
            'Region', 
            'Route To Market', 
            'Opportunity Result' # This is our Target
        ]

    def load_and_clean(self, file_path):
        """
        Loads the IBM Watson dataset and performs initial cleaning.
        """
        try:
            df = pd.read_csv(file_path)
            # Remove redundant columns or identifiers
            cols_to_drop = ['Opportunity Number', 'Supplies Group']
            df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
            
            # Ensure target is binary
            if 'Opportunity Result' in df.columns:
                df['Target'] = df['Opportunity Result'].apply(lambda x: 1 if x == 'Won' else 0)
            
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None

    def get_pipeline(self):
        """
        Creates a Scikit-Learn Pipeline for reproducible preprocessing.
        """
        # Numerical Pipeline: Impute missing values then scale
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        # Categorical Pipeline: Impute missing then One-Hot Encode
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        # Combine into a ColumnTransformer
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_features),
                ('cat', categorical_transformer, [f for f in self.categorical_features if f != 'Opportunity Result'])
            ])
        
        return self.preprocessor

    def prepare_split(self, df):
        """
        Splits data into training, validation, and test sets.
        """
        X = df.drop(columns=['Target', 'Opportunity Result'])
        y = df['Target']

        # First split: Train vs Test/Val
        X_train_raw, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Second split: Validation vs Test
        X_val_raw, X_test_raw, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

        # Fit and transform the data
        pipeline = self.get_pipeline()
        X_train = pipeline.fit_transform(X_train_raw)
        X_val = pipeline.transform(X_val_raw)
        X_test = pipeline.transform(X_test_raw)

        return (X_train, X_val, X_test), (y_train, y_val, y_test)

if __name__ == "__main__":
    print("CRM Data Processor Module Loaded.")