"""
perception_model.py
Author: Shanmukha Rao Bodala
Description: Implementation of the Neural Perception Layer for CRM Forecasting.
This script builds a Feedforward Neural Network (FNN) to predict sales outcomes
using the IBM Watson Sales-Win-Loss dataset structure.
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error

class PerceptionModel:
    def __init__(self):
        self.model = None
        self.preprocessor = None

    def build_preprocessor(self, X):
        """
        Creates a column transformer for handling categorical and numerical CRM data.
        """
        # Feature categories based on common CRM schemas
        categorical_features = ['Supplies Subgroup', 'Region', 'Route To Market']
        numeric_features = ['Elapsed Days In Sales Stage', 'Opportunity Amount USD', 'Total Days Identified Through Closing']
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ])
        
        return self.preprocessor

    def create_model(self, input_shape):
        """
        Defines the 3-layer Feedforward Neural Network (FNN) architecture.
        """
        model = models.Sequential([
            layers.Dense(64, activation='relu', input_shape=(input_shape,), name="Perception_Input"),
            layers.Dropout(0.2), # Dropout layer to prevent overfitting
            layers.Dense(32, activation='relu', name="Hidden_Layer_1"),
            layers.Dense(16, activation='relu', name="Hidden_Layer_2"),
            layers.Dense(1, activation='sigmoid', name="Win_Probability_Output")
        ])
        
        model.compile(
            optimizer='adam', 
            loss='binary_crossentropy', 
            metrics=['mae']
        )
        self.model = model
        return model

    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """
        Trains the model with an early stopping callback to ensure optimal weights.
        """
        early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop],
            verbose=1
        )
        return history

    def evaluate_performance(self, X_test, y_test):
        """
        Generates metrics to compare against baseline models (Linear Regression/Random Forest).
        """
        predictions = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        
        print(f"\n--- Model Performance Report ---")
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        return mae, rmse

# Usage Example (can be moved to a main.py)
if __name__ == "__main__":
    # In a real scenario, you would load your Kaggle/IBM dataset here:
    # df = pd.read_csv('data/sales_data.csv')
    print("Perception Model initialized. Ready for training on CRM data.")