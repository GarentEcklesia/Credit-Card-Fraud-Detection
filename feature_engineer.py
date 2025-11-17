from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom transformer to create engineered features.
    """
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        
        # Transaction velocity
        X_copy['transaction_velocity'] = (
            X_copy['distance_from_last_transaction'] / 
            (X_copy['distance_from_last_transaction'].mean() + 1)
        )
        
        # Home distance risk
        X_copy['home_distance_risk'] = pd.cut(
            X_copy['distance_from_home'], 
            bins=[0, 50, 100, 200, float('inf')],
            labels=[0, 1, 2, 3]
        ).astype(int)

        # Purchase anomaly
        X_copy['purchase_anomaly'] = (
            X_copy['ratio_to_median_purchase_price'] > 3
        ).astype(int)
        
        # Security level
        X_copy['security_level'] = (
            X_copy['used_chip'] + X_copy['used_pin_number']
        )
        
        # Interaction features
        X_copy['online_no_chip'] = (
            (X_copy['online_order'] == 1) & (X_copy['used_chip'] == 0)
        ).astype(int)
        
        X_copy['high_amount_far_home'] = (
            (X_copy['ratio_to_median_purchase_price'] > 2) & 
            (X_copy['distance_from_home'] > 100)
        ).astype(int)
        

        return X_copy

class OutlierHandler(BaseEstimator, TransformerMixin):
    """
    Handle outliers using IQR method.
    """
    def __init__(self, method='iqr', threshold=1.5):
        self.method = method
        self.threshold = threshold
        self.bounds = {}
    
    def fit(self, X, y=None):
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - self.threshold * IQR
            upper_bound = Q3 + self.threshold * IQR
            self.bounds[col] = (lower_bound, upper_bound)
        
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        
        for col, (lower, upper) in self.bounds.items():
            X_copy[col] = X_copy[col].clip(lower=lower, upper=upper)
        
        return X_copy


