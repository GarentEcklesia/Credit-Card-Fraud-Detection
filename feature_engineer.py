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