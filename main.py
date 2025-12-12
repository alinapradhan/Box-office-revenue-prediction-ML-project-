"""
Box Office Revenue Prediction ML Project

This script demonstrates a complete machine learning pipeline for predicting
box office revenue using various algorithms including scikit-learn, LightGBM, and XGBoost.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import lightgbm as lgb
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')


def create_sample_dataset():
    """
    Create a sample movie dataset for demonstration purposes.
    In a real scenario, this would be replaced with Kaggle dataset loading.
    """
    np.random.seed(42)
    n_samples = 1000
    
    # Generate synthetic movie data
    data = {
        'budget': np.random.uniform(1e6, 3e8, n_samples),
        'runtime': np.random.uniform(80, 180, n_samples),
        'vote_average': np.random.uniform(4.0, 9.0, n_samples),
        'vote_count': np.random.uniform(10, 5000, n_samples),
        'popularity': np.random.uniform(1, 100, n_samples),
        'genre_action': np.random.randint(0, 2, n_samples),
        'genre_comedy': np.random.randint(0, 2, n_samples),
        'genre_drama': np.random.randint(0, 2, n_samples),
        'release_month': np.random.randint(1, 13, n_samples),
        'has_sequel': np.random.randint(0, 2, n_samples),
    }
    
    # Create revenue based on features with some noise
    revenue = (
        data['budget'] * 2.5 +
        data['vote_average'] * 1e7 +
        data['popularity'] * 1e6 +
        data['vote_count'] * 5000 +
        data['genre_action'] * 5e7 +
        np.random.normal(0, 5e7, n_samples)
    )
    revenue = np.maximum(revenue, 0)  # Ensure non-negative
    
    data['revenue'] = revenue
    
    return pd.DataFrame(data)


def load_data(filepath=None):
    """
    Load movie dataset from CSV file.
    If no file is provided, create a sample dataset.
    
    Args:
        filepath (str): Path to CSV file with movie data
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    if filepath:
        try:
            df = pd.read_csv(filepath)
            print(f"Loaded data from {filepath}")
            return df
        except FileNotFoundError:
            print(f"File {filepath} not found. Creating sample dataset.")
            return create_sample_dataset()
    else:
        print("No filepath provided. Creating sample dataset.")
        return create_sample_dataset()


def preprocess_data(df, target_column='revenue'):
    """
    Preprocess the movie data for machine learning.
    
    Args:
        df (pd.DataFrame): Raw movie data
        target_column (str): Name of the target variable column
        
    Returns:
        tuple: X_train, X_test, y_train, y_test, scaler
    """
    print("\n=== Data Preprocessing ===")
    
    # Drop rows with missing target values
    df = df.dropna(subset=[target_column])
    
    # Separate features and target
    y = df[target_column]
    X = df.drop(columns=[target_column])
    
    # Handle missing values in features
    # Fill numeric columns with median
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
    
    # For categorical columns, use label encoding if present
    # Fill categorical NaN with 'Unknown' before encoding
    encoders = {}
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = X[col].fillna('Unknown')
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le
    
    print(f"Dataset shape: {X.shape}")
    print(f"Features: {list(X.columns)}")
    print(f"Target variable: {target_column}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def train_linear_regression(X_train, y_train):
    """Train Linear Regression model."""
    print("\n=== Training Linear Regression ===")
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train):
    """Train Random Forest model."""
    print("\n=== Training Random Forest ===")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def train_lightgbm(X_train, y_train):
    """Train LightGBM model."""
    print("\n=== Training LightGBM ===")
    model = lgb.LGBMRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=7,
        random_state=42,
        verbose=-1
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train):
    """Train XGBoost model."""
    print("\n=== Training XGBoost ===")
    model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=7,
        random_state=42,
        verbosity=0
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate a trained model and print metrics.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target values
        model_name (str): Name of the model
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
    y_pred = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n{model_name} Results:")
    print(f"  RMSE: ${rmse:,.2f}")
    print(f"  MAE:  ${mae:,.2f}")
    print(f"  R²:   {r2:.4f}")
    
    return {
        'model_name': model_name,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }


def compare_models(results):
    """
    Compare results from different models.
    
    Args:
        results (list): List of result dictionaries from evaluate_model
    """
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('rmse')
    
    print("\nRanked by RMSE (lower is better):")
    print(results_df.to_string(index=False))
    
    best_model = results_df.iloc[0]
    print(f"\n🏆 Best Model: {best_model['model_name']}")
    print(f"   RMSE: ${best_model['rmse']:,.2f}")
    print(f"   R²:   {best_model['r2']:.4f}")


def predict_revenue(model, scaler, features_dict):
    """
    Predict revenue for a new movie.
    
    Args:
        model: Trained model
        scaler: Fitted StandardScaler
        features_dict (dict): Dictionary of feature values
        
    Returns:
        float: Predicted revenue
    """
    features_df = pd.DataFrame([features_dict])
    features_scaled = scaler.transform(features_df)
    prediction = model.predict(features_scaled)[0]
    return max(0, prediction)  # Ensure non-negative


def main():
    """Main execution function."""
    print("="*60)
    print("BOX OFFICE REVENUE PREDICTION ML PROJECT")
    print("="*60)
    
    # Load data
    # To use a Kaggle dataset, download it and provide the path:
    # df = load_data('path/to/your/kaggle/dataset.csv')
    df = load_data()
    
    # Preprocess data
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
    
    # Train multiple models
    models = {}
    models['Linear Regression'] = train_linear_regression(X_train, y_train)
    models['Random Forest'] = train_random_forest(X_train, y_train)
    models['LightGBM'] = train_lightgbm(X_train, y_train)
    models['XGBoost'] = train_xgboost(X_train, y_train)
    
    # Evaluate all models
    results = []
    for name, model in models.items():
        result = evaluate_model(model, X_test, y_test, name)
        results.append(result)
    
    # Compare models
    compare_models(results)
    
    # Example prediction with the best model (based on lowest RMSE)
    print("\n" + "="*60)
    print("EXAMPLE PREDICTION")
    print("="*60)
    
    # Select the best model based on RMSE
    best_result = min(results, key=lambda x: x['rmse'])
    best_model = models[best_result['model_name']]
    
    example_movie = {
        'budget': 150000000,
        'runtime': 120,
        'vote_average': 7.5,
        'vote_count': 2000,
        'popularity': 50.0,
        'genre_action': 1,
        'genre_comedy': 0,
        'genre_drama': 0,
        'release_month': 6,
        'has_sequel': 0
    }
    
    predicted_revenue = predict_revenue(best_model, scaler, example_movie)
    print(f"\nMovie Features:")
    for key, value in example_movie.items():
        print(f"  {key}: {value}")
    print(f"\nPredicted Box Office Revenue: ${predicted_revenue:,.2f}")
    
    print("\n" + "="*60)
    print("DONE!")
    print("="*60)


if __name__ == "__main__":
    main()
