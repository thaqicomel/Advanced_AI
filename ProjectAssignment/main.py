import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import json
from scipy import stats

def create_output_directory():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f'real_estate_analysis_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
    return output_dir

def save_dataframe_info(df, output_dir, prefix=''):
    info_file = os.path.join(output_dir, f'{prefix}dataset_info.txt')
    with open(info_file, 'w') as f:
        # Basic info
        f.write('Dataset Information:\n\n')
        df.info(buf=f)
        
        # Statistics
        f.write('\n\nDescriptive Statistics:\n')
        f.write(df.describe().to_string())
        
        # Missing values
        f.write('\n\nMissing Values:\n')
        f.write(df.isnull().sum().to_string())
        
        # Data types
        f.write('\n\nData Types:\n')
        f.write(df.dtypes.to_string())
        
        # Distribution statistics
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        if not numeric_cols.empty:
            f.write('\n\nSkewness:\n')
            f.write(df[numeric_cols].skew().to_string())
            f.write('\n\nKurtosis:\n')
            f.write(df[numeric_cols].kurtosis().to_string())

def handle_missing_values(df):
    """Handle missing values in the dataset"""
    for column in df.columns:
        if df[column].isnull().sum() > 0:
            if df[column].dtype in ['int64', 'float64']:
                df[column].fillna(df[column].median(), inplace=True)
            else:
                df[column].fillna(df[column].mode()[0], inplace=True)
    return df

def handle_outliers(df, columns=None):
    if columns is None:
        columns = df.select_dtypes(include=['float64', 'int64']).columns
    
    for col in columns:
        if col != 'Price':  
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    return df

def feature_engineering(df):
    df['House_Age'] = 2024 - df['Year_Built']
    df['Total_Rooms'] = df['Num_Bedrooms'] + df['Num_Bathrooms']
    df['Room_Density'] = df['Total_Rooms'] / df['Num_Floors']
    df['Amenities_Score'] = df['Has_Garden'] + df['Has_Pool'] + (df['Garage_Size'] > 0).astype(int)
    df['Location_Value_Ratio'] = df['Location_Score'] / (df['Distance_to_Center'] + 1)
    return df

def load_and_preprocess_data(filepath, output_dir):
    """Load and preprocess the dataset"""
    # Read the CSV file
    print("Loading dataset...")
    df = pd.read_csv(filepath)
    
    # Save initial dataset information
    print("Saving initial dataset info...")
    save_dataframe_info(df, output_dir, 'initial_')
    
    # Handle missing values
    print("Handling missing values...")
    df = handle_missing_values(df)
    
    # Handle outliers
    print("Handling outliers...")
    df = handle_outliers(df)
    
    # Feature engineering
    print("Performing feature engineering...")
    df = feature_engineering(df)
    
    # Save processed dataset information
    print("Saving processed dataset info...")
    save_dataframe_info(df, output_dir, 'processed_')
    
    # Create correlation heatmap
    print("Creating correlation heatmap...")
    plt.figure(figsize=(12, 10))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plots', 'correlation_heatmap.png'))
    plt.close()
    
    # Drop ID column if exists
    if 'ID' in df.columns:
        df = df.drop('ID', axis=1)
    
    # Split features and target
    X = df.drop('Price', axis=1)
    y = df['Price']
    
    return X, y

def train_and_evaluate_models(X, y, output_dir):
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize models
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=1.0),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
    }
    
    results = {}
    predictions = {}
    
    # Train and evaluate each model
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train the model
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        train_pred = model.predict(X_train_scaled)
        test_pred = model.predict(X_test_scaled)
        predictions[name] = test_pred
        
        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        
        results[name] = {
            'Train RMSE': train_rmse,
            'Test RMSE': test_rmse,
            'Train R2': train_r2,
            'Test R2': test_r2
        }
        
        # Print results
        print(f"{name} Results:")
        print(f"Train RMSE: ${train_rmse:,.2f}")
        print(f"Test RMSE: ${test_rmse:,.2f}")
        print(f"Train R2: {train_r2:.4f}")
        print(f"Test R2: {test_r2:.4f}")
        
        # Save feature importance/coefficients
        if hasattr(model, 'feature_importances_'):
            importance = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            importance.to_csv(os.path.join(output_dir, f'{name.lower().replace(" ", "_")}_importance.csv'))
        elif hasattr(model, 'coef_'):
            coef = pd.DataFrame({
                'feature': X.columns,
                'coefficient': model.coef_
            }).sort_values('coefficient', ascending=False)
            coef.to_csv(os.path.join(output_dir, f'{name.lower().replace(" ", "_")}_coefficients.csv'))
    
    # Save results
    with open(os.path.join(output_dir, 'model_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    return models, predictions, results, X_test, y_test

def plot_results(y_test, predictions, output_dir):
    # Actual vs Predicted
    plt.figure(figsize=(10, 6))
    for name, pred in predictions.items():
        plt.scatter(y_test, pred, alpha=0.5, label=name)
    
    plt.plot([y_test.min(), y_test.max()], 
            [y_test.min(), y_test.max()], 
            'k--', lw=2, label='Perfect Prediction')
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title('Actual vs Predicted House Prices')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'plots', 'actual_vs_predicted.png'))
    plt.close()
    
    # Residual plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Residual Plots')
    axes = axes.ravel()
    
    for idx, (name, pred) in enumerate(predictions.items()):
        residuals = y_test - pred
        axes[idx].scatter(pred, residuals, alpha=0.5)
        axes[idx].axhline(y=0, color='r', linestyle='--')
        axes[idx].set_xlabel('Predicted Price')
        axes[idx].set_ylabel('Residuals')
        axes[idx].set_title(f'{name} Residuals')
        axes[idx].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plots', 'residual_plots.png'))
    plt.close()

def main():
    # Create output directory
    output_dir = create_output_directory()
    print(f"\nOutput will be saved to: {output_dir}")
    
    try:
        # Load and preprocess data
        X, y = load_and_preprocess_data('group/real_estate_dataset.csv', output_dir)
        
        # Train and evaluate models
        models, predictions, results, X_test, y_test = train_and_evaluate_models(X, y, output_dir)
        
        # Create plots
        print("\nGenerating plots...")
        plot_results(y_test, predictions, output_dir)
        
        print(f"\nAnalysis complete. Results have been saved to: {output_dir}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()