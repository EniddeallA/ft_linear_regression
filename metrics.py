import pickle
import pandas as pd
import numpy as np

def estimatePrice(km, theta0, theta1, km_mean, km_std, price_mean, price_std):
    """Estimate price for given km using trained parameters"""
    km_normalized = (km - km_mean) / km_std
    price_normalized = theta0 + theta1 * km_normalized
    return price_normalized * price_std + price_mean

def calculate_metrics(y_true, y_pred):
    """
    Calculate regression metrics to evaluate model precision.
    
    Metrics:
    - RMSE (Root Mean Squared Error): Square root of Average of squared differences
    - R² Score: Coefficient of determination (0 to 1, higher is better)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Mean Squared Error
    mse = np.mean((y_true - y_pred) ** 2)
    # Root Mean Squared Error
    rmse = np.sqrt(mse)
    
    # R² Score (Coefficient of Determination)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2_score = 1 - (ss_res / ss_tot)
    
    return {
        'RMSE': rmse,
        'R²': r2_score
    }

def display_metrics():
    """Load model and display precision metrics on all training data"""
    df = pd.read_csv('data.csv')
    
    dbfile = open('dbfile.pkl', 'rb')
    data = pickle.load(dbfile)
    dbfile.close()
    
    # Calculate predictions on all training data
    predictions = estimatePrice(df['km'], data['theta0'], data['theta1'], 
                               data['km_mean'], data['km_std'], 
                               data['price_mean'], data['price_std'])
    metrics = calculate_metrics(df['price'], predictions)
    
    # Calculate theoretical maximum R² based on data correlation
    correlation = df['km'].corr(df['price'])
    max_r2 = correlation ** 2
    
    print("\n" + "="*50)
    print("MODEL PRECISION METRICS")
    print("="*50)
    print(f"RMSE (Root Mean Squared Error): {metrics['RMSE']:.2f}")
    print(f"R² Score:                     {metrics['R²']:.4f}")
    print("="*50)
    if metrics['R²'] >= 0.8:
        print("✓ Excellent model fit!")
    elif metrics['R²'] >= 0.6:
        print("✓ Good model fit!")
    else:
        print("△ Room for improvement")
    print("="*50 + "\n")

if __name__ == '__main__':
    display_metrics()
