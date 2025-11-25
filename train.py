import pandas as pd
import pickle
import matplotlib.pyplot as plt

def normalize_data(df):
    df_normalized = {}
    df_normalized['km_normalized'] = (df['km'] - df['km'].mean()) / df['km'].std()
    df_normalized['price_normalized'] = (df['price'] - df['price'].mean()) / df['price'].std()
    return df_normalized


def estimatePrice(km, theta0, theta1):
    return theta0 + theta1 * km


def gradientDescent(df, theta0, theta1, lr):
    """
    Performs gradient descent on the cost function.

    Calculates the partial derivatives:
    ∂J/∂θ₀ = (1/m) Σ(h(xᵢ) - yᵢ)
    ∂J/∂θ₁ = (1/m) Σ(h(xᵢ) - yᵢ) * xᵢ
    
    And updates parameters:
    θ₀ := θ₀ - α * ∂J/∂θ₀
    θ₁ := θ₁ - α * ∂J/∂θ₁
    """
    m = len(df)
    
    # Calculate predictions for all training examples
    predictions = df['km_normalized'].apply(lambda x: estimatePrice(x, theta0, theta1))
    
    # The Cost function
    errors = predictions - df['price_normalized']
    
    # Calculate partial derivatives (gradients)
    grad_theta0 = (1 / m) * errors.sum()
    grad_theta1 = (1 / m) * (errors * df['km_normalized']).sum()
    
    # Update parameters using gradient descent rule
    new_theta0 = theta0 - lr * grad_theta0
    new_theta1 = theta1 - lr * grad_theta1
    
    return new_theta0, new_theta1


def training(df, lr, iterations):
    """
    Trains the linear regression model using gradient descent.
    Iteratively applies gradient descent to find optimal parameters θ₀ and θ₁
    that minimize the cost function J(θ₀, θ₁).
    """

    theta0, theta1 = 0, 0
    
    for i in range(iterations):
        theta0, theta1 = gradientDescent(df, theta0, theta1, lr)
    
    return theta0, theta1


if __name__ == '__main__':
    df = pd.read_csv('data.csv')
    
    # Plot the data distribution before training
    plt.figure(figsize=(10, 6))
    plt.scatter(df['km'], df['price'], color='blue', alpha=0.6, edgecolors='k', label='Training Data')
    plt.xlabel('Kilometers (km)', fontsize=12)
    plt.ylabel('Price ($)', fontsize=12)
    plt.title('Car Price vs Mileage Distribution', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    data = {}
    data['km_mean'], data['km_std'], data['price_mean'], data['price_std'] = df['km'].mean(), df['km'].std(), df['price'].mean(), df['price'].std()
    df_normalized = normalize_data(df)

    data['theta0'], data['theta1'] = training(df_normalized, lr=0.02, iterations=100)


    # Convert normalized parameters back to original scale for plotting
    theta0_original = (data['theta0'] - data['theta1'] * data['km_mean'] / data['km_std']) * data['price_std'] + data['price_mean']
    theta1_original = data['theta1'] * data['price_std'] / data['km_std']
    
    # Plot the regression line
    km_min, km_max = df['km'].min(), df['km'].max()
    regression_line = theta0_original + theta1_original * pd.Series([km_min, km_max])
    plt.plot([km_min, km_max], regression_line, color='red', linewidth=2, label='Linear Regression Line')

    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig('plot.png', dpi=100, bbox_inches='tight')

    dbfile = open('dbfile.pkl', 'wb')
    pickle.dump(data, dbfile)
    dbfile.close()