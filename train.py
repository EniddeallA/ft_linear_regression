import pandas as pd
import pickle

def normalize_data(df):
    df_normalized = df.copy()
    df_normalized['km_normalized'] = (df['km'] - df['km'].mean()) / df['km'].std()
    df_normalized['price_normalized'] = (df['price'] - df['price'].mean()) / df['price'].std()
    return df_normalized, df['km'].mean(), df['km'].std(), df['price'].mean(), df['price'].std()

def estimatePrice(km, theta0, theta1):
    return theta0 + theta1 * km

def gradientDescent(df, theta0, theta1, lr):
    m = len(df)
    tmp_theta0 = 0
    tmp_theta1 = 0
    
    for i in range(m):
        hypothesis = estimatePrice(df['km_normalized'].iloc[i], theta0, theta1) - df['price_normalized'].iloc[i]
        tmp_theta0 += hypothesis
        tmp_theta1 += hypothesis * df['km_normalized'].iloc[i]
    
    new_theta0 = theta0 - (lr * tmp_theta0) / m
    new_theta1 = theta1 - (lr * tmp_theta1) / m
    
    return new_theta0, new_theta1

def training(df, lr, iterations):
    theta0, theta1 = 0, 0    
    for i in range(iterations):
        theta0, theta1 = gradientDescent(df, theta0, theta1, lr)    
    return theta0, theta1

if __name__ == '__main__':
    df = pd.read_csv('data.csv')
    
    data = {}
    
    df_normalized, data['km_mean'], data['km_std'], data['price_mean'], data['price_std'] = normalize_data(df)
    data['theta0'], data['theta1'] = training(df_normalized, lr=0.1, iterations=1000)

    dbfile = open('dbfile.pkl', 'wb')
    pickle.dump(data, dbfile)
    dbfile.close()