import pandas as pd


def estimatePrice(mileage, theta0, theta1):
    return theta0 + (theta1 * mileage)

def train_data(file):
    data = pd.read_csv(file)
    mileage = data[0]
    price = data[1]
    learningRate = 0.01
    m = data.length()
    tetha0 = 0
    tetha1 = 0
    for i in range(m):
        tetha0 += learningRate * (1/m) * (estimatePrice(mileage[i]) - price[i])
        tetha1 += learningRate * (1/m) * (estimatePrice(mileage[i]) - price[i]) * mileage[i]
        


if __name__ == '__main__':
    train_data("data.csv")
    

