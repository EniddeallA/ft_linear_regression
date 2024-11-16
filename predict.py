import pickle
import sys

def estimatePrice(km, theta0, theta1, km_mean, km_std, price_mean, price_std):
    km_normalized = (km - km_mean) / km_std
    price_normalized = theta0 + theta1 * km_normalized
    return price_normalized * price_std + price_mean

def main():
    if len(sys.argv) != 2 or not sys.argv[1].isdigit():
        print('Usage: python predict_price.py <km>')
        sys.exit(1)
        
    dbfile = open('dbfile.pkl', 'rb')
    data = pickle.load(dbfile)
    
    test_km = int(sys.argv[1])
    predicted_price = estimatePrice(test_km, data['theta0'], data['theta1'], data['km_mean'], data['km_std'], data['price_mean'], data['price_std'])
    
    print(f'Predicted price for {test_km}km: {predicted_price:.2f}')
    
    dbfile.close()

if __name__ == '__main__':
    main()