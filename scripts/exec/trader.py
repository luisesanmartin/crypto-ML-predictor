import time
import datetime
import pandas as pd
import pickle
import sys

sys.path.insert(1, '../utils')
import data_fetching_utils as dfu
import trading_utils as trading
import feature_engineering_utils as feu

# Parameters
model_design = 'brute-force'
model_type = 'xgb'
model_month = 'apr2021'
crypto = 'BTC'
amount = 10
time_delta = 1 # in hours
frequency = 30 # in minutes
frequency_dataset = 10 # in minutes

# Iterator variables
hold = 0
profits_total = 0

# Loading model
model_file = '../../classifiers/' + model_design + '/' + model_month + model_type + '.txt'
with open(model_file, 'rb') as f:
    model = pickle.load(f)

# Logging in
login = trading.login()

# DF cols
price_cols = [
    'price_close',
    'price_high',
    'price_low',
    'volume_traded',
    'trades_count'
]

# Parameters to standardize
mean_sd_file = '../../data/working/train/X/' + model_design + '/' + model_month + 'mean_sd.txt'
with open(mean_sd_file, 'rb') as f:
    mean_sd_list = pickle.load(f)

while True:

    print('\nStarting new iteration')
    print(datetime.datetime.now())

    # Data
    data = dfu.get_data_time_delta(crypto, time_delta=time_delta)
    df = feu.arrange_deployment_data(data, price_cols, frequency_dataset, time_delta)
    df_std = feu.standardize_df(df, stats=mean_sd_list).drop(columns=['time'])

    # Prediction
    prediction = model.predict(df_std)[0]

    # Trader
    if hold == 0:
        print('Currently not holding '+crypto)
        if prediction == 1:
            buy_order = trading.buy_crypto(crypto=crypto, usd_amount=amount)
            crypto_quantity = buy_order['quantity']
            amount_spent = float(crypto_quantity) * float(buy_order['price'])
            print('Sent an order to buy '+crypto_quantity+' for $'+str(round(amount_spent, 2)))
            hold = 1
        else: #prediction == 0
            print('Price is predicted to decrease, not buying')
            pass

    else: #hold == 1
        print('Currently holding '+crypto_quantity+' of '+crypto)
        if prediction == 0:
            sell_order = trading.sell_crypto(crypto=crypto, crypto_amount=float(crypto_quantity))
            amount_sold = float(crypto_quantity) * float(sell_order['price'])
            profits = amount_sold - amount_spent
            profits_total += profits
            print('Sent an order to sell '+crypto_quantity+' for $'+str(round(amount_sold, 2)))
            print('Profits with this operation: $'+str(round(profits, 2)))
            print('Total profits: $'+str(round(profits_total, 2)))
            hold = 0
        else:
            print('Price is predicted to increase, not selling')
            pass

    print('Now we go to sleep for '+str(frequency)+' more minutes...')
    time.sleep(frequency*60)
