import pandas as pd
import sys
import pickle

sys.path.insert(1, '../utils')
import simulation_utils as sim
import feature_engineering_utils as feu

months = [
    'mar2021',
    'feb2021',
    'jan2021',
    'dec2020',
    'nov2020',
    'oct2020',
    'sep2020',
    'aug2020',
    'jul2020',
    'jun2020',
    'may2020'
]

y_path = '../../data/working/test/Y/'
predictions_path = '../../data/working/test/predictions/'
time_length = 30 # in minutes
amount = 100 # in dollars
total_profits = 0

data_file = '../../data/raw/maxdata_BTC_10min_2021-03-17.txt'
with open(data_file, 'rb') as f:
    data = pickle.load(f)
data_dic = feu.transform_data_to_dict(data)

for month in months:

    Y_real = pd.read_csv(y_path+month+'_Y.csv')
    Y_real = Y_real.set_index('time')
    Y_pred = pd.read_csv(predictions_path+month+'_predictions.csv')
    Y_pred = Y_pred.set_index('time')
    dates = list(Y_real.index)
    dates = sim.get_time_multiples(dates, time_length)

    hold = 0
    profits = 0

    for date in dates:

        prediction = Y_pred.loc[date]['predictions']
        price = data_dic[date]['price_close']

        if hold == 0:
            if prediction == 0:
                continue
            else:
                crypto = amount / price
                hold = 1

        else:
            if prediction == 1:
                continue
            else:
                sell_amount = crypto * price
                profits += sell_amount - amount
                hold = 0

    print('\nPeriod simulated:', month)
    print('Profits: $' + str(round(profits, 2)))
    print('Return: ' + str(round(profits/amount*100, 2)) + '%')
    total_profits += profits

print('\nTotal period simulated: ' + months[-1] + '-' + months[0])
print('Months: ' + str(len(months)))
print('Amount used: $' + str(amount))
print('Total profits: $' + str(round(total_profits, 2)))
