from datetime import datetime, timedelta
import requests
import json
import pandas as pd

URL = 'https://rest.coinapi.io/'

def get_api_key(text='../data/key.txt'):

    with open(text) as file:
        key = file.read()

    return key

def time_bounds(gap=6):
    '''
    gap is measured in hours
    '''
    now = datetime.now()
    before = (now - timedelta(hours=gap))

    upper_bound = now.isoformat()[:-7]
    lower_bound = before.isoformat()[:-7]

    rv = {
        'upper bound': upper_bound,
        'lower bound': lower_bound
    }

    return rv

def get_data(crypto='BTC', period='10MIN', time_delta=6):

    '''
    time_delta is measured in hours
    '''

    times = time_bounds(gap=time_delta)

    request_url = URL + 'v1/ohlcv/' + crypto + '/USD/history?period_id=' + \
        period + '&time_start=' + times['lower bound'] + \
        '&time_end=' + times['upper bound'] + '&limit=100000'

    headers = {"X-CoinAPI-Key": get_api_key()}
    response = requests.get(request_url, headers=headers)

    data = json.loads(response.text)

    return data
