from datetime import datetime, timedelta
import requests
import json
import pandas as pd

URL = 'https://rest.coinapi.io/'
FMT = '%Y-%m-%dT%H:%M:%S'

def get_api_key(text='../data/key.txt'):

    with open(text) as file:
        key = file.read()

    return key

def time_in_datetime(time):

    '''
    Transforms a time string to datetime
    '''

    return datetime.strptime(time, FMT)

def time_in_string(time):

    '''
    Transforms a datetime to a time string
    '''

    return time.isoformat()[:-7]

def time_bounds(gap=6):
    '''
    gap is measured in hours
    '''
    now = datetime.now()
    before = (now - timedelta(hours=gap))

    upper_bound = time_in_string(now)
    lower_bound = time_in_string(now)

    rv = {
        'upper bound': upper_bound,
        'lower bound': lower_bound
    }

    return rv

def get_data_time_delta(crypto='BTC', period='10MIN', time_delta=6):

    '''
    time_delta is measured in hours
    '''

    times = time_bounds(gap=time_delta)
    data = get_data(crypto, period, times['lower_bound'], times['upper_bound'])

    return data

def get_data_max_possible(crypto='BTC', period=10, end=time_in_string(datetime.now())):

    '''
    get the max observations allowed by the API in a single call (100k), given
    the end time
    period is the frequency of the data, must be in minutes
    '''

    minutes_delta = period * 99990
    end_dt = time_in_datetime(end)
    start_dt = end_dt - timedelta(minutes=minutes_delta)
    start = time_in_string(start_dt)
    frequency = str(period) + 'MIN'

    data = get_date(crypto, frequency, start, end)

    return data


def get_data(crypto, period, start, end):

    '''
    '''

    request_url = URL + 'v1/ohlcv/' + crypto + '/USD/history?period_id=' + \
        period + '&time_start=' + start + \
        '&time_end=' + end + '&limit=100000'

    headers = {"X-CoinAPI-Key": get_api_key()}
    response = requests.get(request_url, headers=headers)

    data = json.loads(response.text)

    return data

def calculate_observations(start, end, frequency):

    '''
    '''

    obs = None
    start_dt = time_in_datetime(start)
    end_dt = time_in_datetime(end)

    if 'MIN' in frequency:

        freq_minutes = int(frequency.split('M')[0])
        minutes = (end_dt - start_dt).total_seconds() / 60
        return int((minutes / freq_minutes)) + 1

    return obs
