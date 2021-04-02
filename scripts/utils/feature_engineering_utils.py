from datetime import timedelta
import data_fetching_utils
import pandas as pd

def transform_data_to_dict(data):

    '''
    transforms a list of data retrieved from coinapi to a dict
    where the keys are the end of each time period
    '''

    data_dic = {}

    for obs in data:
        time = obs['time_period_end'].split('.')[0]
        data_dic[time] = obs

    return data_dic

def subset_for_training(data_dic, end, freq=10, time_range=360):

    '''
    receives the data dict and an end time and returns a set
    of observations within the time range specified (backwards)
    and in the frequencies defined
    freq and time_range are defined in minutes
    '''

    end_td = data_fetching_utils.time_in_datetime(end)
    start_td = end_td - timedelta(minutes=time_range)

    current = end_td
    data_return = []

    while current >= start_td:

        current_str = data_fetching_utils.time_in_string(current)

        if current_str in data_dic:
            data_return.append(data_dic[current_str])

        else: # we take the data of the most recent previous obs
            print('\nWarning: Data for obs ' + current_str + ' not found')
            not_in_dic = True
            freq2 = 0 + freq
            while not_in_dic:
                current2 = current - timedelta(minutes=freq2)
                current_str2 = data_fetching_utils.time_in_string(current2)
                if current_str2 in data_dic:
                    print('Using data from ' + current_str2 + ' instead')
                    data_return.append(data_dic[current_str2])
                    not_in_dic = False
                else:
                    freq2 += freq

        current = current - timedelta(minutes=freq)

    return data_return

def filter_subset(subset, cols):

    '''
    '''

    start = subset[0]['time_period_end'].split('.')[0]
    data_return = [start]

    for obs in subset:

        data_return += [obs[attr] for attr in cols]

    return data_return

def initial_train_X_brute_force(
    data_dic,
    end,
    time_range_obs=360,
    time_range_train=360,
    freq=10
    ):

    '''
    time_range_obs is in days
    time_range_train is in minutes
    freq is in minutes
    '''

    # Time range for the observations
    end_td = data_fetching_utils.time_in_datetime(end)
    start_td = end_td - timedelta(days=time_range_obs)
    start = data_fetching_utils.time_in_string(start_td)
    n_obs = data_fetching_utils.calculate_observations(start, end, freq)

    # Time range for the train X cols
    start_obs_td = end_td - timedelta(minutes=time_range_train)
    start_obs = data_fetching_utils.time_in_string(start_obs_td)
    n_train = data_fetching_utils.calculate_observations(start_obs, end, freq)

    # Train cols of the df
    cols = ['time']
    price_cols = [
        'price_close',
        'price_high',
        'price_low',
        'volume_traded',
        'trades_count'
    ]
    for i in range(n_train):
        cols += [col+str(i+1) for col in price_cols]
    df_X = pd.DataFrame(columns = cols)
    df_Y = pd.DataFrame(columns = ['time', 'label'])

    # Building up the df
    print('\nStarting to build up the df...')
    print('Expected obs:', n_obs, '\n')
    current = end_td
    while current >= start_td:

        current_str = data_fetching_utils.time_in_string(current)
        subset = subset_for_training(
            data_dic,
            current_str,
            freq,
            time_range_train
        )
        label = get_label(
            data_dic,
            current_str
        )

        i_X = len(df_X)
        row_X = filter_subset(subset, price_cols)
        df_X.loc[i_X] = row_X

        if label: # only append if label is not None
            i_Y = len(df_Y)
            row_Y = [current_str, label]
            df_Y.loc[i_Y] = row_Y

        current = current - timedelta(minutes=freq)

        if i_X % 500 == 0:
            print('Progress: ' + str(round(i_X/n_obs*100)) + '%')

    return df_X, df_Y

def get_label(data_dic, time, time_range=30):

    '''
    returns a binary value indicating if the price went up
    after the time specified in time_range
    time_range is in minutes
    '''

    time_td = data_fetching_utils.time_in_datetime(time)
    after_td = time_td + timedelta(minutes=time_range)
    after = data_fetching_utils.time_in_string(after_td)

    if time in data_dic and after in data_dic:
        initial_price = data_dic[time]['price_close']
        final_price = data_dic[after]['price_close']

        if final_price > initial_price:
            return 1

        else:
            return 0

    else:
        return None

def standardize(df, stats=False):

    '''
    Standardizes every column but the first, which is assumed to be "date"
    If stats=True, returns the mean and std as well.
    '''

    df_sd = pd.DataFrame()
    df_sd['time'] = df['time']

    for col in df:
        if col == 'time':
            continue
        else:
            mean = df[col].mean()
            sd  = df[col].std()
            df_sd[col] = (df[col] - mean) / sd

    if stats:
        return df_sd, (mean, sd)

    else:
        return df_sd

def match_dates(df_X, df_Y):

    '''
    compares X and Y datasets and excludes the dates (obs)
    that are not included in both df
    '''

    dates_X = df_X['time']
    dates_Y = df_Y['time']

    df_X_matched = df_X.merge(df_Y['time'], how='inner', on='time')
    df_Y_matched = df_Y.merge(df_X['time'], how='inner', on='time')

    return df_X_matched, df_Y_matched
