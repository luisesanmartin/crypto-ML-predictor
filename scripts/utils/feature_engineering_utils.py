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

def subset_for_testing(data_dic, start, freq=10, time_range=360):

    '''
    receives the data dict and start time and returns a set
    of observations within the time range specified (forward-looking)
    and in the frequencies defined
    freq and time_range are defined in minutes
    '''

    start_td = data_fetching_utils.time_in_datetime(start)
    end_td = start_td + timedelta(minutes=time_range)

    current = start_td
    data_return = []

    while current <= end_td:

        current_str = data_fetching_utils.time_in_string(current)

        if current_str in data_dic:
            data_return.append(data_dic[current_str])
        else:
            # if one of the times is not in the dic,
            # we return a None
            return None

        current = current + timedelta(minutes=freq)

    return data_return

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

def test_set_brute_force(
    data_dic,
    start,
    time_range_obs=30,
    time_range_test=360,
    obs_freq=10,
    prediction_freq=30
    ):

    '''
    time_range_obs is in days
    time_range_test is in minutes
    obs_freq is in minutes
    prediction_freq is in minutes
    '''

    # Time range for the observations
    start_td = data_fetching_utils.time_in_datetime(start)
    end_td = start_td + timedelta(days=time_range_obs)
    end = data_fetching_utils.time_in_string(end_td)
    n_obs = data_fetching_utils.calculate_observations(start, end, obs_freq)

    # Time range for the train X cols
    end_obs_td = start_td + timedelta(minutes=time_range_test)
    end_obs = data_fetching_utils.time_in_string(end_obs_td)
    n_train = data_fetching_utils.calculate_observations(start, end_obs, obs_freq)

    # Cols of the df
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
    current = start_td
    while current <= end_td:

        current_str = data_fetching_utils.time_in_string(current)
        subset = subset_for_testing(
            data_dic,
            current_str,
            obs_freq,
            time_range_test
        )
        label = get_label(
            data_dic,
            current_str,
            prediction_freq
        )

        i_X = None
        if subset: # only append if label is not None
            i_X = len(df_X)
            row_X = filter_subset(subset, price_cols)
            df_X.loc[i_X] = row_X

        if label is not None: # only append if label is not None
            i_Y = len(df_Y)
            row_Y = [current_str, label]
            df_Y.loc[i_Y] = row_Y

        current = current + timedelta(minutes=obs_freq)

        if i_X:
            if i_X % 500 == 0:
                print('Progress: ' + str(round(i_X/n_obs*100)) + '%')

    return df_X, df_Y

def train_set_brute_force(
    data_dic,
    end,
    time_range_obs=360,
    time_range_train=360,
    obs_freq=10,
    prediction_freq=30
    ):

    '''
    time_range_obs is in days
    time_range_train is in minutes
    obs_freq is in minutes
    prediction_freq is in minutes
    '''

    # Time range for the observations
    end_td = data_fetching_utils.time_in_datetime(end)
    start_td = end_td - timedelta(days=time_range_obs)
    start = data_fetching_utils.time_in_string(start_td)
    n_obs = data_fetching_utils.calculate_observations(start, end, obs_freq)

    # Time range for the train X cols
    start_obs_td = end_td - timedelta(minutes=time_range_train)
    start_obs = data_fetching_utils.time_in_string(start_obs_td)
    n_train = data_fetching_utils.calculate_observations(start_obs, end, obs_freq)

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
            obs_freq,
            time_range_train
        )
        label = get_label(
            data_dic,
            current_str,
            prediction_freq
        )

        i_X = len(df_X)
        row_X = filter_subset(subset, price_cols)
        df_X.loc[i_X] = row_X

        if label is not None: # only append if label is not None
            i_Y = len(df_Y)
            row_Y = [current_str, label]
            df_Y.loc[i_Y] = row_Y

        current = current - timedelta(minutes=obs_freq)

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

def standardize_df(df, stats=None, stats_out=False):

    '''
    Standardizes every column but the time column
    stats is a list of tuples with the mean and sd to be used for every column
    if stats_out is True, returns a list of tuples with the mean and sd
    '''

    df_sd = pd.DataFrame()
    df_sd['time'] = df['time']
    i = 0

    if stats_out:
        mean_sd_list = []

    for col in df:
        if col == 'time':
            continue
        else:
            if stats:
                df_sd[col] = pd.to_numeric(standardize(df[col], stats[i]))
            else:
                if stats_out:
                    standardized_results = standardize(df[col], stats_out=True)
                    df_sd[col] = standardized_results[0]
                    mean_sd_list.append(standardized_results[1])
                else:
                    df_sd[col] = standardize(df[col])
            i += 1

    if stats_out:
        return df_sd, mean_sd_list
    else:
        return df_sd

def standardize(col, stats=None, stats_out=False):

    '''
    standardizes a column if stats is not provided.
    stats is a tuple or list with the mean and sd to be used if provided
    if stats_out is True, returns a tuple with the mean and sd
    '''

    if stats:
        mean, sd = stats
    else:
        mean = col.mean()
        sd = col.std()

    rv = (col - mean) / sd

    if stats_out:
        return rv, (mean, sd)
    else:
        return rv

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

def arrange_deployment_data(data, price_cols, freq, time_delta):

    '''
    arranges data for deployment
    data: data as we get it from CoinAPI
    price_cols: list of column names that will contain our data
    freq: frequency of observations, in minutes
    time_delta: time we're retrieving data from, in hours
    '''

    # Number of times the price_cols will be repeated
    end = data[-1]['time_period_end'][:19]
    start = data[0]['time_period_end'][:19]
    n_col_iterations = data_fetching_utils.calculate_observations(start, end, freq)

    # Data in reversed order, so most recent obs are first
    data_rev = data[::-1]
    relevant_data = filter_subset(data_rev, price_cols)

    # Cols
    cols = ['time']
    for i in range(n_col_iterations):
        cols += [col+str(i+1) for col in price_cols]

    # Dataframe
    df = pd.DataFrame(columns = cols)
    df.loc[0] = relevant_data
    # Note: this is not standardized

    return df
