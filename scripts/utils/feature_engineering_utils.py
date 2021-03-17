from datetime import datetime, timedelta
import data_fetching_utils

def transform_data_to_dict(data):

    '''
    '''

    dic = {}

    for obs in data:
        time = obs['time_period_end'].split('.')[0]
        dic[time] = obs

    return dic

def arrange_dataset_obs(data_dic, time, previous_time):

    '''
    Gets all the observations from the last
    '''

    return None

#minutes = time_dt.minute
#seconds = time_dt.second
