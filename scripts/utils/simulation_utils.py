import data_fetching_utils as dfu

def get_time_multiples(times, multiple=30):

    '''
    Receives a list of times (strings) in the format: '2020-09-01T00:00:00'
    and returns the same list only with the observations whose modulus
    for the minute part of the time and the parameter multiple is 0
    '''

    rv = []

    for time in times:

        time_dt = dfu.time_in_datetime(time)
        minutes = time_dt.minute

        if minutes % multiple == 0:
            rv.append(time)

    return rv
