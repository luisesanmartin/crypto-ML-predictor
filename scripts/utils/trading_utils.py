import robin_stocks
import robin_stocks.robinhood as rh
import json

def login():

    '''
    Logs in
    '''

    with open('../../data/credentials.json') as f:
        login_info = json.load(f)

    login = rh.login(login_info['username'], login_info['password'])

    return login

def buy_crypto(crypto='BTC', usd_amount=10):

    '''
    Buys the specified amount of dollars of crypto
    '''

    order_details = rh.order_buy_crypto_by_price(crypto, usd_amount)

    return order_details

def sell_crypto(crypto='BTC', crypto_amount=0.000001):

    '''
    Sells the specified amount of crypto at the current price
    '''

    order_details =rh.order_sell_crypto_by_quantity(crypto, crypto_amount)

    return order_details
