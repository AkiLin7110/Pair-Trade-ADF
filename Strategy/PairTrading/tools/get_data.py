# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 00:32:49 2023

@author: USER
"""

from binance import Client, ThreadedWebsocketManager, ThreadedDepthCacheManager
from binance.enums import HistoricalKlinesType 

import numpy as np 
import pandas as pd
import datetime
import time
import threading
from queue import Queue
import pickle


client = Client()

# symbol_list = ['SOLUSDT', 'MATICUSDT', 'DOTUSDT', 'LTCUSDT','ETHUSDT','BTCUSDT','DOGEUSDT', 'BNBUSDT','ADAUSDT']
def get_period_data(start_time, end_time , symbol_list = ['BTCUSDT', 'ETHUSDT'], freq = '1m' ):
    
    if end_time == None:
        end_time = datetime.datetime.now().strftime('%Y-%m-%d')
    
    if start_time == None:
        start_time = pd.to_datetime(end_time) - datetime.timedelta(days=7)
        start_time = start_time.strftime('%Y-%m-%d')
        
    def get_kline(client, symbol, freq ,time_start, time_end, q):
        try:
            klines = client.get_historical_klines( symbol, freq, time_start, time_end, klines_type=HistoricalKlinesType.FUTURES)
            q.put([symbol, klines])
        except  Exception as e:
            print(e)
        
    q = Queue()
    thread_list = list()
    for symbol in symbol_list:
        print(symbol)
        thread_1 = threading.Thread(
                                    target= get_kline , args = (client,
                                                                symbol, 
                                                                freq,
                                                                start_time, 
                                                                end_time,  
                                                                q)
                                    )
        thread_1.start()
        thread_list.append(thread_1)
    for thread_1 in thread_list:
        thread_1.join()

        
    return_list = list()
    for _ in range(q.qsize()):
        try:
            return_list.append(q.get(timeout = 2))
        except Exception as e: print(e)


    symbol_dict = dict()
    for see in return_list:
        try:
            if len(see[1]) == 0: continue
            symbol_dict[see[0]] = see[1]
        except Exception as e: print(e)    
    
    
    symbol_dict_adj = dict()
    for symbol, value in symbol_dict.items():
        print(symbol)
        try:
            columns_name = ['openTime', 'Open', 'High', 'Low', 'Close', 'Volume', 'closeTime', 'quoteVolume', 'numTrade', 'takerBuyVolume', 'takerBuyQuoteVolume', 'ignore']
            df_ = pd.DataFrame(value)
            df_.columns = columns_name
            df_['openTime']= pd.to_datetime(df_['openTime'], unit='ms')
            # df_['closeTime']= pd.to_datetime(df_['closeTime'], unit='ms')
            df_ = df_.drop(['ignore', 'closeTime'], axis=1)
            df_ = df_.sort_values('openTime', ascending=True)
            df_ = df_.set_index('openTime')
            df_ = df_.astype(float)
            df_['takerSellVolume'] = df_['Volume'] - df_['takerBuyVolume']
            df_['takerSellQuoteVolume'] = df_['quoteVolume'] - df_['takerBuyQuoteVolume']
            df_['avgTradeVolume'] = df_['quoteVolume'] / df_['numTrade']
            df_ = df_[~df_.index.duplicated(keep='first')]
            symbol_dict_adj[symbol] = df_
        except Exception as e:print(e)
    
    
    return symbol_dict_adj


def resample_symbol(df_symbol, rule='1H'):

    df_ = pd.DataFrame()

    df_['Open'] = df_symbol.resample(rule=rule, closed='left', label='left').first()['Open']
    df_['High'] = df_symbol.resample(rule=rule, closed='left', label='left').max()['High']
    df_['Low'] = df_symbol.resample(rule=rule, closed='left', label='left').min()['Low']
    df_['Close'] = df_symbol.resample(rule=rule, closed='left', label='left').last()['Close']

    summ = df_symbol.resample(rule=rule, closed='left', label='left').sum()

    df_['Volume'] = summ['Volume']
    df_['quoteVolume'] = summ['quoteVolume']
    # df_['numTrade'] = summ['numTrade']
    # df_['takerBuyVolume'] = summ['takerBuyVolume']
    # df_['takerBuyQuoteVolume'] = summ['takerBuyQuoteVolume']
    # df_['takerSellVolume'] = summ['takerSellVolume']
    # df_['takerSellQuoteVolume'] = summ['takerSellQuoteVolume']
    # df_['avgTradeVolume'] = df_['quoteVolume'] / df_['numTrade']

    return df_


if __name__ == '__main__' :
    symbol_list = ['BTCUSDT', 'ETHUSDT']
    start_time = '2023-07-22 11:00:00'
    end_time ='2023-07-23 01:00:00'
    freq = '1m'
    symbol_dict = dict()
    for symbol in symbol_list:
        print(symbol)
        klines = client.get_historical_klines(symbol, 
                                              freq, 
                                              start_time, 
                                              end_time,
                                              klines_type=HistoricalKlinesType.FUTURES)
        symbol_dict[symbol] = klines
        # print(klines[0])
    
    










