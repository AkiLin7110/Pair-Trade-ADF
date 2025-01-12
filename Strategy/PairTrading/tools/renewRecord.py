# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 17:10:41 2023

@author: david
"""



import os
import time
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import pickle
from tqdm import trange
import re
from itertools import product
from itertools import combinations
import quantstats as qs


from get_data import get_period_data
from utils import *
from utils import PAIR_TRADE_MODEL


# 獲取之前的交易紀錄
mode_name  = 'model1'
mode_list = ['model1','model2','model3','model4']

for mode_name in mode_list:
    print(mode_name)
    record_df = pd.read_pickle("record\\backtest2_pair_%s.pickle"%mode_name)
    record_df.index = record_df['回測開始']
    Asssets_Pool =['SOLUSDT', 'MATICUSDT', 'DOTUSDT', 'LTCUSDT',
                   'ETHUSDT','BTCUSDT','DOGEUSDT', 'BNBUSDT','ADAUSDT', ]
    comb = list(combinations(Asssets_Pool, 2))
    
    
    
    start_time_ = sorted(record_df['回測開始'])[-1]
    start_time = (pd.to_datetime(start_time_) - dt.timedelta(days=8)).strftime('%Y-%m-%d')
    end_time = dt.datetime.now().strftime('%Y-%m-%d')
    raw_symbol_dict = get_period_data(start_time, end_time , Asssets_Pool) #用畢安的API下載歷史資料
    
    
    # 用alex的套件下載資料，但要先去那邊更新到最新的
    # os.chdir(r"I:\我的雲端硬碟\PAIR_TRADE_BICENTIVE\Trading-Universe-main")
    # print(os.getcwd())
    # from module.data import get_tidyData, DataPair, resample_symbol
    # raw_symbol_dict = dict()
    # for symbol in Asssets_Pool:
    #     print(symbol)
    #     raw_symbol_dict[symbol] = get_tidyData(symbol = symbol, data_type='ufutures', start = '2023-06-01' , end = '2023-07-10')
    # with open(r'C:\Users\USER\Desktop\raw_symbol_dict.pickle', "wb") as f:
    #     pickle.dump(raw_symbol_dict, f)
    
    
    
    
    
    print('start_time :', start_time)
    print('end_time :', end_time)
    root_model = dict()
    root_model['model1'] = {'mode' : 'COINT_OLS',
                              'ATR_q' : 0,
                              'speed' : 0.8}
    
    root_model['model2'] = {'mode' : 'COINT_TLS',
                              'ATR_q' : 0,
                              'speed' : 0.8}
    
    root_model['model3'] = {'mode' : 'COINT_OLS',
                              'ATR_q' : 0,
                              'speed' : None}
    
    root_model['model4'] = {'mode' : 'COINT_TLS',
                              'ATR_q' : 0,
                              'speed' : None}
    
    

    
    print(mode_name) #['model1', 'model2', 'model3','model4']
    mode = root_model[mode_name]['mode']
    ATR_q = root_model[mode_name]['ATR_q']
    speed = root_model[mode_name]['speed']
    new_record_list = []
    for pair in comb:
        print(pair)
        df_symbolA = raw_symbol_dict[pair[0]]
        df_symbolB = raw_symbol_dict[pair[1]]
        model = PAIR_TRADE_MODEL(mode,ATR_q,speed , pair)
        model.get_time_list(start = start_time , end = end_time)
        model.load_data(df_symbolA , df_symbolB, rule = '5min')
        day_dict = model.trade_pair()
        if len(day_dict.values()) == 0 : continue
        df = pd.concat([*day_dict.values()])
        new_record_list.append(df)    
    
    
    new_record_df = pd.concat(new_record_list)
    new_record_df.index = new_record_df['回測開始']
    last_time = sorted(record_df['回測開始'])[-1]
    new_record_df = new_record_df.loc[new_record_df['回測開始'] > last_time] #只更新最新的
    new_record_df_ = pd.concat([record_df, new_record_df]) #更新結束
    new_record_df_.to_pickle("record\\backtest2_pair_%s.pickle"%mode_name)
    











# #  看績效
# mode_name = 'model1'
# record_df = pd.read_pickle("record\\backtest2_pair_%s.pickle"%mode_name)
# record_df.index = record_df['進場時間']
# record_df['ID'] = range(len(record_df))
# # record_df['diff_ewm_in_abs'] = record_df['diff_ewm_TLS_in'].abs()
# # record_df['ewm_lag1_in_abs'] = record_df['ewm_lag1_TLS_in'].abs()
# record_df['adj_profit_fee_realized'] = record_df['profit_fee_realized']
# record_df.loc[(record_df['profit_fee_realized']< -1.8), 'adj_profit_fee_realized'] = -1.8  #停損壓回-1.8%
# # record_df.loc[(record_df['profit_fee_realized']> 2), 'adj_profit_fee_realized'] = 1.8
# record_df['ATR_big'] = record_df[['B_ATR_in', 'A_ATR_in']].max(axis = 1)
# record_df['ATR_diff'] = abs(record_df['B_ATR_in'] -  record_df['A_ATR_in'])
# record_df['pair'] = record_df[['A','B']].apply(lambda x: tuple(x), axis=1)
# ll = []
# for pair, v in record_df.groupby(by = 'pair' , axis = 0):
#     if 'TRXUSDT' in pair or 'XRPUSDT' in pair :continue
#     else:
#         ll.append(v)
# record_df2 =  pd.concat(ll)



# def see_perform(tmp):
#     winRate = sum(tmp['profit_fee_realized'] > 0)/ len(tmp['profit_fee_realized'])
#     winRate = np.round(winRate*100,2)
#     mean_ret = tmp['profit_fee_realized'].mean()
#     mean_ret_adj = tmp['adj_profit_fee_realized'].mean()
#     trade_number = len(tmp['adj_profit_fee_realized'])
    
#     print(f'winRate: {winRate}%')
#     print(f'Mean_Return: {mean_ret}%')
#     print(f'Mean_Return_ADJ: {mean_ret_adj}%')
#     print(f'Trade_Number: { trade_number}')



# # 交易表現好的組合
# # 1. 看過去6個月每組的表現，選總報酬大於0的前10組組合(若沒有到10組，舊址交易那些，N組)
# # 2. 選過去6個月的交易績效，計算 "ATR_big"的第75%的數字，當作下個月的參數
# # 3. 下個月最多只會開N組，rolling一個月去做交易。


# time_list = pd.date_range(start = '2022-05', end = '2023-09' ,freq= '1M').strftime('%Y-%m' )
# ll = []
# for i in range(len(time_list[6:])):
    
#     train_start = time_list[i]
#     train_end = time_list[i+ 5]
#     test_end = time_list[i+6]
#     # break
#     tmp = record_df2[train_start : train_end]
#     mom = pd.DataFrame([[i , v['adj_profit_fee_realized'].sum()] for i, v in tmp.groupby(by = 'pair' , axis = 0)])
#     need_pair = mom.loc[mom[1] > -0.5].sort_values([1], ascending = False )[0].to_list()


#     tmp2 = record_df2.loc[test_end]
#     mom_ll = []
#     for pair, v in tmp2.groupby(by = 'pair' , axis = 0):
#         if pair in need_pair[:10]:
#             mom_ll.append(v)

#     tmp2 =  pd.concat(mom_ll)
    
    
#     tmp2 = tmp2[
#         (tmp2['ATR_big'] > tmp['ATR_big'].quantile(0.75, interpolation='midpoint')) 
#         # (tmp2['ATR_big'] > 0.005) &
#         # (tmp['market_ATR'] > tmp['market_ATR'].quantile(0.5, interpolation='midpoint')) & 
#         # (tmp2['ATR_diff'] > tmp['ATR_diff'].quantile(0.7, interpolation='midpoint'))&
#         # (tmp2['diff_ewm_in_abs'] > tmp['diff_ewm_in_abs'].quantile(0.75, interpolation='midpoint'))&
#         # (record_df['market_ATR'] > record_df['market_ATR'].quantile(0.8, interpolation='midpoint'))&
#         # (tmp2['pvalue2'] < 1) 
#         ]
#     tmp2 = tmp2.sort_index()
#     ll.append(tmp2)
    
# tmp = pd.concat(ll)
# cumret_ll = []
# for df in ll:
#     len_pair = 10
#     cumret_ll.append(df[['profit_fee_realized', 'adj_profit_fee_realized','profit_realized']]/len_pair)
# cumret_df = pd.concat(cumret_ll)

# ll = pd.concat([v.sum() for i, v in cumret_df.groupby(by=cumret_df.index)], axis = 1)
# ll.columns = [i for i, v in cumret_df.groupby(by=cumret_df.index)]
# cumret_df = ll.T
# (cumret_df/100).cumsum().plot()


