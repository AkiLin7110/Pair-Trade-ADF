# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 17:35:11 2023

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


def select_trading_record(record_df2 , atr_q = 0.75):
    time_list = pd.date_range(start = '2022-05', end = '2023-09' ,freq= '1M').strftime('%Y-%m' )
    ll = []
    for i in range(len(time_list[6:])):
        train_start = time_list[i]
        train_end = time_list[i+ 5]
        test_end = time_list[i+6]
    
        tmp = record_df2[train_start : train_end]
        mom = pd.DataFrame([[i , v['adj_profit_fee_realized'].sum()] for i, v in tmp.groupby(by = 'pair' , axis = 0)])
        need_pair = mom.loc[mom[1] > -0.5].sort_values([1], ascending = False )[0].to_list()
        tmp2 = record_df2.loc[test_end]
        mom_ll = []
        for pair, v in tmp2.groupby(by = 'pair' , axis = 0):
            if pair in need_pair[:10]:
                mom_ll.append(v)
        tmp2 =  pd.concat(mom_ll)
        tmp2 = tmp2[
            (tmp2['ATR_big'] > tmp['ATR_big'].quantile(atr_q, interpolation='midpoint')) &
            (tmp2['pvalue2'] < 1) 
            ]
        tmp2 = tmp2.sort_index()
        ll.append(tmp2)
    return pd.concat(ll).sort_index()



mode_list = ['model1',
              'model2',
                'model3',
                'model4'
             ]
path = os.getcwd()

mode_dict = dict()
for mode_name in mode_list:
    record_df = pd.read_pickle(path + "\\record\\backtest2_pair_%s.pickle"%mode_name)
    record_df.index = record_df['進場時間']
    record_df.insert(0, 'mode', mode_name)
    # record_df['mode'] = mode_name
    # record_df['ID'] = range(len(record_df))
    # record_df['diff_ewm_in_abs'] = record_df['diff_ewm_TLS_in'].abs()
    # record_df['ewm_lag1_in_abs'] = record_df['ewm_lag1_TLS_in'].abs()
    record_df['adj_profit_fee_realized'] = record_df['profit_fee_realized']
    record_df.loc[(record_df['profit_fee_realized']< -1.65), 'adj_profit_fee_realized'] = -1.8  #停損壓回-1.8%
    # record_df.loc[(record_df['profit_fee_realized']> 2), 'adj_profit_fee_realized'] = 1.8
    record_df['ATR_big'] = record_df[['B_ATR_in', 'A_ATR_in']].max(axis = 1)
    record_df['ATR_diff'] = abs(record_df['B_ATR_in'] -  record_df['A_ATR_in'])
    record_df['pair'] = record_df[['A','B']].apply(lambda x: tuple(x), axis=1)
    record_df2 = record_df.iloc[:].copy()
    if '3' in mode_name or '4' in  mode_name:
        atr_q = 0.8
    else: atr_q = 0.75
    mode_dict[mode_name] = select_trading_record(record_df2, atr_q)



mode_df = pd.concat([*mode_dict.values()])
mode_df['ID'] = range(len(mode_df))
mode_dict = dict(tuple(mode_df.groupby(by = mode_df.index)))




'''
冰塊盒投資組合:
1. 任何時間最多持有 N組(N = 10)
2. 同一個時間pair最多只能出現K組(K = 2)
'''

from collections import Counter
save_list = []
exit_list = []
pair_list = []
n_posisition = []
max_portfolio  = 10
max_samePair  = 3



def trade_orNot( trade1):
    if len(pair_list) >=  max_portfolio :return
    n_pair = Counter(pair_list)[trade1['pair']]
    if n_pair >=max_samePair:return
    exit_list.append(trade1['出場時間'])
    pair_list.append(trade1['pair'])
    save_list.append(trade1)



def get_element_positions(lst, element):
    return [index for index, value in enumerate(lst) if value < element]

def remove_elements_at_positions(lst, positions):
    return [x for i, x in enumerate(lst) if i not in positions]



for day, trade in mode_dict.items():
    # print(day, len(pair_list))
    n_posisition.append([day, len(pair_list)])
    trade = trade.sort_values(by = 'mode')
    trade.index = range(len(trade))
    
    #要不要出場
    exits_position= get_element_positions(exit_list, day)
    if len(exits_position) != 0:
        pair_list = remove_elements_at_positions(pair_list, exits_position)
        exit_list = remove_elements_at_positions(exit_list, exits_position)
    
    #要不要做交易
    for idx in trade.index: 
       trade1 =  trade.loc[idx]
       trade_orNot(trade1) 

    


def see_perform(tmp):
    winRate = sum(tmp['profit_fee_realized'] > 0)/ len(tmp['profit_fee_realized'])
    winRate = np.round(winRate*100,2)
    mean_ret = tmp['profit_fee_realized'].mean()
    mean_ret_adj = tmp['adj_profit_fee_realized'].mean()
    trade_number = len(tmp['adj_profit_fee_realized'])
    
    print(f'winRate: {winRate}%')
    print(f'Mean_Return: {mean_ret}%')
    print(f'Mean_Return_ADJ: {mean_ret_adj}%')
    print(f'Trade_Number: { trade_number}')


df = pd.DataFrame(n_posisition)
df.set_index(0, inplace = True)
df.columns = ['n_position']


leverage = 10
tmp = pd.concat(save_list, axis = 1).T
tmp.index = tmp['進場時間']
cumret_df = tmp[['profit_fee_realized', 'adj_profit_fee_realized','profit_realized']]/max_portfolio
cumret_df = ((leverage * cumret_df)/100).cumsum()
see_perform(tmp)


# 累績報酬
fig, (ax1, ax2) = plt.subplots(2, 1, 
                               sharex=True,  #共用X軸
                               figsize=(14,10),
                               gridspec_kw={'height_ratios': [3, 1]} #上下比例
                               )
ax1.plot(cumret_df.index, cumret_df.iloc[:,:3]*100)
ax1.set_ylabel('cumret (%)')
ax1.legend(cumret_df.columns.to_list()[:3], loc='upper left')

ax2.plot(df.index, df['n_position'])
ax2.set_xlabel('time')
ax2.set_ylabel('n_position')
plt.subplots_adjust(hspace=0.05) #上下距離
plt.show()



# 報酬分布
fig = plt.figure(figsize = (5,10) , dpi=300) 
ax2 = fig.add_subplot(2,1,2)  # 創建子圖2
tmp['adj_profit_fee_realized'].hist(bins=30,alpha = 0.4,ax = ax2)
tmp['adj_profit_fee_realized'].plot(kind = 'kde',ax=ax2)
ax2.set_ylabel('NUMBER')
plt.title('Return Distribution')
plt.axvline(x = 0, color = 'black', linestyle = '-', alpha = 0.3)
plt.show()



# sp_strats = (tmp[['profit_fee_realized', 'adj_profit_fee_realized','profit_realized']]/max_portfolio).resample(rule = '1d').sum()/100 
# df_metrics = pd.DataFrame()
# for col in sp_strats.columns:
#     df_metrics[col] = qs.reports.metrics(sp_strats[col], mode='full',display=False, periods_per_year= 365)






