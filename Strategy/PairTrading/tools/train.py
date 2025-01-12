# -*- coding: utf-8 -*-
"""
Created on Mon May  1 21:26:38 2023

@author: USER
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


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


from utils import *
from module.data import get_tidyData, DataPair, resample_symbol
from module.backtest import backtestingPair2, dictType_output
from module.visualize import Performance, PercentageReturnPlot


# raw_symbol_dict = dict()
# raw_symbol_dict['BTCUSDT'] = get_tidyData(symbol='BTCUSDT', data_type='ufutures')
# raw_symbol_dict['ETHUSDT'] =  get_tidyData(symbol='ETHUSDT', data_type='ufutures')
# raw_symbol_dict['DOGEUSDT'] =  get_tidyData(symbol='DOGEUSDT', data_type='ufutures')
# raw_symbol_dict['BNBUSDT'] = get_tidyData(symbol='BNBUSDT', data_type='ufutures')
# raw_symbol_dict['ADAUSDT'] = get_tidyData(symbol='ADAUSDT', data_type='ufutures')


# raw_symbol_dict['SOLUSDT'] = get_tidyData(symbol='SOLUSDT', data_type='ufutures')
# raw_symbol_dict['MATICUSDT'] = get_tidyData(symbol='MATICUSDT', data_type='ufutures')
# raw_symbol_dict['DOTUSDT'] = get_tidyData(symbol='DOTUSDT', data_type='ufutures')
# raw_symbol_dict['LTCUSDT'] = get_tidyData(symbol='LTCUSDT', data_type='ufutures')



# with open(r'G:\我的雲端硬碟\PAIR_TRADE_BICENTIVE\raw_symbol_dict.pickle', 'wb') as f:
#     pickle.dump(raw_symbol_dict, f)
with open(r'I:\我的雲端硬碟\PAIR_TRADE_BICENTIVE\raw_symbol_dict.pickle', 'rb') as f:
    raw_symbol_dict = pickle.load(f)


Asssets_Pool =['SOLUSDT', 'MATICUSDT', 'DOTUSDT', 'LTCUSDT',
               'ETHUSDT','BTCUSDT','DOGEUSDT', 'BNBUSDT','ADAUSDT', ]
comb = list(combinations(Asssets_Pool, 2))



max_trade = 12*24
rolling_window = int(12*24*4)
fund = 100
mode = 'COINT_OLS'  #[COINT_OLS, COINT_TLS, VAR_HIS_MODEL, COINT_OLS_ADJ, VAR_HIS_MODEL_ADJ, None]
time_list = pd.date_range(start = '2023-06-01', end = '2023-06-19' ,freq= '1D').strftime('%Y-%m-%d' )


pair_dict = dict()
for pair in comb:break
    data = DataPair(df_symbolA = raw_symbol_dict[pair[0]], df_symbolB=raw_symbol_dict[pair[1]], 
                       rule='5min',)
    day_dict = dict()
    for idd in trange(0, len(time_list)-10, desc='total pair'):
        
        train_start = time_list[idd]
        train_end =   time_list[idd + 6]
        backtest_end_ = time_list[idd + 7]
        backtest_end = time_list[idd + 8]
        # print(backtest_end_)
        data.startTime = backtest_end_
        data.endTime = backtest_end
        data.idx = data.df_pair.loc[ data.startTime : data.endTime ].index
        
        pre_process = PRE_PROCESS(train_start = train_start,
                                  train_end = train_end,
                                  backtest_end = backtest_end,
                                  rolling_window = rolling_window,
                                  pair = pair,
                                  df_A = data.dfA[train_start:backtest_end],
                                  df_B = data.dfB[train_start:backtest_end],)
        static_dict = pre_process.static_func()
        dynamic_df = pre_process.dynamic_func()
        
        
        data.static_dict = static_dict
        df_pair = pd.concat([data.df_pair[train_start:backtest_end], dynamic_df], axis = 1, join = 'inner')
        df_pair['ATR_big'] = df_pair[['B_ATR', 'A_ATR']].max(axis = 1)
        
        '''
        codition
        '''
        if mode == 'COINT_OLS':
            args = COINT_OLS(df_pair, upper_std = 3, lower_std = 1.5, speed = 0.8)
            beta, constant = pre_process.reg_ols
            
        elif mode == 'COINT_TLS':
            args = COINT_TLS(df_pair, upper_std = 3, lower_std = 1.5, speed = 0.8)
            beta, constant = pre_process.reg_tls
            
        
        if beta == None or beta > 0 :pass  #把小於0的刪掉
        else:continue
            
        entryLong, entrySellShort, exitShort, exitBuyToCover = args
        entryLong.iloc[-max_trade:] = False
        entrySellShort.iloc[-max_trade:] = False
        
        if sum(entryLong) == 0 and  sum(entrySellShort) == 0:
            continue
        
        '''
        上面都在生成資料，下面跑回測
        '''
        
        
        data.type_setting(entryLong, entrySellShort, exitShort, exitBuyToCover)
        output_dict = dictType_output(backtestingPair2(data.input_arr, takerFee=0.0005,slippage=0,
                                                      exit_timeOut=True, exParam1 = max_trade, #代表最多可以跑幾期
                                                      exit_profitOut=True, exParam2=0.02, 
                                                      fund=fund,
                                                      exit_lossOut=True, lossOut_condition=1, exParam3=0.015, 
                                                      stopLoss_slippageAdd=0.0001,
                                                      A_beta = beta
                                                      ))
               
        record_df, df_pair = MERGE_RECORD(df_pair, output_dict, data.idx,  pair, static_dict,  backtest_end_ , backtest_end)   
        signal_df = pd.DataFrame([entryLong, exitShort, entrySellShort, exitBuyToCover], 
                                 index = ['entryLong', 'exitShort', 'entrySellShort', 'exitBuyToCover']).T
        # df_pair = pd.concat([df_pair,signal_df],axis = 1)
        day_dict[backtest_end_] = [record_df]
        
    pair_dict[pair] = day_dict



    with open(r'I:\我的雲端硬碟\PAIR_TRADE_BICENTIVE\BACKTEST_RECORD\7-4_3-1.5_%s.pickle'%mode, 'wb') as f:
        pickle.dump(pair_dict, f)

# with open(r'C:\Users\USER\Desktop\7-4_3-1.5_%s.pickle'%mode, 'wb') as f:
#     pickle.dump(pair_dict, f)



'''
評價
#這些都是關艙的條件
if exitBuyToCover_arr[i]:
    order_type.append(3)
    flag = True
    
elif  i == len(openA_arr)-2:
    order_type.append(4)
    flag = True
    
elif stopProfit:
    order_type.append(5)
    flag = True
    
elif stopLoss:
    order_type.append(6)
    flag = True
    
elif stopTime:
    order_type.append(7)
'''


# 開啟檔案
# mode = 'VAR_HIS_MODEL_ADJ'  #有加濾網組
# mode = None  #對照組，沒加濾網
mode = 'COINT_OLS'  #對照組，沒加濾網
with open(r'I:\我的雲端硬碟\PAIR_TRADE_BICENTIVE\BACKTEST_RECORD\全年度的\7-4_3-1.5_%s.pickle'%mode, 'rb') as f:
    pair_dict = pickle.load(f)

with open(r'C:\Users\USER\Desktop\7-4_3-1.5_%s.pickle'%mode, 'rb') as f:
    pair_dict = pickle.load(f)



# raw_btc = resample_symbol(raw_symbol_dict['BTCUSDT'], rule='5min')
# market_atr = CAL_ATR(raw_btc, name = "market").shift(-1)
ll = []
for v in pair_dict.values():
    for p in v.values():
        pp = p[0].iloc[:]
        pp.index = pp['進場時間']
        # pp = pd.concat([pp, market_atr], axis = 1, join = 'inner')
        ll.append(pp)
record_df = pd.concat(ll)




record_df['diff_ewm_in_abs'] = record_df['diff_ewm_OLS_in'].abs()
record_df['ewm_lag1_in_abs'] = record_df['ewm_lag1_OLS_in'].abs()
record_df['adj_profit_fee_realized'] = record_df['profit_fee_realized']
record_df.loc[(record_df['profit_fee_realized']< -1.8), 'adj_profit_fee_realized'] = -1.8  #停損壓回-1.8%
# record_df.loc[(record_df['profit_fee_realized']> 2), 'adj_profit_fee_realized'] = 1.8
record_df['ATR_big'] = record_df[['B_ATR_in', 'A_ATR_in']].max(axis = 1)
record_df['ATR_diff'] = abs(record_df['B_ATR_in'] -  record_df['A_ATR_in'])
record_df['pair'] = record_df[['A','B']].apply(lambda x: tuple(x), axis=1)




'''
報酬率分配圖
'''
fig = plt.figure(figsize = (4,6) , dpi=300) 
ax2 = fig.add_subplot(2,1,2)  # 創建子圖2
record_df['profit_fee_realized'].hist(bins=100,alpha = 0.5,ax = ax2)
record_df['profit_fee_realized'].plot(kind = 'kde',ax=ax2)
ax2.set_ylabel('NUMBER')
plt.grid()
plt.axvline(x = 0, color = 'black', linestyle = '-', alpha = 0.3)
plt.show()





'''
TRXUSDT 和 XRPUSDT 績效不太好拿掉
record_df2是拿掉  TRXUSDT 和 XRPUSDT後
'''
ll = []
for pair, v in record_df.groupby(by = 'pair' , axis = 0):
    if 'TRXUSDT' in pair or 'XRPUSDT' in pair :continue
    else:
        ll.append(v)
record_df2 =  pd.concat(ll)





'''
回歸檢定
'''
plt.scatter(record_df['profit_fee_realized'] , record_df['ATR_big'].abs())
x_reg = sm.add_constant(record_df[[
        # 'pvalue2',   
        # 'corr',
        # 'ssd',
        # 'ewm_lag1_in_abs',
        'diff_ewm_in_abs',
        # 'market_ATR',
        # 'ATR_diff',
        # 'B_ATR_in',
        # 'ATR_big',
        ]])    
y = record_df['adj_profit_fee_realized']
results1 = sm.OLS(y,x_reg).fit()
print(results1.summary())








'''
參數高原
'''
# import plotly.io as pio
# import plotly.graph_objs as go
# pio.renderers.default='browser'

# x = record_df2['ATR_big']
# y = record_df2['diff_ewm_in_abs']
# z = record_df2['adj_profit_fee_realized']
# trace = go.Scatter3d(
#    x = x, y = y, z = z,mode = 'markers', marker = dict(
#       size = 12,
#       color = z, # set color to an array/list of desired values
#       colorscale = 'Viridis'
#       )
#    )

# layout = go.Layout(title = '3D Scatter plot')
# fig = go.Figure(data = [trace], layout = layout)
# fig.update_layout(scene = dict(
#                     xaxis_title='ATR_big',
#                     yaxis_title='diff_ewm_in_abs',
#                     zaxis_title='adj_profit_fee_realized'),
#                     # width=700,
#                     margin=dict(r=20, b=10, l=10, t=10)
#                     )
# fig.write_html(file = '3d_plot.html' , auto_open  = False)#圖片儲存並自動展開





'''
看績效
'''

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





'''
交易表現好的組合

1. 看過去5個月每組的表現，選總報酬大於0的前10組組合(若沒有到10組，舊址交易那些，N組)
2. 選過去5個月的交易績效，計算 "ATR_big"的第75%的數字，當作下個月的參數
3. 下個月最多只會開N組，rolling一個月去做交易。
'''

time_list = pd.date_range(start = '2022-05', end = '2023-08' ,freq= '1M').strftime('%Y-%m' )
ll = []
for i in range(len(time_list[6:])):
    
    train_start = time_list[i]
    train_end = time_list[i+ 5]
    test_end = time_list[i+6]
    # break
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
        (tmp2['ATR_big'] > tmp['ATR_big'].quantile(0.75, interpolation='midpoint')) &
        # (tmp2['ATR_big'] > 0.005) &
        # (tmp['market_ATR'] > tmp['market_ATR'].quantile(0.5, interpolation='midpoint')) & 
        # (tmp2['ATR_diff'] > tmp['ATR_diff'].quantile(0.7, interpolation='midpoint'))&
        # (tmp2['diff_ewm_in_abs'] > tmp['diff_ewm_in_abs'].quantile(0.75, interpolation='midpoint'))&
        # (record_df['market_ATR'] > record_df['market_ATR'].quantile(0.8, interpolation='midpoint'))&
        (tmp2['pvalue2'] < 1) 
        ]
    tmp2 = tmp2.sort_index()
    ll.append(tmp2)
    
tmp = pd.concat(ll)
cumret_ll = []
for df in ll:
    len_pair = len(set(df['pair']))
    cumret_ll.append(df[['profit_fee_realized', 'adj_profit_fee_realized','profit_realized']]/len_pair)
cumret_df = pd.concat(cumret_ll)

ll = pd.concat([v.sum() for i, v in cumret_df.groupby(by=cumret_df.index)], axis = 1)
ll.columns = [i for i, v in cumret_df.groupby(by=cumret_df.index)]
cumret_df = ll.T
(cumret_df/100).cumsum().plot()
see_perform(tmp)
(tmp['進場位置'] -  tmp['出場位置']).hist(bins=30)
tmp['period'] = tmp['出場位置'] -  tmp['進場位置']






x_reg = sm.add_constant(tmp[[
        # 'pvalue2',   
        # 'corr',
        # 'ssd',
        # 'ewm_lag1_in_abs',
        'diff_ewm_in_abs',
        # 'market_ATR',
        # 'ATR_diff',
        # 'B_ATR_in',
        # 'ATR_big',
        ]])    
y = tmp['adj_profit_fee_realized']
results1 = sm.OLS(y,x_reg).fit()
print(results1.summary())


