# -*- coding: utf-8 -*-
"""
Created on Mon May  1 21:47:55 2023

@author: USER
"""


import numpy as np
from statsmodels.tsa.stattools import  adfuller
import pandas as pd
import numpy as np 
import time
from scipy import stats   
from scipy.odr import Model as odr_model
from scipy.odr import RealData, ODR


from itertools import combinations
import copy
from tqdm import trange
import re
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

pathOrin = os.getcwd()
pathNew = pathOrin.split('\\Strategy')[0]
os.chdir(pathNew)
print(pathNew)
from Strategy.PairTrading.tools.module.data import get_tidyData, DataPair, resample_symbol
from Strategy.PairTrading.tools.module.backtest import backtestingPair2, dictType_output
from Strategy.PairTrading.tools.module.visualize import Performance, PercentageReturnPlot
os.chdir(pathOrin)


'''
to load the function of pair trade
PRE_PROCESS
'''
class PRE_PROCESS():
    def __init__(self,train_start : str,
                     train_end: str,
                     backtest_end : str, 
                     rolling_window : int,
                     pair : tuple,
                     df_A : pd.DataFrame,  
                     df_B :  pd.DataFrame
                     ):

        self.train_start = train_start
        self.train_end = train_end
        self.backtest_end = backtest_end
        self.rolling_window = rolling_window
        self.pair = pair
        self.df_A = df_A.loc[train_start : backtest_end]
        self.df_B = df_B.loc[train_start : backtest_end]
        self.pair_df = pd.DataFrame([self.df_A.loc[train_start : backtest_end, 'Close'] 
                                  ,self.df_B.loc[train_start : backtest_end, 'Close']], 
                                    index = ['A','B']).T
        self.residual = None
        
        
    def static_func(self)-> dict:
        args =  self.pair_df, self.train_start, self.train_end, self.backtest_end        
        pvalue_A, pvalue_B = CAL_ADF1(args)
        pvalue2, self.residual, self.reg_ols = CAL_ADF2(args)   
        pvalue3,  self.residual_TLS, self.reg_tls = CAL_TLS(args)   
        
        args2 = self.residual,  self.train_start, self.train_end
        hurst = CAL_HURST(args2, max_lag=20)
        corr = CAL_CORR(args)
        ssd = CAL_SSD(args)
        
        static_dict = dict()
        static_dict['pvalue_A'] = pvalue_A
        static_dict['pvalue_B'] = pvalue_B
        static_dict['pvalue2'] = pvalue2
        # static_dict['pvalue3'] = pvalue3
        static_dict['hurst'] = hurst
        static_dict['corr'] = corr[0]
        static_dict['ssd'] = ssd
        return static_dict
        
        
    def dynamic_func(self)-> pd.DataFrame:
        n_std = self.residual.rolling(self.rolling_window).std()
        n_std_ = DIVERSE_SPEED(self.residual, self.rolling_window , alpha = 0.3, mode = 'OLS')
        n_std_tls = DIVERSE_SPEED(self.residual_TLS, self.rolling_window , alpha = 0.3, mode = 'TLS')
        
        atr_A = CAL_ATR(self.df_A, name = "A")
        atr_B = CAL_ATR(self.df_B, name = "B")
        self.residual.name = 'residual_OLS'
        self.residual_TLS.name = 'residual_TLS'        
        
        var = CAL_VAR(self.df_A, self.df_B)
        dynamic_df = pd.concat([n_std_, n_std_tls, atr_A, atr_B, self.residual, self.residual_TLS ,var], axis = 1)
        return dynamic_df
    



        
'''
STATIC FUNC
'''
def CAL_ADF1(args)-> float:
    pair_df, train_start, train_end, backtest_end = args
    pvalue_A = adfuller(pair_df.loc[: train_end, 'A'])[1]
    pvalue_B = adfuller(pair_df.loc[: train_end, 'B'])[1]
    return pvalue_A, pvalue_B


# B = constant + A*beta + reidual
def CAL_ADF2(args)-> float:
    pair_df, train_start, train_end, backtest_end = args
    regre_para = np.polyfit(pair_df.loc[train_start :train_end, 'A'], pair_df.loc[train_start :train_end,'B'], 1) #[beta1 , alpha]
         
    # if results[0]  < 0: continue #對沖比例，小於0
    predict = np.poly1d(regre_para)
    residual = pair_df['B'] - predict(pair_df['A'])    #找出訓練期間的殘差                
    pvalue2 = adfuller(residual.loc[train_start : train_end] , regression="nc")[1]   # engle 推薦使用截距模型作檢定較佳    , regression="nc" 
    return pvalue2,  residual, regre_para #[beta1 , alpha]



def CAL_HURST(args2, max_lag=20)-> float:
    '''
    路人寫的另一種算 hurst的比較簡單的方法，個人覺得在樣本比較大的時候這個好像比較好
    https://towardsdatascience.com/introduction-to-the-hurst-exponent-with-code-in-python-4da0414ca52e

    '''
    """Returns the Hurst Exponent of the time series"""
    
    residual, train_start, train_end = args2
    time_series = residual[train_start :train_end].values #只能放訓練期的阿
    lags = range(2, max_lag)
    tau = [np.std(np.subtract(time_series[lag:], time_series[:-lag])) for lag in lags] # variances of the lagged differences
    reg = np.polyfit(np.log(lags), np.log(tau), 1)# calculate the slope of the log plot -> the Hurst Exponent
    return reg[0]



def CAL_CORR(args)-> float:
    pair_df, train_start, train_end, backtest_end = args
    pair_ret_df = pair_df.loc[train_start :train_end].pct_change().dropna()
    cor = stats.spearmanr(pair_ret_df) #斯皮爾曼等級相關係數(母體分配未知) p-value < 0.05，H0: ρ = 0，不成立，X、Y有關。
    return cor



def CAL_SSD(args)-> float:
    pair_df, train_start, train_end, backtest_end = args
    
    def normalize(x):
        mu = x.mean()
        std = x.std()
        nor = (x - mu)/std
        return nor
    
    A_nor = normalize( pair_df.loc[train_start : train_end, 'A'])
    B_nor = normalize( pair_df.loc[train_start : train_end, 'B'])
    ssd = np.power(A_nor - B_nor, 2).sum()
    return ssd





'''
DYNAMIC FUNC
'''
def DIVERSE_SPEED(residual, rolling_window , alpha = 0.3, mode = 'OLS')-> pd.DataFrame:
    n_std = (residual - residual.rolling(rolling_window).mean())/residual.rolling(rolling_window).std()  #紀錄現在是N被標準差
    n_std_ = n_std.iloc[:].copy()
    n_std_.name = 'n_std'
    n_std_ = n_std_.to_frame()
    n_std_['ewm'] = n_std_.ewm(alpha = 0.3).mean()  #作天的EWM
    n_std_['ewm_lag1'] = n_std_['ewm'].shift(1)  #作天的EWM
    n_std_['diff_ewm'] = n_std_['n_std'] -  n_std_['ewm_lag1'] 

    cols = n_std_.columns.to_list()
    for col in cols:
        n_std_.rename(columns = {col : col+'_%s'%mode} , inplace = True)
    return n_std_




def CAL_ATR(df_A, name = "A")-> pd.Series:
    df_A_ = df_A.iloc[:]
    df_A_['Close_lag1'] = df_A_['Close'].shift(1)
    df_A_['H-L'] = abs(df_A_['High'] - df_A_['Low'])/df_A_['Close']
    df_A_['H-PC'] = abs(df_A_['High'] - df_A_['Close_lag1'])/df_A_['Close']
    df_A_['L-PC'] = abs(df_A_['Low'] - df_A_['Close_lag1'])/df_A_['Close']
    df_A_['TR'] = df_A_[['H-L', 'H-PC', 'L-PC']].max(axis =1, skipna =False)
    df_A_['%s_ATR'%name] = df_A_['TR'].ewm(alpha = 0.3).mean()   
    return df_A_['%s_ATR'%name]




def CAL_VAR(df_A, df_B, upper = .99, lower = .50, window = 12*24*4): #value
    db = (df_B['Close'].pct_change() - df_A['Close'].pct_change() )
    df = pd.DataFrame()
    df['spread'] = db
    df['upper_in'] = db.rolling(window).quantile(upper, interpolation='midpoint')
    df['lower_in'] = db.rolling(window).quantile(1-upper, interpolation='midpoint')
    
    df['upper_out'] = db.rolling(window).quantile(lower, interpolation='midpoint')
    df['lower_out'] = db.rolling(window).quantile(1-lower, interpolation='midpoint')
    return df




def CAL_TLS(args):
    pair_df, train_start, train_end, backtest_end = args
    X = pair_df.loc[train_start :train_end, 'A']
    y = pair_df.loc[train_start :train_end, 'B']
    
    def linear_func(p, x):
        m, c = p
        return m*x + c
    linear_model = odr_model(linear_func)

    data = RealData(X, y)
    odr = ODR(data, linear_model, beta0=[0., 1.])
    out = odr.run()
    m, c = out.beta

    residual_TLS = pair_df['B'] - (m * pair_df['A'] + c)
    pvalue3 = adfuller(residual_TLS.loc[train_start : train_end] , regression="nc")[1]
    regre_para = [m, c]
    return pvalue3, residual_TLS, regre_para




'''
SELECT MODEL
'''
def COINT_OLS(df_pair, upper_std = 3, lower_std = 1.5, speed = 0.7):
    if type(speed) == float:
        entryLong = (df_pair['n_std_OLS'] > upper_std) & (df_pair['diff_ewm_OLS'] > speed)
        entrySellShort = (df_pair['n_std_OLS'] < - upper_std) & (df_pair['diff_ewm_OLS'] < -speed)
        exitShort = (df_pair['n_std_OLS'] < lower_std)
        exitBuyToCover = (df_pair['n_std_OLS'] > - lower_std)
        
    else:
        entryLong = (df_pair['n_std_OLS'] > upper_std) 
        entrySellShort = (df_pair['n_std_OLS'] < - upper_std) 
        exitShort = (df_pair['n_std_OLS'] < lower_std)
        exitBuyToCover = (df_pair['n_std_OLS'] > - lower_std)
        
    return entryLong, entrySellShort, exitShort, exitBuyToCover



def COINT_OLS_ADJ(df_pair, upper_std = 3, lower_std = 1.5, speed = 0.7, ATR = 0.016018):
    entryLong = (df_pair['n_std_OLS'] > upper_std) & (df_pair['diff_ewm_OLS'] > speed) & (df_pair['ATR_big'] > ATR)
    entrySellShort = (df_pair['n_std_OLS'] < - upper_std) & (df_pair['diff_ewm_OLS'] < -speed) & (df_pair['ATR_big'] > ATR)
    exitShort = (df_pair['n_std_OLS'] < lower_std)
    exitBuyToCover = (df_pair['n_std_OLS'] > - lower_std)
   
    return entryLong, entrySellShort, exitShort, exitBuyToCover



def COINT_TLS(df_pair, upper_std = 3, lower_std = 1.5, speed = 0.7):
    if type(speed) == float:
        entryLong = (df_pair['n_std_TLS'] > upper_std) & (df_pair['diff_ewm_TLS'] > speed)
        entrySellShort = (df_pair['n_std_TLS'] < - upper_std) & (df_pair['diff_ewm_TLS'] < -speed)
        exitShort = (df_pair['n_std_TLS'] < lower_std)
        exitBuyToCover = (df_pair['n_std_TLS'] > - lower_std)
    else:
        entryLong = (df_pair['n_std_TLS'] > upper_std) 
        entrySellShort = (df_pair['n_std_TLS'] < - upper_std) 
        exitShort = (df_pair['n_std_TLS'] < lower_std)
        exitBuyToCover = (df_pair['n_std_TLS'] > - lower_std)   
        
    return entryLong, entrySellShort, exitShort, exitBuyToCover



def VAR_HIS_MODEL(df_pair):
    entryLong = (df_pair['spread'] > df_pair['upper_in'])
    entrySellShort = df_pair['spread'] < df_pair['lower_in']
    
    exitShort = (df_pair['spread'] < df_pair['upper_out'])
    exitBuyToCover = df_pair['spread'] > df_pair['lower_out']
    
    return entryLong, entrySellShort, exitShort, exitBuyToCover



def VAR_HIS_MODEL_ADJ(df_pair, ATR = 0.014898, speed = 0.91364):
    entryLong = (df_pair['spread'] > df_pair['upper_in']) &  (df_pair['diff_ewm_OLS'].abs() > speed) & (df_pair['ATR_big'] > ATR)
    entrySellShort = (df_pair['spread'] < df_pair['lower_in']) &  (df_pair['diff_ewm_OLS'].abs() > speed) & (df_pair['ATR_big'] > ATR)
    
    exitShort = (df_pair['spread'] < df_pair['upper_out'])
    exitBuyToCover = df_pair['spread'] > df_pair['lower_out']
    
    return entryLong, entrySellShort, exitShort, exitBuyToCover









'''
MERGE RECORD DATA OF VISUALIZE
'''
# df_pair = df_pair.loc[data.idx, 'n_std']
# idx = data.idx
def MERGE_RECORD(df_pair, output_dict, idx, pair, static_dict, backtest_end_ , backtest_end):
    '''
    找進場出場的時間
    '''
    # aa =output_dict['profit_fee_list']
    df_pair.loc[idx, 'profit_list'] = output_dict['profit_list']
    df_pair.loc[idx, 'idx'] =range(len(idx))
    df_pair.loc[:, ['profit_list']] = df_pair.loc[:, ['profit_list']].shift(-1)
    
    sellshort = df_pair.loc[idx].iloc[output_dict['sellshort']].index.values #沒有按照順序排
    buytocover = df_pair.loc[idx].iloc[output_dict['buytocover']].index.values #沒有按照順序排
    sell = df_pair.loc[idx].iloc[output_dict['sell']].index.values #沒有按照順序排
    buy = df_pair.loc[idx].iloc[output_dict['buy']].index.values #沒有按照順序排
    
    
    in_time = pd.to_datetime(np.sort(np.append(buy , sellshort)))
    out_time = pd.to_datetime(np.sort(np.append(sell , buytocover)))
    in_time_idx =np.sort(np.append(output_dict['buy'] , output_dict['sellshort'])) 
    out_time_idx = np.sort(np.append(output_dict['sell'] ,output_dict['buytocover'])) 
    
    def split_list(l, n):
    # 將list分割 (l:list, n:每個matrix裡面有n個元素)
        for idx in range(0, len(l), n):
            yield l[idx:idx+n]
    
    order_type = list(split_list(output_dict['order_type'], 2)) #將list分割成每份中有3個元素

    
    '''
    合併交易紀錄和動態數據
    '''
    profit_fee_list_realized = [i for i in output_dict['profit_fee_list_realized'] if i != 0]
    profit_list_realized = [i for i in output_dict['profit_list_realized'] if i != 0]
    record_list = list(zip(in_time, out_time, 
                        in_time_idx, out_time_idx ,
                        # result.tradePeriod,
                        profit_fee_list_realized, #有按照順序排
                        profit_list_realized,
                        output_dict['order_value'],
                        order_type))
    record_list_ = [pair + (backtest_end_ , backtest_end) + i for i in record_list]
    cols = ['A', 'B', '回測開始','回測結束','進場時間','出場時間','進場位置','出場位置', 'profit_fee_realized', 'profit_realized','order_value', 'order_type']
    record_df = pd.DataFrame(record_list_, columns =cols)
    
    
    '''
    合併靜態數據
    '''
    static_df = pd.DataFrame([static_dict.values()]*len(in_time_idx),columns=static_dict.keys())
    record_df = pd.concat([record_df ,static_df] , axis = 1 , join = 'inner')
    
    def rename_column(by,df_pair_in):
        df_pair_in_ = df_pair_in.iloc[:]
        cols = df_pair_in_.columns.to_list()
        for col in cols:
            df_pair_in_.rename(columns = {col : col+by} , inplace = True)
        return df_pair_in_
    
    df_pair_in = rename_column('_in', df_pair.iloc[:,8:]).shift(1)  #這邊的資料會跟即時不一樣
    df_pair_out = rename_column('_out', df_pair.iloc[:,8:]).shift(1)
    
    record_df.index =  record_df['進場時間']
    record_df = pd.concat([record_df ,df_pair_in] , axis = 1 , join = 'inner')
    record_df.index =  record_df['出場時間']
    record_df = pd.concat([record_df ,df_pair_out] , axis = 1 , join = 'inner')
    return record_df , df_pair.iloc[:,8:]




'''
Go pairt trading backtest
'''
class PAIR_TRADE_MODEL:
    def __init__(self, mode, ATR_q, speed, pair) -> None:
        self.mode = mode
        self.ATR_q = ATR_q
        self.speed = speed
        self.pair = pair




    def load_data(self, df_symbolA , df_symbolB, rule = '5min',) -> None:
        data = DataPair(df_symbolA = df_symbolA, 
                        df_symbolB = df_symbolB, 
                        rule = rule)
        self.data = data


    

    def get_time_list(self, start, end):
        # start = data.df_pair.loc[ data.startTime : data.endTime ].index[0].strftime('%Y-%m-%d')
        # end = data.df_pair.loc[ data.startTime : data.endTime ].index[-1].strftime('%Y-%m-%d')
        self.time_list = pd.date_range(start = start, 
                                       end = end,
                                       freq= '1D').strftime('%Y-%m-%d' )
        
        
        
    def trade_pair(self, max_trade = 12*24, rolling_window = int(12*24*4) , fund = 100):
        self.day_dict = dict()
        for idd in trange(0, len( self.time_list)-8, desc='total pair'):
            
            train_start = self.time_list[idd]
            train_end =   self.time_list[idd + 6]
            backtest_end_ = self.time_list[idd + 7]
            backtest_end = self.time_list[idd + 8]
            # print(backtest_end_)
            self.data.startTime = backtest_end_
            self.data.endTime = backtest_end
            self.data.idx = self.data.df_pair.loc[ self.data.startTime : self.data.endTime ].index
            
            pre_process = PRE_PROCESS(train_start = train_start,
                                      train_end = train_end,
                                      backtest_end = backtest_end,
                                      rolling_window = rolling_window,
                                      pair = self.pair,
                                      df_A = self.data.dfA[train_start:backtest_end],
                                      df_B = self.data.dfB[train_start:backtest_end],)
            static_dict = pre_process.static_func()
            dynamic_df = pre_process.dynamic_func()
            
            
            self.data.static_dict = static_dict
            df_pair = pd.concat([self.data.df_pair[train_start:backtest_end], dynamic_df], axis = 1, join = 'inner')
            df_pair['ATR_big'] = df_pair[['B_ATR', 'A_ATR']].max(axis = 1)
            
            
            
            '''
            codition
            '''
            if self.mode == 'COINT_OLS':
                args = COINT_OLS(df_pair, upper_std = 3, lower_std = 1.5, speed = self.speed)
                beta, constant = pre_process.reg_ols
                
            elif self.mode == 'COINT_TLS':
                args = COINT_TLS(df_pair, upper_std = 3, lower_std = 1.5, speed = self.speed)
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
            

            self.data.type_setting(entryLong, entrySellShort, exitShort, exitBuyToCover)
            output_dict = dictType_output(backtestingPair2(self.data.input_arr, takerFee=0.0005,slippage=0,
                                                          exit_timeOut=True, exParam1 = max_trade, #代表最多可以跑幾期
                                                          exit_profitOut=True, exParam2=0.02, 
                                                          fund=fund,
                                                          exit_lossOut=True, lossOut_condition=1, exParam3=0.015, 
                                                          stopLoss_slippageAdd=0.0001,
                                                          A_beta = beta
                                                          ))
                   
            record_df, df_pair_ = MERGE_RECORD(df_pair, output_dict, self.data.idx,  self.pair, static_dict,  backtest_end_ , backtest_end)
            # signal_df = pd.DataFrame([entryLong, exitShort, entrySellShort, exitBuyToCover], 
            #                          index = ['entryLong', 'exitShort', 'entrySellShort', 'exitBuyToCover']).T
            # df_pair = pd.concat([df_pair,signal_df],axis = 1)
            self.day_dict[backtest_end_] = record_df
            
        return self.day_dict
    
    




if __name__ == "__main__":
    from get_data import get_period_data
    symbol_list =  ['ETHUSDT','BTCUSDT','DOGEUSDT']
    start_time = '2023-06-20'
    end_time = '2023-07-12'
    raw_symbol_dict = get_period_data(start_time = start_time, end_time = end_time , symbol_list = symbol_list)


    root_model = dict() 
    root_model['model1'] = {'mode' : 'COINT_TLS',
                              'ATR_q' : 0,
                              'speed' : 0.8}
    
    pair = ('BTCUSDT', 'ETHUSDT')
    df_symbolA = raw_symbol_dict[pair[0]]
    df_symbolB = raw_symbol_dict[pair[1]]
    
    mode = root_model['model1']['mode']
    ATR_q = root_model['model1']['ATR_q']
    speed = root_model['model1']['speed']
    
    model = PAIR_TRADE_MODEL(mode,ATR_q,speed , pair)
    model.get_time_list(start = start_time, end = end_time)
    model.load_data(df_symbolA , df_symbolB, rule = '5min')
    day_dict = model.trade_pair()
    df = pd.concat([*day_dict.values()])
    print(df)



