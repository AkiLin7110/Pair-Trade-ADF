# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 14:53:30 2023

@author: 仔仔
"""


import pandas as pd
import numpy as np
from typing import *
import datetime as dt
import nest_asyncio
nest_asyncio.apply() # for jupyter notebook

from Utils.logger import Logger
from Strategy.PairTrading.model import *
# from Strategy.PairTrading.pool import *
# from Strategy.PairTrading.cubebox import *
from Strategy.PairTrading.config import *   #當下交易的訊號
from Strategy.PairTrading.tools.get_data import get_period_data, resample_symbol

from Data.datasrc import BinanceDataSource
from OrderSystem.client import OrderClient
client = OrderClient('test', '127.0.0.1', 8888)


class PairTradeTest:
    def __init__(self, pair_strategy, 
                 stopLoss_sec = 0.017, 
                 symbols_pool =  ['ETHUSDT','BTCUSDT'],
                 interval = '5m') -> None:
        
        self.pair_strategy = pair_strategy
        self.symbols_pool = symbols_pool
        self.interval = interval
        self.stopLoss_sec = stopLoss_sec , #扣完fee的報酬
        self.datastream = BinanceDataSource(symbols_pool)
        self.logger = Logger('pairTradeTest')
        self.pair = (self.symbols_pool[0], self.symbols_pool[1] , )
        
    # 每秒更新訊號
    def tick(self):
        if self.pair_strategy.now_poistion == 0:
            return
        price_dict = self.datastream.getRealtime()
        ret, profit = self.pair_strategy.cal_now_return(price_dict)

        os.makedirs('./Record',exist_ok = True)
        os.makedirs(f'./Record/{self.pair[0]}-{self.pair[1]}',exist_ok = True)
        name = f"./Record/{self.pair[0]}-{self.pair[1]}_{self.pair_strategy.idx}.json"
        
        print(round(ret, 5), round(profit, 5))
        if ret < -1*self.stopLoss_sec:
            unit_dict = self.pair_strategy.reverse_unit() #產生相反的下單數量平倉
            cost_dict_row = client.placeOrder(unit_dict)
            cost_dict = self.get_cost_dict(cost_dict_row)
            self.pair_strategy.get_close_position( cost_dict, STOP_LOSS2, price_dict)
            self.logger.info('%s order_type : %s ==> success'%( self.pair , STOP_LOSS2))
        
        return_dict = dict()
        return_dict['ret'] = ret
        return_dict['profit'] = profit 

        with open(name, "a") as outfile:
            json.dump(return_dict, outfile)
            outfile.write('\n')
            
    # 每五分鐘更新進場訊號
    # 進行Pool, CubeBox的訊號檢查
    def checkSignals(self, startTime, endTime):  #TODO 
        new_k_dict = self.datastream.getKlines(startTime, endTime, self.interval, self.symbols_pool) #位置改
        unit_dict, order_type = self.pair_strategy.get_signal(new_k_dict) #訊號進，訊號出，不動作
        if order_type == ENTRY1 or order_type == ENTRY2:
            print(unit_dict)
            unit_dict[self.pair[0]] = round(unit_dict[self.pair[0]], 2)  #暫時這樣改
            unit_dict[self.pair[1]] = round(unit_dict[self.pair[1]], 3)  #暫時這樣改

            cost_dict_row = client.placeOrder(unit_dict)
            cost_dict = self.get_cost_dict(cost_dict_row)
            self.pair_strategy.get_enter_position(cost_dict, unit_dict, new_k_dict, order_type)
            self.logger.info('%s order_type : %s ==> success'%( self.pair , order_type))        
        elif order_type  in [SIGNAL_EXIT, STOP_PROFIT, STOP_LOSS1, LASTPERIOD, STOP_LOSS2] :
            price_dict = dict()
            price_dict[self.pair[0]] = new_k_dict[self.pair[0]]['Close'].values[-1]
            price_dict[self.pair[1]] = new_k_dict[self.pair[1]]['Close'].values[-1]            
            
            cost_dict_row = client.placeOrder(unit_dict)
            cost_dict = self.get_cost_dict(cost_dict_row)
            self.pair_strategy.get_close_position( cost_dict, order_type, price_dict)
            self.logger.info('%s order_type : %s ==> success'%( self.pair , order_type))
        else:
            pass
        self.logger.info( 'renew finish!   order_type : %s'%order_type)


    def get_cost_dict(self, cost_dict_row):
        cost_dict_row = cost_dict_row['response']
        cost_dict = dict()
        for symbol , record_list in cost_dict_row.items():
            qty = 0  #數量
            realize_principal = 0 #本金(價值)
            for record in record_list:
                realize_principal += float(record['quoteQty'])  #買的本金 = 數量*價格
                qty += float(record['qty']) #買的數量
            cost_dict[symbol] = realize_principal/qty
        return cost_dict
            






# self = pt_test.pair_strategy.model#.df_pair
# unit_dict = {'ETHUSDT' : 0.01, 
#           'BTCUSDT' : 0.001}
# unit_dict = {'ETHUSDT': 0.02141324, 'BTCUSDT': -0.0031414}


# cost_dict_row = {'id': 'test',
#                     'status': 'OK',
#                     'response': {'ETHUSDT': [{'symbol': 'ETHUSDT',
#                         'id': 3191322860,
#                         'orderId': 8389765613625389930,
#                         'side': 'BUY',
#                         'price': '1663.53',
#                         'qty': '0.010',
#                         'realizedPnl': '0',
#                         'marginAsset': 'USDT',
#                         'quoteQty': '16.63530',
#                         'commission': '0.00665412',
#                         'commissionAsset': 'USDT',
#                         'time': 1692698287146,
#                         'positionSide': 'BOTH',
#                         'buyer': True,
#                         'maker': False}],
                                 
#                       'BTCUSDT': [{'symbol': 'BTCUSDT',
#                         'id': 4027761417,
#                         'orderId': 182762021122,
#                         'side': 'BUY',
#                         'price': '26040.30',
#                         'qty': '0.001',
#                         'realizedPnl': '0',
#                         'marginAsset': 'USDT',
#                         'quoteQty': '26.04030',
#                         'commission': '0.01041611',
#                         'commissionAsset': 'USDT',
#                         'time': 1692698287140,
#                         'positionSide': 'BOTH',
#                         'buyer': True,
#                         'maker': False}]}}




# import sys
# sys.exit()

'''
Initialize
'''

t = dt.datetime.utcnow()
print('現在時間 UTC-0 : ' , t)
minute_need = int(t.minute%5)
t_need =  t- dt.timedelta(minutes = minute_need) 
time_end = t_need.strftime('%Y-%m-%d %H:%M')
time_start =  t_need - dt.timedelta(hours =7*24)  # UTC0
time_start = time_start.strftime('%Y-%m-%d %H:%M') # UTC1
print(time_start, time_end)


#設定交易時間
t_start =  pd.to_datetime(time_end) + dt.timedelta(minutes = 5)
t_start = str(t_start)
t_end =  pd.to_datetime(time_end) + dt.timedelta(hours = 24) #只交易1天，剩下1天只出不進
t_end = str(t_end)
update_time_list = pd.date_range(start = t_start, end = t_end  ,freq = '5min').to_list()
print('交易時間 %s ~~ %s'%(t_start,t_end) )


# 初始化配對
symbol_list =  ['ETHUSDT','BTCUSDT']
raw_symbol_dict = get_period_data(start_time = time_start, end_time = time_end , symbol_list = symbol_list)
root_model = dict() 
root_model['model1'] = {'mode' : 'COINT_TLS',
                          'ATR_q' : 0,
                          'speed' : None,
                          'comb' : [('ETHUSDT', 'BTCUSDT'), ('DOGEUSDT', 'BTCUSDT'),('DOGEUSDT', 'ADAUSDT') ]}

pair = ('ETHUSDT', 'BTCUSDT')
df_symbolA = raw_symbol_dict[pair[0]]
df_symbolB = raw_symbol_dict[pair[1]]

mode = root_model['model1']['mode']
ATR_q = root_model['model1']['ATR_q']
speed = root_model['model1']['speed']
model = OLS(upperStd = 0.025, lowerStd = 0, atr_q = ATR_q, speed = speed)
pair_strategy = Pair(model, df_symbolA, df_symbolB,
                     stopProfit = 0.002,
                     stopLoss = 0.002,
                     pair = pair, 
                     pair_idx = 10,
                     max_trade = 12)



pt_test = PairTradeTest(pair_strategy, 
                 stopLoss_sec = 0.0035, 
                 symbols_pool =  pair,
                 interval = '5m') 


'''
五秒鐘報時
'''
from threading import Timer
class RepeatTimer(Timer):
    def run(self):
        while not self.finished.wait(self.interval):
            self.function(*self.args, **self.kwargs)


test_dict= dict()
test_dict['5sec'] = False
def dummyfn():
    if test_dict['5sec'] == False:
        test_dict['5sec'] = True
    # print('報時')
timer = RepeatTimer(5, dummyfn )
timer.start()
# timer.cancel() #取消線程



'''
開始交易
'''

while True:
    # try:
    time.sleep(0.001)
    update_time_now = update_time_list[0] # 最新的更新時間
    t = dt.datetime.utcnow() 
    if test_dict['5sec'] == True:
        print( t.strftime('%Y-%m-%d %H:%M:%S'))
        pt_test.tick()
        
        test_dict['5sec'] = False
        print('')
        print('')           
        print('')         
        print('')         

    if t > update_time_now:
        # t_need =  t - dt.timedelta(minutes = minute_need) 
        endTime = t.strftime('%Y-%m-%d %H:%M')
        startTime =  (t - dt.timedelta(minutes=10)).strftime('%Y-%m-%d %H:%M')  # UTC0
        pt_test.checkSignals(startTime, endTime)
        print(pt_test.pair_strategy.model.df_pair[['A', 'B', 'n_std_OLS','ATR_big', 'ewm_OLS','idx','ret', 'profit_list']].tail(5))
        del update_time_list[0]  #刪除最新經過的   
    # except Exception as e:
    #     print(e)
    #     break
    



