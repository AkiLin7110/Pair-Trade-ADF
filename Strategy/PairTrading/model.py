# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 16:06:49 2023

@author: USER
"""



from typing import *
from typing import List
import os
import sympy as sympy
import datetime 
import json
nowTime = lambda : datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


from Strategy.PairTrading.tools.utils import *
from Strategy.PairTrading.tools.get_data import get_period_data
from Strategy.PairTrading.config import *   #當下交易的訊號


'''
進出場訊號存檔
1. LS == 'L'
2. LS == 'S'

四種出場訊號
3. exitShort_arr[i] == True 
4. i == len(openA_arr)-2 
5. stopProfit
6. stopLoss 
7. stopTime:
8. stopLoss_sec
'''




# 命名規則
# pair[0] = A
# pair[1] = B
class Model:
    def __init__(self,
                 name,
                  upperStd = 3, 
                  lowerStd = 1.5,  
                  atr_q = 0, 
                  speed = None,
                 rolling_window = int(12*24*4),
                 rule = '5min',
                       ):
        self.name = name
        self.rolling_window = rolling_window
        self.rule = rule

        self.atr_q = atr_q
        self.speed = speed
        self.upperStd = upperStd
        self.lowerStd = lowerStd
        
        self.pair  = None
        self.beta = None
        self.constant = None
        self.df_pair = None

    def getRegressionParam(self):
        raise NotImplementedError

    # 每個月重新產生新的Pairs
    def initial_pair(self, df_symbolA, df_symbolB, pair)-> None:
        self.pair = pair
        data = DataPair(df_symbolA = df_symbolA, 
                        df_symbolB = df_symbolB, 
                        rule = self.rule)
        
        pre_process = PRE_PROCESS(train_start = None,
                                train_end = None,
                                backtest_end = None,
                                rolling_window =  self.rolling_window,
                                pair = self.pair,
                                df_A = data.dfA,
                                df_B = data.dfB)
        static_dict = pre_process.static_func()
        dynamic_df = pre_process.dynamic_func()
        
        
        data.static_dict = static_dict
        df_pair = pd.concat([data.df_pair, dynamic_df], axis = 1, join = 'inner')
        df_pair['ATR_big'] = df_pair[['B_ATR', 'A_ATR']].max(axis = 1)
        self.beta, self.constant = self.getRegressionParam(pre_process)
        df_pair = df_pair.drop(columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Mean', 'Volatility', 'Qt5'])
        df_pair.loc[:,'A'] = df_symbolA['Close']
        df_pair.loc[:,'B'] = df_symbolB['Close']
        df_pair['profit_list'] = None
        df_pair['ret'] = None
        df_pair['idx'] = None
        self.df_pair = df_pair

    def renew_kbar(self, new_k_dict, now_poistion, idx)-> None:
        '''
        new_k_dict -> dict() :
            {'BTCUSDT':              Open      High      Low     Close
             openTime                                                  
             2023-07-17 00:00:00  30217.69  30217.70  30200.0  30209.78
             2023-07-17 00:05:00  30209.77  30227.28  30200.0  30217.31,

             
             
             'ETHUSDT':              Open     High      Low    Close
             openTime                                               
             2023-07-17 00:00:00  1919.43  1919.50  1918.24  1918.68
             2023-07-17 00:05:00  1918.67  1920.38  1917.89  1919.15}
        
        
        
        更新五分鐘的參數
            'residual_%s'%name
            'A_ATR', 'B_ATR','ATR_big' 
            'n_std_%s'%name, 'ewm_%s'%name, 'ewm_lag1_%s'%name, 'diff_ewm_%s'%name,
           'upper_in', 'lower_in',
           'upper_out', 'lower_out', 
        '''
        
        pA_now = new_k_dict[self.pair[0]]['Close'].values[-1]  #現價A
        pB_now = new_k_dict[self.pair[1]]['Close'].values[-1]  #現價B
        self.t_now = new_k_dict[self.pair[1]].index[-1]
        residual_now = pB_now - self.beta*pA_now - self.constant        
        A_ATR = CAL_ATR(new_k_dict[self.pair[0]], name = "A").values[-1]
        B_ATR = CAL_ATR(new_k_dict[self.pair[1]], name = "B").values[-1]
        ATR_big = max(A_ATR, B_ATR)
        
        name = self.name
        
        cols = ['residual_%s'%name, 'A_ATR', 'B_ATR','ATR_big']
        k_df = pd.DataFrame([residual_now, A_ATR, B_ATR, ATR_big], columns = [self.t_now], index = cols).T
        self.df_pair.loc[self.t_now, ['A', 'B', 'residual_%s'%name, 'A_ATR', 'B_ATR','ATR_big','idx']] = pA_now, pB_now, residual_now, A_ATR, B_ATR, ATR_big, idx
        self.df_pair['n_std_%s'%name] = (self.df_pair['residual_%s'%name] - self.df_pair['residual_%s'%name].rolling( self.rolling_window).mean())/ \
        self.df_pair['residual_%s'%name].rolling( self.rolling_window).std() 
        self.df_pair['ewm_%s'%name] = self.df_pair['n_std_%s'%name].ewm(alpha = 0.3).mean()  #作天的EWM
        self.df_pair['ewm_lag1_%s'%name] = self.df_pair['ewm_%s'%name].shift(1)  #作天的EWM
        self.df_pair['diff_ewm_%s'%name] = self.df_pair['n_std_%s'%name] -  self.df_pair['ewm_lag1_%s'%name] 
        
         
        signal_df = self.df_pair.iloc[-1]
        order_type = self.signal(signal_df, now_poistion)
        return order_type
    
    def cal_trade_unit(self, order_type, pA_now, pB_now, fund): #算下單的數量
        k = sympy.symbols("k")
        ans = sympy.solve([pB_now*k + self.beta*pA_now*k  - fund] , [k])
        k = float(ans[k])
        
        if order_type == 1:  
            orderWeightA = self.beta*pA_now*k
            orderWeightB =  pB_now*k*-1
            orderSizeA = orderWeightA / pA_now
            orderSizeB = orderWeightB / pB_now
            
        elif  order_type == 2:
            orderWeightA = self.beta*pA_now*k*-1
            orderWeightB =  pB_now*k
            orderSizeA = orderWeightA / pA_now
            orderSizeB = orderWeightB / pB_now
            
        return orderSizeA, orderSizeB, orderWeightA, orderWeightB
        
    def signal(self, signal_df, now_poistion ) -> int:  #判斷有沒有滿足某種條件，執行動作
        name = self.name
        if now_poistion == NOACTION :  
            if self.speed != None :   #mode1
                if (signal_df['n_std_%s'%name] > self.upperStd) & (signal_df['diff_ewm_%s'%name] > self.speed) & (signal_df['ATR_big'] > self.atr_q): #mode1進場
                    return ENTRY1
                
                elif (signal_df['n_std_%s'%name] < - self.upperStd) & (signal_df['diff_ewm_%s'%name] < -self.speed)  & (signal_df['ATR_big'] > self.atr_q): #mode1進場
                    return ENTRY2
                
                else: #不做動作
                    return NOACTION
                
                
                
            elif  self.speed == None :   #mode3
                if (signal_df['n_std_%s'%name] > self.upperStd) & (signal_df['ATR_big'] > self.atr_q): #mode3進場
                    return ENTRY1
                
                elif (signal_df['n_std_%s'%name] < - self.upperStd) & (signal_df['ATR_big'] > self.atr_q): #mode3進場
                    return ENTRY2
    
                else: #不做動作
                    return NOACTION
                
                
        elif  now_poistion == ENTRY1 :  #出場
            if signal_df['n_std_%s'%name] <  self.lowerStd:
                return SIGNAL_EXIT
            
        elif now_poistion == ENTRY2 :
            if signal_df['n_std_%s'%name] > self.lowerStd*-1:
                return SIGNAL_EXIT
            
        else:  
            return 0


    # 調整Pair的參數
    def updatePair(self, pair):
        raise NotImplementedError



'''
B = constant + A*beta + epsilon
ordertype = 1  ==>  epsilon > 0 : short B, long A
ordertype = 2  ==>  epsilon < 0 : long B, short A
'''

class OLS(Model):
    def __init__(self, upperStd = 3, lowerStd = 1.5,  atr_q = 0, speed = None) -> None:
        super().__init__('OLS', upperStd = upperStd, lowerStd = lowerStd,  atr_q = atr_q, speed = speed)
        
    def getRegressionParam(self, pre_process: PRE_PROCESS):
        return pre_process.reg_ols


class TLS(Model):
    def __init__(self, upperStd = 3, lowerStd = 1.5,  atr_q = 0, speed = None) -> None:
        super().__init__('TLS', upperStd = upperStd, lowerStd = lowerStd,  atr_q = atr_q, speed = speed)

    def getRegressionParam(self, pre_process: PRE_PROCESS):
        return pre_process.reg_tls

class VAR(Model):
    pass



class Pair:
    def __init__(self, model, dataA, dataB, 
                        pair , # 名字
                        pair_idx, # 編號
                        fund = 100,
                        stopProfit = 0.02,
                        stopLoss = 0.016,  #扣完fee的報酬
                        stopLoss_sec = 0.017, #扣完fee的報酬
                        feeRate = 0.04/100,
                        max_trade = 12*24) -> None:
       
        
        # 初始階段
        self.model = model # 使用的模型
        self.dataA = dataA
        self.dataB = dataB
        self.pair = pair
        self.pair_idx = pair_idx  #用ID來紀錄是哪一組
        
        
        # 基本交易參數
        self.fund = fund# 打算買進的金額
        self.stopProfit = stopProfit
        self.stopLoss_sec = stopLoss_sec
        self.stopLoss = stopLoss
        self.feeRate = feeRate
        self.max_trade = max_trade
        
        
        
        # 紀錄交易的過程，交易結束要初始化
        self.now_poistion = 0
        self.record_dict = dict()
        self.costPriceA = None  #A成本價
        self.costPriceB = None  #B成本價
        self.orderSizeA = None  #A進場數量
        self.orderSizeB = None  #B進場數量
        self.orderWeightA = None
        self.orderWeightB = None
        self.costFund = None #總成本  (實際買進的金額，用來計算停損停利)
        self.idx = 0 #紀錄現在更新幾次，最多更新 max_trade次
        self.t_now = None #最近更新的一次時間
        self.comission = None  #進場時的手續費(元)
        
        
        self.model.initial_pair(dataA, dataB, self.pair)   #初始化配對，包括產稱pair_df、算beta、constant(如果有需要的話)
        if self.model.beta < 0:  #這個pair會被關掉
            print('model beta smaller than 0, do not use')
            
    
        
    
    
    # 根據資料產生訊號，每五分鐘
    # 會由Pool, CubeBox去呼叫
    def get_signal(self, new_k_dict) -> None:  #這邊要用utils的工具去寫，而且更新訊號同時也要下單
        '''
        new_k_dict -> dict() :
            {'BTCUSDT':              Open      High      Low     Close
             openTime                                                  
             2023-07-17 00:04:00  30217.69  30217.70  30200.0  30209.78
             2023-07-17 00:05:00  30209.77  30227.28  30200.0  30217.31,
             
             
             'ETHUSDT':              Open     High      Low    Close
             openTime                                               
             2023-07-17 00:04:00  1919.43  1919.50  1918.24  1918.68
             2023-07-17 00:05:00  1918.67  1920.38  1917.89  1919.15}
        
        
        
        更新五分鐘的參數
            'residual_%s'%name
            'A_ATR', 'B_ATR','ATR_big' 
            'n_std_%s'%name, 'ewm_%s'%name, 'ewm_lag1_%s'%name, 'diff_ewm_%s'%name,
           'upper_in', 'lower_in',
           'upper_out', 'lower_out', 
           
           
           
        輸出要給的單位
        回傳的是買到的價格，當作成本價
        
        回傳格式 :
            unit_dict = {symbolA : 500, 
                      symbolB : -600}
        
        這邊測試用 price_dict 當作價格
            price_dict  = {'BTCUSDT': 29812.1, 
                           'ETHUSDT': 1891.83}
           

        '''

        if self.now_poistion == ENTRY1 or  self.now_poistion == ENTRY2:  #有部位
            self.idx += 1
            order_type = self.model.renew_kbar(new_k_dict,  self.now_poistion, self.idx)
            price_dict = {self.pair[0] : new_k_dict[self.pair[0]]['Close'].values[-1],
                          self.pair[1] : new_k_dict[self.pair[1]]['Close'].values[-1]}
            ret, profit = self.cal_now_return(price_dict)
            self.t_now =  new_k_dict[self.pair[1]].index[-1]
            self.model.df_pair.loc[self.t_now,['ret', 'profit_list', 'idx']] = ret, profit , self.idx

            if order_type == SIGNAL_EXIT: #訊號出場
                unit_dict = self.reverse_unit()
                return unit_dict, SIGNAL_EXIT
            
            else:  #其他出場
                if  ret > self.stopProfit:
                    unit_dict = self.reverse_unit()
                    return unit_dict, STOP_PROFIT
                
                elif  ret < -1*self.stopLoss:
                    unit_dict = self.reverse_unit()
                    return unit_dict, STOP_LOSS1
                    
                if self.idx >= self.max_trade:
                    unit_dict = self.reverse_unit()
                    return unit_dict, LASTPERIOD
                
                else: #沒要做動作
                    return {}, NOACTION
    
        
        elif self.now_poistion == NOACTION: #沒部位
            order_type = self.model.renew_kbar(new_k_dict,  self.now_poistion, self.idx)
            if order_type == ENTRY1 or order_type == ENTRY2:  #進場
                pA_now = new_k_dict[self.pair[0]]['Close'].values[-1]  #現價A
                pB_now = new_k_dict[self.pair[1]]['Close'].values[-1]  #現價B
                
                orderSizeA, orderSizeB, self.orderWeightA, self.orderWeightB = self.model.cal_trade_unit(order_type, pA_now, pB_now, self.fund)
                unit_dict = {self.pair[0]: orderSizeA, 
                             self.pair[1] : orderSizeB}
                return unit_dict, order_type
            
            else : #沒做動作
                return {}, NOACTION
    
    
    
    

    def reverse_unit(self): #反向平倉出訊號
        unit_dict = {self.pair[0]: self.orderSizeA*-1,  
                     self.pair[1] : self.orderSizeB*-1}
        return unit_dict
        
    
    
    
        
    def get_enter_position(self, cost_dict, unit_dict, new_k_dict, order_type): # 有交易到，把交易的資訊記下來
        print('entry')
        self.t_now = new_k_dict[self.pair[1]].index[-1]
        self.now_poistion = order_type
        self.costPriceA = cost_dict[self.pair[0]]  #A成本價
        self.costPriceB = cost_dict[self.pair[1]]  #B成本價
        self.orderSizeA = unit_dict[self.pair[0]]  #A進場數量
        self.orderSizeB = unit_dict[self.pair[1]]  #B進場數量
        self.costFund = abs(self.orderSizeA * self.costPriceA) +  abs(self.orderSizeB *self.costPriceB)
        self.model.df_pair.loc[self.t_now,['ret', 'profit_list', 'idx']] = 0, 0, 0
        self.comission = self.costFund* self.feeRate
        
        entry_dict = dict()
        entry_dict['(A,B)'] = self.pair
        entry_dict['model'] = str(self.model)
        entry_dict['cost'] = self.costFund
        entry_dict['exit_time'] = nowTime()
        entry_dict['order_type'] = order_type
        entry_dict['receivePriceA'] = new_k_dict[self.pair[0]]['Close'].values[-1]
        entry_dict['receivePriceB'] = new_k_dict[self.pair[1]]['Close'].values[-1]
        entry_dict['costPriceA'] = self.costPriceA 
        entry_dict['costPriceB'] = self.costPriceB
        entry_dict['orderWeightA'] = self.orderWeightA
        entry_dict['orderWeightB'] = self.orderWeightB
        entry_dict['orderSizeA'] = self.orderSizeA
        entry_dict['orderSizeB'] = self.orderSizeB
        entry_dict['snapShot'] = self.model.df_pair.iloc[-1].to_json()
        self.record_dict['entry'] = entry_dict
        print("Dump entry record")
        self.dump_recod('entry') #將交易紀錄另存出去

    
        
        
    def get_close_position(self, cost_dict, order_type, price_dict): #接收平倉訊號
        name = f"./log/{self.pair_idx}/entry_log.json"
        with open(name, 'w+') as f: #刪除第一列json
                pass
        ret, profit = self.cal_now_return(cost_dict) #出場的價格
        exit_dict = dict()
        exit_dict['(A,B)'] = self.pair
        exit_dict['model'] = str(self.model)
        # print('here')
        exit_dict['exit_time'] = nowTime()
        exit_dict['order_type'] = order_type
        exit_dict['receivePriceA'] = price_dict[self.pair[0]]
        exit_dict['receivePriceB'] = price_dict[self.pair[1]]
        exit_dict['costPriceA'] = cost_dict[self.pair[0]]
        exit_dict['costPriceB'] = cost_dict[self.pair[1]]
        exit_dict['profit'] = profit
        exit_dict['ret'] = ret
        exit_dict['snapShot'] = self.model.df_pair.iloc[-1].to_json()
        self.record_dict['exit'] = exit_dict
        print("Dump exit record")
        self.dump_recod('exit') #將交易紀錄另存出去Ｓ
        self.initial_para() #初始化交易紀錄



    
    def dump_recod(self, order_type):  #交易完成，另存出去
        # self.record_dict  傳出去
        # self.record_dict = dict

        os.makedirs('./log',exist_ok = True)
        os.makedirs(f'./log/{self.pair_idx}',exist_ok=True)

        # now = datetime.datetime.now()
        # dt_string = now.strftime("%Y%m%d-%H:%M:%S")
        print(order_type, type(order_type))
        name = f"./log/{self.pair_idx}/{order_type}_log.json"
        with open(name, "a") as outfile:
            json.dump(self.record_dict[order_type], outfile)
            outfile.write('\n')
    


    def initial_para(self):
        self.now_poistion = NOACTION
        self.record_dict = dict()
        self.costPriceA = None  #A成本價
        self.costPriceB = None  #B成本價
        self.orderSizeA = None  #A進場數量
        self.orderSizeB = None  #B進場數量
        self.orderWeightA = None
        self.orderWeightB = None
        self.costFund = None #總成本
        self.idx = 0 #紀錄現在更新幾次，最多更新 max_trade次
        self.t_now = None #最近更新的一次時間
        self.comission = None #進場時的手續費(元)
    
    
    
    def cal_now_return(self, price_dict):
        pA_now = price_dict[self.pair[0]] #現價A
        pB_now = price_dict[self.pair[1]] #現價B
        if self.now_poistion == 1:
            profit = self.orderSizeA * (pA_now - self.costPriceA ) +  self.orderSizeB * ( self.costPriceB - pB_now) - (self.orderSizeA*pA_now + -1*self.orderSizeB* pB_now)*self.feeRate - self.comission
        elif  self.now_poistion == 2:
            profit = self.orderSizeA * (self.costPriceA - pA_now ) +  self.orderSizeB * (pB_now - self.costPriceB) - (-1*self.orderSizeA*pA_now + self.orderSizeB* pB_now)*self.feeRate - self.comission
        ret = profit/self.costFund
        return ret, profit    
    
    
    
    # 更新args (eg. beta, const, residual)，每日觸發
    def update(self): #這邊要用utils的工具去寫
        self.model.updatePair(self)
        pass




    def updateCost(self, price):
        self.cost = price




# 每個月:每個Model各選十個pair
# 每天:重新訓練每個Pair的beta跟const
# 每五分鐘: 更新90分位數
# if __name__ == "__main__":
#     from binance import Client, ThreadedWebsocketManager, ThreadedDepthCacheManager
#     import ccxt 
#     import pandas as pd
#     # api_key = 'dri7XIEU5DRs2wWTxw0QSwtRb6IFXUIhkVyyoFDxIhl4hL74QAnV5AN0Ct8hSGiT' 
#     # api_secret = 'DQ96HNz40FmDbNNzP8w8JnX2hEPI7CIpO5bdXcjPQnpuYyDMlujwq22stIHI2zjI'
    
#     symbol_list =  ['ETHUSDT','BTCUSDT']
#     client = Client()
#     prices = client.get_all_tickers()
#     price_dict = dict()
#     for i in prices:
#         if i['symbol'] in symbol_list:
#             price_dict[i['symbol']] = float(i['price'])


#     new_k_dict = get_period_data(start_time =  '2023-07-17', end_time = '2023-07-18' , symbol_list = symbol_list)
#     for k,v in new_k_dict.items():
#         new_k_dict[k] = v.iloc[4:6,:4]

    
#     start_time = '2023-07-10'
#     end_time = '2023-07-17'
#     raw_symbol_dict = get_period_data(start_time = start_time, end_time = end_time , symbol_list = symbol_list)


#     root_model = dict() 
#     root_model['model1'] = {'mode' : 'COINT_TLS',
#                               'ATR_q' : 0,
#                               'speed' : 0.8}
    
#     pair = ('BTCUSDT', 'ETHUSDT')
#     df_symbolA = raw_symbol_dict[pair[0]]
#     df_symbolB = raw_symbol_dict[pair[1]]
    
#     mode = root_model['model1']['mode']
#     ATR_q = root_model['model1']['ATR_q']
#     speed = root_model['model1']['speed']
    
#     # model  = Model( max_trade = 12*24,rolling_window = int(12*24*4),fund = 100, rule = '5min')
#     model = OLS(atr_q = ATR_q, speed = 0.8)
#     pair_strategy = Pair(model, df_symbolA, df_symbolB, ('ETHUSDT','BTCUSDT'), 10 )
#     # pair_strategy.get_initial_pair(df_symbolA, df_symbolB, ('ETHUSDT','BTCUSDT'), 10)  #初始化


















