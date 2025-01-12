# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 17:25:15 2023

@author: 仔仔
"""

'''
當下交易的訊號
'''


NOACTION = 0
ENTRY1 = 1
ENTRY2 = 2
SIGNAL_EXIT = 3
STOP_PROFIT = 5 
STOP_LOSS1 = 6 #五分鐘止損 
LASTPERIOD = 7 #最多可以交易N期
STOP_LOSS2 = 8 #1秒鐘止損

STOPLOSS_MINUTE_RATE = 0.015
STOPPROFIT_RATE = 0.02
STOPLOSS_SECOND_RATE = 0.0165