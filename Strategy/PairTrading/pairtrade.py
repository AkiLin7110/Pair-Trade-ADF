from Strategy.PairTrading.pool import *
from Strategy.PairTrading.model2 import *
from Strategy.PairTrading.cubebox import *
from OrderSystem.server import OrderServer
from Data.datasrc import BinanceDataSource
from typing import *

class PairTrade:
    def __init__(self) -> None:
        self.pool = Pool()
        self.cubebox = CubeBox()
        self.orderserver = OrderServer()
        self.datastream = BinanceDataSource()

    # 每秒更新訊號
    def tick(self):
        pass
    
    # 每五分鐘更新進場訊號
    # 進行Pool, CubeBox的訊號檢查
    def checkSignals(self):
        cubebox_order = self.cubebox.updateSignal()
        cubebox_result = self.placeOrder(cubebox_order)
        pairs = self.cubebox.getResult(cubebox_result)

        pool_order = self.pool.updateSignal()
        pool_result = self.placeOrder(pool_order)
        # TODO
        # 將pool_result回傳回pool
        self.pool.add(pairs)



    
    # 對orderserver下單
    def placeOrder(self, compOrder):
        orderIds = self.orderserver.placeCompositeOrder(compOrder)
        # TODO
        # orderId 查詢資料
        result = None
        return result
    
    # 迴圈
    def run(self):
        pass
    












