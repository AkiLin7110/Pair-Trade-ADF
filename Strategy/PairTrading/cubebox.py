from typing import *
from Strategy.PairTrading.model import Pair

class CubeBox:
    def __init__(self) -> None:
        # 最多(10)個Pair的list
        self.box: Dict[int, Pair] = {} # key: pair id, value: Pair
        self.tradingId = []
    
    # 每秒讀取即時資料，作停損停利
    def snapshot(self, data):
        pass

    # 每五分鐘更新當前資料，產生新訊號，決定進出場
    def updateSignal(self) -> dict:
        pass
        # TODO
        # 更新tradingId
    
    def getResult(self, result):
        self.updateTradePrice(result)
        removed = self.remove()
        return removed

    def remove(self):
        target = dict()
        for id in self.tradingId:
            target[id] = self.box[id]
            del self.box[id]
        self.tradingId = []
        return target

    def updateTradePrice(self, result):
        for id in self.tradingId:
            self.box[id].updateCost(result)
        