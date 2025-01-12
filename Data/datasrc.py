import asyncio
import pandas as pd
from binance import BinanceSocketManager, ThreadedWebsocketManager, AsyncClient, DepthCacheManager
from binance.depthcache import FuturesDepthCacheManager
from binance.enums import HistoricalKlinesType
from typing import *

from Utils import util

class DataSource(object):
    def __init__(self) -> None:
        pass

    def _run(self, *awaitables: Awaitable, timeout = None):
        return util.run(*awaitables, timeout=timeout)
    
    def getRealtime(self, symbols: List[str]=None):
        raise NotImplementedError
    
    def getKlines(self, startTime, endTime, interval='1m', symbols=None):
        raise NotImplementedError

class BinanceDataSource(DataSource):
    def __init__(self, symbols = []) -> None:
        super().__init__()
        self.loop = None
        self.client = None
        self.bm = None
        self.dcms : Dict[str, FuturesDepthCacheManager] = None
        self.symbols = symbols
        self._run(self.init())

    async def init(self):
        self.client = await AsyncClient.create()
        self.bm = BinanceSocketManager(self.client)
        self.dcms = {symbol: FuturesDepthCacheManager(self.client, symbol, bm=self.bm) for symbol in self.symbols}
    
    def close(self):
        self._run(self.client.close_connection())
    
    def getRealtime(self, symbols: List[str]=None):
        if not symbols:
            symbols = self.symbols
        tasks = [self.getRealtimeAsync(symbol) for symbol in symbols]
        results = self._run(*tasks)
        return {k: v for k, v in results}
    
    async def getRealtimeAsync(self, symbol):
        async with self.dcms[symbol] as dcm_socket:
            depth_cache = await dcm_socket.recv()
            bid = depth_cache.get_bids()[0][0]
            ask = depth_cache.get_asks()[0][0]
            return symbol, (bid + ask) / 2
        
    def getKlines(self, startTime, endTime, interval='1m', symbols=None):
        if not symbols:
            symbols = self.symbols
        startTime = int(pd.to_datetime(startTime).timestamp()*1000)
        endTime = int(pd.to_datetime(endTime).timestamp()*1000) - 1

        tasks = [self.getKlinesAsync(symbol, interval, startTime, endTime) for symbol in symbols]
        results = self._run(*tasks)
        
        def formatKlines(klines):
            columnsname = ['openTime', 'Open', 'High', 'Low', 'Close', 'Volume', 'closeTime', 'quoteVolume', 'numTrade', 'takerBuyVolume', 'takerBuyQuoteVolume', 'ignore']
            df = pd.DataFrame(klines)
            df.columns = columnsname
            df['openTime']= pd.to_datetime(df['openTime'], unit='ms')
            df = df.drop(['ignore', 'closeTime'], axis=1)
            df = df.sort_values('openTime', ascending=True)
            df = df.set_index('openTime')
            df = df.astype(float)
            df = df[~df.index.duplicated(keep='first')]
            df['takerSellVolume'] = df['Volume'] - df['takerBuyVolume']
            df['takerSellQuoteVolume'] = df['quoteVolume'] - df['takerBuyQuoteVolume']
            df['avgTradeVolume'] = df['quoteVolume'] / df['numTrade']
            return df
        
        return {k: formatKlines(v) for k, v in results}
    
    async def getKlinesAsync(self, symbol, interval, startTime, endTime):
        # klines = await self.client.get_klines(symbol=symbol, interval=interval, startTime=startTime, endTime=endTime)
        klines = await self.client.get_historical_klines(symbol=symbol, 
                                                         interval=interval, 
                                                         start_str=startTime, 
                                                         end_str=endTime, 
                                                         klines_type=HistoricalKlinesType.FUTURES)#
        # klines = await self.client.get_klines(symbol=symbol, interval=interval)
        return symbol, klines
    


if __name__ == '__main__':
    ds = BinanceDataSource(['BTCUSDT', 'ETHUSDT'])
    # while True:
    #    data = ds.getRealtime()
    #    print(data)
    data = ds.getKlines(['BTCUSDT', 'ETHUSDT'], interval='5m', startTime='2023-08-01 12:40:00', endTime='2023-08-01 12:50:00')
    print(data)
    ds.close()