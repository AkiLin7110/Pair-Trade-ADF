from binance import AsyncClient, ThreadedWebsocketManager, ThreadedDepthCacheManager
import ccxt
import asyncio
import pickle
import struct
import traceback
import pandas as pd
from typing import *
from datetime import datetime, timedelta

from Utils import util
from Utils.logger import Logger

class BinanceOrderServer:
    def __init__(self, host: str, port: int, api_key: str, api_secret: str, ruleUpdateFreq: timedelta = timedelta(days=1)) -> None:
        self.host = host
        self.port = port
        self.api_key = api_key
        self.api_secret = api_secret
        self.client = None
        self.server = None
        self.exchangeRule = None
        self.ruleLastUpdate = None
        self.ruleUpdateFreq = ruleUpdateFreq
        self.logger = Logger("BinanceOrderServer")
        util.run(self.init())

    async def init(self):
        self.client = await AsyncClient.create(api_key = self.api_key, api_secret = self.api_secret)
        await self.updateRule()

    def run(self):
        util.run(self._run())

    async def _run(self):
        self.server = await asyncio.start_server(self.handle_client, self.host, self.port)
        self.addr = self.server.sockets[0].getsockname()
        self.logger.info(f'Serving on {self.addr}')
        async with self.server:
            await self.server.serve_forever()
    
    async def updateRule(self):
        rules = await self.client.futures_exchange_info()
        stepSizeDict = self.exchangeRule['stepSize'] if self.exchangeRule else dict()
        for value in rules['symbols']:
            stepSize = value['filters'][2]['stepSize']
            try:
                stepSizeDict[value['symbol']] = float(stepSize)
            except Exception as e:
                if not self.exchangeRule:
                    raise e
                else:
                    self.logger.warning(f"StepSize for {value['symbol']} cannot be recognized.")
        self.exchangeRule = dict()
        self.exchangeRule['stepSize'] = stepSizeDict
        self.ruleLastUpdate = datetime.now()
    
    def roundOrder(self, order: Dict[str, float]):
        if self.ruleLastUpdate is None or datetime.now() - self.ruleLastUpdate > self.ruleUpdateFreq:
            util.run(self.updateRule())
        for symbol, qt in order.items():
            order[symbol] = util.round(qt, self.exchangeRule['stepSize'][symbol])
        return order

    async def handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        # Handle data from a single client connection
        addr = writer.get_extra_info('peername')
        self.logger.info(f"Connected to {addr}")

        try:
            while True:
                try:
                    request = await self.readData(reader)
                    self.logger.info(f"Received from {addr}: {request}")
                    asyncio.create_task(self.handleRequest(request, reader, writer))
                except:
                    self.logger.info(f"Connection to {addr} reset or closed by the client")
                    break
        except Exception as e:
            self.logger.warning(f"Connection to {addr} closed due to excpetion {e}")
        finally:
            writer.close()
            # await writer.wait_closed()
    
    async def handleRequest(self, request: Dict[str, Any], reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        status = 'OK'
        self.logger.info(f"Handling request, content: {request}")
        try:
            clientId = request['id']
            order = request['msg']
            target = request['target']
        except:
            self.logger.error("Cannot correctly handle request with unknown format.")
            status = 'Wrong Request Format'
        
        if status == 'OK': 
            try:
                order = self.roundOrder(order)
                orderInfos = await self.placeOrder(order)
                """
                sample orderInfos:
                [{'orderId': 8389765611938178142, 'symbol': 'ETHUSDT', 'status': 'NEW',
                    'clientOrderId': '8DBq0yOLxH89D31Otq5h9w', 'price': '0.00', 
                    'avgPrice': '0.00', 'origQty': '0.010', 'executedQty': '0.000', 
                    'cumQty': '0.000', 'cumQuote': '0.00000', 'timeInForce': 'GTC', 
                    'type': 'MARKET', 'reduceOnly': True, 'closePosition': False, 'side': 'SELL', 
                    'positionSide': 'BOTH', 'stopPrice': '0.00', 'workingType': 'CONTRACT_PRICE', 
                    'priceProtect': False, 'origType': 'MARKET', 'updateTime': 1691565860674}, 
                    {'orderId': 178872065511, 'symbol': 'BTCUSDT', 'status': 'NEW', 
                    'clientOrderId': 'OMbUtt6TVIhNx3vFAgABYv', 'price': '0.00', 'avgPrice': '0.00', 
                    'origQty': '0.002', 'executedQty': '0.000', 'cumQty': '0.000', 
                    'cumQuote': '0.00000', 'timeInForce': 'GTC', 'type': 'MARKET', 
                    'reduceOnly': True, 'closePosition': False, 'side': 'SELL', 'positionSide': 'BOTH', 
                    'stopPrice': '0.00', 'workingType': 'CONTRACT_PRICE', 'priceProtect': False, 
                    'origType': 'MARKET', 'updateTime': 1691565860674}]
                """
            except Exception as e:
                traceback.print_exc()
                self.logger.error(f"Cannot correctly place order due to {e}")
                status = 'Place Order Failed'
        
        if status == 'OK':
            try:
                tradeResults = await self.getTrades(orderInfos)
            except Exception as e:
                self.logger.error(f"Cannot retrieve trading results due to {e}")
                status = 'Retrieve Trade Info Failed'
        
        responseDict = dict()
        responseDict['id'] = clientId
        responseDict['status'] = status
        if status == 'OK':
            responseDict['response'] = tradeResults
        else:
            responseDict['response'] = None
        
        self.logger.info(f"Responding client <{clientId}> with: {responseDict}")

        await self.writeData(writer, responseDict)

            
    async def placeOrder(self, orders: Dict[str, int]):
        """
        batchOrders = []
        for symbol, quantity in orders.items():
            orderDict = {"type":"MARKET",
                        "symbol":symbol,
                        "side":"BUY" if quantity > 0 else "SELL",
                        "quantity":str(abs(quantity))}
            batchOrders.append(orderDict)
        self.logger.info(f"Placing batched order: {batchOrders}")
        return await self.client.futures_place_batch_order(batchOrders=batchOrders)
        """
        tasks = []
        for symbol, quantity in orders.items():
            tasks.append(self.client.futures_create_order(type='MARKET',
                                                          symbol=symbol,
                                                          side="BUY" if quantity > 0 else "SELL",
                                                          quantity=abs(quantity)))
        return await asyncio.gather(*tasks)


    async def getTrades(self, orderInfos):
        tasks = []
        for info in orderInfos:
            orderId = info["orderId"]
            symbol = info["symbol"]
            tasks.append(self.getTrade(symbol=symbol, orderId=orderId))
        trades = await asyncio.gather(*tasks)
        return {trade[0]: trade[1] for trade in trades}

    async def getTrade(self, symbol: str, orderId: int, order_timeout: float = 1.0, trade_timeout: float = 1.0, order_attempt: int = 3, trade_attempt: int = 3):
        while True:
            try:
                attempt = 0
                while attempt < order_attempt:
                    try: 
                        # check order status
                        order_check = await asyncio.wait_for(self.client.futures_get_order(symbol=symbol, orderId=orderId), timeout=order_timeout)
                        order_status = order_check['status']
                        break
                    except asyncio.TimeoutError:
                        attempt += 1
                if attempt >= order_attempt:
                    self.logger.error(f"Check order status failed in {order_attempt}, symbol: {symbol}, orderID: {orderId}")
                    return symbol, []

                if order_status == "FILLED":
                    self.logger.info(f"{symbol} Sell Order Filled, order ID {orderId}")
                    record = f'Retrieving trading record, symbol: {symbol}, orderID: {orderId}'
                    attempt = 0
                    while attempt < trade_attempt:
                        try:
                            trade = await asyncio.wait_for(self.client.futures_account_trades(symbol=symbol, orderId=orderId), timeout=trade_timeout)
                            if len(trade) > 0:
                                record += f"\nAttempt {attempt}: Success."
                                self.logger.info(record)
                                return symbol, trade
                            else:
                                record += f"\nAttempt {attempt}: Empty Value."
                        except asyncio.TimeoutError:
                            record += f"\nAttempt {attempt}: Timeout in {trade_timeout} seconds."
                        finally:
                            attempt += 1
                    record += f"\nRetreive trading record failed in {order_attempt}, symbol: {symbol}, orderID: {orderId}"
                    self.logger.error(record)
                    return symbol, []
                            
                await asyncio.sleep(0)
            except Exception as e:
                pass

    async def readData(self, reader: asyncio.StreamReader):
        # Read the message length (4 bytes, as an unsigned integer)
        raw_msglen = await reader.readexactly(4)
        msglen = struct.unpack('>I', raw_msglen)[0]
        # Read the message data using the message length
        data = await reader.readexactly(msglen)
        # Process the received data
        try:
            # If the data is a pickled dictionary, deserialize it
            message = pickle.loads(data)
        except pickle.UnpicklingError:
            # If the data is not a pickled dictionary, treat it as regular text
            message = data.decode().strip()
        return message
    
    async def writeData(self, writer: asyncio.StreamWriter, data):
        # Serialize the data using pickle
        serialized_data = pickle.dumps(data)

        # Prefix the message with its length (as 4-byte unsigned integer)
        msglen = struct.pack('>I', len(serialized_data))

        # Send the length-prefixed message to the server
        writer.write(msglen + serialized_data)
        await writer.drain()

if __name__ == '__main__':
    server = BinanceOrderServer('127.0.0.1', 8888, 1, 1)
    server.run()