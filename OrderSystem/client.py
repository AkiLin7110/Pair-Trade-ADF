import asyncio
import pickle
import struct
from typing import *

from Utils.logger import Logger
from Utils import util

class OrderClient:
    def __init__(self, id: str, host: str, port: int) -> None:
        self.id = id
        self.host = host
        self.port = port
        self.reader: asyncio.StreamReader = None
        self.writer: asyncio.StreamWriter = None
        self.logger = Logger(name=f'Client [{self.id}]')
        self._run(self.connect())

    async def connect(self):
        self.reader, self.writer = await asyncio.open_connection(self.host, self.port)

    def sleep(self, time):
        self._run(asyncio.sleep(time))

    def sendData(self, data):
        self._run(self.sendDataAsync(data))

    async def sendDataAsync(self, data):
        if self.writer is None:
            await self.connect()

        # Serialize the data using pickle
        serialized_data = pickle.dumps(data)

        # Prefix the message with its length (as 4-byte unsigned integer)
        msglen = struct.pack('>I', len(serialized_data))

        # Send the length-prefixed message to the server
        self.writer.write(msglen + serialized_data)
        await self.writer.drain()
    
    def receiveData(self):
        return self._run(self.receiveDataAsync())

    async def receiveDataAsync(self):
        # Read the response from the server (length-prefixed)
        raw_response_len = await self.reader.readexactly(4)
        response_len = struct.unpack('>I', raw_response_len)[0]
        response = await self.reader.readexactly(response_len)

        # Process the response (optional)
        try:
            received_data = pickle.loads(response)
            # self.logger.info("Received from server:", received_data)
        except pickle.UnpicklingError:
            received_data = response.decode().strip()
            # self.logger.info("Received from server:", response.decode().strip())  # Treat as regular text
        return received_data

    def placeOrder(self, order: Dict[str, int], target : bool = False):
        message_dict = dict()
        message_dict["id"] = self.id
        message_dict["msg"] = order
        message_dict["target"] = target
        # self.sendData(message_dict)
        _, execTime = util.timer(self.sendData, message_dict)
        self.logger.debug(f"Send data from client takes {execTime}")
        response, execTime = util.timer(self.receiveData)
        self.logger.debug(f"Receive data from server takes {execTime}")
        if response['status'] == 'OK':
            self.logger.info(f"Order successfully placed, content: {order}.")
        else:
            self.logger.error(f"Placing order failed due to {response['status']}.")
        return response

    
    def _run(self, *awaitables, timeout : int = None):
        return util.run(*awaitables, timeout=timeout)