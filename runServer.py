from OrderSystem.server import BinanceOrderServer
import asyncio
from Utils import util
from dotenv import load_dotenv
import os

load_dotenv()

api_key = 'api_key'
api_secret = 'api_secret'



if __name__ == '__main__':
    server = BinanceOrderServer('127.0.0.1', 8888, api_key=api_key, api_secret=api_secret)
    server.run()
