from OrderSystem.client import OrderClient


client = OrderClient('test', '127.0.0.1', 8888)
# for i in range(2):
#     client._run(client.sendData('test'))
#     client.sleep(2)

client.placeOrder({'ETHUSDT' : -0.03,  
        'BTCUSDT' : 0.002})
