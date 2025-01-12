import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import regex as re


nominal = 1000
data = []

# entry_log # 實際上應該要抓個檔案的最後一筆
with open('log/ETHUSDT-BTCUSDT/entry_log.json', 'r', encoding='utf-8') as jsonfile:
    data.append(jsonfile.readline())
    while data[-1]:
        data.append(jsonfile.readline())
data.pop() # ->str


# String -> Dictionary
data_json = []
for item in data:
    data_json.append(json.loads(item)) # ->dictionary


def data_model(df): # 轉output
    pattern = re.compile('OLS|TLS')
    model = re.findall(pattern, df)[0]
    return model

def get_Return(df):
    ret = json.loads(df)['ret']
    return ret

def get_Profit(df):
    profit = json.loads(df)['profit_list']
    # print(profit)
    return profit

def get_hold_Period(df):
    period = json.loads(df)['idx']
    return period

def get_nstd(df):
    n_std = json.loads(df)['n_std_OLS']
    return n_std

# 1.1 # 目前仍持有部位
df = pd.DataFrame.from_dict(data_json)
df['model'] = df['model'].apply(lambda x: data_model(x))
df['ret'] = df['snapShot'].apply(lambda x: get_Return(x)).apply(lambda x:round(x,2))
df['profit'] = df['snapShot'].apply(lambda x : get_Profit(x))
df['period'] = df['snapShot'].apply(lambda x: get_hold_Period(x))
df['nstd'] = df['snapShot'].apply(lambda x: get_nstd(x))


# print(df.columns)
# print(data_json[-1].keys())
# print(json.loads(data_json[-1]['snapShot']))

db = pd.DataFrame()
db['Pair'] = df['(A,B)'].copy()
db['model'] = df['model'].copy()
db['ret'] = df['ret'].copy()
db['profit'] = df['profit'].copy()
db['period'] = df['period'].copy()
db['nstd'] = df['nstd'].copy()
db['cost'] = df['cost'].copy()

round_db = db.copy()
round_db['ret'] = db['ret'].apply(lambda x:round(x,2))
round_db['profit'] = db['profit'].apply(lambda x:round(x,2))
round_db['period'] = db['period'].apply(lambda x:round(x,2))
round_db['nstd'] = db['nstd'].apply(lambda x:round(x,2))
round_db['cost'] = db['cost'].apply(lambda x:round(x,2))


# exit_log # 實際上應該要抓個檔案的最後一筆
data1 = []
with open('log/ETHUSDT-BTCUSDT/exit_log.json', 'r', encoding='utf-8') as jsonfile:
    data1.append(jsonfile.readline())
    while data1[-1]:
        data1.append(jsonfile.readline())
data1.pop()

# String -> Dictionary
data_json = []
for item in data1:
    data_json.append(json.loads(item)) # ->dictionary

DF = pd.DataFrame.from_dict(data_json)
DF['model'] = DF['model'].apply(lambda x: data_model(x))
DF['ret'] = DF['snapShot'].apply(lambda x: get_Return(x))
DF['profit'] = DF['snapShot'].apply(lambda x : get_Profit(x))
DF['period'] = DF['snapShot'].apply(lambda x: get_hold_Period(x))
DF['nstd'] = DF['snapShot'].apply(lambda x: get_nstd(x))
# print(DF)
# print(data_json[-1].keys())
# print(json.loads(data_json[0]['snapShot']))

DB = pd.DataFrame()
DB['Pair'] = DF['(A,B)'].copy()
DB['model'] = DF['model'].copy()
DB['ret'] = DF['ret'].copy()
DB['profit'] = DF['profit'].copy()
DB['period'] = DF['period'].copy()
DB['nstd'] = DF['nstd'].copy()
# DB['cost'] = DF['cost'].copy()
# print(DB['ret'])

def time_filter():
    '''比較各時間抓個時間最後一筆，並確認其是否出場，歸類已實現及未實現
        如果進場就未實現，再抓tick的data算return
        如果已出場就已實現
    '''
    pass



# 1.2
data = pd.DataFrame()
data['總資金'] = [10000] # 應該有地方可以拿 # 寫html的時候寫
data['總曝險'] = [round(db.shape[0]*db['cost'][0],2)]
data['目前總報酬'] = [round(db['ret'].sum(),2)]
data['目前總獲利'] = [round(db['profit'].sum(),2)]
data['已實現報酬'] = [round(DB['ret'].sum(),2)]
data['已實現獲利'] = [round(DB['profit'].sum(),2)]
# print(DB['ret'])
# print(DB['profit'])
# print(data['已實現獲利'])

# 2.1 
realized_df = pd.DataFrame()
realized_df['Cumulative Return'] = DB['ret'].copy()

# 2.2
unrealized_df = pd.DataFrame()
unrealized_df['Cumulative Profits'] = db['profit'].copy()



'''目前開倉部位'''
fig = make_subplots(rows = 2, cols = 2, 
            specs=[[{"type": "table", "colspan": 1}, {"type": "table", "colspan": 1}],
                   [{"type": "scatter", "colspan": 1}, {"type": "scatter", "colspan": 1}],],
            subplot_titles = ("目前開倉", "狀態", "已實現報酬","未實現報酬"))

fig.add_trace(go.Table(
    header = dict(values = list(round_db.columns),
                fill_color='paleturquoise',
                align='left'),
    cells = dict(values = round_db.transpose().values.tolist(),
               fill_color='lavender',
               align='left')),1,1)

fig.add_trace(go.Table(
    header = dict(values=['名稱','數值'],
                fill_color='paleturquoise',
                align='left'),
    cells = dict(values=[list(data.columns),data.transpose().values.tolist()],
               fill_color='lavender',
               align='left')),1,2)

fig.add_trace(go.Scatter(x = df['exit_time'], y = realized_df['Cumulative Return'],
                    mode='lines+markers',
                    name = '累積報酬',
                    showlegend = False)
            , 2, 1)

fig.add_trace(go.Scatter(x = df['exit_time'], y = unrealized_df['Cumulative Profits'],
                    mode='lines+markers',
                    name = '累積報酬',
                    showlegend = False)
            , 2, 2)

fig.show()

