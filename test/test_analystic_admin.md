# Report for the analystic (Admin branch)

## 1. Stock overiview

## Screenshot on the Website

![%E6%88%AA%E5%B1%8F2023-04-24%2001.35.50.png](attachment:%E6%88%AA%E5%B1%8F2023-04-24%2001.35.50.png)

### Function for the Figure


```python
import time
from datetime import datetime
import datetime as dt
import yfinance as yf
def get_latest_date(ticker):
    today = datetime.strptime("2022-09-23", "%Y-%m-%d")
    while True:
        data = yf.download(ticker, start=today)
        if today.strftime("%Y-%m-%d") not in data.index:
            today = today - dt.timedelta(days=1)
        else:
            break
    return today

def get_yesterday_date(ticker,dates):
    yesterday = datetime.strptime(dates, "%Y-%m-%d") - dt.timedelta(days=1)
    while True:
        data = yf.download(ticker, start=yesterday, end=dates)
        if yesterday.strftime("%Y-%m-%d") not in data.index:
            yesterday = yesterday - dt.timedelta(days=1)
        else:
            break
    return yesterday

def stockoverview(stocks):
    data = []
    for stock in stocks:
        ticker = stock[0]
        today = get_latest_date(ticker)
        yesterday = get_yesterday_date(ticker, today.strftime("%Y-%m-%d"))
        today_price = round(yf.download(ticker,start=today).loc[today.strftime("%Y-%m-%d"),'Adj Close'],4)
        yesterday_price = round(yf.download(ticker,start=yesterday).loc[yesterday,'Adj Close'],4)
        data.append([ticker,today_price, str(round((today_price-yesterday_price)/yesterday_price,6)*100)+"%", stock[1], round(today_price*stock[1],6)])
    return data
```

### Test for the Function


```python
import unittest

class TestStockOverview(unittest.TestCase):
    
    def test_stockoverview(self):
        stocks = [["AAPL", 10], ["MSFT", 20]]
        result = stockoverview(stocks)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0][0], "AAPL")
        self.assertIsInstance(result[0][1], float)
        self.assertIsInstance(result[0][2], str)
        self.assertEqual(result[0][3], 10)
        self.assertIsInstance(result[0][4], float)
unittest.main(argv=[''], exit=False)
```

    [*********************100%***********************]  1 of 1 completed
    [*********************100%***********************]  1 of 1 completed
    [*********************100%***********************]  1 of 1 completed
    [*********************100%***********************]  1 of 1 completed
    [*********************100%***********************]  1 of 1 completed
    [*********************100%***********************]  1 of 1 completed
    [*********************100%***********************]  1 of 1 completed
    [*********************100%***********************]  1 of 1 completed

    .

    


    
    ----------------------------------------------------------------------
    Ran 1 test in 2.491s
    
    OK





    <unittest.main.TestProgram at 0x7fe22894d240>



# 2.transactionrecord

## Screenshot on the Website

![%E6%88%AA%E5%B1%8F2023-04-24%2001.36.52.png](attachment:%E6%88%AA%E5%B1%8F2023-04-24%2001.36.52.png)

### Function for the Figure


```python
def transactionrecord(raw_data):
    data = {}
    for row in raw_data:
        if row[2]==0:
            data[datetime.strftime(row[0], "%Y-%m-%d")+row[1]] = [row[0],row[1],row[3],0]
    for row in raw_data:
        if row[2]==1:
            if datetime.strftime(row[0], "%Y-%m-%d")+row[1] in data.keys():
                data[datetime.strftime(row[0], "%Y-%m-%d")+row[1]][3] = -row[3]
            else:
                data[datetime.strftime(row[0], "%Y-%m-%d")+row[1]] =[row[0],row[1],0, -row[3]]
    return list(data.values())
```

### Test for the Function


```python
import unittest
from datetime import datetime

class TestTransactionRecord(unittest.TestCase):
    
    def setUp(self):
        self.raw_data = [
            [datetime(2023, 4, 22), 'ABC', 0, 1000],
            [datetime(2023, 4, 22), 'XYZ', 1, 500],
            [datetime(2023, 4, 23), 'DEF', 0, 2000],
            [datetime(2023, 4, 23), 'GHI', 1, 1000]
        ]
    
    def test_transactionrecord(self):
        expected_output = [
            [datetime(2023, 4, 22), 'ABC', 1000, 0],
            [datetime(2023, 4, 23), 'DEF', 2000, 0],
            [datetime(2023, 4, 22), 'XYZ', 0, -500],
            [datetime(2023, 4, 23), 'GHI', 0, -1000]
        ]
        self.assertEqual(transactionrecord(self.raw_data), expected_output)
        
unittest.main(argv=[''], exit=False)
```

    [*********************100%***********************]  1 of 1 completed
    [*********************100%***********************]  1 of 1 completed
    [*********************100%***********************]  1 of 1 completed
    [*********************100%***********************]  1 of 1 completed
    [*********************100%***********************]  1 of 1 completed
    [*********************100%***********************]  1 of 1 completed
    [*********************100%***********************]  1 of 1 completed
    [*********************100%***********************]  1 of 1 completed

    ..

    


    
    ----------------------------------------------------------------------
    Ran 2 tests in 2.434s
    
    OK





    <unittest.main.TestProgram at 0x7fe220a819e8>



## 3. Risk overview

## Screenshot on the Website

![%E6%88%AA%E5%B1%8F2023-04-24%2001.47.22.png](attachment:%E6%88%AA%E5%B1%8F2023-04-24%2001.47.22.png)

### Function for the Figure


```python
import time

import yfinance as yf
import numpy as np
import scipy.optimize as opt
import pandas_datareader.data as web

import matplotlib.pyplot as plt
def get_summary(stocks,start,end,srw,mock_weight):
    result = {}
    tickers = {}
    for sk in stocks:
        tickers[sk] = ''

    stock_data = web.DataReader(stocks, 'stooq', '2016-01-01', '2017-12-31')["Close"]
    stock_data.rename(columns=tickers, inplace=True)
    stock_data = stock_data.iloc[::-1]
    stock_data.head()
    R = stock_data / stock_data.shift(1) - 1
    R.head()
    log_r = np.log(stock_data / stock_data.shift(1))
    log_r.head()

    # 年化收益率
    r_annual = np.exp(log_r.mean() * 250) - 1
    # 风险
    std = np.sqrt(log_r.var() * 250)  # 假设协方差为0

    # 投资组合的收益和风险
    def gen_weights(n):
        w = np.random.rand(n)
        return w / sum(w)

    n = len(list(tickers))
    w = gen_weights(n)

    list(zip(r_annual.index, w))

    # 投资组合收益
    def port_ret(w):
        return np.sum(w * r_annual)

    port_ret(w)

    # 投资组合的风险
    def port_std(w):
        return np.sqrt((w.dot(log_r.cov() * 250).dot(w.T)))

    port_std(w)

    # 若干投资组合的收益和风险
    def gen_ports(times):
        for _ in range(times):  # 生成不同的组合
            w = gen_weights(n)  # 每次生成不同的权重
            yield (port_std(w), port_ret(w), w)  # 计算风险和期望收益 以及组合的权重情况

    import pandas as pd
    df = pd.DataFrame(gen_ports(25000), columns=["std", "ret", "w"])
    df.head()

    selected = {'fill': "#64b5f6", 'stroke': "#64b5f6"}
    normal = {'fill': "#64b5f6", 'stroke': "#64b5f6"}
    hovered = {'fill': "#64b5f6", 'stroke': "#64b5f6"}

    chartdata = []
    ret_list = list(df['ret'])
    std_list = list(df['std'])
    for i in range(len(ret_list)):
        chartdata.append({'x': std_list[i],
                          'value':ret_list[i],
                          'normal': normal,
                          'hovered': hovered,
                          'selected': selected})


    df['sharpe'] = (df['ret'] - 0.015) / df['std']  # 定义夏普比率
    list(zip(r_annual.index, df.loc[df.sharpe.idxmax()].w))


    res = opt.minimize(lambda x: -((port_ret(x) - 0.03) / port_std(x)),
                       x0=((1 / n),) * n,
                       method='SLSQP',
                       bounds=((0, 1),) * n,
                       constraints={"fun": lambda x: (np.sum(x) - 1), "type": "eq"})
    result['zy_weight'] = list(res.x.round(3))
    result['zy_point'] = [port_std(res.x),port_ret(res.x)]
    result['market_line'] = [[0, .03], [.27, -res.fun * .27 + .03]]
    result['max_sharpe_ratio'] = (-res.fun * .27 + .03-.03)/.27
    result['return_max_sharp'] = port_ret(res.x)
    result['chartdata'] = chartdata
    srmm = (port_ret(np.array(srw))-0.03)/ port_std(np.array(srw))
    result['sharpe_raion_now'] = srmm
    result['sharpe_point_now'] = [{"x": port_std(np.array(srw)), "value": port_ret(np.array(srw)),
      'normal': {'fill': "#000000", 'stroke': "#000000"},
      'hovered': {'fill': "#000000", 'stroke': "#000000"},
      'selected': {'fill': "#000000", 'stroke': "#000000"},
      'size': 5}]
    
    result['sharpe_raion_mock'] = (port_ret(np.array(mock_weight))-0.03)/ port_std(np.array(mock_weight)),
    result['sharpe_point_mock'] = [{"x": port_std(np.array(mock_weight)), "value": port_ret(np.array(mock_weight)),
      'normal': {'fill': "#9900FF", 'stroke': "#9900FF"},
      'hovered': {'fill': "#9900FF", 'stroke': "#9900FF"},
      'selected': {'fill': "#9900FF", 'stroke': "#9900FF"},
      'size': 5}]

    slist = sorted(std_list)

    result['zc_point'] = [slist[0], ret_list[std_list.index(slist[0])]]
    return result


```

### Test for the Function


```python
import unittest

class TestGetSummary(unittest.TestCase):
    
    def test_get_summary(self):
        # Set up test inputs
        stocks = ['AAPL', 'GOOGL', 'TSLA']
        start = '2016-01-01'
        end = '2017-12-31'
        srw = [0.25, 0.25, 0.5]
        mock_weight = [0.2, 0.3, 0.5]
        
        # Call the function and get the result
        result = get_summary(stocks, start, end, srw, mock_weight)
        
        # Check that the result is a dictionary
        self.assertIsInstance(result, dict)
        
        # Check that all the required keys are present in the result dictionary
        required_keys = ['zy_weight', 'zy_point', 'market_line', 'max_sharpe_ratio', 'return_max_sharp', 'chartdata', 
                         'sharpe_raion_now', 'sharpe_point_now', 'sharpe_raion_mock', 'sharpe_point_mock', 'zc_point']
        for key in required_keys:
            self.assertIn(key, result.keys())
        
        # Check that the 'chartdata' key in the result dictionary is a list of dictionaries
        self.assertIsInstance(result['chartdata'], list)
        for item in result['chartdata']:
            self.assertIsInstance(item, dict)
            self.assertIn('x', item.keys())
            self.assertIn('value', item.keys())
            self.assertIn('normal', item.keys())
            self.assertIn('hovered', item.keys())
            self.assertIn('selected', item.keys())
        
        # Check that the 'sharpe_raion_now' key in the result dictionary is a float or an int
        self.assertTrue(isinstance(result['sharpe_raion_now'], (float, int)))
        
        # Check that the 'sharpe_point_now' key in the result dictionary is a list of dictionaries with one item
        self.assertIsInstance(result['sharpe_point_now'], list)
        self.assertEqual(len(result['sharpe_point_now']), 1)
        self.assertIsInstance(result['sharpe_point_now'][0], dict)
        self.assertIn('x', result['sharpe_point_now'][0].keys())
        self.assertIn('value', result['sharpe_point_now'][0].keys())
        self.assertIn('normal', result['sharpe_point_now'][0].keys())
        self.assertIn('hovered', result['sharpe_point_now'][0].keys())
        self.assertIn('selected', result['sharpe_point_now'][0].keys())
        self.assertIn('size', result['sharpe_point_now'][0].keys())
        
        # Check that the 'sharpe_raion_mock' key in the result dictionary is a tuple with one item, which is a float or an int
        self.assertIsInstance(result['sharpe_raion_mock'], tuple)
        self.assertEqual(len(result['sharpe_raion_mock']), 1)
        self.assertTrue(isinstance(result['sharpe_raion_mock'][0], (float, int)))
        
        # Check that the 'sharpe_point_mock' key in the result dictionary is a list of dictionaries with one item
        self.assertIsInstance(result['sharpe_point_mock'], list)
        self.assertEqual(len(result['sharpe_point_mock']), 1)
        self.assertIsInstance(result['sharpe_point_mock'][0], dict)
        self.assertIn('value', result['sharpe_point_mock'][0].keys())
        self.assertIn('normal', result['sharpe_point_mock'][0].keys())
        self.assertIn('hovered', result['sharpe_point_mock'][0].keys())
        self.assertIn('selected', result['sharpe_point_mock'][0].keys())
        self.assertIn('size', result['sharpe_point_mock'][0].keys())
        # Check that the 'zc_point' key in the result dictionary is a list with two items, both of which are numbers
        self.assertIsInstance(result['zc_point'], list)
        self.assertEqual(len(result['zc_point']), 2)
        self.assertTrue(isinstance(result['zc_point'][0], (float, int)))
        self.assertTrue(isinstance(result['zc_point'][1], (float, int)))
    
unittest.main(argv=[''], exit=False)
```

    /Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages/pandas_datareader/data.py:431: ResourceWarning: unclosed <ssl.SSLSocket fd=80, family=AddressFamily.AF_INET, type=SocketKind.SOCK_STREAM, proto=0, laddr=('192.168.1.6', 57350), raddr=('78.47.75.66', 443)>
      session=session,
    .

    [*********************100%***********************]  1 of 1 completed
    [*********************100%***********************]  1 of 1 completed
    [*********************100%***********************]  1 of 1 completed
    [*********************100%***********************]  1 of 1 completed
    [*********************100%***********************]  1 of 1 completed
    [*********************100%***********************]  1 of 1 completed
    [*********************100%***********************]  1 of 1 completed
    [*********************100%***********************]  1 of 1 completed

    ..

    


    
    ----------------------------------------------------------------------
    Ran 3 tests in 32.057s
    
    OK





    <unittest.main.TestProgram at 0x7fe218d84da0>


