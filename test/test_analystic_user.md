# Report on portfolio

## 1.Quantitative Finance' Figure 14.1 price plot

### screenshot on the website (AAPL from 2022-9-23 to 2023-4-22)

## ![Figure14.1](attachment:%E6%88%AA%E5%B1%8F2023-04-21%2010.11.20.png)

### Function for the Figure


```python
def close_data(dates,js):
    chart_data = {}
    chart_close = []
    for dat in dates:
        chart_close.append({'x':time.strftime("%Y-%m-%d",time.localtime(dat/1000)),'value':'{:.4f}'.format(js['Adj Close'][str(dat)])})
    return chart_close
```

### Test for the Function


```python
import unittest
import time

class TestCloseData(unittest.TestCase):

    def test_close_data(self):
        # sample input data
        dates = [1641868800000, 1641955200000, 1642041600000]
        js = {'Adj Close': {'1641868800000': 123.45, '1641955200000': 234.56, '1642041600000': 345.67}}

        # expected output
        expected_output = [
            {'x': '2022-01-10', 'value': '123.4500'},
            {'x': '2022-01-11', 'value': '234.5600'},
            {'x': '2022-01-12', 'value': '345.6700'}
        ]

        # call the function
        output = close_data(dates, js)

        # assert that the output matches the expected output
        self.assertEqual(len(output), 3,"1")
        for i in range(3):
            self.assertDictEqual(output[i],expected_output[i],"not ")

res = unittest.main(argv=[''], verbosity=3, exit=False)
```

    test_close_data (__main__.TestCloseData) ... ok
    
    ----------------------------------------------------------------------
    Ran 1 test in 0.007s
    
    OK


# 2.'Quantitative Finance' Figure 14.2 return plot

### Screenshot on the website (AAPL from 2022-9-23 to 2023-4-22)

![%E6%88%AA%E5%B1%8F2023-04-21%2011.57.02.png](attachment:%E6%88%AA%E5%B1%8F2023-04-21%2011.57.02.png)

### Function for this Figure


```python
def daily_return(dates,js):
    chart_compairson = []
    for i,dat in enumerate(dates,0):
        if i==0:
            continue
        chart_compairson.append([time.strftime("%Y-%m-%d",time.localtime(dat/1000)),'{:.4f}'.format((js['Adj Close'][str(dat)]-js['Adj Close'][str(dates[i-1])])/js['Adj Close'][str(dates[i-1])])])
    return chart_compairson
```

### Test for the Function


```python
import unittest
import time

class TestReturnData(unittest.TestCase):

    def test_return_data(self):
        # sample input data
        dates = [1641868800000, 1641955200000, 1642041600000]
        js = {'Adj Close': {'1641868800000': 123.45, '1641955200000': 234.56, '1642041600000': 345.67}}

        # expected output
        expected_output = [['2022-01-11', '0.9000'], ['2022-01-12', '0.4737']]

        # call the function
        output = daily_return(dates, js)

        # assert that the output matches the expected output
        self.assertEqual(len(output), 2,"1")
        for i in range(2):
            self.assertListEqual(output[i],expected_output[i])

unittest.main(argv=[''], exit=False)
```

    ..
    ----------------------------------------------------------------------
    Ran 2 tests in 0.002s
    
    OK





    <unittest.main.TestProgram at 0x7fa9e0729e80>



## 3.'Quantitative Finance' Figure 14.3 return plot

### Screenshot on the website (AAPL from 2022-9-23 to 2023-4-22)

![%E6%88%AA%E5%B1%8F2023-04-22%2010.07.22.png](attachment:%E6%88%AA%E5%B1%8F2023-04-22%2010.07.22.png)

### Function for the Figure


```python
def comparison(dates,js):
    chart_data = {}
    chart_compairson = []
    chart_daily_charge=[]
    drr = []
    for i,dat in enumerate(dates,0):
        if i==0:
            continue
        cc = []
        cc.append(time.strftime("%Y-%m-%d",time.localtime(dat/1000)))
        drr.append((js['Adj Close'][str(dat)]-js['Adj Close'][str(dates[i-1])])/js['Adj Close'][str(dates[i-1])])
        cc.append('{:.4f}'.format((js['Adj Close'][str(dat)]-js['Adj Close'][str(dates[i-1])])/js['Adj Close'][str(dates[i-1])]))
        if i<=1:
            cc.append(0)
        else:
            if drr[i-2]==0:
                cc.append('{:.4f}'.format(drr[i-1]-drr[i-2]))
            else:
                cc.append('{:.4f}'.format((drr[i-1]-drr[i-2])/drr[i-2]))
        chart_daily_charge.append([cc[0],cc[2]])

        chart_compairson.append(cc)
    return chart_compairson
```

### Test for the Function


```python
import unittest

class TestComparison(unittest.TestCase):
    def setUp(self):
        self.dates = [1620169200000, 1620255600000, 1620342000000, 1620428400000]
        self.js = {
            'Adj Close': {
                '1620169200000': 10,
                '1620255600000': 15,
                '1620342000000': 20,
                '1620428400000': 25
            }
        }

    def test_comparison(self):
        expected_output = [
            ['2021-05-05', '0.5000', 0],
            ['2021-05-06', '0.3333', '-0.3333'],
            ['2021-05-07', '0.2500', '-0.2500']
        ]
        self.assertEqual(comparison(self.dates, self.js), expected_output)

unittest.main(argv=[''], exit=False)
```

    ...
    ----------------------------------------------------------------------
    Ran 3 tests in 0.006s
    
    OK





    <unittest.main.TestProgram at 0x7faa00ebdd68>



## 4.'Quantitative Finance' Figure 14.4 return plot

### Screenshot on the website (AAPL from 2022-9-23 to 2023-4-22)

![%E6%88%AA%E5%B1%8F2023-04-22%2010.44.15.png](attachment:%E6%88%AA%E5%B1%8F2023-04-22%2010.44.15.png)

### Function for the Figure


```python
from itertools import groupby
def histogram(dates,js):
    chart_histogram = []
    step = 0.001
    for i,dat in enumerate(dates,0):
        if i==0:
            continue
        chart_histogram.append((js['Adj Close'][str(dat)] - js['Adj Close'][str(dates[i-1])]) / js['Adj Close'][str(dates[i-1])])

    datas = []
    for k, g in groupby(sorted(chart_histogram), key=lambda x: x // step):
        datas.append(['{}-{}'.format(k * step, (k + 1) * step + 1), len(list(g))])
    chart_histogram = datas
    return chart_histogram
```

### Test for the Function


```python
import unittest

class TestHistogram(unittest.TestCase):
    
    def setUp(self):
        # Define some example data to use in the tests
        self.dates = [1, 2, 3, 4, 5]
        self.js = {'Adj Close': {'1': 10, '2': 20, '3': 30, '4': 25, '5': 35}}
        
    def test_histogram(self):
        # Define the expected output
        expected_output = [['-0.167-0.834', 1], ['0.4-1.401', 1], ['0.499-1.5', 1], ['0.999-2.0', 1]]
        
        # Call the function with the example data
        output = histogram(self.dates, self.js)
        
        # Check that the output matches the expected output
        self.assertEqual(output, expected_output)

unittest.main(argv=[''], exit=False)
```

    ....
    ----------------------------------------------------------------------
    Ran 4 tests in 0.009s
    
    OK





    <unittest.main.TestProgram at 0x7faa00ebd6d8>



## 5.'What Hedge Funds Do', Figure 7.1 stock price versus index price chart

### Screenshot on the website (AAPL and Index from 2022-9-23 to 2023-4-22)

![%E6%88%AA%E5%B1%8F2023-04-22%2010.50.28.png](attachment:%E6%88%AA%E5%B1%8F2023-04-22%2010.50.28.png)

### Function for the Figure


```python
def close_vs(dates,symboljs,sp500js):
    adj=[]
    adj1 = []
    adj2 = []
    #chat data adj close
    for dat in dates:
        adj1.append({'x':time.strftime("%Y-%m-%d", time.localtime(dat / 1000)),
                     'value':'{:.4f}'.format((symboljs['Adj Close'][str(dat)])),})
        adj2.append({'x':time.strftime("%Y-%m-%d", time.localtime(dat / 1000)),
                     'value': (sp500js['Adj Close'][str(dat)]),})
    adj.append(adj1)
    adj.append(adj2)
    return adj
```

### Test for the Function


```python
import unittest

class TestCloseVs(unittest.TestCase):
    def test_close_vs(self):
        # Define sample input data
        dates = [1618606800000, 1618693200000, 1618779600000]
        symboljs = {'Adj Close': {'1618606800000': 134.16, '1618693200000': 133.35, '1618779600000': 131.94}}
        sp500js = {'Adj Close': {'1618606800000': 4180.17, '1618693200000': 4163.29, '1618779600000': 4167.59}}

        # Define expected output
        expected_output = [[{'x': '2021-04-16', 'value': '134.1600'}, 
                            {'x': '2021-04-17', 'value': '133.3500'}, 
                            {'x': '2021-04-18', 'value': '131.9400'}], 
                           [{'x': '2021-04-16', 'value': 4180.17}, 
                            {'x': '2021-04-17', 'value': 4163.29}, 
                            {'x': '2021-04-18', 'value': 4167.59}]]

        # Call the function and assert output
        self.assertEqual(close_vs(dates, symboljs, sp500js), expected_output)


unittest.main(argv=[''], exit=False)
```

    .....
    ----------------------------------------------------------------------
    Ran 5 tests in 0.011s
    
    OK





    <unittest.main.TestProgram at 0x7fa9f0829400>



## 6.'What Hedge Funds Do', Figure 7.2 stock return versus index return time series chart

### Screenshot on the website (AAPL and Index from 2022-9-23 to 2023-4-22)

![%E6%88%AA%E5%B1%8F2023-04-22%2011.05.39.png](attachment:%E6%88%AA%E5%B1%8F2023-04-22%2011.05.39.png)

### Function for the Figure


```python
def return_vs(dates,symboljs,sp500js):
    daily = []
    daily1 = []
    daily2 = []
    #chat data daily return
    for i,dat in enumerate(dates,0):
        if i==0:
            continue
        daily1.append({'x': time.strftime("%Y-%m-%d", time.localtime(dat / 1000)),
                     'value': '{:.4f}'.format((symboljs['Adj Close'][str(dat)] - symboljs['Adj Close'][str(dates[i-1])]) / symboljs['Adj Close'][str(dates[i-1])]),})
        daily2.append({'x': time.strftime("%Y-%m-%d", time.localtime(dat / 1000)),
                     'value': '{:.4f}'.format((sp500js['Adj Close'][str(dat)] - sp500js['Adj Close'][str(dates[i-1])]) / sp500js['Adj Close'][str(dates[i-1])]),})
    daily.append(daily1)
    daily.append(daily2)
    return daily
```

### Test for the Function


```python
import unittest
import json
from datetime import datetime

class TestReturnVs(unittest.TestCase):
    def test_return_vs(self):
        dates = [1630425600000, 1630512000000, 1630771200000, 1630857600000]
        symboljs = json.loads('{"Adj Close": {"1630425600000": 100.0, "1630512000000": 102.0, "1630771200000": 105.0, "1630857600000": 107.0}}')
        sp500js = json.loads('{"Adj Close": {"1630425600000": 4500.0, "1630512000000": 4550.0, "1630771200000": 4600.0, "1630857600000": 4650.0}}')
        
        expected_daily = [[{'x': '2021-09-01', 'value': '0.0200'},
                          {'x': '2021-09-04', 'value': '0.0294'},
                          {'x': '2021-09-05', 'value': '0.0190'}],
                         [{'x': '2021-09-01', 'value': '0.0111'},
                          {'x': '2021-09-04', 'value': '0.0110'},
                          {'x': '2021-09-05', 'value': '0.0109'}]]
        
        self.assertEqual(return_vs(dates, symboljs, sp500js), expected_daily)

unittest.main(argv=[''], exit=False)
```

    ......
    ----------------------------------------------------------------------
    Ran 6 tests in 0.012s
    
    OK





    <unittest.main.TestProgram at 0x7faa00ecffd0>



## 7.'What Hedge Funds Do', Figure 7.3 scatter plot of stock return versus index return

### Screenshot on the website (AAPL and Index from 2022-9-23 to 2023-4-22)

![%E6%88%AA%E5%B1%8F2023-04-22%2011.20.40.png](attachment:%E6%88%AA%E5%B1%8F2023-04-22%2011.20.40.png)

### Function for the Figure


```python
def correlation(dates,symboljs,sp500js):
    chart_sandian=[]
    for i,dat in enumerate(dates,0):
        if i==0:
            continue
        index_daily_return = '{:.4f}'.format((sp500js['Adj Close'][str(dat)]-sp500js['Adj Close'][str(dates[i-1])])/sp500js['Adj Close'][str(dates[i-1])])
        stock_daily_return= '{:.4f}'.format( (symboljs['Adj Close'][str(dat)]-symboljs['Adj Close'][str(dates[i-1])])/symboljs['Adj Close'][str(dates[i-1])])
        chart_sandian.append([ index_daily_return, stock_daily_return ])
    return chart_sandian
```

### Test for the Function


```python
import unittest

class TestCorrelation(unittest.TestCase):
    
    def test_correlation(self):
        # mock data
        dates = ['2022-01-01', '2022-01-02', '2022-01-03', '2022-01-04', '2022-01-05']
        symboljs = {'Adj Close': {'2022-01-01': 100, '2022-01-02': 110, '2022-01-03': 105, '2022-01-04': 115, '2022-01-05': 120}}
        sp500js = {'Adj Close': {'2022-01-01': 200, '2022-01-02': 210, '2022-01-03': 205, '2022-01-04': 215, '2022-01-05': 220}}
        
        # expected output
        expected_output = [['0.0500', '0.1000'],
                            ['-0.0238', '-0.0455'],
                             ['0.0488', '0.0952'],
                             ['0.0233', '0.0435']]
        
        # test the function
        result = correlation(dates, symboljs, sp500js)
        self.assertEqual(result, expected_output)

unittest.main(argv=[''], exit=False)

```

    .......
    ----------------------------------------------------------------------
    Ran 7 tests in 0.013s
    
    OK





    <unittest.main.TestProgram at 0x7faa00ebdc88>



## 8.efficient frontier plot for the portfolio

### screenshot on the website for the situation

![%E6%88%AA%E5%B1%8F2023-04-22%2011.46.35.png](attachment:%E6%88%AA%E5%B1%8F2023-04-22%2011.46.35.png)

![%E6%88%AA%E5%B1%8F2023-04-22%2011.29.09.png](attachment:%E6%88%AA%E5%B1%8F2023-04-22%2011.29.09.png)

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

    ..../Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages/pandas_datareader/data.py:431: ResourceWarning: unclosed <ssl.SSLSocket fd=79, family=AddressFamily.AF_INET, type=SocketKind.SOCK_STREAM, proto=0, laddr=('192.168.1.6', 51689), raddr=('78.47.75.66', 443)>
      session=session,
    ....
    ----------------------------------------------------------------------
    Ran 8 tests in 25.578s
    
    OK





    <unittest.main.TestProgram at 0x7fa9f1144be0>



## 9. Porfolio return vs Index

### Screenshot on the website (Portfolio and Index from 2023-01-01 to 2023-4-23)

![%E6%88%AA%E5%B1%8F2023-04-23%2021.56.34.png](attachment:%E6%88%AA%E5%B1%8F2023-04-23%2021.56.34.png)

### Function for the Figure


```python
import yfinance as yf
import json
import time
from datetime import datetime
def portfolioreturn(start,end,stocks):
    stock_data_arr = {}
    ss = {}
    sts = set()
    for stock in stocks:
        sts.add(stock[0])
        ss[stock[0]] = stock[1]
    for st in stocks:
        d = yf.download(st[0],start=start,end=end)
        stockjson = d.to_json()
        stockjson = json.loads(stockjson)
        stock_data_arr[st[0]] = stockjson

    d = yf.download('^GSPC',start=start,end=end)
    sp500js = d.to_json()
    sp500js = json.loads(sp500js)

    dates = set()
    for c in sp500js['Adj Close']:
        dates.add(int(c))
    dates = sorted(dates)

    def sumPrice(column,dat):
        sum = 0
        for stock_data in stock_data_arr:
            dd = stock_data_arr[stock_data]
            sum+=dd[column][str(dat)]*ss[stock_data]
        return sum

    datas = []
    for i,dat in enumerate(dates,0):
        if i ==0:
            continue
        portfolio_return = (sumPrice('Adj Close',dat)-sumPrice('Adj Close',dates[i-1]))/sumPrice('Adj Close',dates[i-1])
        index_return = (sp500js['Adj Close'][str(dat)]-sp500js['Adj Close'][str(dates[i-1])])/sp500js['Adj Close'][str(dates[i-1])]
        datas.append([datetime.utcfromtimestamp(dat/1000).strftime('%Y-%m-%d'),'{:.4f}'.format(portfolio_return),'{:.4f}'.format(index_return),'{:.4f}'.format(index_return-portfolio_return)])
    return datas
```

### Test for the Function


```python
import unittest
import yfinance as yf
import json

class TestPortfolioReturn(unittest.TestCase):
    
    def setUp(self):
        self.start = '2022-01-01'
        self.end = '2022-01-10'
        self.stocks = [('AAPL', 0.5), ('TSLA', 0.5)]

    def test_portfolioreturn(self):
        result = portfolioreturn(self.start, self.end, self.stocks)
        self.assertIsInstance(result, list)
        self.assertIsInstance(result[0], list)
        self.assertEqual(len(result[0]), 4)
        self.assertIsInstance(result[0][0], str)
        self.assertIsInstance(result[0][1], str)
        self.assertIsInstance(result[0][2], str)
        self.assertIsInstance(result[0][3], str)


unittest.main(argv=[''], exit=False)
```

    ..../Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages/pandas_datareader/data.py:431: ResourceWarning: unclosed <ssl.SSLSocket fd=80, family=AddressFamily.AF_INET, type=SocketKind.SOCK_STREAM, proto=0, laddr=('172.30.3.119', 51767), raddr=('78.47.75.66', 443)>
      session=session,
    ..

    [*********************100%***********************]  1 of 1 completed
    [*********************100%***********************]  1 of 1 completed
    [*********************100%***********************]  1 of 1 completed

    ...

    


    
    ----------------------------------------------------------------------
    Ran 9 tests in 27.989s
    
    OK





    <unittest.main.TestProgram at 0x7faa01c4e6d8>


