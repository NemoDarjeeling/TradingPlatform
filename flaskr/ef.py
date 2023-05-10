import time

import yfinance as yf
import json
import numpy as np
import scipy.optimize as opt
import pandas_datareader.data as web

import matplotlib.pyplot as plt
def get_summary(stocks,start,end,srw,mock_weight):
    result = {}
    tickers = {}
    for sk in stocks:
        tickers[sk] = ''
    ef_data = yf.download("^TNX", start, end)
    js = ef_data.to_json()
    js = json.loads(js)
    dates = set()
    for c in js["Adj Close"]:
        dates.add(int(c))
    dates = sorted(dates)
    ef=js["Adj Close"][str(dates[-1])]/100
    print(dates)
    print(ef)
    stock_data = yf.download(stocks,start,end)["Adj Close"]
    stock_data.rename(columns=tickers, inplace=True)
    stock_data = stock_data.iloc[::-1]
    stock_data.head()
    R = stock_data.shift(1)/ stock_data - 1
    R = R.drop(R.index[0])
   # log_r = np.log(stock_data.shift(1)/stock_data )
   # print(R.mean())


    # 年化收益率
    #r_annual = np.exp(log_r.mean() * 252) - 1
    r_annual = list((((R+1).cumprod().tail(1)** (1/R.shape[0]) - 1)*252).iloc[0])
    #print(r_annual)
    # 风险
    #std = np.sqrt(log_r.var() * 252)  # 假设协方差为0
    #std = std(r_annual)*np.sqrt(252)

    # 投资组合的收益和风险
    def gen_weights(n):
        w = np.random.rand(n)
        return w / sum(w)

    n = len(list(tickers))
    w = gen_weights(n)

    #list(zip(r_annual.index, w))

    # 投资组合收益
    def port_ret(w):
        return np.sum(w * r_annual)

    port_ret(w)

    # 投资组合的风险
    def port_std(w):
        return np.sqrt(np.dot(w, np.dot(R.cov()*252, w)))

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


    df['sharpe'] = (df['ret'] - ef) / df['std']  # 定义夏普比率
    #list(zip(r_annual.index, df.loc[df.sharpe.idxmax()].w))

    res = opt.minimize(lambda x: -((port_ret(x) - ef) / port_std(x)),
                       x0=((1 / n),) * n,
                       method='SLSQP',
                       bounds=((0, 1),) * n,
                       constraints=[{"fun": lambda x: (np.sum(x) - 1), "type": "eq"},
                       {'type': 'ineq', 'fun': lambda w: w}])
    print(list(res.x.round(3)))
    result['zy_weight'] = list(res.x.round(3))
    result['zy_point'] = [port_std(res.x),port_ret(res.x)]
    result['market_line'] = [[0, ef], [port_std(res.x), -res.fun * port_std(res.x) + ef]]
    result['max_sharpe_ratio'] = -res.fun
    result['return_max_sharp'] = port_ret(res.x)
    result['chartdata'] = chartdata
    srmm = (port_ret(np.array(srw))-ef)/ port_std(np.array(srw))
    print(type(srmm))
    result['sharpe_raion_now'] = srmm
    result['sharpe_point_now'] = [{"x": port_std(np.array(srw)), "value": port_ret(np.array(srw)),
      'normal': {'fill': "#000000", 'stroke': "#000000"},
      'hovered': {'fill': "#000000", 'stroke': "#000000"},
      'selected': {'fill': "#000000", 'stroke': "#000000"},
      'size': 5}]
    
    result['sharpe_raion_mock'] = (port_ret(np.array(mock_weight))-ef)/ port_std(np.array(mock_weight)),
    result['sharpe_point_mock'] = [{"x": port_std(np.array(mock_weight)), "value": port_ret(np.array(mock_weight)),
      'normal': {'fill': "#9900FF", 'stroke': "#9900FF"},
      'hovered': {'fill': "#9900FF", 'stroke': "#9900FF"},
      'selected': {'fill': "#9900FF", 'stroke': "#9900FF"},
      'size': 5}]
    print(result['sharpe_raion_now'])

    slist = sorted(std_list)

    result['zc_point'] = [slist[0], ret_list[std_list.index(slist[0])]]
    print(result['zc_point'])
    return result