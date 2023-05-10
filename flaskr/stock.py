import functools
import json
import time

from datetime import datetime
from flask import Blueprint
from flask import flash
from flask import g
from flask import redirect
from flask import render_template
from flask import request
from flask import session
from flask import url_for
from werkzeug.security import check_password_hash
from werkzeug.security import generate_password_hash

from flaskr.db import get_db
import yfinance as yf
from flaskr import auth
from datetime import date
import datetime as dt
from datetime import datetime

from flaskr.ef import get_summary

bp = Blueprint("stock", __name__, url_prefix="/stock")

def get_portfolio(userid, dates):
    db = get_db()
    portfolio = db.execute(
    "SELECT ticker, SUM(shares) FROM trade "
    "WHERE trade.userid = ? AND date(trade.dates) <= date(?) "
    "GROUP BY ticker",
    (userid, dates),
    )
    return portfolio

def get_yesterday_data(ticker,dates):
    yesterday = datetime.strptime(dates, "%Y-%m-%d") - dt.timedelta(days=1)
    while True:
        data = yf.download(ticker, start=yesterday, end=dates)
        if data.empty:
            yesterday = yesterday - dt.timedelta(days=1)
        else:
            break
    return yesterday

@auth.user_login_required
@bp.route("/detail", methods=("GET", "POST"))
def detail():
    start = "2023-01-01"
    end = time.strftime("%Y-%m-%d",time.localtime(time.time()))
    if request.method == "POST":
        date_range = request.form['daterange']
        start_date_str, end_date_str = date_range.split(" - ")
        start = datetime.strptime(start_date_str, "%m/%d/%Y").strftime('%Y-%m-%d')
        end = datetime.strptime(end_date_str, "%m/%d/%Y").strftime('%Y-%m-%d')
    symbol = request.args.get('symbol')
    d = yf.download(symbol, start=start, end=end)
    js = d.to_json()
    js = json.loads(js)

    dates = set()
    ths = []
    for item in js:
        ths.append(item)
        content = js[item]
        for c in content:
            dates.add(int(c))
    dates = sorted(dates)
    datas = []
    for i,dat in enumerate(dates,0):
        if i==0:
            continue
        lines = []
        lines.append(datetime.utcfromtimestamp(dat/1000).strftime('%Y-%m-%d'))
        for th in ths:
            lines.append('{:.4f}'.format(js[th][str(dat)]))

        lines.append('{:.4f}'.format((js['Adj Close'][str(dat)]-js['Adj Close'][str(dates[i-1])])/js['Adj Close'][str(dates[i-1])]))
        datas.append(lines)
    print(datas)

    #chart data
    chart_data = {}
    chart_close = []
    for dat in dates:
        chart_close.append({'x':datetime.utcfromtimestamp(dat/1000).strftime('%Y-%m-%d'),'value':'{:.4f}'.format(js['Adj Close'][str(dat)])})
    chart_data['close'] = chart_close

    chart_compairson = []
    for i,dat in enumerate(dates,0):
        if i==0:
            continue
        chart_compairson.append([datetime.utcfromtimestamp(dat/1000).strftime('%Y-%m-%d'),'{:.4f}'.format((js['Adj Close'][str(dat)]-js['Adj Close'][str(dates[i-1])])/js['Adj Close'][str(dates[i-1])])])
    chart_data['compairson'] = chart_compairson

    chart_data = json.dumps(chart_data)

    ths.insert(0,"Date")
    ths.append("Daily Return")

    start = datetime.strptime(start, '%Y-%m-%d').strftime('%m/%d/%Y')
    end = datetime.strptime(end, '%Y-%m-%d').strftime('%m/%d/%Y')
    return render_template("stock/detail.html" ,symbol=symbol, start=start,end=end, ths = ths, datas=datas,chart_data=chart_data)

from itertools import groupby

@auth.user_login_required
@bp.route("/comparison", methods=("GET", "POST"))
def comparison():
    start = "2023-01-01"
    end = time.strftime("%Y-%m-%d",time.localtime(time.time()))
    step = 0.001
    if request.method == "POST":
        date_range = request.form['daterange']
        start_date_str, end_date_str = date_range.split(" - ")
        start = datetime.strptime(start_date_str, "%m/%d/%Y").strftime('%Y-%m-%d')
        end = datetime.strptime(end_date_str, "%m/%d/%Y").strftime('%Y-%m-%d')
        step = float(request.form["step"])
    symbol = request.args.get('symbol')
    d = yf.download(symbol, start=start, end=end)
    js = d.to_json()
    js = json.loads(js)

    dates = set()
    ths = []
    for item in js:
        ths.append(item)
        content = js[item]
        for c in content:
            dates.add(int(c))
    dates = sorted(dates)

    #chart data
    chart_data = {}
    chart_compairson = []
    chart_daily_charge=[]
    drr = []
    for i,dat in enumerate(dates,0):
        if i==0:
            continue
        cc = []
        cc.append(datetime.utcfromtimestamp(dat/1000).strftime('%Y-%m-%d'))
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
    chart_histogram = []
    for i,dat in enumerate(dates,0):
        if i==0:
            continue
        chart_histogram.append((js['Adj Close'][str(dat)] - js['Adj Close'][str(dates[i-1])]) / js['Adj Close'][str(dates[i-1])])

    datas = []
    for k, g in groupby(sorted(chart_histogram), key=lambda x: x // step):
        datas.append(['{}-{}'.format(k * step, (k + 1) * step + 1), len(list(g))])
    chart_histogram = datas

    datas = []
    datas.extend(chart_compairson)
    chart_data['histogram']=chart_histogram
    chart_data['compairson']=chart_compairson
    chart_data['charge']=chart_daily_charge
    chart_data = json.dumps(chart_data)
    print(chart_data)
    ths = ['Date','Daily Return','Daily Change']
    start = datetime.strptime(start, '%Y-%m-%d').strftime('%m/%d/%Y')
    end = datetime.strptime(end, '%Y-%m-%d').strftime('%m/%d/%Y')
    return render_template("stock/comparison.html" ,symbol=symbol, start=start,end=end, datas = datas, ths = ths, chart_data=chart_data,step=step)




@auth.user_login_required
@bp.route("/market", methods=("GET", "POST"))
def market():
    start = "2023-01-01"
    end = time.strftime("%Y-%m-%d",time.localtime(time.time()))
    if request.method == "POST":
        date_range = request.form['daterange']
        start_date_str, end_date_str = date_range.split(" - ")
        start = datetime.strptime(start_date_str, "%m/%d/%Y").strftime('%Y-%m-%d')
        end = datetime.strptime(end_date_str, "%m/%d/%Y").strftime('%Y-%m-%d')

    symbol = request.args.get('symbol')
    d = yf.download('^GSPC',start=start,end=end)

    sp500js = d.to_json()
    sp500js = json.loads(sp500js)

    d = yf.download(symbol, start=start, end=end)
    symboljs = d.to_json()
    symboljs = json.loads(symboljs)
    dates = set()
    for item in symboljs:
        content = symboljs[item]
        for c in content:
            dates.add(int(c))
    dates = sorted(dates)


    tabledatas = []
    #table data
    for i,dat in enumerate(dates,0):
        if i==0:
            continue
        tabledatas.append([time.strftime("%Y-%m-%d", time.localtime(dat / 1000)),
                           '{:.4f}'.format(symboljs['Adj Close'][str(dat)]),'{:.4f}'.format(sp500js['Adj Close'][str(dat)]),
                           '{:.4f}'.format((symboljs['Adj Close'][str(dat)] - symboljs['Adj Close'][str(dates[i-1])]) / symboljs['Adj Close'][str(dates[i-1])]),
                           '{:.4f}'.format((sp500js['Adj Close'][str(dat)] - sp500js['Adj Close'][str(dates[i-1])]) / sp500js['Adj Close'][str(dates[i-1])])])

    chart1_normal = {'fill': "none",'stroke': "#9fa8da"}
    chart1_hovered={'fill': "none",'stroke': "#9fa8da"}
    chart1_selected={'fill':"none",'stroke': "#9fa8da"}
    chart2_normal = {'fill': "none",'stroke': "#40c4ff"}
    chart2_hovered={'fill': "none",'stroke': "#40c4ff"}
    chart2_selected={'fill': "none",'stroke': "#40c4ff"}
    adj=[]
    adj1 = []
    adj2 = []
    #chat data adj close
    for dat in dates:
        adj1.append({'x':time.strftime("%Y-%m-%d", time.localtime(dat / 1000)),
                     'value':'{:.4f}'.format((symboljs['Adj Close'][str(dat)])),
                     'normal':chart1_normal,
                     'hovered':chart1_hovered,
                     'selected':chart1_selected})
        adj2.append({'x':time.strftime("%Y-%m-%d", time.localtime(dat / 1000)),
                     'value': (sp500js['Adj Close'][str(dat)]),
                     'normal': chart2_normal,
                     'hovered': chart2_hovered,
                     'selected': chart2_selected})
    adj.append(adj1)
    adj.append(adj2)

    daily = []
    daily1 = []
    daily2 = []
    #chat data daily return
    for i,dat in enumerate(dates,0):
        if i==0:
            continue
        daily1.append({'x': time.strftime("%Y-%m-%d", time.localtime(dat / 1000)),
                     'value': '{:.4f}'.format((symboljs['Adj Close'][str(dat)] - symboljs['Adj Close'][str(dates[i-1])]) / symboljs['Adj Close'][
                         str(dates[i-1])]),
                     'normal': chart1_normal,
                     'hovered': chart1_hovered,
                     'selected': chart1_selected})
        daily2.append({'x': time.strftime("%Y-%m-%d", time.localtime(dat / 1000)),
                     'value': '{:.4f}'.format((sp500js['Adj Close'][str(dat)] - sp500js['Adj Close'][str(dates[i-1])]) / sp500js['Adj Close'][str(dates[i-1])]),
                     'normal': chart2_normal,
                     'hovered': chart2_hovered,
                     'selected': chart2_selected})
    daily.append(daily1)
    daily.append(daily2)
    chart_datas = []
    chart_datas.append(adj)
    chart_datas.append(daily)
    chart_sandian=[]
    for i,dat in enumerate(dates,0):
        if i==0:
            continue
        index_daily_return = '{:.4f}'.format((sp500js['Adj Close'][str(dat)]-sp500js['Adj Close'][str(dates[i-1])])/sp500js['Adj Close'][str(dates[i-1])])
        stock_daily_return= '{:.4f}'.format( (symboljs['Adj Close'][str(dat)]-symboljs['Adj Close'][str(dates[i-1])])/symboljs['Adj Close'][str(dates[i-1])])
        chart_sandian.append([ index_daily_return, stock_daily_return ])

    chart_datas.append(chart_sandian)
    chart_datas= json.dumps(chart_datas)
    ths = ['Date',symbol+' Close','Market Close',symbol+' Daily Return','Market Daily Return']
    start = datetime.strptime(start, '%Y-%m-%d').strftime('%m/%d/%Y')
    end = datetime.strptime(end, '%Y-%m-%d').strftime('%m/%d/%Y')
    return render_template("stock/market.html",symbol=symbol,start=start,end=end, ths=ths, datas=tabledatas,chart_datas=chart_datas)

import yfinance as yf
import numpy as np
import scipy.optimize as opt
import pandas_datareader.data as web

import matplotlib.pyplot as plt
@auth.user_login_required
@bp.route("/ef", methods=("GET", "POST"))
def ef():
    dates = "2023-01-01"
    sts = set()
    stocks = get_portfolio(g.user['id'], time.strftime("%Y-%m-%d",time.localtime(time.time()))).fetchall()
    ss = {}
    for stock in stocks:
        if stock[1] !=0 :
            sts.add(stock[0])
            ss[stock[0]] = stock[1]

    if(len(stocks)<2):
        nums = 2
        return render_template("stock/gobuy.html",nums=nums)
    density = 10000
    start = "2023-01-01"
    end = time.strftime("%Y-%m-%d",time.localtime(time.time()))
    stocks = list(sts)
    mock_weight = []
    for l in range(len(stocks)):
        mock_weight.append(round(1/len(stocks),4))
    mock_weight[0] = 1-(len(stocks)-1)*mock_weight[0]
    if request.method == "POST":
        date_range = request.form['daterange']
        start_date_str, end_date_str = date_range.split(" - ")
        start = datetime.strptime(start_date_str, "%m/%d/%Y").strftime('%Y-%m-%d')
        end = datetime.strptime(end_date_str, "%m/%d/%Y").strftime('%Y-%m-%d')
        density = int(request.form["density"])
        mock_weight = []
        for l in range(len(stocks)):
            mock_weight.append(float(request.form[stocks[l]]))
    chart_datas={}

    selected={'fill': "#0000ff",'stroke': "#0000ff"}
    normal={'fill': "#0000ff",'stroke': "#0000ff"}
    hovered={'fill': "#0000ff",'stroke': "#0000ff"}

    min_weight = 0.04
    max_weight = 0.9
    df = yf.download(stocks, start=start, end=end)
    df = np.log(1 + df['Adj Close'].pct_change())

    def portfolio_return(weights):
        return (1 + (np.dot(df.mean(), weights))) ** 253 - 1

    # Scalable way to calculate Portfolio Standard Deviation
    def portfolio_standard_deviation(weights):
        return np.dot(np.dot(df.cov(), weights), weights) ** (1 / 2) * np.sqrt(250)

    def weights_creator(df):
        rand = np.random.uniform(low=min_weight, high=max_weight, size=len(df.columns))
        rand /= rand.sum()
        return rand

    stock_data_arr = {}
    dates = set()
    for st in sts:
        d = yf.download(st, start=start, end=end)
        stockjson = d.to_json()
        stockjson = json.loads(stockjson)
        for c in stockjson['Adj Close']:
            dates.add(int(c))
        stock_data_arr[st] = stockjson
    dates = sorted(dates)

    def sharpe_ratio():
        sumClose = 0
        for sdd in stocks:
            stjs = stock_data_arr[sdd]
            sumClose+=stjs['Adj Close'][str(dates[-1])]*ss[sdd]
        weights = []
        for sdd in stocks:
            stjs = stock_data_arr[sdd]
            weights.append(stjs['Adj Close'][str(dates[-1])]*ss[sdd]/sumClose)
        return weights

    srw = sharpe_ratio()
    result = get_summary(stocks,start,end,srw,mock_weight)
    zy_weight = result['zy_weight']
    max_sharpe = result['max_sharpe_ratio']
    return_max_sharp = result['return_max_sharp']
    zy_point = result['zy_point']
    zc_point = result['zc_point']

    sharpe_raion_val = result['sharpe_raion_now']
    now_sr=result['sharpe_point_now']
    chart_datas['now_sr'] = now_sr

    chart_datas['chart_datas'] = result['chartdata']

    mock_sr = result['sharpe_point_mock']
    chart_datas['mock_sr'] = mock_sr


    market_line = result['market_line']
    chart_datas['market_line'] =  [{"x": market_line[0][0], "value": market_line[0][1],
               'normal': {'fill': "#777777", 'stroke': "#777777"},
               'hovered': {'fill': "#777777", 'stroke': "#777777"},
               'selected': {'fill': "#777777", 'stroke': "#777777"},
               'size': 5
               },
               {"x": market_line[1][0], "value": market_line[1][1],
                'normal': {'fill': "#777777", 'stroke': "#777777"},
                'hovered': {'fill': "#777777", 'stroke': "#777777"},
                'selected': {'fill': "#777777", 'stroke': "#777777"},
                'size': 5
                }
               ]

    weights_max_sharp = []
    for j in range(len(stocks)):
        weights_max_sharp.append(
            {
                "stock": stocks[j],
                "mw": '{:.4f}'.format(zy_weight[j]),
                "nw": '{:.4f}'.format(srw[j]),
                "mcw": '{:.4f}'.format(mock_weight[j])
            }
        )

    max_sr = [{"x": '{:.4f}'.format(zy_point[0]), "value": '{:.4f}'.format(zy_point[1]),
               'normal': {'fill': "#ff0000", 'stroke': "#ff0000"},
               'hovered': {'fill': "#ff0000", 'stroke': "#ff0000"},
               'selected': {'fill': "#ff0000", 'stroke': "#ff0000"},
               'size': 5
               }]
    chart_datas['max_sr'] = max_sr
    min_vp=[{"x":zc_point[0],"value": zc_point[1],
             'normal': {'fill': "#00ff00",'stroke': "#00ff00"},
             'hovered': {'fill': "#00ff00",'stroke': "#00ff00"},
             'selected': {'fill': "#00ff00",'stroke': "00ff00"},
             'size':5
             }]

    chart_datas['min_vp'] = min_vp


    chart_datas = json.dumps(chart_datas)

    start = datetime.strptime(start, '%Y-%m-%d').strftime('%m/%d/%Y')
    end = datetime.strptime(end, '%Y-%m-%d').strftime('%m/%d/%Y')
    return render_template("stock/ef.html",
                           density=density,start=start,end=end,
                           chart_datas =chart_datas,
                           max_sharpe=max_sharpe,
                           return_max_sharp=return_max_sharp,
                           weights_max_sharp=weights_max_sharp,
                           sharpe_raion = sharpe_raion_val,
                           srw=srw )

@auth.user_login_required
@bp.route("/portfolioreturn", methods=("GET", "POST"))
def portfolioreturn():
    dates = "2023-01-01"
    sts = set()
    stocks = get_portfolio(g.user['id'], time.strftime("%Y-%m-%d",time.localtime(time.time()))).fetchall()
    ss = {}
    for stock in stocks:
        sts.add(stock[0])
        ss[stock[0]] = stock[1]

    print(ss)
    if(len(stocks)<1):
        nums = 1
        return render_template("stock/gobuy.html",nums=nums)

    start = "2023-01-01"
    end = time.strftime("%Y-%m-%d",time.localtime(time.time()))
    if request.method == "POST":
        date_range = request.form['daterange']
        start_date_str, end_date_str = date_range.split(" - ")
        start = datetime.strptime(start_date_str, "%m/%d/%Y").strftime('%Y-%m-%d')
        end = datetime.strptime(end_date_str, "%m/%d/%Y").strftime('%Y-%m-%d')

    stock_data_arr = {}
    for st in sts:
        d = yf.download(st,start=start,end=end)
        stockjson = d.to_json()
        stockjson = json.loads(stockjson)
        stock_data_arr[st] = stockjson

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
    print(datas)



    chart1_normal = {'fill': "#ff0000",'stroke': "#ff0000"}
    chart1_hovered={'fill': "#ff0000",'stroke': "#ff0000"}
    chart1_selected={'fill': "#ff0000",'stroke': "#ff0000"}
    chart2_normal = {'fill': "#0000ff",'stroke': "#0000ff"}
    chart2_hovered={'fill': "#0000ff",'stroke': "#0000ff"}
    chart2_selected={'fill': "#0000ff",'stroke': "#0000ff"}
    daily1 = []
    daily2 = []
    for data in datas:
        daily1.append({'x': data[0],
                     'value': data[1],
                     'normal': chart1_normal,
                     'hovered': chart1_hovered,
                     'selected': chart1_selected})
        daily2.append({'x': data[0],
                     'value': data[2],
                     'normal': chart2_normal,
                     'hovered': chart2_hovered,
                     'selected': chart2_selected})
    chart_data = {}
    chart_data['portfolio'] = daily1
    chart_data['indexdata'] = daily2
    chart_data = json.dumps(chart_data)
    start = datetime.strptime(start, '%Y-%m-%d').strftime('%m/%d/%Y')
    end = datetime.strptime(end, '%Y-%m-%d').strftime('%m/%d/%Y')
    return render_template("stock/portfolioreturn.html",start=start,end=end,datas=datas,chart_data=chart_data)