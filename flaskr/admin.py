import functools
import time
import numpy as np
import json
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
import pandas_datareader.data as web
import scipy.optimize as opt
from flaskr.ef import get_summary
bp = Blueprint("admin", __name__, url_prefix="/admin")

def get_all_portfolio():
    db = get_db()
    portfolio = db.execute(
    "SELECT ticker, SUM(shares) FROM trade "
    "GROUP BY ticker"
    )
    return portfolio
def get_all_portfolio_summary():
    db = get_db()
    portfolio = db.execute(
    "SELECT dates, ticker, types, SUM(shares) FROM trade "
    "GROUP BY dates,ticker, types"

    )
    return portfolio
def get_yesterday_date(ticker,dates):
    yesterday = datetime.strptime(dates, "%Y-%m-%d") - dt.timedelta(days=1)
    while True:
        data = yf.download(ticker, start=yesterday, end=dates)
        if yesterday.strftime("%Y-%m-%d") not in data.index:
            yesterday = yesterday - dt.timedelta(days=1)
        else:
            break
    return yesterday
def get_latest_date(ticker):
    today = date.today()
    #today = datetime.strptime("2022-09-23", "%Y-%m-%d")
    while True:
        data = yf.download(ticker, start=today)
        if today.strftime("%Y-%m-%d") not in data.index:
            today = today - dt.timedelta(days=1)
        else:
            break
    return today
@auth.admin_login_required
@bp.route("/adminindex", methods=("GET", "POST"))
def adminindex():
    return render_template('admin/adminindex.html')

@auth.admin_login_required
@bp.route("/riskoverview", methods=("GET", "POST"))
def riskoverview():
    sts = set()
    stocks = get_all_portfolio().fetchall()
    ss = {}
    for stock in stocks:
        sts.add(stock[0])
        ss[stock[0]] = stock[1]

    if(len(stocks)<2):
        nums = 2
        return render_template("admin/admingobuy.html",nums=nums)
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
    return render_template("admin/riskoverview.html",
                           density=density,start=start,end=end,
                           chart_datas =chart_datas,
                           max_sharpe=max_sharpe,
                           return_max_sharp=return_max_sharp,
                           weights_max_sharp=weights_max_sharp,
                           sharpe_raion = sharpe_raion_val,
                           srw=srw )


@auth.admin_login_required
@bp.route("/stockoverview", methods=("GET", "POST"))
def stockoverview():
    stocks = get_all_portfolio().fetchall()
    data = []
    for stock in stocks:
        ticker = stock[0]
        today = get_latest_date(ticker)
        yesterday = get_yesterday_date(ticker, today.strftime("%Y-%m-%d"))
        today_price = round(yf.download(ticker,start=today).loc[today.strftime("%Y-%m-%d"),'Adj Close'],4)
        yesterday_price = round(yf.download(ticker,start=yesterday).loc[yesterday,'Adj Close'],4)
        data.append([ticker,today_price, str(round((today_price-yesterday_price)/yesterday_price,6)*100)+"%", stock[1], round(today_price*stock[1],6)])
    return render_template('admin/stockoverview.html', data=data)

@auth.admin_login_required
@bp.route("/transactionrecord", methods=("GET", "POST"))
def transactionrecord():
    data = {}
    raw_data = get_all_portfolio_summary().fetchall()
    for row in raw_data:
        if row[2]==0:
            data[datetime.strftime(row[0], "%Y-%m-%d")+row[1]] = [row[0],row[1],row[3],0]
    for row in raw_data:
        if row[2]==1:
            if datetime.strftime(row[0], "%Y-%m-%d")+row[1] in data.keys():
                data[datetime.strftime(row[0], "%Y-%m-%d")+row[1]][3] = -row[3]
            else:
                data[datetime.strftime(row[0], "%Y-%m-%d")+row[1]] =[row[0],row[1],0, -row[3]]
    return render_template('admin/transactionrecord.html', data = list(data.values()))