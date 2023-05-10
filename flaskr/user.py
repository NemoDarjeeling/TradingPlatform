import functools
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
from flask import jsonify
import os
bp = Blueprint("user", __name__, url_prefix="/user")

def get_yesterday_data(ticker,dates):
    yesterday = datetime.strptime(dates, "%Y-%m-%d") - dt.timedelta(days=1)
    while True:
        data = yf.download(ticker, start=yesterday, end=dates)
        if data.empty:
            yesterday = yesterday - dt.timedelta(days=1)
        else:
            break
    return yesterday
def get_portfolio(userid, dates):
    db = get_db()
    portfolio = db.execute(
    "SELECT ticker, SUM(shares) FROM trade "
    "WHERE trade.userid = ? AND date(trade.dates) <= date(?) "
    "GROUP BY ticker",
    (userid, dates),
    )
    return portfolio
def get_portfolio_by_ticker(userid, ticker, dates):
    db = get_db()
    portfolio = db.execute(
    "SELECT SUM(shares) FROM trade "
    "WHERE trade.userid = ? AND date(trade.dates) <= date(?) AND trade.ticker == ticker",
    (userid, dates),
    )
    return portfolio
def get_latest_date(ticker, date):
    today = date
    flag = 1
    while True:
        flag+=1
        data = yf.download(ticker, start=today, end=today+dt.timedelta(days=1))
        if today.strftime("%Y-%m-%d") not in data.index:
            today = today - dt.timedelta(days=1)
            if flag>20:
                return None
        else:
            break
    return today
def istradingtime(symbol, dat="now"):
    if dat=="now":
        start_time = dt.time(hour=9, minute=30)
        end_time = dt.time(hour=16)   
        current_time = dt.datetime.now().time()
        dat = date.today()
        if current_time <= start_time or current_time >= end_time:
            return False
    data = yf.download(symbol, start=dat, end=dat+dt.timedelta(days=1), interval='1m')
    if not data.empty:
        return True
    else:
        return False


@bp.route('/get_stock_price', methods=['POST'])
def get_stock_price():
    include_history = request.json.get('include_history')
    symbol = request.json.get('symbol')
    if include_history:
        year = request.json.get('year')
        month = request.json.get('month')
        day = request.json.get('day')
        if int(month)<10:
            month = '0'+str(month)
        if int(day)<10:
            day = '0'+str(day)
        date_str = year + '-' + month + '-' + day
        try:
            dates = datetime.strptime(date_str, "%Y-%m-%d")
        except:
            return jsonify({'success': False},200)
        if int(year)<1800 or int(year)>2030:
            return jsonify({'success': False},200)
        dates = get_latest_date(symbol, datetime.date(dates))
        if dates is None:
            return jsonify({'success': False},200)
        dates = dates.strftime("%Y-%m-%d")
        data = yf.download(symbol, start=dates)
        Adj_Close = data.loc[dates, 'Adj Close']
    else:
        dates = date.today()
        dates = get_latest_date(symbol, dates)
        if dates is None:
            return jsonify({'success': False},200)
        data = yf.download(symbol, start=dates, interval='1m')
        if data.empty:
            data = yf.download(symbol, start=dates)
        Adj_Close = list(data.loc[:,'Adj Close'])[-1]
    if data.empty:
        return jsonify({'success': False},200)
    return jsonify({'success': True,'price': Adj_Close},200)

    
@auth.user_login_required
@bp.route("/index", methods=("GET", "POST"))
def index():
    return render_template("user/index.html")

@auth.user_login_required
@bp.route("/trade", methods=("GET", "POST"))
def trade():
    if request.method == "POST":
        error = None
        try:
            request.form["symbol"]
            request.form["shares"]
            request.form["type"]
        except:
            flash("please fill out the form")
            return render_template("user/trade.html")
        symbol = request.form["symbol"]
        dates = "now"
        share = int(request.form["shares"])
        credit = float(g.user["credit"])  # convert to float
        userid = g.user["id"]
        type = request.form["type"]
        try:
            request.form["his"]
            his = True
        except:
            his = False
        if his:
            try:
                if request.form["year"]== '' or request.form["month"]=='' or request.form["day"]=='':
                    flash("please fill out the form")
                    return render_template("user/trade.html")
            except:
                flash("please fill out the form")
                return render_template("user/trade.html")
            year = request.form["year"]
            month = request.form["month"]
            day =  request.form["day"]
            if int(month)<10:
                month = '0'+str(month)
            if int(day)<10:
                day = '0'+str(day)
            date_str = year + '-' + month + '-' + day
            try:
                dates = datetime.strptime(date_str, "%Y-%m-%d")
            except:
                flash("invalid date")
                return render_template("user/trade.html")
            if int(year)<1800 or int(year)>2030:
                flash("invalid date")
                return render_template("user/trade.html")
        if not istradingtime(symbol, dates):
            notice = "it's not a valid trading period but will trade with the most recent available price."
        else:
            notice = ""
        if not his:
            dates = date.today()
            dates = get_latest_date(symbol, dates)
            if(dates==None):
                flash("incorrect date")
                return render_template("user/trade.html")
            dates = dates.strftime("%Y-%m-%d")
            data = yf.download(symbol, start=dates, interval='1m')
            if data.empty:
                data = yf.download(symbol, start=dates)
                notice +="There is no live market data for that stock. Will use the most recent one."
            Adj_Close = list(data.loc[:,'Adj Close'])[-1]
        else:
            dates = get_latest_date(symbol, datetime.date(dates))
            if(dates==None):
                flash("incorrect date")
                return render_template("user/trade.html")
            dates = dates.strftime("%Y-%m-%d")
            data = yf.download(symbol, start=dates)
            Adj_Close = data.loc[dates, 'Adj Close']
        db = get_db()
        if share<=0:
            error = "Shares should be positive"
        totalcredit = Adj_Close * share

        if type == 'buy':
            type = 0
            if credit < totalcredit:
                error = 'insufficient fund'
        else:
            type = 1
            portfolio = get_portfolio_by_ticker(userid, symbol, dates).fetchall()[0][0]  # fetch the result
            if share > portfolio:
                error = 'insufficient shares'
        if error is None:
            db.execute(
                "INSERT INTO trade (userid, shares, types, ticker, dates) VALUES (?, ?, ?, ?, ?)",
                (userid, (1 - type * 2) * share, type, symbol, dates),

            )
            db.execute(
                "UPDATE user SET credit = ? WHERE id = ?", (credit + (type * 2 - 1) * totalcredit, userid)
            )
            db.commit()
            notice+="Success"
            flash(notice)
            return redirect(url_for("user.trade"))
        else:
            flash(error)
        #return render_template("user/trade.html")
    return render_template("user/trade.html")

@auth.user_login_required
@bp.route("/withdraw", methods=("GET", "POST"))
def withdraw():
    if request.method == "POST":
        amount = float(request.form["amount"])
        error = None
        db = get_db()
        password = request.form["password"]
        if not check_password_hash(g.user["password"], password):
            error = "Incorrect password."
            flash(error)
            return render_template("user/withdraw.html")
        if amount<0:
            error = "Negative amount"
        elif amount>g.user['credit']:
            error = "Insufficient balance"
        else:
            db.execute(
                "UPDATE user SET credit = ? WHERE id = ?", (g.user['credit']-amount, g.user['id'])
            )
            #bootstrap alert
            db.commit()
            return redirect(url_for("user.withdraw"))
        flash(error)
        return render_template("user/withdraw.html")
    return render_template("user/withdraw.html")

@auth.user_login_required
@bp.route("/deposit", methods=("GET", "POST"))
def deposit():
    if request.method == "POST":
        amount = float(request.form["amount"])
        error = None
        password = request.form["password"]
        if not check_password_hash(g.user["password"], password):
            error = "Incorrect password."
            flash(error)
            return render_template("user/deposit.html")
        if amount<0:
            error = "Negative amount"
            flash(error)
            return render_template("user/deposit.html")
        db = get_db()
        db.execute(
                "UPDATE user SET credit = ? WHERE id = ?", (g.user['credit']+amount, g.user['id'])
            )
            #bootstrap alert
        db.commit()
        return redirect(url_for("user.deposit"))
    return render_template("user/deposit.html")

@auth.user_login_required
@bp.route("/stock")
def stock():
    dates = date.today()
    stocks = get_portfolio(g.user['id'], dates).fetchall()
    results = []
    for stock in stocks:
        if stock[1]!=0:
            ticker = stock[0]
            dates = date.today()
            dates = get_latest_date(ticker, dates)
            todayClosePrice = yf.download(ticker, start=dates, end = dates+dt.timedelta(days=1)).loc[dates.strftime("%Y-%m-%d"), 'Adj Close']
            dates = dates.strftime("%Y-%m-%d")
            yesterday = get_yesterday_data(ticker,dates)
            yesterdayClosePrice = yf.download(ticker, start=yesterday).loc[yesterday, 'Adj Close']
            rt = (todayClosePrice-yesterdayClosePrice)/yesterdayClosePrice*100
            results.append([stock[0],stock[1],"{:.3f}".format(todayClosePrice),"{:.3f}".format(rt)])
    return render_template("user/stock.html", stocks=results)
