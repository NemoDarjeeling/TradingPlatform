import pytest
from flask import g
from flask import session

from flaskr.db import get_db
from datetime import datetime
from flask import url_for

def test_trade(client, app, auth):
        auth.login()
        response = client.post("/user/trade", data=dict(
                symbol='MSFT',
                shares='100',
                type='buy',
                his = 'True',
                year='2021',
                month='1',
                day='20'
            ), follow_redirects=True)
        assert b"Success" in response.data
        with client:
            client.get("/")
            assert round(g.user['credit']) == 1000000-21986
        
        response = client.post("/user/trade", data=dict(
                symbol='MSFT',
                shares='100',
                type='sell',
                his = 'True',
                year='2021',
                month='1',
                day='27'
            ), follow_redirects=True)
        assert b"Success" in response.data
        with client:
            client.get("/")
            assert round(g.user['credit']) == 1000000-21986+22825

def test_portfolio(client, app, auth):
    auth.login()
    response = client.post("/user/trade", data=dict(
                symbol='TSLA',
                shares='100',
                type='buy',
            ), follow_redirects=True)
    assert b"Success" in response.data
    response = client.post("/user/trade", data=dict(
                symbol='AAPL',
                shares='100',
                type='buy',
            ), follow_redirects=True)
    assert b"Success" in response.data
    response = client.post("/user/trade", data=dict(
                symbol='AMZN',
                shares='100',
                type='buy',
            ), follow_redirects=True)
    assert b"Success" in response.data
    response = client.post("/user/trade", data=dict(
                symbol='F',
                shares='1000',
                type='buy',
            ), follow_redirects=True)
    assert b"Success" in response.data
    response = client.post("/user/trade", data=dict(
                symbol='BTC',
                shares='1',
                type='buy',
            ), follow_redirects=True)
    assert b"Success" in response.data