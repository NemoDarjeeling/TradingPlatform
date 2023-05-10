import unittest
from flask import Flask
from flask.testing import FlaskClient
from unittest.mock import patch
import user


class UserBlueprintTestCase(unittest.TestCase):
    def setUp(self):
        self.app = Flask(__name__)
        self.app.config["TESTING"] = True
        self.app.register_blueprint(user.bp)
        self.client = self.app.test_client()
    
    def tearDown(self):
        pass
    
    @patch("yfinance.download")
    def test_index_buy_successful(self, mock_yfinance_download):
        mock_yfinance_download.return_value = {
            "Adj Close": {
                "2022-01-01": 100.0
            }
        }
        with self.app.app_context():
            with self.client as c:
                # create a test user
                db = user.get_db()
                db.execute(
                    "INSERT INTO user (username, password, email, credit) VALUES (?, ?, ?, ?)",
                    ("testuser", generate_password_hash("password"), "testuser@example.com", 10000),
                )
                db.commit()
                user_id = db.execute("SELECT id FROM user WHERE username = 'testuser'").fetchone()[0]
                
                # make a successful buy trade
                response = c.post("/user/index", data={
                    "symbol": "AAPL",
                    "date": "2022-01-01",
                    "shares": "10",
                    "credit": "9999.0",
                    "userid": str(user_id),
                    "type": "buy",
                }, follow_redirects=True)
                
                # assert the response is successful
                self.assertEqual(response.status_code, 200)
                
                # assert the trade is recorded in the database
                trades = db.execute(
                    "SELECT * FROM trade WHERE userid = ?", (user_id,)
                ).fetchall()
                self.assertEqual(len(trades), 1)
                self.assertEqual(trades[0]["shares"], 10)
                self.assertEqual(trades[0]["types"], 0)
                self.assertEqual(trades[0]["ticker"], "AAPL")
                self.assertEqual(trades[0]["dates"], "2022-01-01")
                
                # assert the user's credit is updated in the database
                user = db.execute(
                    "SELECT * FROM user WHERE id = ?", (user_id,)
                ).fetchone()
                self.assertEqual(user["credit"], 0)
                
                # assert the user's credit is displayed in the template
                self.assertIn("Credit: $0.00", response.data.decode("utf-8"))
                
                # cleanup
                db.execute("DELETE FROM trade")
                db.execute("DELETE FROM user")
                db.commit()
    
    @patch("yfinance.download")
    def test_index_buy_insufficient_fund(self, mock_yfinance_download):
        mock_yfinance_download.return_value = {
            "Adj Close": {
                "2022-01-01": 100.0
            }
        }
        with self.app.app_context():
            with self.client as c:
                # create a test user
                db = user.get_db()
                db.execute(
                    "INSERT INTO user (username, password, email, credit) VALUES (?, ?, ?, ?)",
                    ("testuser", generate_password_hash("password"), "testuser@example.com", 10000),
                )
                db.commit()
                user_id = db.execute("SELECT id FROM user WHERE username = 'testuser'").fetchone()[0]
                
                # make a buy trade with insufficient fund
                response = c.post("/user/index", data={
                    "symbol": "AAP",
                    "date": "2022-01-01",
                    "shares": "10",
                    "credit": "10.0",
                    "userid": str(user_id),
                    "type": "buy",
            }, follow_redirects=True)
            
            # assert the response is unsuccessful
            self.assertEqual(response.status_code, 200)
            
            # assert the error message is displayed in the template
            self.assertIn("insufficient fund", response.data.decode("utf-8"))
            
            # cleanup
            db.execute("DELETE FROM user")
            db.commit()

@patch("yfinance.download")
def test_index_sell_successful(self, mock_yfinance_download):
    mock_yfinance_download.return_value = {
        "Adj Close": {
            "2022-01-01": 100.0
        }
    }
    with self.app.app_context():
        with self.client as c:
            # create a test user and make a buy trade
            db = user.get_db()
            db.execute(
                "INSERT INTO user (username, password, email, credit) VALUES (?, ?, ?, ?)",
                ("testuser", generate_password_hash("password"), "testuser@example.com", 10000),
            )
            db.execute(
                "INSERT INTO trade (userid, shares, types, ticker, dates) VALUES (?, ?, ?, ?, ?)",
                (1, 10, 0, "AAPL", "2022-01-01"),
            )
            db.commit()
            user_id = db.execute("SELECT id FROM user WHERE username = 'testuser'").fetchone()[0]
            
            # make a successful sell trade
            response = c.post("/user/index", data={
                "symbol": "AAPL",
                "date": "2022-01-01",
                "shares": "5",
                "credit": "10000.0",
                "userid": str(user_id),
                "type": "sell",
            }, follow_redirects=True)
            
            # assert the response is successful
            self.assertEqual(response.status_code, 200)
            
            # assert the trade is recorded in the database
            trades = db.execute(
                "SELECT * FROM trade WHERE userid = ?", (user_id,)
            ).fetchall()
            self.assertEqual(len(trades), 2)
            self.assertEqual(trades[1]["shares"], 5)
            self.assertEqual(trades[1]["types"], 1)
            self.assertEqual(trades[1]["ticker"], "AAPL")
            self.assertEqual(trades[1]["dates"], "2022-01-01")
            
            # assert the user's credit is updated in the database
            user = db.execute(
                "SELECT * FROM user WHERE id = ?", (user_id,)
            ).fetchone()
            self.assertEqual(user["credit"], 5000)
            
            # assert the user's credit is displayed in the template
            self.assertIn("Credit: $5000.00", response.data.decode("utf-8"))
            
            # cleanup
            db.execute("DELETE FROM trade")
            db.execute("DELETE FROM user")
            db.commit()

@patch("yfinance.download")
def test_index_sell_insufficient_shares(self, mock_yfinance_download):
    mock_yfinance_download.return_value = {
        "Adj Close": {
            "2022-01-01": 100.0
        }
    }
    with self.app.app_context():
        with self.client as c:
            # create a test user and make a buy trade
            db = user.get_db()
            db.execute(
                "INSERT INTO user (username, password, email, credit) VALUES (?, ?, ?, ?)",
                ("testuser", generate_password_hash("password"),
                "testuser@example.com", 10000),
            )
            db.execute(
                "INSERT INTO trade (userid, shares, types, ticker, dates) VALUES (?, ?, ?, ?, ?)",
                (1, 10, 0, "AAPL", "2022-01-01"),
            )
            db.commit()
            user_id = db.execute("SELECT id FROM user WHERE username = 'testuser'").fetchone()[0]
            
            # make a sell trade with insufficient shares
            response = c.post("/user/index", data={
                "symbol": "AAPL",
                "date": "2022-01-01",
                "shares": "15",
                "credit": "10000.0",
                "userid": str(user_id),
                "type": "sell",
            }, follow_redirects=True)
            
            # assert the response is unsuccessful
            self.assertEqual(response.status_code, 200)
            
            # assert the error message is displayed in the template
            self.assertIn("insufficient shares", response.data.decode("utf-8"))
            
            # cleanup
            db.execute("DELETE FROM trade")
            db.execute("DELETE FROM user")
            db.commit()

if __name__ == "__main__":
    unittest.main()
