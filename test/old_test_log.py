import unittest
from flask import Flask
from werkzeug.security import generate_password_hash
import auth


class AuthTestCase(unittest.TestCase):
    def setUp(self):
        self.app = Flask(__name__)
        self.app.register_blueprint(bp)
        self.client = self.app.test_client()

        # Create a test user
        self.username = "testuser"
        self.password = "testpass"
        with self.app.app_context():
            db = get_db()
            db.execute(
                "INSERT INTO user (username, password, credit) VALUES (?, ?, ?)",
                (self.username, generate_password_hash(self.password), 1000000),
            )
            db.commit()

    def tearDown(self):
        with self.app.app_context():
            db = get_db()
            db.execute("DELETE FROM user WHERE username=?", (self.username,))
            db.commit()

    def test_register(self):
        # Test registration with valid credentials
        response = self.client.post(
            "/auth/register",
            data={"username": "newuser", "password": "newpass"},
            follow_redirects=True,
        )
        self.assertEqual(response.status_code, 200)

        # Test registration with an existing username
        response = self.client.post(
            "/auth/register",
            data={"username": self.username, "password": "newpass"},
            follow_redirects=True,
        )
        self.assertIn(b"User testuser is already registered.", response.data)

        # Test registration with missing username and/or password
        response = self.client.post(
            "/auth/register",
            data={"username": "", "password": ""},
            follow_redirects=True,
        )
        self.assertIn(b"Username is required.", response.data)
        self.assertIn(b"Password is required.", response.data)

    def test_login_logout(self):
        # Test login with valid credentials
        response = self.client.post(
            "/auth/login",
            data={"username": self.username, "password": self.password},
            follow_redirects=True,
        )
        self.assertEqual(response.status_code, 200)

        # Test login with incorrect username
        response = self.client.post(
            "/auth/login",
            data={"username": "wronguser", "password": self.password},
            follow_redirects=True,
        )
        self.assertIn(b"Incorrect username.", response.data)

        # Test login with incorrect password
        response = self.client.post(
            "/auth/login",
            data={"username": self.username, "password": "wrongpass"},
            follow_redirects=True,
        )
        self.assertIn(b"Incorrect password.", response.data)

        # Test logout
        response = self.client.get("/auth/logout", follow_redirects=True)
        self.assertEqual(response.status_code, 200)
