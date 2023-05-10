import functools
import os
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

from flaskr import create_app
from flaskr.db import get_db

bp = Blueprint("auth", __name__, url_prefix="/auth")

@bp.route('/save-secret', methods=['POST'])
def save_secret():
  secret = request.json.get('secret')
  if os.path.exists("password.txt"):
    os.remove("password.txt")
  with open("password.txt", "w") as file:
    file.write(secret)
  return 'Secret saved!'

def user_login_required(view):
    """View decorator that redirects anonymous users to the login page."""

    @functools.wraps(view)
    def wrapped_view(**kwargs):
        if g.user is None:
            return redirect(url_for("auth.login"))
        if g.admin:
            #bootstrap alert
            return redirect(url_for("user.adminindex"))
        return view(**kwargs)

    return wrapped_view
def admin_login_required(view):
    """View decorator that redirects anonymous users to the login page."""

    @functools.wraps(view)
    def wrapped_view(**kwargs):
        if g.user is None:
            return redirect(url_for("auth.login"))
        if not g.admin:
            #bootstrap alert
            return redirect(url_for("user.index"))
        return view(**kwargs)

    return wrapped_view

@bp.before_app_request
def load_logged_in_user():
    """If a user id is stored in the session, load the user object from
    the database into ``g.user``."""
    user_id = session.get("user_id")
    admin = session.get("admin")
    if user_id is None:
        g.user = None
        g.admin = None
    elif admin:
        g.user = (
            get_db().execute("SELECT * FROM admin WHERE id = ?", (user_id,)).fetchone()
        )
        g.admin = True
    else:
        g.user = (
            get_db().execute("SELECT * FROM user WHERE id = ?", (user_id,)).fetchone()
        )
        g.admin = False


@bp.route("/register", methods=("GET", "POST"))
def register():
    """Register a new user.

    Validates that the username is not already taken. Hashes the
    password for security.
    """
    if g.user and g.admin:
        redirect ('/user/adminindex.html')
    elif g.user and not g.admin:
        redirect ('/user/index.html')
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        adminPassword = request.form["adminpassword"]
        db = get_db()
        error = None

        if not username:
            error = "Username is required."
        elif not password:
            error = "Password is required."

        if error is None:
            if adminPassword:
                with open("password.txt", "r") as file:
                    correctpassword = file.read().strip()
                if adminPassword== correctpassword:
                    try:
                        db.execute(
                        "INSERT INTO admin (username, password) VALUES (?, ?)",
                        (username, generate_password_hash(password)),
                        )
                        db.commit()
                    except db.IntegrityError:
                        error = f"Admin {username} is already registered."
                    else:
                        return redirect(url_for("auth.login"))
                else:
                    error = "Admin password not match"
            else:
                try:
                    db.execute(
                    "INSERT INTO user (username, password, credit) VALUES (?, ?, ?)",
                    (username, generate_password_hash(password), 1000000),
                    )
                    db.commit()
                except db.IntegrityError:
                # The username was already taken, which caused the
                # commit to fail. Show a validation error.
                    error = f"User {username} is already registered."
                else:
                # Success, go to the login page.
                    return redirect(url_for("auth.login"))

        flash(error)

    return render_template("auth/register.html")


@bp.route("/login", methods=("GET", "POST"))
def login():
    """Log in a registered user by adding the user id to the session."""
    if g.user and g.admin:
        redirect ('/user/adminindex.html')
    elif g.user and not g.admin:
        redirect ('/user/index.html')
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        try:
            request.form["admin"]
            admin = True
        except:
            admin = False
        db = get_db()
        error = None
        if not admin:
            user = db.execute(
            "SELECT * FROM user WHERE username = ?", (username,)
            ).fetchone()
        else:
            user = db.execute(
            "SELECT * FROM admin WHERE username = ?", (username,)
            ).fetchone()

        if user is None:
            error = "Incorrect username."
        elif not check_password_hash(user["password"], password):
            error = "Incorrect password."

        if error is None:
            # store the user id in a new session and return to the index
            session.clear()
            session["user_id"] = user["id"]
            session["admin"] = admin
            if not admin:
                return redirect(url_for("user.index", user=user))
            else:
                return redirect(url_for("admin.adminindex", user=user))

        flash(error)

    return render_template("auth/login.html")


@bp.route("/logout")
def logout():
    """Clear the current session, including the stored user id."""
    session.clear()
    return redirect(url_for("hello"))


if __name__ == '__main__':
    create_app().run(debug=True,host='0.0.0.0',port=5001)