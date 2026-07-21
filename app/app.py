import os
from flask import Flask, render_template

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static")
)

app.secret_key = "secret_key"


# Home Page
@app.route("/")
def index():
    return render_template("login.html")


# Login Page
@app.route("/login")
def login():
    return render_template("login.html")


# Register Page
@app.route("/register")
def register():
    return render_template("register.html")


# Dummy Dashboard
@app.route("/dashboard")
def dashboard():
    return "<h1>User Dashboard (Coming in Day 3)</h1>"


# Dummy Submit Complaint
@app.route("/submit_complaint")
def submit_complaint():
    return "<h1>Submit Complaint Page (Coming Soon)</h1>"


# Dummy Admin Dashboard
@app.route("/admin_dashboard")
def admin_dashboard():
    return "<h1>Admin Dashboard (Coming Soon)</h1>"


# Dummy Logout
@app.route("/logout")
def logout():
    return "<h1>Logout Page (Coming Soon)</h1>"


if __name__ == "__main__":
    app.run(debug=True)