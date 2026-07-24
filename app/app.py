"""
AI Complaint Management System
Flask application entry point.

Run:
    1. python dataset/generate_dataset.py       (already generated, optional)
    2. python machine_learning/train_models.py  (already trained, optional)
    3. python app.py
    4. Open http://127.0.0.1:5000
"""

import os
import sqlite3
from datetime import datetime
from functools import wraps

import joblib
from flask import Flask, render_template, request, redirect, url_for, session, flash, g
from werkzeug.security import generate_password_hash, check_password_hash

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "database.db")
MODELS_DIR = os.path.join(BASE_DIR, "machine_learning", "models")

app = Flask(__name__)
app.secret_key = "change-this-secret-key-in-production"


# ------------------------------------------------------------------
# ML models - loaded once at startup
# ------------------------------------------------------------------
def load_ml_models():
    required = [
        "tfidf_vectorizer.pkl", "category_model.pkl", "category_encoder.pkl",
        "priority_model.pkl", "priority_encoder.pkl",
        "department_model.pkl", "department_encoder.pkl",
    ]
    missing = [f for f in required if not os.path.exists(os.path.join(MODELS_DIR, f))]
    if missing:
        print("WARNING: ML models not found:", missing)
        print("Run: python machine_learning/train_models.py")
        return None

    return {
        "vectorizer": joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")),
        "category_model": joblib.load(os.path.join(MODELS_DIR, "category_model.pkl")),
        "category_encoder": joblib.load(os.path.join(MODELS_DIR, "category_encoder.pkl")),
        "priority_model": joblib.load(os.path.join(MODELS_DIR, "priority_model.pkl")),
        "priority_encoder": joblib.load(os.path.join(MODELS_DIR, "priority_encoder.pkl")),
        "department_model": joblib.load(os.path.join(MODELS_DIR, "department_model.pkl")),
        "department_encoder": joblib.load(os.path.join(MODELS_DIR, "department_encoder.pkl")),
    }


ML = load_ml_models()


def predict_complaint(text):
    """Run the 3 ML models on complaint text and return predictions."""
    if ML is None:
        return {"category": "Unclassified", "priority": "Medium", "department": "Admin Office"}

    X = ML["vectorizer"].transform([text])

    cat_pred = ML["category_model"].predict(X)[0]
    category = ML["category_encoder"].inverse_transform([cat_pred])[0]

    pri_pred = ML["priority_model"].predict(X)[0]
    priority = ML["priority_encoder"].inverse_transform([pri_pred])[0]

    dept_pred = ML["department_model"].predict(X)[0]
    department = ML["department_encoder"].inverse_transform([dept_pred])[0]

    return {"category": category, "priority": priority, "department": department}


# ------------------------------------------------------------------
# Database helpers
# ------------------------------------------------------------------
def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(exception):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row
    db.executescript("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        role TEXT NOT NULL DEFAULT 'customer',   -- customer | admin | staff
        created_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS complaints (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        complaint_text TEXT NOT NULL,
        category TEXT,
        priority TEXT,
        department TEXT,
        status TEXT NOT NULL DEFAULT 'Open',   -- Open | In Progress | Resolved
        created_at TEXT NOT NULL,
        FOREIGN KEY (user_id) REFERENCES users (id)
    );
    """)

    # create a default admin account if none exists
    cur = db.execute("SELECT COUNT(*) as c FROM users WHERE role = 'admin'")
    if cur.fetchone()["c"] == 0:
        db.execute(
            "INSERT INTO users (name, email, password_hash, role, created_at) VALUES (?, ?, ?, ?, ?)",
            ("System Admin", "admin@cms.local", generate_password_hash("admin123"), "admin",
             datetime.now().isoformat()),
        )
    db.commit()
    db.close()


# ------------------------------------------------------------------
# Auth helpers / decorators
# ------------------------------------------------------------------
def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in to continue.", "warning")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return wrapper


def admin_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if session.get("role") != "admin":
            flash("Admin access required.", "danger")
            return redirect(url_for("dashboard"))
        return f(*args, **kwargs)
    return wrapper


# ------------------------------------------------------------------
# Routes - Auth
# ------------------------------------------------------------------
@app.route("/")
def index():
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        if not name or not email or not password:
            flash("All fields are required.", "danger")
            return redirect(url_for("register"))

        db = get_db()
        existing = db.execute("SELECT id FROM users WHERE email = ?", (email,)).fetchone()
        if existing:
            flash("An account with this email already exists.", "danger")
            return redirect(url_for("register"))

        db.execute(
            "INSERT INTO users (name, email, password_hash, role, created_at) VALUES (?, ?, ?, ?, ?)",
            (name, email, generate_password_hash(password), "customer", datetime.now().isoformat()),
        )
        db.commit()
        flash("Account created successfully. Please log in.", "success")
        return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        db = get_db()
        user = db.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()

        if user and check_password_hash(user["password_hash"], password):
            session["user_id"] = user["id"]
            session["name"] = user["name"]
            session["role"] = user["role"]
            flash(f"Welcome back, {user['name']}!", "success")
            return redirect(url_for("dashboard"))

        flash("Invalid email or password.", "danger")
        return redirect(url_for("login"))

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("login"))


# ------------------------------------------------------------------
# Routes - Customer
# ------------------------------------------------------------------
@app.route("/dashboard")
@login_required
def dashboard():
    if session.get("role") == "admin":
        return redirect(url_for("admin_dashboard"))

    db = get_db()
    complaints = db.execute(
        "SELECT * FROM complaints WHERE user_id = ? ORDER BY created_at DESC",
        (session["user_id"],),
    ).fetchall()

    stats = {
        "total": len(complaints),
        "open": len([c for c in complaints if c["status"] == "Open"]),
        "in_progress": len([c for c in complaints if c["status"] == "In Progress"]),
        "resolved": len([c for c in complaints if c["status"] == "Resolved"]),
    }

    return render_template("dashboard.html", complaints=complaints, stats=stats)


@app.route("/submit-complaint", methods=["GET", "POST"])
@login_required
def submit_complaint():
    if request.method == "POST":
        text = request.form.get("complaint_text", "").strip()
        if not text:
            flash("Please describe your complaint.", "danger")
            return redirect(url_for("submit_complaint"))

        prediction = predict_complaint(text)

        db = get_db()
        db.execute(
            """INSERT INTO complaints
               (user_id, complaint_text, category, priority, department, status, created_at)
               VALUES (?, ?, ?, ?, ?, 'Open', ?)""",
            (session["user_id"], text, prediction["category"], prediction["priority"],
             prediction["department"], datetime.now().isoformat()),
        )
        db.commit()

        flash(
            f"Complaint submitted! AI classified it as '{prediction['category']}' "
            f"({prediction['priority']} priority) -> routed to {prediction['department']}.",
            "success",
        )
        return redirect(url_for("dashboard"))

    return render_template("submit_complaint.html")


# ------------------------------------------------------------------
# Routes - Admin
# ------------------------------------------------------------------
@app.route("/admin")
@login_required
@admin_required
def admin_dashboard():
    db = get_db()
    complaints = db.execute(
        """SELECT complaints.*, users.name as user_name, users.email as user_email
           FROM complaints JOIN users ON complaints.user_id = users.id
           ORDER BY complaints.created_at DESC"""
    ).fetchall()

    stats = {
        "total": len(complaints),
        "open": len([c for c in complaints if c["status"] == "Open"]),
        "in_progress": len([c for c in complaints if c["status"] == "In Progress"]),
        "resolved": len([c for c in complaints if c["status"] == "Resolved"]),
        "high_priority": len([c for c in complaints if c["priority"] == "High"]),
    }

    category_counts = {}
    department_counts = {}
    for c in complaints:
        category_counts[c["category"]] = category_counts.get(c["category"], 0) + 1
        department_counts[c["department"]] = department_counts.get(c["department"], 0) + 1

    return render_template(
        "admin_dashboard.html",
        complaints=complaints,
        stats=stats,
        category_counts=category_counts,
        department_counts=department_counts,
    )


@app.route("/admin/complaint/<int:complaint_id>/status", methods=["POST"])
@login_required
@admin_required
def update_status(complaint_id):
    new_status = request.form.get("status")
    if new_status not in ("Open", "In Progress", "Resolved"):
        flash("Invalid status.", "danger")
        return redirect(url_for("admin_dashboard"))

    db = get_db()
    db.execute("UPDATE complaints SET status = ? WHERE id = ?", (new_status, complaint_id))
    db.commit()
    flash("Complaint status updated.", "success")
    return redirect(url_for("admin_dashboard"))


# ------------------------------------------------------------------
if __name__ == "__main__":
    if not os.path.exists(DB_PATH):
        init_db()
    else:
        init_db()  # safe to call - uses IF NOT EXISTS
    app.run(debug=True)
