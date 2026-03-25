from flask import Flask, render_template, request
import sqlite3
from datetime import datetime
import os

app = Flask(__name__)

# Use same DB as attendance_taker.py
DB_PATH = os.path.join(os.path.dirname(__file__), "attendance.db")

@app.route('/')
def index():
    return render_template('index.html', selected_date='', no_data=False)

@app.route('/attendance', methods=['POST'])
def attendance():
    selected_date = request.form.get('selected_date')

    # Format date to YYYY-MM-DD (matches DB)
    date_obj = datetime.strptime(selected_date, '%Y-%m-%d')
    formatted_date = date_obj.strftime('%Y-%m-%d')

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT name, time FROM attendance WHERE date = ?", (formatted_date,))
    attendance_data = cursor.fetchall()

    conn.close()

    if not attendance_data:
        return render_template('index.html', selected_date=selected_date, no_data=True)

    return render_template('index.html', selected_date=selected_date, attendance_data=attendance_data)

if __name__ == '__main__':
    app.run(debug=True)
