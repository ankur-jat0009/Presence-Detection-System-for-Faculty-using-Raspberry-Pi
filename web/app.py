from flask import Flask, render_template
import json
import os

app = Flask(__name__)

STATUS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../status.json'))

@app.route('/')
def show_status():
    try:
        with open(STATUS_PATH, 'r') as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = {}

    return render_template('status.html', data=data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
