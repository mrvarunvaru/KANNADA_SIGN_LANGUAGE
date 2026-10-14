from flask import Flask, render_template
from recognition import start_recognition  # ✅ import the function
import threading  # run in background thread

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/cam1')
def cam1():
    # Run recognition in a separate thread so Flask doesn’t freeze
    threading.Thread(target=start_recognition).start()
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
