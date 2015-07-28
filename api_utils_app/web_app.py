from flask import Flask, jsonify, render_template, request, session, url_for
import json
from collections import defaultdict, OrderedDict
from luminoso_api import LuminosoClient

app = Flask(__name__)
#app.secret_key = 'some_secret'

@app.route('/')
def home():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login():
    """From the given token, get name and id of surveys"""
    session['username'] = request.form['username']
    session['password'] = request.form['password']
    session['apps_to_show'] = {'app1':url_for('app1')}
    try:
        LuminosoClient.connect('/projects/', username=session['username'],
                                             password=session['password'])
        return redirect(url_for('index'))
    except:
        error = 'Invalid credentials'
        return render_template('login.html', error=error)

@app.route('/index')
def index():
    return render_template('index.html', apps=session['apps_to_show'])

@app.route('/app1')
def app1():
	return render_template('lumi_app1.html')

if __name__ == '__main__':
    app.run(debug=True)