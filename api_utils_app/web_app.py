from flask import Flask, jsonify, render_template, request
import json
from collections import defaultdict, OrderedDict
from luminoso_api import LuminosoClient

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('login.html')

@app.route('/apps')
def show_apps():
    """From the given token, get name and id of surveys"""
    return render_template('apps.html')
    #token = request.args.get('token', 0, type=str)
    #info = get_name_id(token) #returns a dictionary
    #return jsonify(**info)

if __name__ == '__main__':
    app.run(debug=True)