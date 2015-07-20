from flask import Flask, jsonify, render_template, request
import requests
import json
import urllib
import zipfile
import os
app = Flask(__name__)

BASE_URI = 'https://s.qualtrics.com/API/v1/'

#trivial function
def add_apple(a):
	new_word = a + ' apple'
	return new_word


def get_surveys(token):
    return requests.get(BASE_URI+'surveys?apiToken='+token+'&fileType=json').json()

def get_name_id(token):
    d = get_surveys(token)
    s_info = {}
    for i in range(len(d['result'])):
        s_info[d['result'][i]['name']] = d['result'][i]['id']
    return s_info


@app.route('/_step1')
def step1():
    """From the given token, get name and id of surveys"""
    token = request.args.get('token', 0, type=str)
    info = get_name_id(token) #returns a dictionary
    return jsonify(**info)


@app.route('/_step2')
def add_numbers():
    """Add two numbers server side, ridiculous but well..."""
    sid = request.args.get('sid', 0, type=str)
    token = request.args.get('token', 0, type=str)
    #Tim's code
    word = add_apple(sid+token) #trivial function
    return jsonify(result=word)

@app.route('/')
def index():
    return render_template('simple.html')

if __name__ == '__main__':
    app.run()

