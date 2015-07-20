from flask import Flask, jsonify, render_template, request
import requests
import json
import urllib
import zipfile
import os
from collections import defaultdict
import time
import glob
from urllib.request import urlretrieve
import zipfile
import re
import arrow
from luminoso_api import LuminosoClient

app = Flask(__name__)

BASE_URI = 'https://s.qualtrics.com/API/v1/'
lumi_token = 'JWsmYfNCyqz2-nkbXY1aeggioGfQNN-N'
lumi_account = 'admin'

def chunks(l, n):
    n = max(1, n)
    return [l[i:i + n] for i in range(0, len(l), n)]

def get_surveys(token):
    return requests.get(BASE_URI+'surveys?apiToken='+token+'&fileType=json').json()

def get_survey(sid, token):
    return requests.get(BASE_URI+'surveys/'+sid+'?apiToken='
                        +token+'&fileType=json').json()

def get_responses(sid, token):
    return requests.get(BASE_URI+'surveys/'+sid+'/responseExports/?apiToken='
                        +token+'&fileType=JSON').json()

def get_name_id(token):
    surveys = get_surveys(token)
    return {s['name']:s['id'] for s in surveys['result']}

def get_question_descriptions(sid, token):
    ret = defaultdict(list)
    survey = get_survey(sid, token)
    for q in survey['result']['questions']:
        q_type = survey['result']['questions'][q]['questionType']['type']
        q_text = survey['result']['questions'][q]['questionText']
        q_name = ''.join(re.search("(Q)ID(\d+)", q).group(1,2))
        if q_type == "TE":
            ret["text"].append((q_name, q_text))
        else:
            ret["subsets"].append((q_name, q_text))
    return dict(ret)

def _create_project(acct, token, name, docs):
    cli = LuminosoClient.connect('/projects/'+acct, token = lumi_token)
    pid = cli.post('/', name = name)['project_id']
    cli = cli.change_path(pid)
    batches = chunks(docs, 1000)
    for b in batches:
        cli.upload('/docs/', b)
    cli.post('/docs/recalculate/')
    return 'https://dashboard.luminoso.com/v4/explore.html?account='+acct+'&projectId='+pid

def build_analytics_project(sid, token, text_q_id, subset_q_ids, acct, lumi_token, name):
    responses_json = get_survey_json(ice_cream, token)
    docs = []
    for r in responses_json['responses']:
        subsets = [qid+": "+r[qid] for qid in subset_q_ids]
        docs.append({"text":r[text_q_id],
                     "date":arrow.get(r['EndDate']).timestamp,
                     "subsets":subsets})
    proj_url = _create_project(acct, lumi_token, name, docs)
    return proj_url

@app.route('/')
def index():
    return render_template('simple.html')

@app.route('/_step1')
def step1():
    """From the given token, get name and id of surveys"""
    token = request.args.get('token', 0, type=str)
    info = get_name_id(token) #returns a dictionary
    return jsonify(**info)

@app.route('/_step2')
def step2():
    """Given sid and token, return the question info"""
    sid = request.args.get('sid', 0, type=str)
    token = request.args.get('token', 0, type=str)
    return jsonify(get_question_descriptions(sid, token))

@app.route('/_step3')
def step3():
    """Given sid, token, text question ID, subset question IDs,
       a luminoso account and token, create a proj in Analytics"""
    sid = request.args.get('sid', 0, type=str)
    token = request.args.get('token', 0, type=str)
    text_q = request.args.get('text_q', 0, type=str)
    subset_qs = request.args.get('subset_qs', 0, type=list)
    account = lumi_account
    token = lumi_token
    proj_url = build_analytics_project(sid, token, text_q, subset_q_ids,
                                       lumi_acct, lumi_token, "Qualtrics Import")
    return proj_url

if __name__ == '__main__':
    app.run(debug=True)