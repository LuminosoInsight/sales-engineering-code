from flask import Flask, jsonify, render_template, request
import requests
import json
import urllib
import zipfile
import os
from collections import defaultdict, OrderedDict
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
    return requests.get(BASE_URI+'surveys/'+sid+'?apiToken='+token+'&fileType=json').json()

def get_responses(sid, token):
    return requests.get(BASE_URI+'surveys/'+sid+'/responseExports/?apiToken='+token+'&fileType=JSON').json()

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
        ret["text"] = sorted(ret["text"], key = lambda x: int(x[0][1:]))
        ret["subsets"] = sorted(ret["subsets"], key = lambda x: int(x[0][1:]))
    return ret

def get_survey_json(sid, token):
    #strip the url from get requests
    responses_json = get_responses(sid, token)
    url1 = responses_json['result']['exportStatus']
    file_json = requests.get(url1+'?apiToken='+token).json()
    while file_json['result']['percentComplete'] < 100:
        print("waiting for zip file to download. Percent complete: "
               + str(int(file_json['result']['percentComplete'])))
        time.sleep(1)
        file_json = requests.get(url1+'?apiToken='+token).json()
    url2 = file_json['result']['fileUrl']
    #make a new folder
    foldername='qualtrics_download'
    if not os.path.isdir(foldername):
        os.mkdir(foldername)
    #download the file
    timestamp = str(round(time.time()))
    urlretrieve(url2,'./'+foldername+'/'+sid+"_"+timestamp+".zip")
    #unzip the file, put it into the new folder
    with zipfile.ZipFile('./'+foldername+'/'+sid+"_"+timestamp+".zip", "r") as z:
        z.extractall(path=foldername)
        newest = max(glob.iglob(foldername+'/*.json'), key=os.path.getctime)
        return json.load(open(newest, 'r'))

def _create_project(acct, token, name, docs):
    cli = LuminosoClient.connect('/projects/'+acct, token = lumi_token)
    pid = cli.post('/', name = name)['project_id']
    cli = cli.change_path(pid)
    batches = chunks(docs, 1000)
    for b in batches:
        cli.upload('/docs/', b)
    cli.wait_for(cli.post('/docs/recalculate/'))
    return 'https://dashboard.luminoso.com/v4/explore.html?account='+acct+'&projectId='+pid

def build_analytics_project(sid, token, text_q_ids, subset_q_ids, acct, lumi_token, name):
    def make_subset_mapping(survey):
        survey = survey['result']['questions']
        ret = {}
        for qid in survey:
            if 'choices' in survey[qid].keys():
                qid2 = ''.join(re.search("(Q)ID(\d+)", qid).group(1,2))
                ret[qid2] = {s:survey[qid]['choices'][s]['description'] for s in survey[qid]['choices']}
        return ret
    responses_json = get_survey_json(sid, token)
    subset_mapping = make_subset_mapping(get_survey(sid, token))
    docs = []
    for r in responses_json['responses']:
        subsets = [qid+": "+subset_mapping[qid][r[qid]] for qid in subset_q_ids]
        docs.append({"text":' ¶ '.join([r[q] for q in text_q_ids]),
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
    text_qs = eval(request.args.get('text_qs', 0, type=str))
    subset_qs = eval(request.args.get('subset_qs', 0, type=str))
    title = request.args.get('title', 0, type=str)
    proj_url = build_analytics_project(sid, token, text_qs, subset_qs,
                                       lumi_account, lumi_token, title+" (Imported from Qualtrics)")
    return jsonify({"url":proj_url})

if __name__ == '__main__':
    app.run(debug=True)