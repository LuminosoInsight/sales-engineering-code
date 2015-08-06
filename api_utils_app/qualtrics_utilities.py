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

BASE_URI = 'https://co1.qualtrics.com/API/v1/'
lumi_account = 'admin'

def __get_token(cli):
    cli2 = cli.change_path('/')
    return cli2.get('/user/tokens/')[0]['token']

def __get_account(cli):
    cli2 = cli.change_path('/')
    return cli2.get('user/profile')['default_account']

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
    print(responses_json)
    url1 = responses_json['result']['exportStatus']
    print(url1)
    file_json = requests.get(url1+'?apiToken='+token).json()
    print(file_json)
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

def _create_project(user, passwd, name, docs):
    cli = LuminosoClient.connect('/', username=user, password=passwd)
    acct = __get_account(cli)
    cli = LuminosoClient.connect('/projects/'+acct, username=user, password=passwd)
    pid = cli.post('/', name = name)['project_id']
    cli = cli.change_path(pid)
    batches = chunks(docs, 1000)
    for b in batches:
        cli.upload('/docs/', b)
    cli.wait_for(cli.post('/docs/recalculate/'))
    return 'https://dashboard.luminoso.com/v4/explore.html?account='+acct+'&projectId='+pid

def build_analytics_project(sid, token, text_q_ids, subset_q_ids, user, passwd, name):
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
    proj_url = _create_project(user, passwd, name, docs)
    return proj_url