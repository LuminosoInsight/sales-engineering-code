import re
from luminoso_api import LuminosoClient
import datetime
import requests
import time

BASE_URL = 'https://compass.luminoso.com/api/'
STAGING_URL = 'https://compass-test.services.luminoso.com/api/'

def parse_com_url(url):
    proj = re.search("projects%2F(.*)%2F", url).group(1)
    return proj

def get_headers(username, password, url):
    login_resp = requests.post(url + 'login/',
                    data={'username':username, 'password':password})
    headers = {"authorization": "Token " + login_resp.json()["token"],
               "Content-Type": "application/json"}
    return headers

def get_messages(project, username, password, include_spam=False, staging=False):
    messages = []
    spam = ''
    if include_spam:
        spam = '?spam=true'
    url = BASE_URL + 'projects/'+project+'/p/messages/'+spam
    if staging:
        url = STAGING_URL + 'projects/'+project+'/p/messages/'+spam
    headers = get_headers(username, password, BASE_URL)
    while True:
        res = requests.get(url, headers=headers).json()
        messages = messages + res['results']
        url = res['next']
        if not url:
            break
    return messages

def get_all_docs(client):
    docs = []
    while True:
        new_docs = client.get('docs', limit=20000, offset=len(docs))
        if new_docs:
            docs.extend(new_docs)
        else:
            return docs
        
def format_messages(docs):
    messages = []
    for doc in docs:
        message = {}
        message['text'] = doc['text']
        message['timestamp'] = datetime.datetime.strftime(datetime.datetime.fromtimestamp(time.time()+4*60*60), '%Y-%m-%d %H:%M:%S')
        messages.append(message)
    return messages

def post_messages(api_url, docs, interval, username, password):
    headers = get_headers(username, password, BASE_URL)
    response = requests.post(api_url + 'messages/', json=docs, headers=headers)
    if not response.ok:
        print(response.text)
    time.sleep(interval)