import re

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