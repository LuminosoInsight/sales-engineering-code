
# coding: utf-8

# In[180]:

import requests
import json
import urllib
import zipfile
import os
from flask import Flask, jsonify, render_template, request


# In[136]:

BASE_URI = 'https://s.qualtrics.com/API/v1/'


# In[137]:

token = 'YGt8lpZHTOfbUip5puXwQVlARk2CfMaGeusxyPHD'
ice_cream = 'SV_1YumlPcR5CHiOuF'


# In[138]:

def get_surveys(token):
    return requests.get(BASE_URI+'surveys?apiToken='+token+'&fileType=json').json()


# In[139]:

def get_survey(sid, token):
    return requests.get(BASE_URI+'surveys/'+sid+'?apiToken='+token+'&fileType=json').json()


# In[140]:

def get_responses(sid, token):
    return requests.get(BASE_URI+'surveys/'+sid+'/responseExports/?apiToken='+token+'&fileType=JSON')


# In[187]:

get_surveys(token)


# In[165]:

len(d['result'])


# In[183]:

def get_name_id(token):
    d = get_surveys(token)
    s_info = {}
    for i in range(len(d['result'])):
        s_info[d['result'][i]['name']] = d['result'][i]['id']
    return s_info


# In[184]:

get_name_id(token)


# In[186]:

get_survey('SV_1YumlPcR5CHiOuF', token)


# In[129]:

get_responses(ice_cream, token).text


# In[130]:

test = requests.get('http://co1.qualtrics.com/API/v1/responseExports/ES_2nLLosccnLdgljT?apiToken='+token)


# In[131]:

test.text


# In[132]:

#strip the url from get requests
cool = str(get_responses(ice_cream, token).text)
d = json.loads(cool)
url1 = str(d['result']['exportStatus'])
test= requests.get(url1+'?apiToken='+token)
print(test.text)


# In[133]:

def download_unzip():
    #strip the url from get requests
    cool = str(get_responses(ice_cream, token).text)
    d = json.loads(cool)
    url1 = str(d['result']['exportStatus'])
    test= requests.get(url1+'?apiToken='+token)

    #strip the url from the test.text
    new = str(test.text)
    d = json.loads(new)
    url2 = str(d['result']['fileUrl'])

    #make a new folder
    foldername='qualtrics_download'
    os.mkdir(foldername)
    #download the file
    urllib.urlretrieve(url2,"a.zip")
    #unzip the file, put it into the new folder
    with zipfile.ZipFile("a.zip", "r") as z:
        z.extractall(path=foldername)
        return


# In[134]:

download_unzip()


# In[ ]:




# In[59]:





# In[2]:




# In[ ]:




# In[ ]:




# In[ ]:



