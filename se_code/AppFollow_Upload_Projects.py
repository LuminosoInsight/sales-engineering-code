
# coding: utf-8

# # Prepare AppFollow data for Upload to Luminoso Projects
# ## It will also filter languages and save to JSON files to your folder as backup

# ## The Luminoso API documentation is  [Here](https://analytics.luminoso.com/api/v4/#Prediction)

# given json format of AppFollow data:
# 
# {"reviews":
# [
# {"rating":2,"date":"2016-06-01","created":"2016-08-19 20:18:23","content":"SOME TEXT",
# "store":"as","rating_prev":0,"user_id":"USERZ","id":5555,"ext_id":"5555",
# "is_answer":0,"app_version":"2.2","app_id":46045,"was_changed":0,"locale":"de","author":"ABC","review_id":"555","title":"SOME TITLE","updated":"2017-06-03 19:49:16"},
# 
# {"rating":1,"date":"2016-06-01","created":"2016-08-19 20:18:23","content":"SOME TEXT 2",
# "store":"as","rating_prev":0,"user_id":"USERX","id":4444,"ext_id":"4444",
# "app_version":"2.2","is_answer":0,"app_id":46045,"was_changed":0,"locale":"be","author":"XYZ","review_id":"111","title":"SOME TITLE 2","updated":"2017-06-03 19:49:16"}
# ]
# }

# ### Load libraries and some utility functions

# In[1]:


import json
import os
import csv


# In[2]:


#load and save to JSON type files
import json

def load_json_data(filename):
    with open(filename) as json_data:
        return json.load(json_data)

def save_as_json_file(data, filename):
    if filename[-5:] != '.json':
        file = filename + '.json'
    with open(file, 'w') as f:
        json.dump(data, f)
        print("JSON data saved to file: ", file)


# In[3]:


# create a stats dict from a list of dict data, and a given dict entry:
def dict_stats(entry, dictdata):
    stats_list = {}
    for n in dictdata:
        k = n[entry]
        if k in stats_list:
            stats_list[k] = stats_list[k] + 1
        else:
            stats_list[k] = 1
    print("Stats count for each type of " + entry + ":\n", stats_list)
    return stats_list


# In[4]:


import argparse, cld2

def remove_foreign_languages(docs, lang_code, threshold=0):
    good_documents = []
    bad_documents = []
    for doc in docs:
        isbad = False
        try:
            isReliable, textBytesFound, details = cld2.detect(doc['text'])
        except ValueError:
            isbad = True
            continue
        if not details[0][1] == lang_code and isReliable or details[0][2] < threshold:
                isbad = True
        if isbad:
            bad_documents.append(doc)
        else:
            good_documents.append(doc)
    print('{} documents not identified as "{}" removed from project.'.format(len(bad_documents),lang_code))
    return good_documents, bad_documents


# ## Load the Input JSON data from AppFollow

# In[5]:


INPUT_FILENAME = "AppFollowReviews.json"
#currentPath = os.getcwd()


# In[6]:


raw_data = load_json_data(INPUT_FILENAME)


# In[7]:


data = raw_data['reviews']
len(data)


# In[8]:


data[0]


# In[9]:


data[1]


# In[9]:


locale_counts = dict_stats('locale', data)


# ## Keep only the necessary fields and sort by language
# #### The only *compulsory* field is the Text, which here is 'content'. All other extra fields are metadata.
# #### We also need to seperate  between different languages (locale). Here we show how to keep German reviews

# In[10]:


# we split the data by locale:
loc_data = {}
for loc in locale_counts:
    loc_data[loc] = [d for d in data if d['locale'] == loc] 


# In[11]:


loc_data.keys()


# In[12]:


#in the german data, there is some english but very few, so that's not a concern for luminoso.
for x in loc_data['de']:
    print(x['content'])


# In[13]:


# The belgian data is a more even mix between English, French ad Dutch languages. 
# We can thus create a project for each language using a pre-defined language mapping, by upload all to one porect and copying twice, then 
# removing the other languages using another script we have.
for x in loc_data['be']:
    print(x['content'])


# In[14]:


#for lccale = canada, the lanuguage is mainly english
for x in loc_data['ca']:
    print(x['content'])


# In[15]:


#for lccale = denmark, the lanuguage is mainly danish but this is not suppored in Luminoso. 
for x in loc_data['da']:
    print(x['content'])


# ## Create a locale-to-languages mapping
# #### This will be empty if language not supported by luminoso, by may also have several values if there needs to be separation into several projects (as for Belgium)

# In[16]:


# to be expanded if more locales are included in original input file
locale_languages_mapping = {'de': ['de'], 'be': ['fr', 'nl', 'en'], 'da': [], 'ca': ['en']}


# In[17]:


for loc in locale_counts:
    print(loc, locale_languages_mapping[loc])


# In[18]:


def format_for_upload (data):
     return [{'text': r['content'],
              'title' : r['title'], 
              'app_id':r['app_id'], 
              'app_version':r['app_version'], 
              'date' : r['date'], 
              'rating' : r['rating'], 
              'store' : r['store'] 
                } for r in data]


# In[19]:


PROJECTS_MAIN_TITLE = 'AppFollow TEST'
LANG_THRESHOLD = 0.6


# In[20]:


project_list = []

for (loc, data) in loc_data.items():
    langs = locale_languages_mapping[loc]
    print(langs)
    if langs == []:
        print('The dataset for locale:', loc, 'will not be uploaded as no language is defined')
    else:
        for lingua in langs:
            proj = {}
            unused_docs = []
            proj['language'] = lingua
            proj['location'] = loc
            proj['name'] = PROJECTS_MAIN_TITLE + ' -location:' + loc + ' -language:' + lingua
            if len(langs) > 1:
                documents, unused_docs = remove_foreign_languages(format_for_upload(data), lingua, LANG_THRESHOLD)
            else:
                documents = format_for_upload(data)
            proj['docs'] = documents
            proj['unused_docs'] = unused_docs
            print(proj['name'], '   with ' + str(len(proj['docs'])), 'docs')
            project_list.append(proj)


# In[21]:


len(project_list)


# In[22]:


for x in project_list:
    print(x['name'], 'docs total:', len(x['docs']), 'unused-docs total:', len(x['unused_docs']))


# ## Save data to Json files 
# ### In case needed later -- it will be easy to check data or re-upload

# In[23]:


for p in project_list:
    save_as_json_file(p, p['name'])


# ## Uploading data to new project

# In[24]:


from luminoso_api import LuminosoClient


# In[25]:


account_id = 'https://eu-analytics.luminoso.com/api/v4/projects/u46p858s/'  #Adidas Training
uname = 'bodier@luminoso.com'


# In[26]:


#connect
connection = LuminosoClient.connect(account_id, username = uname)


# In[35]:


for proj in project_list:
    if len(proj['docs']) != 0:    
        # Create a new project
        new_project = connection.post(name = proj['name'])
        new_project_id = new_project['project_id']

        new_project_path = connection.change_path(new_project_id)
        new_project_path.upload('docs', proj['docs'])

        print('Uploading of docs complete for project:', proj['name'])

        job_id = new_project_path.post('docs/recalculate', language = proj['language'])
        end_job = new_project_path.wait_for(job_id)
        if end_job['success'] == True:
            print('Project created successfully')
        else:
            print('Warning: recalculation failed')

print('All done')

