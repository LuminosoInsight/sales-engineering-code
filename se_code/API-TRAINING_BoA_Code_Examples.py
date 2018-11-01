
# coding: utf-8

# # API Quickstart for Luminoso Daylight (v4)

# # Examples on BoA app-Reviews dataset

# ### Author: Boaz Odier - bodier@luminoso.com

# ## General helper functions

# In[1]:


import csv

def save_to_CSV(filename, data):
    if len(filename) < 4 or filename[-4:] != '.csv':
        filename += '.csv'
    with open(filename, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)
    print('Data saved to file:  ', filename)

def load_from_CSV(filename):
    with open(filename, 'r') as file:
        data = [row for row in csv.DictReader(file)]
    print('Data loaded from file:  ', filename)
    return data


# In[2]:


import datetime

def get_iso_date(d):
    return datetime.datetime.fromtimestamp(d).strftime('%Y-%m-%d')


# In[3]:


def substitute_keys_in_dict(data_dict, new_keys_dict):
    '''
    Substitute the keys used in the data_dict, to the values returned by the new_keys_dict
    '''
    result = {}
    for k, v in data_dict.items():
        result[new_keys_dict[k]] = v
    return result


# In[4]:


import dateutil.parser 
import datetime
import time

#Transform a date in string format into Unix Epoch date

def extract_date(string, dayfirst = False):
    #tries to find a date in a given string, and returns it as a float (number of seconds)
    #if no date found, returns None
    #default setting is to expect US format (month/day/year)
    try:
        d = dateutil.parser.parse(string, dayfirst = dayfirst) 
    except:
        return None
    else:
        return time.mktime(d.timetuple())

    
DATE_FORMAT_EU= "%d/%m/%Y"  # %Y = Year as 4 digits, %y is 2 digits
DATE_FORMAT_dby = "%d-%b-%y"  # %b = Month as locale’s abbreviated name, eg: Sep
DATE_FORMAT_EU_time = '%d/%m/%Y %H:%M'
DATE_FORMAT_ISO = '%Y-%m-%d'
    
def extract_formatted_date(date, date_format):
    #To be used when a specific date format is expected
    #use standard strptime function to get datetime object from a string
    try:
        d = datetime.datetime.strptime(date, date_format) 
    except:
        return None
    else:
        return time.mktime(d.timetuple())


# ## Helper functions specific to Luminoso usage

# In[5]:


#may want to refactor this so output of 2nd value is a dict with key = subsettype, value = list of possible values
#then the get_subset_elements function can be subsumed.  
def get_subsets_info(connection):
    '''
    Given a Luminoso project connection, this will simply give the 'subsets/stats' API endpoint info,
    but also as a second value a list of the Subset Types. 
    '''
    substats = connection.get('subsets/stats')
    type_list = []
    for s in substats:
        if s['subset'] == '__all__':
            t = '__all__'
            s['subset_type'] = t
            s['subset_element'] = ''
        else:
            (t, e) = s['subset'].split(': ')
            s['subset_type'] = t
            s['subset_element'] = e
        
        if t not in type_list: 
            type_list.append(t)
    
    return substats, type_list

def get_subset_elements(subsets_info, subset_type):
    results = []
    for s in subsets_info:
        if s['subset_type'] == subset_type:
            results.append(s['subset'])
    return results


# In[6]:


def get_all_docs(client, doc_fields=None):
    '''
    Get all docs from a Luminoso connection to a project (client)
    '''
    docs = []
    while True:
        if doc_fields:
            newdocs = client.get('docs', limit=25000, offset=len(docs), doc_fields=doc_fields)
        else:
            newdocs = client.get('docs', limit=25000, offset=len(docs))
        if newdocs:
            docs.extend(newdocs)
        else:
            return docs


# In[7]:


def subset_array_to_dict(subsets_array):
    '''
    Given an array of subset values (as per Luminoso output from docs download),
    transforms it into a dictionary format
    '''
    obj={}
    for s in subsets_array:
        if  s != '__all__':
            (sub, val) = s.split(': ')
            obj[sub] = val
    return obj


# In[8]:


def build_ID_TOPICNAME_dict(topicsdata):
    '''
    Given a Luminoso list of topics, as per API call to get('topics'), 
    build a topic ID to NAME mapping table (as a dict)
    '''
    result = {}
    for t in topicsdata:
        result[t['_id']] = t['name']
    return result


# In[9]:


def create_or_update_topics(connection, new_topics):
    '''
    This function will create the given new topics into the project
    If a topic already exists (by name), then it will be overwritten with the new terms given
    Returns a dict mapping table of topicIDs to TopicNames
    '''
    project_info = connection.get()
    existing_topics = connection.get('topics')
    
    for n in new_topics:
        newName = n['Topic Name']
        newTerms = n['Topic Terms']
        isnew = True
        for x in existing_topics:
            if newName == x['name']:
                connection.put('topics/id/' + x['_id'], text = newTerms, name = newName) #we have to give the name as parameter as well, if not the name defaults to list of terms
                isnew = False
                print('The topic', newName, 'already exists - Overwritten with: ', newTerms )
        if isnew:
            connection.post('topics', text = newTerms, name = newName)
            print('The topic', newName, 'is new - Created with', newTerms)
    
    print('\nThe topics have been updated for project:',  project_info['name'], ', on workspace:', project_info['account_name'] )
    
    return connection.get('topics')


# # -----------------------------CONNECTION & INFO----------------------------------

# ## Connecting to an Account (workspace) & list its projects

# In[10]:


#some parameters:
user_name = 'bodier@luminoso.com'  #change to your own email that you use to login to Luminoso

#use the correct one from below:
api_url = 'https://analytics.luminoso.com/api/v4' 
#api_url = 'https://eu-analytics.luminoso.com/api/v4' 

account_id = 'n22d432u' #this is the accountID for your workspace


# In[11]:


from luminoso_api import LuminosoClient


# In[12]:


account_url = api_url + '/projects/' + account_id + '/'
account_client = LuminosoClient.connect(account_url, username = user_name)
all_projects = account_client.get()


# In[31]:


print('Total number of projects:', len(all_projects))
filterstr = 'BoA'.lower()
for p in all_projects:
    pn = p['name']
    if filterstr in pn.lower():
        print(p['account_name'], '  ',  p['project_id'], '  ',  pn)


# ## Connect to a specific Luminoso Project

# In[50]:


project_id = 'prjrm7d5'  
#change the end-point to it:
client = LuminosoClient.connect(account_url + project_id, username=user_name)
print('Connected to project: ' +  client.get()['name'])
direct_url = (account_url + project_id).replace('api/v4','app/#')
print(direct_url)


# In[51]:


#Get all the top level info about the project
project_info = client.get()
project_info


# ## Look at the available Subsets (ie: metadata / filters)

# In[52]:


project_subsets, project_subsetTypes = get_subsets_info(client)
project_subsetTypes


# In[53]:


project_subsets[0:2]


# In[54]:


example_subset = get_subset_elements(project_subsets, 'Store')
example_subset


# # ---------------------------------------TOPICS----------------------------------------

# ##  Add a new Topic

# In[40]:


test_topic = {'Topic Name': 'SOME FANTASTOPIC', 'Topic Terms': 'Fantastic, topic, for testing !'}
test_topic


# In[41]:


#insert a new topic
new_topic = client.post('topics', text = test_topic['Topic Terms'], name = test_topic['Topic Name'])
new_topic
#you can now check that this topic exists in the UI of the project !


# In[57]:


#  Now you can see it in the project, just refresh it in browser or click below
print(direct_url)


# ##  Delete a Topic

# In[59]:


#delete this topic:
outcome_deletion = client.delete('topics/id/' + new_topic['_id'])
outcome_deletion
#you can now check that this topic is no longer in the UI of the project ! 


# In[58]:


#  Again you can check on your project, link below:
print(direct_url)


# ## Get all Topics currently in the Project

# In[60]:


topics = client.get('topics')


# In[61]:


topics[0:2]


# In[62]:


for x in topics:
    print( x['_id'], x['name'] , '  ',  x['text'])


# In[63]:


#If needed, you can use the helper function from above to create a mapping table from topic IDs to their Names
id_topicnames_table = build_ID_TOPICNAME_dict(topics)


# ## Import topics data from a CSV

# In[64]:


#API_TRAINING_BoA_topics_list.csv
topic_file = 'API_TRAINING_BoA_topics_list.csv'  #file from wich the topics will be loaded

input_topics = load_from_CSV(topic_file)

print('')
for x in input_topics:
    print(x['Topic Name'], ':' ,  x['Topic Terms'])


# ## Upload all new Topics onto Project, overwriting the definition for each if it already exists

# In[65]:


refreshed_topics = create_or_update_topics(client, input_topics)


# In[66]:


#If needed, use the helper function to get a mapping table from topic IDs to their Names
id_topicnames_table = build_ID_TOPICNAME_dict(refreshed_topics)
id_topicnames_table


# # --------------------GETTING DOCUMENTS FROM A PROJECT------------------

# ## Download all the documents

# In[67]:


DOCS = get_all_docs(client)
len(DOCS)


# In[68]:


#Structure of a document in a project:
DOCS[105]


# ## Saving documents into CSV

# In[69]:


# choose which info to keep for export:
docs_output = []
for d in DOCS:
    obj = {}
    obj['Luminoso_docID'] = d['_id']
    obj['text'] = d['text']
    obj['title'] = d['title']
    metadata = subset_array_to_dict(d['subsets'])
    for subset, value in metadata.items():
        obj[subset] = value
    docs_output.append(obj)


# In[70]:


len(docs_output)


# In[71]:


docs_output[0]


# In[73]:


# Write to Output File
#docs_output_file = 'API_TRAINING_OUPUT_documents.csv'
docs_output_file = 'DOCS-DOWNLOAD_' + project_info['name'] #using name of project
save_to_CSV(docs_output_file, docs_output)


# # ----------------------------------------------------------------------------------------

# ## The 'terms/search' Endpoint

# In[74]:


examples_terms = 'cheque'


# In[75]:


# We can feed the API 'terms/search' endpoint with a list of topics, 
# and it will give us a list of search results with terms most closely associated with this list
exple_search_results = client.get('terms/search', text=examples_terms,limit=100)
exple_search_results.keys()


# In[76]:


len(exple_search_results['search_results'])


# In[82]:


# The results are in the 'search_results' value, and we also get the associated search vector, which we discard.
# Lets look at, for example, the 17th result:
exple_search_results['search_results'][2]


# In[83]:


#the first element is the actual search results
exple_search_results['search_results'][2][0]['term']


# In[84]:


#second element is the score
exple_search_results['search_results'][2][1]


# ### Another example of a search result

# In[86]:


# Another example, look at the th result:
exple_search_results['search_results'][1][0]['term']


# ### Using the 'terms/doc_counts' endpoint

# In[91]:


exple_terms_array = []
exple_terms_array.append(exple_search_results['search_results'][1][0]['term'])
exple_terms_array.append(exple_search_results['search_results'][2][0]['term'])
exple_terms_array.append(exple_search_results['search_results'][9][0]['term'])
exple_terms_array


# In[92]:


exple_stats = client.get('terms/doc_counts', terms=exple_terms_array, format='json')


# In[93]:


#the stats gives us the number of Exact & Related matches
for x in exple_stats:
    print(x)


# ### Counting on a Subset only

# In[94]:


example_subset


# In[96]:


#get counts on only a subset:
chosen_subset = example_subset[0]
print('terms/doc_counts Results on a few terms, with subset =', chosen_subset)
exple_stats_SUBSET = client.get('terms/doc_counts', terms=exple_terms_array, subset=chosen_subset, format='json')
exple_stats_SUBSET                                                                            


# ## Using the 'docs/vectors' endpoint to transform a list of terms in another format...

# In[97]:


a_list = input_topics[3]['Topic Terms']
a_list


# In[98]:


#putting in JSON format:
[{'text': a_list }]


# In[99]:


vector_ping = client.upload('docs/vectors', [{'text': a_list }] )
vector_ping


# In[100]:


vector_ping[0]['terms']


# In[101]:


#We select the data we need from result above, so as to have it in another format that we can re-use later
vector_ping_terms = [t for t,_,_ in vector_ping[0]['terms'] ]
vector_ping_terms


# ## ... to be used directly in the 'docs/search' endpoint, where we can filter by subset and get stats

# In[105]:


chosen_subset


# In[106]:


#here we use a subset filter on which to do the search, and a total of 3 (limit) of docs results.
#We get the total stats on the subset, and the full docs search as well. 
ping_count = client.get('docs/search', terms = vector_ping_terms, limit = 3, subset = chosen_subset)
ping_count


# # -----------------------------Topics-Docs Correlations------------------------------

# ## Using the topic document-counts endpoint (with or without a subset)

# In[107]:


#Get topic document-counts
topics_docs_counts = client.get('topics/doc_counts', format='json')
topics_docs_counts2 = substitute_keys_in_dict(topics_docs_counts, id_topicnames_table)
topics_docs_counts2


# In[109]:


#Get topic document-counts WITH a Subset
print('Subset =', chosen_subset)
topics_docs_counts_Subset = client.get('topics/doc_counts', format='json', subset=chosen_subset)
topics_docs_counts_Subset2 = substitute_keys_in_dict(topics_docs_counts_Subset, id_topicnames_table)
topics_docs_counts_Subset2


# ## Finding the topic correlations on a new text

# In[111]:


#Get topic correlation to text
newtext = 'This phone app does some weird connection on my internet'
newdoc_topics_correl = client.put('topics/text_correlation/', text = newtext )
newdoc_topics_correl2 = substitute_keys_in_dict(newdoc_topics_correl, id_topicnames_table)
print(newtext)
newdoc_topics_correl2


# In[114]:


newtext = "the app is bloody good, can't wait to see the new features"
newdoc_topics_correl = client.put('topics/text_correlation/', text = newtext )
newdoc_topics_correl2 = substitute_keys_in_dict(newdoc_topics_correl, id_topicnames_table)
print(newtext)
newdoc_topics_correl2


# ## Getting all the docs-topics correlations

# In[115]:


#for each document, this gives the topics-document correlations
docs_topics_correl = client.get('docs/correlations')
len(docs_topics_correl)


# In[116]:


nbr = 45
print(DOCS[nbr]['_id'])
DOCS[nbr]['text']


# In[118]:


one_doc_tc = docs_topics_correl['uuid-3395ea1fbdd44d8f82e164f7790fd95d']
one_doc_tc2 = substitute_keys_in_dict(one_doc_tc, id_topicnames_table)
one_doc_tc2


# # ------------------------ NEW PROJECTS & DOCS UPLOAD ----------------------

# ## Loading data from a CSV

# In[119]:


data_file_for_upload = 'API-TRAINING_BoA_docs_for_upload_(sample).csv'
data_input = load_from_CSV(data_file_for_upload)


# In[120]:


len(data_input)


# In[121]:


data_input[0]


# In[122]:


#get the data into the correct JSON format for upload, including subsets syntax:
data_for_upload = []
for d in data_input:
    obj = {}
    obj['text'] = d['text']
    obj['title'] = d['title']
    obj['date'] = extract_formatted_date(d['date'], DATE_FORMAT_ISO)
    obj['subsets'] = ['Month: ' + d['Month'], 
                      'Store: ' + d['Store'],
                      'Star Rating: ' + d['Star Rating'],
                      'Store + Rating: ' + d['Store + Rating'] ]
    obj['language'] = 'en'
    data_for_upload.append(obj)


# In[123]:


data_for_upload[0]


# ## Creating a New Project

# In[124]:


new_project_name = 'Example New Project (API training)'

# Create a new project
new_project = account_client.post(name = new_project_name)
new_project_client = account_client.change_path(new_project['project_id'])

new_project_client.upload('docs', data_for_upload)
print('Project created: ' +  new_project['name'])


# In[125]:


jobID = new_project_client.post('docs/recalculate', language='en')
jobID

