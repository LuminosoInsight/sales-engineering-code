
# coding: utf-8

# # Script for Topic Expansion  & Topic Docs Correlations (version v5b)
# ## Part A:  Auto-Topic Expansion, with optional stats on subsets
# ## Part B:  Update Topics in Project & get Topics-Docs correlations (classification)

# ### Boaz Odier, bodier@luminoso.com - v5b: with all parameters at the top

# Note: The script here also takes in topics definitions from a CSV file, and saves those topics onto the given project.
# The format of the topic input file (topics_file = 'PA_Default_Topics.csv' in the code below) should be as follows, in csv format:
# 
# Topic Name,Topic Terms
# DESIGN1,"design, style, color, red"
# DESIGN2,"classic, elegant, simple, beautiful, fashion"
# MATERIAL1,"leather, primeknit"
# MATERIAL2,"soft"
# SERVICE,"refund, delivery, package, customer, service"

# ## Input Parameters

# In[1]:


#Parameters for topic expansion:
ASSOCIATION_THRESHOLD = .6  #only terms with association score above this will appear
chosen_subset = 'Franchise' #if you want to run the topic expans on a specific subset


# In[2]:


#provide the filename, from wich the topics will be loaded - must be in same folder as this file
topics_file = 'PA_Default_Topics.csv'


# In[3]:


#Options for Topic-Correlations export (Classfication)
USE_TOPIC_THRESHOLD = True
TOPIC_THRESHOLD = .35  #the threshold over which a document is classfied as inside a Topic.


# ## Connect to the Luminoso Project on which to run the analysis

# In[4]:


from luminoso_api import LuminosoClient


# In[5]:


#sconnection parameters
user_name = 'bodier@luminoso.com'  #change to your own email that you use to login to Luminoso

#use the correct one from below:
api_url = 'https://eu-analytics.luminoso.com/api/v4' 
#api_url = 'https://analytics.luminoso.com/api/v4' 

#adidas Product Analytics: s75r663v  
#adidas Training acount:  u46p858s
account_id = 'u46p858s' #this is the accountID for your workspace


# In[6]:


project_id = 'pr97kgx5'  

#Connect to that specific project
account_url = api_url + '/projects/' + account_id + '/'

client = LuminosoClient.connect(account_url + '/' + project_id, username=user_name)
project_info =  client.get()
proj_name = project_info['name']
print('Connected to project: ' +  proj_name, '[On workspace: ' + project_info['account_name'] +']')


# ## Helper functions for CSV Input/Output

# In[7]:


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


# ## Helper functions specific to Luminoso projects

# In[8]:


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


# In[9]:


def get_all_docs(client, doc_fields=None):
    '''
    Get all docs
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


# In[10]:


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


# In[11]:


def substitute_keys_in_dict(data_dict, new_keys_dict):
    '''
    Substitute the keys used in the data_dict, to the values returned by the new_keys_dict
    '''
    result = {}
    for k, v in data_dict.items():
        result[new_keys_dict[k]] = v
    return result


# In[12]:


def build_ID_TOPICNAME_dict(topicsdata):
    '''
    Given a Luminoso list of topics, as per API call to get('topics'), 
    build a topic ID to NAME mapping table (as a dict)
    '''
    result = {}
    for t in topicsdata:
        result[t['_id']] = t['name']
    return result


# In[13]:


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


# In[14]:


def delete_all_topics(connection):
    '''
    This function will delete al the topics (saved concepts) in a given project
    This is not reversible, so the output of the function are all the topics before deletion.
    '''
    project_info = connection.get()
    existing_topics = connection.get('topics')
    all_deleted = True
    
    for t in existing_topics:
        t_name = t['name']
        t_id = t['_id']
        result = client.delete('topics/id/' + t_id)
        if 'deleted' not in result.keys():
            print('Warning: The Topic: ', t_name, ' could not be deleted.')
            all_deleted = False

    if all_deleted:
        print('\nAll Topics have been DELETED for Project:',  project_info['name'], ', on workspace:', project_info['account_name'] )
    
    return existing_topics


# ## Import the Topics from a CSV file

# In[15]:


# Load all topics from the CSV Topic File
input_topics = load_from_CSV(topics_file)

print('')
for x in input_topics:
    print(x['Topic Name'], ':' ,  x['Topic Terms'])


# # PART A: Topic Expansion

# In[16]:


# Function to expand a topic list. 
# A connection to a project is needed, and optionally the subset to provide stats upon.

def expand_topic_list(connection, subfilter='__all__'):

    #default subfilter, if not given, is __all__ , which is the same as not giving any subfilter. 

    #get the subsets info and check the provided subfilter actually exists:
    project_subsets, project_subsetTypes = get_subsets_info(connection)
    
    if subfilter not in project_subsetTypes:
        print('Error: subfilter provided does not exist in Project')
        return 'error'

    subset_list = get_subset_elements(project_subsets, subfilter)
    
    #an empty array to fill with results 
    output_list = [] 

    #iterate through each master topic and its given associated list of terms.
    for item in input_topics: 

        input_topic = item['Topic Name']
        input_list = item['Topic Terms']

        # collect only the 'term' value from search results, with a high enough score threshold
        # this are the new expanded topic terms
        new_topic_terms_scores = { 
                    t['text']:[t['term'], s]  #for each text, give the term(s) & the score
                    for t,s in client.get('terms/search', text = input_list, limit = 100)['search_results']
                    if s > ASSOCIATION_THRESHOLD}
        
        new_topic_terms = [ t for t,s in new_topic_terms_scores.values() ]

        #iterate over each subset to get the counts:
        for sl in subset_list:
            
            #collect counting stats on each of the new terms
            topic_counts = client.get('terms/doc_counts', terms = new_topic_terms, subset = sl, format='json')
            
            if sl == '__all__':
                subset_type = ''
                subset_element = '__all__'
            else:
                (subset_type, subset_element) = sl.split(': ')

            for topic in topic_counts:
                topic['master_topic'] = input_topic  #add a label referencing the current main topic
                topic['new_topic'] = '{}'.format(topic['text'] not in input_list) #add label with True/False value
                topic['assoc_score'] = new_topic_terms_scores[topic['text']][1] #get the score using our data above
                topic['Subset_' + subset_type] = subset_element
                
            # Add all the results into our output_list variable (final results)
            output_list.extend(topic_counts)
            
            ## The remainder of the code is to find the Total on the input master topic itself. 

            #we put the input_list in a JSON-encoded array of strings [{}], 
            #and send it to 'docs/vectors' API endpoint
            master_topic_terms = client.upload('docs/vectors', [{'text': input_list }] )[0]['terms']
            master_topic_terms = [t for t,_,_ in master_topic_terms]

            #now we can use the docs/search endpoint to directly get the TOTAL COUNT on the original list of terms
            master_counts = client.get('docs/search', terms = master_topic_terms, limit = 1, subset = sl)

            #so we can now add to our output an extra line with total matching docs count for the master topic:
            output_list.append({'master_topic': input_topic,
                                'text': input_topic, 
                                'new_topic': 'False',
                                'assoc_score': 1,
                                'num_related_matches': master_counts['num_related_matches'],  #total for input list of terms
                                'num_exact_matches': master_counts['num_exact_matches'],      #total for input list of terms
                                'Subset_' + subset_type: subset_element})

    return output_list


# ### Run Topic Expansion without subset

# In[17]:


# Run with no subset
output_TopExpNoSubsets = expand_topic_list(client)


# In[29]:


outputfile_TopExpNS = proj_name + '_TopExp_(NoSubsets).csv'
save_to_CSV(outputfile_TopExpNS, output_TopExpNoSubsets)


# ### Run Topic Expansion with a subset

# In[30]:


#Get the list of subsets available (metadata)
project_subsets, project_subsetTypes = get_subsets_info(client)
project_subsetTypes


# In[31]:


#This line is to just illustrate what's in the Franchise Subset
exple_subset = get_subset_elements(project_subsets, chosen_subset)
exple_subset


# In[32]:


#Run with the chosen Subset:
output_TopExpWithSubset = expand_topic_list(client, subfilter=chosen_subset)


# In[33]:


outputfile_TopExpWiSub = proj_name + '_TopExp_(WithSubset_' + chosen_subset  + ').csv'
save_to_CSV(outputfile_TopExpWiSub, output_TopExpWithSubset)


# # PART B: Topics-Docs Correlations

# In[34]:


#Download all documents from the Project
DOCS = get_all_docs(client)
print(len(DOCS), 'documents loaded from project:', client.get()['name'])


# In[35]:


#Upload all new topics onto the Project, overwrite if they exist
refreshed_topics = create_or_update_topics(client, input_topics)

#use the helper function to get a mapping table from topic IDs to their Names
id_topicnames_table = build_ID_TOPICNAME_dict(refreshed_topics)


# In[36]:


#for each document, this gives the topics-document correlations
docs_topics_correl = client.get('docs/correlations')


# In[37]:


# choose which info to keep for export, and add the corresponding correlations to topics

docs_output = []
for d in DOCS:
    obj = {}
    obj['Luminoso_docID'] = d['_id']
    obj['text'] = d['text']
    obj['title'] = d['title']
    metadata = subset_array_to_dict(d['subsets'])
    for subset, value in metadata.items():
        obj['Subset_' + subset] = value
    topics_correl = docs_topics_correl[d['_id']]
    topics_correl = substitute_keys_in_dict(topics_correl, id_topicnames_table)
    #add correlation scores for each topic:
    for topic, correl in topics_correl.items():
        obj['Topic_' + topic] = correl
        # optionally add a YesNo field based on threshold parameter:
        if USE_TOPIC_THRESHOLD:
            if correl >= TOPIC_THRESHOLD:
                obj['Topic_' + topic + '_YesNo'] = 1
            else:
                obj['Topic_' + topic + '_YesNo'] = 0
    docs_output.append(obj)


# In[38]:


# Write to Output File
docs_output_file = proj_name + '_DocTopCorrel' + '.csv'
save_to_CSV(docs_output_file, docs_output)

