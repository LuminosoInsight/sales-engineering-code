from luminoso_api import LuminosoClient
from pack64 import unpack64
import sys, json, time
import numpy as np
import argparse

def connect(account_id, project_id):
    client = LuminosoClient.connect(
        'https://analytics.luminoso.com/api/v4/projects/{}/{}'.format(account_id, project_id))
    return client

def get_all_docs(client):
    docs = []
    while True:
        new_docs = client.get('docs', limit=25000, offset=len(docs))
        if new_docs:
            docs.extend(new_docs)
        else:
            return docs

def subsets_to_remove(client, count, only, subset_name, more=False):   
    print(only)
    print(subset_name)
    subsets = client.get('subsets/stats')
    subsets_to_remove = []
    if only:
        for subset in subsets:
            if not more:
                if subset['count'] <= count and subset['subset'].partition(':')[0].lower() == subset_name.lower():
                    subsets_to_remove.append(subset['subset'])
            else:
                if subset['count'] >= count and subset['subset'].partition(':')[0].lower() == subset_name.lower():
                    subsets_to_remove.append(subset['subset'])
    else:
        for subset in subsets:
            if not more:
                if subset['count'] <= count:
                    subsets_to_remove.append(subset['subset'])
            else:
                if subset['count'] >= count:
                    subsets_to_remove.append(subset['subset'])
    return subsets_to_remove

def modify_docs(docs, subsets_to_remove):
    for doc in docs:
        subset = []
        for doc_subset in doc['subsets']:
            if doc_subset not in subsets_to_remove:
                subset.append(doc_subset)
        doc['subsets'] = subset
    return docs
        
def filter_subsets(client, account_id, project_id, proj_name, subset_name, count, only, more, batch_size=10000):
    client = client.change_path(account_id + '/' + project_id)
    print('Getting all docs...')
    docs = get_all_docs(client)
    remove = subsets_to_remove(client, count, only, subset_name, more)
    docs = modify_docs(docs, remove)
    
    batch = 0
    total_size = len(docs)
    print('Removing subsets...')
    
    if proj_name != '':
        client = LuminosoClient.connect('https://analytics.luminoso.com/api/v4/projects/{}/'.format(account_id))
        proj_id = client.post(name=proj_name)['project_id']
        client = connect(account_id, proj_id)
        while batch < total_size:
            end = min(total_size, batch + batch_size)
            client.upload('docs', docs=docs[batch:end])
            batch += batch_size
    else:
        while batch < total_size:
            end = min(total_size, batch + batch_size)
            client.put_data('docs', json.dumps(docs[batch:end]), content_type='application/json')
            batch += batch_size
   
    client.post('docs/recalculate')
    while client.get('jobs'):
        time.sleep(1)
    print('Recalculation done.')