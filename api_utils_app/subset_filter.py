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

def subsets_to_remove(client, min_count):        
    subsets = client.get('subsets/stats')
    subsets_to_remove = []
    for subset in subsets:
        if subset['count'] < min_count:
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
        
def filter_subsets(client, account_id, project_id, proj_name, min_count, batch_size=10000):
    client = client.change_path(account_id + '/' + project_id)
    print('Getting all docs...')
    docs = get_all_docs(client)
    remove = subsets_to_remove(client, min_count)
    docs = modify_docs(docs, remove)
    
    client = LuminosoClient.connect('https://analytics.luminoso.com/api/v4/projects/{}/'.format(account_id))
    proj_id = client.post(name=proj_name)['project_id']
    client = connect(account_id, proj_id)

    batch = 0
    total_size = len(docs)
    print('Removing subsets...')
    while batch < total_size:
        end = min(total_size, batch + batch_size)
        client.upload('docs', docs=docs[batch:end])
        batch += batch_size
    client.post('docs/recalculate')
    while client.get('jobs'):
        time.sleep(1)
    print('Recalculation done.')