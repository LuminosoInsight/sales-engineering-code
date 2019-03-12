from luminoso_api import V5LuminosoClient as LuminosoClient
from pack64 import unpack64
import sys, json, time
import numpy as np
import argparse

def get_all_docs(client):
    docs = []
    while True:
        new_docs = client.get('docs', limit=25000, offset=len(docs))['result']
        if new_docs:
            docs.extend(new_docs)
        else:
            return docs

def subsets_to_remove(client, count, only, subset_name, more=False):   
    print(only)
    print(subset_name)
    metadata = [m for m in client_v5.get('metadata')['result'] if m['type'] == 'string']
    for m in metadata:
        if only:
            if m['name'].lower() == subset_name.lower():
                for v in m['values']:
                    if v['count'] <= count and not more:
                        subsets_to_remove.append('%s: %s' % (m['name'], m['value']))
                    if v['count'] >= count and more:
                        subsets_to_remove.append('%s: %s' % (m['name'], m['value']))
        else:
            for v in m['values']:
                    if v['count'] <= count and not more:
                        subsets_to_remove.append('%s: %s' % (m['name'], m['value']))
                    if v['count'] >= count and more:
                        subsets_to_remove.append('%s: %s' % (m['name'], m['value']))
    return subsets_to_remove

def modify_docs(docs, subsets_to_remove):
    new_docs = []
    for doc in docs:
        info = {}
        info['text'] = doc['text']
        info['title'] = doc['title']
        metadata = []
        for m in doc['metadata']:
            if m['type'] != 'string':
                metadata.append(m)
            if '%s: %s' % (m['name'], m['value']) not in subsets_to_remove:
                metadata.append(m)
        info['metadata'] = metadata
        new_docs.append(info)
    return new_docs
        
def filter_subsets(url, proj_name, subset_name, count, only, more, batch_size=10000):
    print('Getting all docs...')
    api_root = url.strip('/ ').split('/app')[0]
    proj_id = url.strip('/ ').split('/')[-1]
    client = LuminosoClient.connect('{}/api/v5/projects/{}'.format(api_root, proj_id))
    docs = get_all_docs(client)
    remove = subsets_to_remove(client, count, only, subset_name, more)
    docs = modify_docs(docs, remove)
    
    language = client.get()['language']
    account = client.get()['account_id']
    
    batch = 0
    total_size = len(docs)
    print('Removing subsets...')
    
    root_client = LuminosoClient.connect('{}/api/v5/projects/'.format(api_root))
    proj_id = root_client.post(name=proj_name, language=language, account_id=account)['project_id']
    client = LuminosoClient.connect('{}/api/v5/projects/{}'.format(api_root, proj_id))
    while batch < total_size:
        end = min(total_size, batch + batch_size)
        client.post('upload', docs=docs[batch:end])
        batch += batch_size
   
    client.post('build')
    client.wait_for_jobs()
    print('Recalculation done.')