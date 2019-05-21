from luminoso_api import V5LuminosoClient as LuminosoClient
from pack64 import unpack64
import sys, json, time
import numpy as np
import argparse

def connect(project_id):
    client = LuminosoClient.connect(
        'https://analytics.luminoso.com/api/v5/projects/{}'.format(project_id))
    return client

def get_all_docs(client):

    '''Pull all docs from project'''

    docs = []
    offset = 0
    while True:
        new_docs = client.get('docs',
                              offset=offset,
                              limit=25000)['result']

        if not new_docs:
            return docs

        docs.extend(new_docs)
        offset += 25000

def search_all_docs(client, text, exact=False):
    docs_list = []
    offset = 0
    while True:
        if exact:
            new_docs = client.get('docs',
                                  search={'texts': [text]},
                                  offset=offset,
                                  limit=20000,
                                  exact_only=True
                                 )['result']
        else:
            new_docs = client.get('docs',
                                  search={'texts': [text]},
                                  offset=offset,
                                  limit=20000
                                 )['result']
        if new_docs:
            docs_list.extend(new_docs)
            offset += 20000
        else:
            return docs_list

def create_id_list(docs, negate):
    docs_id = []
    keep_id = []
    for doc in docs:
        if negate:
            if doc['match_score'] < .3:
                docs_id.append(doc['doc_id'])
            else:
                keep_id.append(doc['doc_id'])
        else:
            if doc['match_score'] >= .3:
                docs_id.append(doc['doc_id'])
            else:
                keep_id.append(doc['doc_id'])
    return docs_id, keep_id

def delete_docs(client, docs_id, name):
    print('Deleting docs...')
    client.post('docs/delete', doc_ids=docs_id)
    print('Docs deleted, project recalculating')
    client.put(name=name)
    client = client.post('build')
    
    
def filter_project(client, text, name, exact=False):
    name = client.get()['name']
    print('Copying project to "Copy of original %s"' % name)
    client.post('copy', name= 'Copy of original %s' % name)
    
    print('Searching for docs...')
    docs = search_all_docs(proj, text, exact)  
    
    docs_id, keep_id = create_id_list(docs, negate)
    delete_docs(client, docs_id, name)
    client.wait_for_jobs()
    print('Recalculation done.')
