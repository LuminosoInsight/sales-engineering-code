import json

from luminoso_api import V5LuminosoClient

def read_documents(client,lumi_filter,max_docs=0):

    '''Pull documents using the specified filter'''

    docs = []
    while True:

        result = client.get('/docs', limit=5000, offset=len(docs),filter=lumi_filter)

        if result['result']:
            docs.extend(result['result'])
        else:
            break

        if max_docs>0 and len(docs)>=max_docs:
            break

    return(docs)

def search_documents(client,lumi_concept_search,max_docs=0):

        '''Pull documents using the specified concept selector'''

    docs = []
    limit = 5000
    while True:
        if max_docs>0 and limit>max_docs:
            limit = max_docs
        result = client.get('/docs', limit=limit, offset=len(docs),search=lumi_concept_search)
        if result['result']:
            docs.extend(result['result'])
        else:
            break
        if max_docs>0 and len(docs)>=max_docs:
            break
    return(docs)

def get_all_docs(client):

    '''Pull all docs from project, filtered by subset if specified'''

    docs = []
    offset = 0
    while True:

        new_docs = client.get('docs',
                                offset=offset,
                                limit=25000)

        if not new_docs:
            return docs

        docs.extend(new_docs)
        offset += 25000
