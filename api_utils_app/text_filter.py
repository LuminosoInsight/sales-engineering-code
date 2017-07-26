
from luminoso_api import LuminosoClient
from pack64 import unpack64
import sys, json, time
import numpy as np
import argparse

def connect(account_id, project_id):
    client = LuminosoClient.connect(
        'https://analytics.luminoso.com/api/v4/projects/{}/{}'.format(account_id, project_id))
    return client

def get_all_docs(client, subset=None):

    '''Pull all docs from project, filtered by subset if specified'''

    docs = []
    offset = 0
    while True:
        if subset:
            new_docs = client.get('docs',
                                  offset=offset,
                                  limit=25000,
                                  subset=subset)
        else:
            new_docs = client.get('docs',
                                  offset=offset,
                                  limit=25000)

        if not new_docs:
            return docs

        docs.extend(new_docs)
        offset += 25000

def search_all_docs(client, iterations, text, exact=False):
    docs_list = []
    offset = 0
    for i in range(0, iterations):
        if exact:
            new_docs = client.get('docs/search',
                                  text=text,
                                  start_at=offset,
                                  limit=20000,
                                  exact_only=True
                                 )
        else:
            new_docs = client.get('docs/search',
                                  text=text,
                                  start_at=offset,
                                  limit=20000
                                 )
        docs_list.append(new_docs)
        offset += 20000
    return docs_list

def split_loop(docs, limit):
    length = len(docs)
    iterations = length / limit
    iterations_int = int(iterations)
    if iterations > iterations_int:
        iterations = 1 + iterations_int
    else:
        iterations = iterations_int
    return iterations

def create_id_list(iterations, docs, negate):
    docs_id = []
    for i in range(0, iterations):
        search_results = docs[i]['search_results']
        length = len(search_results)
        for j in range(0, length):
            if negate:
                if search_results[j][1] < .3:
                    docs_id.append(search_results[j][0]['document']['_id'])
            else:
                if search_results[j][1] >= .3:
                    docs_id.append(search_results[j][0]['document']['_id'])
    return docs_id

def branch_project(client, branch_name, text, docs_id):
    print('Branching Project')
    #branch_name = client.get()['name'] + '_branch({})'.format(text)
    post_results = client.post('project/branch', ids = docs_id, destination='{}'.format(branch_name))

def delete_docs(client, docs_id, index):
    print('Deleting docs...')
    length = len(docs_id)
    start = 0
    end = index
    while length > 0:
        client.delete('docs', ids=docs_id[start:end])
        start += index
        end += index
        length = max(0, length - index)

    print('Docs deleted, project recalculating')
    client = client.post('docs/recalculate')
    
def filter_project(client, acc_id, proj_id, branch_name, text, not_related=False, branch=False, exact=False):
    proj = client.change_path(acc_id + '/' + proj_id)#connect(acc_id, proj_id)
    docs = get_all_docs(proj)
    iterations = split_loop(docs, 20000)
    
    print('Searching for docs...')
    docs = search_all_docs(proj, iterations, text, exact)    
    
    to_branch = text
    
    if exact:
        to_branch = to_branch + '_exact'
    
    if not_related:
        negate = True
        to_branch = 'not_' + to_branch
    else:
        negate = False
    
    
    docs_id = create_id_list(iterations, docs, negate)
   
    if branch:
        if branch_name == '':
            branch_name = proj.get()['name'] + '_branched({})'.format(to_branch)
        branch_project(proj, branch_name, text, docs_id)
    delete_docs(proj, docs_id, 600)
    while proj.get('jobs'):
        time.sleep(1)
    print('Recalculation done.')
    
#def main(args):
#    proj = connect(args.account_id, args.project_id)

#    text = args.text
#    docs = get_all_docs(proj)
#    iterations = split_loop(docs, 20000)
    
#    print('Searching for docs...')
#    docs = search_all_docs(proj, iterations, text, args.exact)

#    docs_id = create_id_list(iterations, docs, args.not_related)

#    if args.branch:
#        branch_project(proj, text, docs_id)

#    delete_docs(proj, docs_id, 600)
    
#if __name__ == '__main__':
#    parser = argparse.ArgumentParser(
#        description = 'Delete documents related to a term'
#    )
#    parser.add_argument(
#        '-u', '--username',
#        help='Username (email) of Luminoso account'
#    )
#    parser.add_argument(
#        '-a', '--api_url',
#        help='Luminoso API endpoint (https://eu-analytics.luminoso.com/api/v4)'     )
#    parser.add_argument(
#        'account_id',
#        help="The ID of the account that owns the project"
#    )
#    parser.add_argument(
#        'project_id',
#        help="The ID of the project"
#    )
#    parser.add_argument(
#        'text',
#        help="The text or term that you want to get rid of"
#    )
#    parser.add_argument(
#        '-n', '--not_related', default=False, action='store_true',
#        help="If you want to delete and branch project for terms NOT related to "
#        "given text."
 #   )
#    parser.add_argument(
#        '-e', '--exact', default=False, action='store_true',
#        help="If you want to find exact matches only"
#    )
#    parser.add_argument(
#        '-b', '--branch', default=False, action='store_true',
#        help="If you want to keep all the deleted documents by branching into separate project"
#    )
#    args = parser.parse_args()
#    main(args)
