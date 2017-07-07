from luminoso_api import LuminosoClient
from pack64 import unpack64
import run_voting_classifier # need accuracy/coverage chart
from conjunctions_disjunctions import get_new_results
from subset_key_terms import subset_key_terms

import csv
import json
import time
import sys
import argparse
import numpy as np


def get_as(vector1, vector2):
    return np.dot(unpack64(vector1), unpack64(vector2))


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def pull_lumi_data(account, project, term_count=1000, interval='day', themes=7, theme_terms=4):

    print('Extracting Lumi data...')
    client = LuminosoClient.connect('/projects/{}/{}'.format(account, project))
    subsets = client.get('subsets/stats')

    docs = []
    while True:
        new_docs = client.get('docs', limit=25000, offset=len(docs))
        if new_docs:
            docs.extend(new_docs)
        else:
            break
    drivers = list(set([key for d in docs for key in d['predict'].keys()]))

    # See if any score drivers are present, if not, create some from subsets
    if not any(drivers):
        drivers = []
        subset_headings = list(set([s['subset'].partition(':')[0] for s in subsets]))
        for subset in subset_headings:
            subset_values = [s['subset'].partition(':')[2] for s in subsets
                             if s['subset'].partition(':')[0] == subset]
            if all([is_number(v) for v in subset_values]):
                drivers.append(subset)
        if drivers:
            add_score_drivers_to_project(client, docs, drivers)

    topics = client.get('topics')
    themes = client.get('/terms/clusters/', num_clusters=themes, num_cluster_terms=theme_terms)
    terms = client.get('terms', limit=term_count)
    terms_doc_count = client.get('terms/doc_counts', limit=term_count, format='json')
    terms = [dict(t, **tdc) for t, tdc in zip(terms, terms_doc_count)]
    terms = {t['text']: t for t in terms}
    timelines = client.get('topics/timeline_correlation', interval=interval, format='json')
    skt = subset_key_terms(client, 20)
    return client, docs, topics, terms, subsets, drivers, skt, timelines, themes


def create_doc_table(client, docs, subsets, themes):

    print('Creating doc table...')
    doc_table = []
    xref_table = []
    subset_headings = list(set([s['subset'].partition(':')[0] for s in subsets]))
    subset_headings = {s: i for i, s in enumerate(subset_headings)}
    xref_table.extend([{'Header': 'Subset {}'.format(n), 'Name': h} for h,n in subset_headings.items()])

    for i, theme in enumerate(themes):
        search_terms = [t['text'] for t in theme['terms']]
        theme['name'] = ', '.join(search_terms)[:-2]
        theme['docs'] = get_new_results(client, search_terms, [], 'docs', 20, 'conjunction', False)
        xref_table.append({'Header': 'Theme {}'.format(i), 'Name': theme['name']})

    for doc in docs:
        row = {}
        row['doc_id'] = doc['_id']
        row['doc_text'] = doc['text']
        if 'date' in doc:
            row['doc_date'] = doc['date']
        else:
            row['doc_date'] = 0
        row.update({'Subset {}'.format(i): '' for i in range(len(subset_headings))})
        row.update({'Subset {}_centrality'.format(i): 0 for i in range(len(subset_headings))})

        for subset in doc['subsets']:
            subset_partition = subset.partition(':')
            if subset_partition[0] in subset_headings:
                row['Subset {}'.format(subset_headings[subset_partition[0]])] = subset_partition[2]
                row['Subset {}_centrality'.format(subset_headings[subset_partition[0]])] = get_as(doc['vector'],
                    [s['mean'] for s in subsets if s['subset'] == subset][0])

        for i, theme in enumerate(themes):
            row['Theme {}'.format(i)] = 0
            if doc['_id'] in [d['_id'] for d in theme['docs']]:
                row['Theme {}'.format(i)] = [d['score'] for d in theme['docs'] if d['_id'] == doc['_id']][0]
        doc_table.append(row)
    return doc_table, xref_table


def create_skt_table(client, skt):

    print('Creating subset key terms table...')
    terms = client.get('terms/doc_counts',
                       terms=[t['term'] for _, t, _, _ in skt],
                       format='json')
    terms = {t['text']: t for t in terms}
    skt_table = [{'term': t['text'],
                  'subset': s.partition(':')[0],
                  'value': s.partition(':')[2],
                  'odds_ratio': o,
                  'p_value': p,
                  'exact_matches': terms[t['text']]['num_exact_matches'],
                  'conceptual_matches': terms[t['text']]['num_related_matches']}
                 for s, t, o, p in skt]
    return skt_table


def add_score_drivers_to_project(client, docs, drivers):
    mod_docs = []
    for doc in docs:
        for subset_to_score in drivers:
            if subset_to_score in [a.split(':')[0] for a in doc['subsets']]:
                mod_docs.append({'_id': doc['_id'],
                                 'predict': {subset_to_score: float([a for a in doc['subsets'] 
                                    if subset_to_score in a][0].split(':')[1])}})
    client.put_data('docs', json.dumps(mod_docs), content_type='application/json')
    client.post('docs/recalculate')

    time_waiting = 0
    while True:
        if time_waiting%30 == 0:
            if len(client.get()['running_jobs']) == 0:
                break
        sys.stderr.write('\r\tWaiting for recalculation ({}sec)'.format(time_waiting))
        time.sleep(30)
        time_waiting += 30
    print('Done recalculating. Training...')
    client.post('prediction/train')
    print('Done training.')


def create_themes_table(client, themes):
    print('Creating themes table...')
    for i, theme in enumerate(themes):
        search_terms = [t['text'] for t in theme['terms']]
        theme['name'] = ', '.join(search_terms)[:-2]
        theme['id'] = i
        theme['docs'] = sum([t['distinct_doc_count'] for t in theme['terms']])
        del theme['terms']
    return themes


def create_drivers_table(client, drivers):
    driver_table = []
    for subset in drivers:
        score_drivers = client.get('prediction/drivers', predictor_name=subset)
        for driver in score_drivers['negative']:
            row = {}
            row['driver'] = driver['text']
            row['subset'] = subset
            row['impact'] = driver['regressor_dot']
            row['score'] = driver['driver_score']
    
            # Use the driver term to find related documents
            search_docs = client.get('docs/search', terms=[driver['term']], limit=500, exact_only=True)
    
            # Sort documents based on their association with the coefficient vector
            for doc in search_docs['search_results']:
                document = doc[0]['document']
                document['driver_as'] = get_as(score_drivers['coefficient_vector'],document['vector'])

            docs = sorted(search_docs['search_results'], key=lambda k: k[0]['document']['driver_as'])
            row['example_doc'] = docs[0][0]['document']['text']
            driver_table.append(row)
        for driver in score_drivers['positive']:
            row = {}
            row['driver'] = driver['text']
            row['subset'] = subset
            row['impact'] = driver['regressor_dot']
            row['score'] = driver['driver_score']

            # Use the driver term to find related documents
            search_docs = client.get('docs/search', terms=[driver['term']], limit=500, exact_only=True)

            # Sort documents based on their association with the coefficient vector
            for doc in search_docs['search_results']:
                document = doc[0]['document']
                document['driver_as'] = get_as(score_drivers['coefficient_vector'],document['vector'])

            docs = sorted(search_docs['search_results'], key=lambda k: -k[0]['document']['driver_as'])
            row['example_doc'] = docs[0][0]['document']['text']
            driver_table.append(row)
    return driver_table


#def create_trends_table():
    
    
#def create_prediction_table():
    
    
#def create_pairings_table():
    
    
def write_table_to_csv(table, filename):

    print('Writing to file {}.'.format(filename))
    with open(filename, 'w') as file:
        writer = csv.DictWriter(file, fieldnames=table[0].keys())
        writer.writeheader()
        writer.writerows(table)


def main():
    parser = argparse.ArgumentParser(
        description='Export data to Tableau compatible CSV files.'
    )
    parser.add_argument('account_id', help="The ID of the account that owns the project, such as 'demo'")
    parser.add_argument('project_id', help="The ID of the project to analyze, such as '2jsnm'")
    args = parser.parse_args()

    client, docs, topics, terms, subsets, drivers, skt, timelines, themes = pull_lumi_data(args.account_id, args.project_id)

    doc_table, xref_table = create_doc_table(client, docs, subsets, themes)
    write_table_to_csv(doc_table, 'doc_table.csv')
    write_table_to_csv(xref_table, 'xref_table.csv')

    themes_table = create_themes_table(client, themes)
    write_table_to_csv(themes_table, 'themes_table.csv')

    skt_table = create_skt_table(client, skt)
    write_table_to_csv(skt_table, 'skt_table.csv')

    driver_table = create_drivers_table(client, drivers)
    write_table_to_csv(driver_table, 'drivers_table.csv')

if __name__ == '__main__':
    main()