from luminoso_api import V5LuminosoClient as LuminosoClient
from pack64 import unpack64
from conjunctions_disjunctions import get_new_results
from subset_key_terms import subset_key_terms, create_skt_table
from score_drivers import get_as, get_all_docs, get_driver_fields, create_drivers_table
from scipy.stats import linregress

import csv
import json
import time
import sys
import datetime
import argparse
import numpy as np


def parse_url(url):
    api_root = url.strip('/ ').split('/app')[0] + '/api/v5'
    project_id = url.strip('/ ').split('/')[-1]
    return api_root, project_id

        
def pull_lumi_data(project, api_url, skt_limit, concept_count=100, interval='day', themes=7, theme_terms=4, token=None):
    '''
    Extract relevant data from Luminoso project
    :param project: Luminoso project id
    :param skt_limit: Number of terms per subset when creating subset key terms
    :param term_count: Number of top terms to include in the analysis
    :param interval: The appropriate time interval for trending ('day', 'week', 'month', 'year')
    :param themes: Number of themes to calculate
    :param theme_terms: Number of terms to represent each theme
    :param token: Authentication token
    :return: Return lists of dictionaries containing project data
    '''
    print('Extracting Lumi data...')
    if token:
        client = LuminosoClient.connect('{}/projects/{}'.format(api_url, project), token=token)
    else:
        client = LuminosoClient.connect('{}/projects/{}'.format(api_url, project))
    
    docs = get_all_docs(client)
    
    metadata = client.get('metadata')['result']
    saved_concepts = client.get('concepts/saved', include_science=True)
    concepts = client.get('concepts/match_counts', 
                          concept_selector={'type': 'top', 
                                            'limit': concept_count})['match_counts']
    
    subset_counts = {}
    for m in metadata:
        if m['type'] != 'date':
            subset_counts[m['name']] = {}
            for v in m['values']:
                subset_counts[m['name']][v['value']] = v['count']
                
    skt = subset_key_terms(client, subset_counts, len(docs), skt_limit)
    driver_fields = get_driver_fields(client)
    
    themes = client.get('concepts', concept_selector={'type': 'suggested', 
                                                      'num_clusters': themes,
                                                      'num_cluster_concepts': theme_terms})
    
    return client, docs, saved_concepts, concepts, metadata, driver_fields, skt, themes


def create_doc_term_table(docs, concepts):
    '''
    Creates a tabulated format for the relationships between docs & terms
    :param docs: List of document dictionaries
    :param concepts: List of concept dictionaries
    :return: List of dicts containing doc_ids, related terms, score & whether an exact match was found
    '''

    doc_term_table = []
    terms_in_docs = []
    term_in_doc = 0
    for doc in docs:
        if doc['vector']:
            for t in doc['terms']:
                terms_in_docs.append(t['term_id'])
            for c in concepts:
                if c['exact_term_ids'][0] in terms_in_docs:
                    term_in_doc = 1
                doc_term_table.append({'doc_id': doc['doc_id'],
                                       'term': c['name'],
                                       'exact_match': term_in_doc})
    return doc_term_table


def create_doc_topic_table(docs, saved_concepts):
    '''
    Create a tabulation of docs and topics they're related to
    :param docs: List of document dictionaries
    :param saved_concepts: List of saved concept dictionaries
    :return: List of document ids associated topic and score
    '''

    doc_topic_table = []
    for doc in docs:
        if doc['vector']:
            doc_vector = [float(v) for v in unpack64(doc['vector'])]
            max_score = 0
            max_topic = ''
            for c in saved_concepts:
                if c['vector']:
                    saved_concept_vector = [float(v) for v in unpack64(c['vector'])]
                    #if np.dot(doc_vector, topic_vector) >= .3:
                    score = np.dot(doc_vector, saved_concept_vector)
                    if score > max_score:
                        max_score = score
                        max_topic = c['name']
            doc_topic_table.append({'doc_id': doc['doc_id'], 
                                    'topic': max_topic,
                                    'association': max_score})
    return doc_topic_table


def create_topic_topic_table(saved_concepts):
    '''
    Create a tabulation of topic to topic relationships
    :param saved_concepts: List of saved concept dictionaries
    :return: List of topic pairs and association score
    '''

    topic_topic_table = []
    for concept in saved_concepts:
        for c in saved_concepts:
            if concept['vector'] and c['vector'] and concept['name'] != c['name']:
                concept_vector = [float(v) for v in unpack64(concept['vector'])]
                c_vector = [float(v) for v in unpack64(c['vector'])]
                topic_topic_table.append({'topic': concept['name'],
                                          'second topic': c['name'],
                                          'association': np.dot(concept_vector, c_vector)})
    return topic_topic_table


def create_term_topic_table(concepts, saved_concepts):
    '''
    Create a tabulation of topic to term relationships
    :param concepts: List of concept dictionaries
    :param saved_concepts: List of saved concept dictionaries
    :return: List of topics, terms and association score
    '''

    term_topic_table = []
    for concept in concepts:
        for saved_concept in saved_concepts:
            if concept['vector'] and saved_concept['vector']:
                concept_vector = [float(v) for v in unpack64(concept['vector'])]
                saved_concept_vector = [float(v) for v in unpack64(saved_concept['vector'])]
                term_topic_table.append({'term': concept['name'],
                                         'topic': saved_concept['name'],
                                         'association': np.dot(concept_vector, saved_concept_vector)})
    return term_topic_table


def create_doc_subset_table(docs, metadata_map):
    '''
    Create a tabulation of documents and associated subsets
    :param docs: List of document dictionaries
    :param metadata: List of metadata objects
    :return: List of document ids, subsets, subset names and subset values
    '''

    doc_subset_table = []
    for doc in docs:
        for m in doc['metadata']:
            doc_subset_table.append({'doc_id': doc['doc_id'],
                                     'subset': metadata_map[m['name']],
                                     'subset_name': m['name'],
                                     'value': m['value']})
    return doc_subset_table


def create_doc_table(client, docs, metadata):
    '''
    Create a tabulation of documents and their related subsets & themes
    :param client: LuminosoClient object set to project path
    :param docs: List of document dictionaries
    :param metadata: List of metadata dictionaries
    :return: List of documents with associated themes and list of cross-references between docs and subsets
    '''

    print('Creating doc table...')
    numeric_metadata = [m for m in metadata if m['type'] == 'number']
    string_metadata = [m for m in metadata if m['type'] == 'string']
    date_metadata = [m for m in metadata if m['type'] == 'date']
    metadata_map = {}
    for i, m in enumerate(numeric_metadata):
        metadata_map[m['name']] = 'Subset %d' % i
    for i, m in enumerate(string_metadata):
        metadata_map[m['name']] = 'Subset %d' % (i + len(numeric_metadata))
    for i, m in enumerate(date_metadata):
        metadata_map[m['name']] = 'Subset %d' % (i + len(numeric_metadata) + len(date_metadata))
        
    doc_table = []
        
    for doc in docs:
        row = {}
        row['doc_id'] = doc['doc_id']
        row['doc_text'] = doc['text']
        date_number = 0
        for m in doc['metadata']:
            if m['type'] == 'date':
                row['doc_date %d' % date_number] = '%s %s' % (m['value'].split('T')[0], 
                                                              m['value'].split('T')[1].split('.')[0])
                date_number += 1
            row[metadata_map[m['name']]] = m['value']
        if date_number == 0:
            row['doc_date 0'] = 0
        doc_table.append(row)
        
    xref_table = [metadata_map]
    return doc_table, xref_table, metadata_map


def create_terms_table(concepts):
    '''
    Create a tabulation of top terms and their exact/total match counts
    :param concepts: List of concept dictionaries
    :return: List of terms, and match counts
    '''

    print('Creating terms table...')
    table = []
    for c in concepts:
        row = {}
        row['Term'] = c['name']
        row['Exact Matches'] = c['exact_match_count']
        row['Related Matches'] = c['match_count'] - c['exact_match_count']
        table.append(row)
    return table


def create_themes_table(client, suggested_topics):
    print('Creating themes table...')
    cluster_labels = {}
    themes = []
    for r in suggested_topics['result']:
        if r['cluster_label'] not in cluster_labels:
            cluster_labels[r['cluster_label']] = {'id': 'Theme %d' % len(cluster_labels),
                                                  'name': []}
        cluster_labels[r['cluster_label']]['name'].append(r['name'])
    for c in cluster_labels:
        row = {}
        row['cluster_label'] = c
        row['name'] = ', '.join(cluster_labels[c]['name'])
        row['id'] = cluster_labels[c]['id']
        selector = [{'texts': [t]} for t in cluster_labels[c]['name']]
        count = 0
        match_counts = client.get('concepts/match_counts', concept_selector={'type': 'specified', 'concepts': selector})['match_counts']
        for m in match_counts:
            count += m['exact_match_count']
        row['docs'] = count
        themes.append(row)
    return themes
                    
"""
def create_trends_table(terms, docs):
    '''
    Creates tabulation of terms and their association per document in order to compute trends over time
    :param terms: List of term dictionaries
    :param docs: List of document dictionaries
    :return: List of documents with associated terms and scores as well as a list of terms and slopes for
    preset percentages of the overall timeine.
    '''

    term_list = []
    for t in terms:
        if t['vector'] != None:
            term_list.append([float(v) for v in unpack64(t['vector'])])
        else:
            term_list.append([0 for i in range(len(term_list[0]))])
    term_vecs = np.asarray(term_list)
    #term_vecs = np.asarray([unpack64(t['vector']) if t['vector'] != None for t in terms])
    concept_list = [t['text'] for t in terms]

    dated_docs = [d for d in docs if 'date' in d]
    dated_docs.sort(key = lambda k: k['date'])
    dates = np.asarray([[datetime.datetime.fromtimestamp(int(d['date'])).strftime('%Y-%m-%d %H:%M:%S')] for d in dated_docs])

    doc_vecs = np.asarray([[float(v) for v in unpack64(t['vector'])] for t in dated_docs])
    
    if len(doc_vecs) > 0:

        results = np.dot(term_vecs, np.transpose(doc_vecs))
        results = np.transpose(results)
        idx = [[x] for x in range(0, len(results))]
        results = np.hstack((idx, results))

        headers = ['Date','Index']
        headers.extend(concept_list)

        tenth = int(.9 * len(results))
        quarter = int(.75 * len(results))
        half = int(.5 * len(results))

        slopes = [linregress(results[:,0], results[:,x+1])[0] for x in range(len(results[0])-1)]
        slope_ranking = zip(concept_list, slopes)
        slope_ranking = sorted(slope_ranking, key=lambda rank:rank[1])
        slope_ranking = slope_ranking[::-1]

        tenth_slopes = [linregress(results[tenth:,0], results[tenth:,x+1])[0] for x in range(len(results[0]) - 1)]
        tenth_slope_ranking = zip(concept_list, tenth_slopes)
        tenth_slope_ranking = sorted(tenth_slope_ranking, key=lambda rank:rank[1])
        tenth_slope_ranking = tenth_slope_ranking[::-1]

        quarter_slopes = [linregress(results[quarter:,0], results[quarter:,x+1])[0] for x in range(len(results[0]) - 1)]
        quarter_slope_ranking = zip(concept_list, quarter_slopes)
        quarter_slope_ranking = sorted(quarter_slope_ranking, key=lambda rank:rank[1])
        quarter_slope_ranking = quarter_slope_ranking[::-1]

        half_slopes = [linregress(results[half:,0], results[half:,x+1])[0] for x in range(len(results[0]) - 1)]
        half_slope_ranking = zip(concept_list, half_slopes)
        half_slope_ranking = sorted(half_slope_ranking, key=lambda rank:rank[1])
        half_slope_ranking = half_slope_ranking[::-1]

        results = np.hstack((dates, results))
        trends_table = []
        for i in range(len(results)):
            for j in range(len(concept_list)):
                trends_table.append({'Date':results[i][0],
                                     'Index':results[i][1],
                                     'Term':concept_list[j],
                                     'Score':results[i][j + 2]})
        trendingterms_table = [{'Term':term, 
                                'Slope':slope, 
                                'Rank':slope_ranking.index((term, slope)) + 1, 
                                'Tenth term slope':tenth_slope, 
                                'Tenth term rank':tenth_slope_ranking.index((term, tenth_slope)) + 1, 
                                'Quarter term slope':quarter_slope,
                                'Quarter term rank':quarter_slope_ranking.index((term, quarter_slope)) + 1, 
                                'Half term slope':half_slope, 
                                'Half term rank':half_slope_ranking.index((term, half_slope)) + 1}
                               for term, slope, tenth_slope, quarter_slope, half_slope in zip(concept_list, slopes, tenth_slopes, quarter_slopes, half_slopes)]
    else:
        trends_table = []
        trendingterms_table = []
    return trends_table, trendingterms_table
"""    
    
def write_table_to_csv(table, filename):
    '''
    Function for writing lists of dictionaries to a CSV file
    :param table: List of dictionaries to be written
    :param filename: Filename to be written to (string)
    :return: None
    '''

    print('Writing to file {}.'.format(filename))
    if len(table) == 0:
        print('Warning: No data to write to {}.'.format(filename))
        return
    with open(filename, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=table[0].keys())
        writer.writeheader()
        writer.writerows(table)


def main():
    parser = argparse.ArgumentParser(
        description='Export data to Tableau compatible CSV files.'
    )
    parser.add_argument('project_url', help="The URL of the Daylight project to export from")
    parser.add_argument('-t', '--token', default=None, help="Enter your Daylight token")
    parser.add_argument('-c', '--concept_count', default=100, help="The number of top concepts to pull from the project")
    parser.add_argument('-sktl', '--skt_limit', default=20, help="The max number of subset key terms to display per subset")
    parser.add_argument('-docs', '--doc', default=False, action='store_true', help="Do not generate doc_table")
    parser.add_argument('-terms', '--terms', default=False, action='store_true', help="Do not generate terms_table")
    parser.add_argument('-theme', '--themes', default=False, action='store_true', help="Do not generate themes_table")
    parser.add_argument('-dterm', '--doc_term', default=False, action='store_true', help="Do not generate doc_term_table")
    parser.add_argument('-tterm', '--term_topic', default=False, action='store_true', help="Do not generate term_topic_table")
    parser.add_argument('-dtopic', '--doc_topic', default=False, action='store_true', help="Do not generate doc_topic_table")
    parser.add_argument('-ttopic', '--topic_topic', default=False, action='store_true', help="Do not generate topic_topic_table")
    parser.add_argument('-dsubset', '--doc_subset', default=False, action='store_true', help="Do not generate doc_subset_table")
    #parser.add_argument('-trends', '--trend_tables', default=False, action='store_true', help="Do not generate trends_table and trendingterms_table")
    parser.add_argument('-skt', '--skt_table', default=False, action='store_true',help="Do not generate skt_tables")
    parser.add_argument('-drive', '--drive', default=False, action='store_true',help="Do not generate driver_table")
    parser.add_argument('-tdrive', '--topic_drive', default=False, action='store_true', help="Generate drivers_table with topics instead of drivers")
    args = parser.parse_args()
    
    api_url, proj = parse_url(args.project_url)
    
    if args.token:
        client, docs, saved_concepts, concepts, metadata, driver_fields, skt, themes = pull_lumi_data(proj, api_url, skt_limit=int(args.skt_limit), concept_count=int(args.concept_count), token=args.token)
    else:
        client, docs, saved_concepts, concepts, metadata, driver_fields, skt, themes = pull_lumi_data(proj, api_url, skt_limit=int(args.skt_limit), concept_count=int(args.concept_count))

    if not args.doc:
        doc_table, xref_table, metadata_map = create_doc_table(client, docs, metadata)
        write_table_to_csv(doc_table, 'doc_table.csv')
        write_table_to_csv(xref_table, 'xref_table.csv')
    
    if not args.terms:
        terms_table = create_terms_table(concepts)
        write_table_to_csv(terms_table, 'terms_table.csv')
        
    if not args.themes:
        themes_table = create_themes_table(client, themes)
        write_table_to_csv(themes_table, 'themes_table.csv')
        
    if not args.doc_term:
        doc_term_table = create_doc_term_table(docs, concepts)
        write_table_to_csv(doc_term_table, 'doc_term_table.csv')
    
    if not args.doc_topic:
        doc_topic_table = create_doc_topic_table(docs, saved_concepts)
        write_table_to_csv(doc_topic_table, 'doc_topic_table.csv')
        
    if not args.topic_topic:
        topic_topic_table = create_topic_topic_table(saved_concepts)
        write_table_to_csv(topic_topic_table, 'topic_topic_table.csv')
        
    if not args.term_topic:
        term_topic_table = create_term_topic_table(concepts, saved_concepts)
        write_table_to_csv(term_topic_table, 'term_topic_table.csv')
        
    if not args.doc_subset:
        doc_subset_table = create_doc_subset_table(docs, metadata_map)
        write_table_to_csv(doc_subset_table, 'doc_subset_table.csv')

    if not args.skt_table:
        skt_table = create_skt_table(client, skt)
        write_table_to_csv(skt_table, 'skt_table.csv')
    
    if not args.drive:
        driver_table = create_drivers_table(client, driver_fields, args.topic_drive)
        write_table_to_csv(driver_table, 'drivers_table.csv')
    
    #if not args.trend_tables:
    #    trends_table, trendingterms_table = create_trends_table(terms, docs)
    #    write_table_to_csv(trends_table, 'trends_table.csv')
    #    write_table_to_csv(trendingterms_table, 'trendingterms_table.csv')

if __name__ == '__main__':
    main()
