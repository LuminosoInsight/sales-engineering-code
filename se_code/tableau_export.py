from luminoso_api import LuminosoClient
from pack64 import unpack64
from conjunctions_disjunctions import get_new_results
from subset_key_terms import subset_key_terms
from scipy.stats import linregress

import csv
import json
import time
import sys
import datetime
import argparse
import numpy as np
import concurrent.futures
import threading


def get_as(vector1, vector2):
    '''
    Calculate the association score between two vectors
    :param vector1: First vector
    :param vector2: Second vector
    :return: Cosine similarity of two vectors
    '''

    return np.dot([float(v) for v in unpack64(vector1)], [float(v) for v in unpack64(vector2)])


def is_number(s):
    '''
    Detect whether a string is a number
    :param s: string to be tested
    :return: True/False
    '''

    try:
        float(s)
        return True
    except ValueError:
        return False


def reorder_subsets(subsets):
    '''
    Convert numeric subsets into numbers within a list
    :param subsets: List of subsets (strings)
    :return: Revised list of subsets
    '''

    new_subsets = []
    for s in subsets:
        if is_number(s['subset'].split(':')[-1]):
            new_subsets.insert(0, s)
        else:
            new_subsets.append(s)
    return new_subsets


session_local = threading.local()


def get_client(api_url=None, account=None, project=None):
    if not hasattr(session_local, "client"):
        session_local.client = LuminosoClient.connect(url='{}/projects/{}/{}'.format(api_url, account, project))
    return session_local.client


def pull_lumi_data(account, project, api_url, skt_limit, term_count=100, interval='day', themes=7, theme_terms=4, rebuild=False):
    '''
    Extract relevant data from Luminoso project
    :param account: Luminoso account id
    :param project: Luminoso project id
    :param skt_limit: Number of terms per subset when creating subset key terms
    :param term_count: Number of top terms to include in the analysis
    :param interval: The appropriate time interval for trending ('day', 'week', 'month', 'year')
    :param themes: Number of themes to calculate
    :param theme_terms: Number of terms to represent each theme
    :param rebuild: Whether or not to rebuild the Luminoso project
    :return: Return lists of dictionaries containing project data
    '''

    print('Extracting Lumi data...')
    client = get_client(api_url, account, project)
    subsets = client.get('subsets/stats')

    docs = []
    while True:
        new_docs = client.get('docs', limit=25000, offset=len(docs))
        if new_docs:
            docs.extend(new_docs)
        else:
            break
    
    topics = client.get('topics')
    themes = client.get('/terms/clusters/', num_clusters=themes, num_cluster_terms=theme_terms)
    terms = client.get('terms', limit=term_count)
    skt = subset_key_terms(client, terms_per_subset=skt_limit)

    drivers = list(set([key for d in docs for key in d['predict'].keys()]))
    exist_flag = True

    # See if any score drivers are present, if not, create some from subsets
    if not any(drivers):
        exist_flag = False
        drivers = []
        subset_headings = list(set([s['subset'].partition(':')[0] for s in subsets]))
        for subset in subset_headings:
            subset_values = [s['subset'].partition(':')[2] for s in subsets
                             if s['subset'].partition(':')[0] == subset]
            if all([is_number(v) for v in subset_values]):
                drivers.append(subset)
    
    if rebuild or not exist_flag:
        add_score_drivers_to_project(client, docs, drivers)
    return client, docs, topics, terms, subsets, drivers, skt, themes


def create_doc_term_table(docs, terms, threshold):
    '''
    Creates a tabulated format for the relationships between docs & terms
    :param docs: List of document dictionaries
    :param terms: List of term dictionaries
    :param threshold: Threshold for similarity when tagging docs with terms
    :return: List of dicts containing doc_ids, related terms, score & whether an exact match was found
    '''

    doc_term_table = []
    for doc in docs:
        if doc['vector']:
            terms_in_docs = []
            for t in doc['terms']:
                terms_in_docs.append(t[0])
            for term in terms:
                term_in_doc = 0
                if term['term'] in terms_in_docs:
                    term_in_doc = 1
                if term['vector']:
                    assoc_score = get_as(doc['vector'], term['vector'])
                    if assoc_score >= threshold:
                        doc_term_table.append({'doc_id': doc['_id'], 
                                               'term': term['text'],
                                               'association': assoc_score,
                                               'exact_match': term_in_doc})
    return doc_term_table


def create_doc_topic_table(docs, topics):
    '''
    Create a tabulation of docs and topics they're related to
    :param docs: List of document dictionaries
    :param topics: List of topic dictionaries
    :return: List of document ids associated topic and score
    '''

    doc_topic_table = []
    for doc in docs:
        if doc['vector']:
            doc_vector = [float(v) for v in unpack64(doc['vector'])]
            max_score = 0
            max_topic = ''
            for topic in topics:
                if topic['vector']:
                    topic_vector = [float(v) for v in unpack64(topic['vector'])]
                    #if np.dot(doc_vector, topic_vector) >= .3:
                    score = np.dot(doc_vector, topic_vector)
                    if score > max_score:
                        max_score = score
                        max_topic = topic['name']
            doc_topic_table.append({'doc_id': doc['_id'], 
                                    'topic': max_topic,
                                    'association': max_score})
    return doc_topic_table


def create_topic_topic_table(topics):
    '''
    Create a tabulation of topic to topic relationships
    :param topics: List of topic dictionaries
    :return: List of topic pairs and association score
    '''

    topic_topic_table = []
    for topic in topics:
        for t in topics:
            if topic['vector'] and t['vector'] and topic['name'] != t['name']:
                topic_vector = [float(v) for v in unpack64(topic['vector'])]
                t_vector = [float(v) for v in unpack64(t['vector'])]
                topic_topic_table.append({'topic': topic['name'],
                                          'second topic': t['name'],
                                          'association': np.dot(topic_vector, t_vector)})
    return topic_topic_table


def create_term_topic_table(terms, topics):
    '''
    Create a tabulation of topic to term relationships
    :param terms: List of term dictionaries
    :param topics: List of topic dictionaries
    :return: List of topics, terms and association score
    '''

    term_topic_table = []
    for term in terms:
        for t in topics:
            if term['vector'] and t['vector']:
                term_vector = [float(v) for v in unpack64(term['vector'])]
                topic_vector = [float(v) for v in unpack64(t['vector'])]
                term_topic_table.append({'term': term['text'],
                                         'topic': t['name'],
                                         'association': np.dot(term_vector, topic_vector)})
    return term_topic_table


def create_doc_subset_table(docs, subsets):
    '''
    Create a tabulation of documents and associated subsets
    :param docs: List of document dictionaries
    :param subsets: List of subsets (strings)
    :return: List of document ids, subsets, subset names and subset values
    '''

    doc_subset_table = []
    subset_headings = list(dict.fromkeys([s['subset'].partition(':')[0] for s in subsets]))
    subset_headings.remove('__all__')
    subset_headings = {s: i for i, s in enumerate(subset_headings)}
    for doc in docs:
        for h, n in subset_headings.items():
            value = ''
            for subset in doc['subsets']: 
                subset_partition = subset.partition(':')
                if subset_partition[0] in h:
                    value = subset_partition[2]
            if value != '' and 'null' not in value.lower():
                doc_subset_table.append({'doc_id': doc['_id'],
                                         'subset': 'Subset {}'.format(n),
                                         'subset_name': h,
                                         'value': value
                })
    return doc_subset_table


def create_doc_table(client, docs, subsets, themes, api_url, account, project):
    '''
    Create a tabulation of documents and their related subsets & themes
    :param client: LuminosoClient object set to project path
    :param docs: List of document dictionaries
    :param subsets: List of subsets (string)
    :param themes: List of theme dictionaries
    :return: List of documents with associated themes and list of cross-references between docs and subsets
    '''

    print('Creating doc table...')
    doc_table = []
    subset_headings = list(dict.fromkeys([s['subset'].partition(':')[0] for s in subsets]))
    subset_headings.remove('__all__')
    subset_headings = {s: i for i, s in enumerate(subset_headings)}
    info = []
    header = []
    for h,n in subset_headings.items():
        header.append('Subset {}'.format(n))
        info.append(h)

    # Get 20 documents related to a particular theme
    for i, theme in enumerate(themes):
        search_terms = [t['text'] for t in theme['terms']]
        theme['name'] = ', '.join(search_terms)[:-2]
        theme['docs'] = get_new_results(client, search_terms, [], 'docs', 20, 'conjunction', False)
        header.append('Theme {}'.format(i))
        info.append(theme['name'])
        
    for doc in docs:
        row = {}
        row['doc_id'] = doc['_id']
        row['doc_text'] = doc['text']
        if 'date' in doc:
            row['doc_date'] = datetime.datetime.fromtimestamp(int(doc['date'])).strftime('%Y-%m-%d %H:%M:%S')
        else:
            row['doc_date'] = 0
        # changed from subset # to subset (name)
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
    xref_table = []
    xref_dic = {}
    for i in range(len(header)):
        ref_dic = {header[i]:info[i]}
        xref_dic.update(ref_dic)
    xref_table.append(xref_dic)
    return doc_table, xref_table


def create_skt_table(client, skt):
    '''
    Create tabulation of subset key terms analysis (terms distinctive within a subset)
    :param client: LuminosoClient object pointed to project path
    :param skt: List of subset key terms dictionaries
    :return: List of subset key terms output with example documents & match counts
    '''

    print('Creating subset key terms table...')
    terms = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(get_term_document_counts, client, t, s) for s, t, o, p in skt]

        for future in concurrent.futures.as_completed(futures):
            terms.extend(future.result())
    
    terms = {t['text']: t for t in terms}
    skt_table = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(get_skt_row, client, terms, s, t, o, p) for s, t, o, p in skt]

        for future in concurrent.futures.as_completed(futures):
            skt_table.append(future.result())

    return skt_table


def get_term_document_counts(client, t, s):
    return client.get('terms/doc_counts', terms=[t['term']], subsets=[s], format='json')


def get_skt_row(client, terms, s, t, o, p):
    docs = client.get('docs/search', limit=3, text=t['text'], subset=s)
    doc_texts = [ids[0]['document']['text'] for ids in docs['search_results']]
    text_length = len(doc_texts)
    text_1 = ''
    text_2 = ''
    text_3 = ''
    if text_length == 1:
        text_1 = doc_texts[0]
    elif text_length == 2:
        text_1 = doc_texts[0]
        text_2 = doc_texts[1]
    elif text_length > 2:
        text_1 = doc_texts[0]
        text_2 = doc_texts[1]
        text_3 = doc_texts[2]
    return {'term': t['text'],
            'subset': s.partition(':')[0],
            'value': s.partition(':')[2],
            'odds_ratio': o,
            'p_value': p,
            'exact_matches': terms[t['text']]['num_exact_matches'],
            'conceptual_matches': terms[t['text']]['num_related_matches'],
            'Text 1': text_1,
            'Text 2': text_2,
            'Text 3': text_3,
            'total_matches': terms[t['text']]['num_exact_matches'] + terms[t['text']]['num_related_matches']}


def wait_for_jobs(client, text):
    '''
    Repeatedly test for project recalculation, display text when complete
    :param client: LuminosoClient object pointed to project path
    :param text: String to print when complete
    :return: None
    '''

    check_interval = 1
    time_waiting = 0

    while len(client.get()['running_jobs']) != 0:
        sys.stderr.write('\r\tWaiting for {} ({}sec)'.format(text, time_waiting))
        sys.stderr.flush()
        time.sleep(check_interval)
        time_waiting += check_interval

    if time_waiting > 0:
        sys.stderr.write('\n')


def add_score_drivers_to_project(client, docs, drivers):
    '''
    Create add data to 'predict' field to support creation of ScoreDrivers if none existed
    :param client: LuminosoClient object pointed to project path
    :param docs: List of document dictionaries
    :param drivers: List of subsets (string) that contain numerics (could be score drivers)
    :return: None
    '''
   
    mod_docs = []
    for doc in docs:
        predict = {}
        for subset_to_score in drivers:
            if subset_to_score in [a.split(':')[0] for a in doc['subsets']]:
                predict.update({subset_to_score: float([a for a in doc['subsets'] 
                         if subset_to_score.strip().lower() == a.split(':')[0].strip().lower()][0].split(':')[-1])})
        mod_docs.append({'_id': doc['_id'],
                         'predict': predict})
    client.put_data('docs', json.dumps(mod_docs), content_type='application/json')
    client.post('docs/recalculate')

    wait_for_jobs(client, 'recalculation')
    print('Done recalculating. Training...')
    client.post('prediction/train')
    wait_for_jobs(client, 'driver training')
    print('Done training.')


def create_terms_table(client, terms):
    '''
    Create a tabulation of top terms and their exact/total match counts
    :param client: LuminosoClient object pointed to a project path
    :param terms: List of term dictionaries
    :return: List of terms, and match counts
    '''

    print('Creating terms table...')
    table = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(docs_search_with_term, client, term) for term in terms]

        for future in concurrent.futures.as_completed(futures):
            row = future.result()
            table.append(row)

    return table


def docs_search_with_term(client, term):
    row = {'Term': term['text']}
    search_result = client.get('docs/search', terms=[term['term']])
    row['Exact Matches'] = search_result['num_exact_matches']
    row['Related Matches'] = search_result['num_related_matches']
    return row


def create_themes_table(themes):
    '''
    Create tabulation of themes and doc count for each theme
    :param themes: List of theme dictionaries
    :return: List of themes, terms, and doc count
    '''
    print('Creating themes table...')
    for i, theme in enumerate(themes):
        search_terms = [t['text'] for t in theme['terms']]
        theme['name'] = ', '.join(search_terms)
        # Changed from numerical to "Theme #"
        theme['id'] = 'Theme {}'.format(i)
        theme['docs'] = sum([t['distinct_doc_count'] for t in theme['terms']])
        del theme['terms']
    return themes


def create_drivers_table(client, drivers, topic_drive, average_score):
    '''
    Create tabulation of ScoreDrivers output, complete with doc counts, example docs, scores and driver clusters
    :param client: LuminosoClient object pointed to project path
    :param drivers: List of drivers (string)
    :param topic_drive: Whether or not to include topics as drivers (bool)
    :param average_score: Whether or not to compute the average score for documents containing drivers (bool)
    :return: List of drivers with scores, example docs, clusters and type
    '''

    driver_table = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for subset in drivers:
            if topic_drive:
                topic_drivers = client.put('prediction/drivers', predictor_name=subset)
                for driver in topic_drivers:
                    futures.append(executor.submit(get_row_for_score_driver, client, driver, subset, average_score,
                                                   driver_type='topic'))

            score_drivers = client.get('prediction/drivers', predictor_name=subset)

            for driver in score_drivers['negative']:
                futures.append(executor.submit(get_row_for_score_driver, client, driver, subset, average_score,
                                               score_drivers, driver_type='negative'))

            for driver in score_drivers['positive']:
                futures.append(executor.submit(get_row_for_score_driver, client, driver, subset, average_score,
                                               score_drivers, driver_type='positive'))

        for future in concurrent.futures.as_completed(futures):
            row = future.result()
            driver_table.append(row)

    return driver_table


def get_row_for_score_driver(client, driver, subset, average_score, score_drivers=None, driver_type=None):
    row = dict()
    row['driver'] = driver['text']
    row['subset'] = subset
    row['impact'] = driver['regressor_dot']
    row['score'] = driver['driver_score']

    if driver_type == 'topic':
        row = get_row_for_topic_score_driver(client, driver, subset, average_score, row)
    elif driver_type == 'negative':
        row = get_row_for_negative_score_driver(client, driver, subset, average_score, score_drivers, row)
    elif driver_type == 'positive':
        row = get_row_for_positive_score_driver(client, driver, subset, average_score, score_drivers, row)

    return row


def get_row_for_topic_score_driver(client, driver, subset, average_score, row):
    row['type'] = 'user_defined'

    # ADDED RELATED TERMS
    related_terms = driver['terms']

    list_terms = client.get('terms', terms=related_terms)
    row['related_terms'] = get_related_text(list_terms)

    row['doc_count'] = get_doc_count_sum(client, related_terms)

    # Use the driver term to find related documents
    search_docs = client.get('docs/search', terms=driver['terms'], limit=500, exact_only=True)

    # Sort documents based on their association with the coefficient vector
    for doc in search_docs['search_results']:
        document = doc[0]['document']
        document['driver_as'] = get_as(driver['vector'], document['vector'])

    docs = sorted(search_docs['search_results'], key=lambda k: k[0]['document']['driver_as'])

    # average score is EXPERIMENTAL
    if average_score:
        row['average_score'] = get_avg_score(docs, subset)

    row.update(get_example_docs(docs))

    return row


def get_row_for_negative_score_driver(client, driver, subset, average_score, score_drivers, row):
    row['type'] = 'auto_found'

    # ADDED RELATED TERMS
    related_terms = driver['similar_terms']

    list_terms = client.get('terms', terms=related_terms)
    row['related_terms'] = get_related_text(list_terms)

    row['doc_count'] = get_doc_count_sum(client, related_terms)

    # Use the driver term to find related documents
    search_docs = client.get('docs/search', terms=[driver['term']], limit=500, exact_only=True)

    # Sort documents based on their association with the coefficient vector
    for doc in search_docs['search_results']:
        document = doc[0]['document']
        document['driver_as'] = get_as(score_drivers['coefficient_vector'], document['vector'])

    docs = sorted(search_docs['search_results'], key=lambda k: k[0]['document']['driver_as'])

    # average score is EXPERIMENTAL
    if average_score:
        row['average_score'] = get_avg_score(docs, subset)

    row.update(get_example_docs(docs))

    return row


def get_row_for_positive_score_driver(client, driver, subset, average_score, score_drivers, row):
    row['type'] = 'auto_found'

    # ADDED RELATED TERMS
    related_terms = driver['similar_terms']

    list_terms = client.get('terms', terms=related_terms)
    row['related_terms'] = get_related_text(list_terms)

    row['doc_count'] = get_doc_count_sum(client, related_terms)

    # Use the driver term to find related documents
    search_docs = client.get('docs/search', terms=[driver['term']], limit=500, exact_only=True)

    # Sort documents based on their association with the coefficient vector
    for doc in search_docs['search_results']:
        document = doc[0]['document']
        document['driver_as'] = get_as(score_drivers['coefficient_vector'], document['vector'])

    docs = sorted(search_docs['search_results'], key=lambda k: -k[0]['document']['driver_as'])
    # average score is EXPERIMENTAL
    if average_score:
        row['average_score'] = get_avg_score(docs, subset)

    row.update(get_example_docs(docs))

    return row


def get_related_text(list_terms):
    related_text = []
    for term in list_terms:
        related_text.append(term['text'])

    return related_text


def get_doc_count_sum(client, related_terms):
    doc_count_terms_list = [related_terms[0]]
    doc_count = client.get('terms/doc_counts', terms=doc_count_terms_list, use_json=True)

    count_sum = 0
    for doc_dict in doc_count:
        count_sum += (doc_dict['num_exact_matches'] + doc_dict['num_related_matches'])
    return count_sum


def get_avg_score(docs, subset):
    avg_score = 0
    for score_doc in docs:
        for category in score_doc[0]['document']['subsets']:
            if subset in category:
                avg_score += int(category.split(':')[-1])
                break
    try:
        avg_score = float(avg_score / len(docs))
    except ZeroDivisionError:
        avg_score = 0

    return avg_score


def get_example_docs(docs):
    example_docs = dict()

    example_docs['example_doc'] = ''
    example_docs['example_doc2'] = ''
    example_docs['example_doc3'] = ''
    if len(docs) >= 1:
        example_docs['example_doc'] = docs[0][0]['document']['text']
    if len(docs) >= 2:
        example_docs['example_doc2'] = docs[1][0]['document']['text']
    if len(docs) >= 3:
        example_docs['example_doc3'] = docs[2][0]['document']['text']

    return example_docs


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
    
    
def write_table_to_csv(table, filename, encoding="utf-8"):
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
    with open(filename, 'w', encoding=encoding, newline='') as file:
        writer = csv.DictWriter(file, fieldnames=table[0].keys())
        writer.writeheader()
        writer.writerows(table)


def main():
    parser = argparse.ArgumentParser(
        description='Export data to Tableau compatible CSV files.'
    )
    parser.add_argument('project_url', help="The URL of the Daylight project to export from")
    parser.add_argument('-t', '--term_count', default=100, help="The number of top terms to pull from the project")
    parser.add_argument('-a', '--assoc_threshold', default=.5, help="The minimum association threshold to display")
    parser.add_argument('-skt', '--skt_limit', default=20, help="The max number of subset key terms to display per subset")
    parser.add_argument('-d', '--doc', default=False, action='store_true', help="If you really do not want doc_table")
    parser.add_argument('-terms', '--terms', default=False, action='store_true', help="Do not generate terms_table")
    parser.add_argument('-dterm', '--doc_term', default=False, action='store_true', help="Generate doc_term_table")
    parser.add_argument('-tterm', '--term_topic', default=False, action='store_true', help="Generate term_topic_table")
    parser.add_argument('-dtopic', '--doc_topic', default=False, action='store_true', help="Generate doc_topic_table")
    parser.add_argument('-ttopic', '--topic_topic', default=False, action='store_true', help="Generate topic_topic_table")
    parser.add_argument('-dsubset', '--doc_subset', default=False, action='store_true', help="Do not generate doc_subset_table")
    parser.add_argument('-themes', '--themes', default=False, action='store_true', help="Do not generate themes")
    parser.add_argument('-trends', '--trend_tables', default=False, action='store_true', help="Generate trends_table and trendingterms_table")
    parser.add_argument('-sktt', '--skt_table', default=False, action='store_true',help="Do not generate skt_tables")
    parser.add_argument('-drive', '--drive', default=False, action='store_true',help="Do not generate driver_table")
    parser.add_argument('-rebuild', '--rebuild', default=False, action='store_true',help="Rebuild drivers even if previous drivers exist")
    parser.add_argument('-tdrive', '--topic_drive', default=False, action='store_true', help="Generate drivers_table with topics instead of drivers")
    parser.add_argument('-avg', '--average_score', default=False, action='store_true', help="Add average scores to drivers_table")
    parser.add_argument('-e', '--encoding', default="utf-8", help="Encoding type of the files to write to")
    args = parser.parse_args()
    
    acct = args.project_url.strip('/').split('/')[-2]
    proj = args.project_url.strip('/').split('/')[-1]
    api_url = args.project_url.split('/app')[0] + '/api/v4'
    #api_url = '/'.join(args.project_url.strip('/').split('/')[:-4]).strip('/') + '/api/v4'
    
    client, docs, topics, terms, subsets, drivers, skt, themes = pull_lumi_data(acct, proj, api_url, skt_limit=int(args.skt_limit), term_count=int(args.term_count), rebuild=args.rebuild)
    subsets = reorder_subsets(subsets)

    if not args.doc:
        doc_table, xref_table = create_doc_table(client, docs, subsets, themes, api_url, acct, proj)
        write_table_to_csv(doc_table, 'doc_table.csv', encoding=args.encoding)
        write_table_to_csv(xref_table, 'xref_table.csv', encoding=args.encoding)
    
    if not args.terms:
        terms_table = create_terms_table(client, terms)
        write_table_to_csv(terms_table, 'terms_table.csv', encoding=args.encoding)
        
    if args.doc_term:
        doc_term_table = create_doc_term_table(docs, terms, float(args.assoc_threshold))
        write_table_to_csv(doc_term_table, 'doc_term_table.csv', encoding=args.encoding)
    
    if args.doc_topic:
        doc_topic_table = create_doc_topic_table(docs, topics)
        write_table_to_csv(doc_topic_table, 'doc_topic_table.csv', encoding=args.encoding)
        
    if args.topic_topic:
        topic_topic_table = create_topic_topic_table(topics)
        write_table_to_csv(topic_topic_table, 'topic_topic_table.csv', encoding=args.encoding)
        
    if args.term_topic:
        term_topic_table = create_term_topic_table(terms, topics)
        write_table_to_csv(term_topic_table, 'term_topic_table.csv', encoding=args.encoding)
        
    if not args.doc_subset:
        doc_subset_table = create_doc_subset_table(docs, subsets)
        write_table_to_csv(doc_subset_table, 'doc_subset_table.csv', encoding=args.encoding)

    if not args.themes:
        themes_table = create_themes_table(themes)
        write_table_to_csv(themes_table, 'themes_table.csv', encoding=args.encoding)

    if not args.skt_table:
        skt_table = create_skt_table(client, skt)
        write_table_to_csv(skt_table, 'skt_table.csv', encoding=args.encoding)
    
    if not args.drive:
        driver_table = create_drivers_table(client, drivers, args.topic_drive, args.average_score)
        write_table_to_csv(driver_table, 'drivers_table.csv', encoding=args.encoding)
    
    if args.trend_tables:
        trends_table, trendingterms_table = create_trends_table(terms, docs)
        write_table_to_csv(trends_table, 'trends_table.csv', encoding=args.encoding)
        write_table_to_csv(trendingterms_table, 'trendingterms_table.csv', encoding=args.encoding)

if __name__ == '__main__':
    main()
