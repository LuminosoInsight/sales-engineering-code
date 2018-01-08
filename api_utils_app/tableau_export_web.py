from luminoso_api import LuminosoClient
from pack64 import unpack64
import run_voting_classifier # need accuracy/coverage chart
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
import os


def get_as(vector1, vector2):
    return np.dot(unpack64(vector1), unpack64(vector2))


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def reorder_subsets(subsets):
    new_subsets = []
    for s in subsets:
        if is_number(s['subset'].partition(':')[2]):
            new_subsets.insert(0, s)
        else:
            new_subsets.append(s)
    return new_subsets

def pull_lumi_data(account, project, skt_limit, term_count=100, interval='day', themes=7, theme_terms=4):

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
    
    topics = client.get('topics')
    themes = client.get('/terms/clusters/', num_clusters=themes, num_cluster_terms=theme_terms)
    terms = client.get('terms', limit=term_count)
    terms_doc_count = client.get('terms/doc_counts', limit=term_count, format='json')
    skt = subset_key_terms(client, skt_limit)

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
    return client, docs, topics, terms, subsets, drivers, skt, themes


def create_doc_term_table(client, docs, terms, threshold):
    doc_term_table = []
    for doc in docs:
        if doc['vector']:
            doc_vector = unpack64(doc['vector'])
            for term in terms:
                if term['vector']:
                    term_vector = unpack64(term['vector'])
                    if np.dot(doc_vector, term_vector) >= threshold:
                        doc_term_table.append({'doc_id': doc['_id'], 
                                               'term': term['text'],
                                               'association': np.dot(doc_vector, term_vector)})
    return doc_term_table
    
def create_doc_topic_table(client, docs, topics):
    doc_topic_table = []
    for doc in docs:
        if doc['vector']:
            doc_vector = unpack64(doc['vector'])
            max_score = 0
            max_topic = ''
            for topic in topics:
                if topic['vector']:
                    topic_vector = unpack64(topic['vector'])
                    #if np.dot(doc_vector, topic_vector) >= .3:
                    score = np.dot(doc_vector, topic_vector)
                    if score > max_score:
                        max_score = score
                        max_topic = topic['text']
            doc_topic_table.append({'doc_id': doc['_id'], 
                                    'topic': max_topic,
                                    'association': max_score})
    return doc_topic_table

def create_doc_subset_table(client, docs, subsets):
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

def create_doc_table(client, docs, subsets, themes, drivers):

    print('Creating doc table...')
    doc_table = []
    xref_table = []
    subset_headings = list(dict.fromkeys([s['subset'].partition(':')[0] for s in subsets]))
    subset_headings.remove('__all__')
    subset_headings = {s: i for i, s in enumerate(subset_headings)}
    info = []
    header = []
    for h,n in subset_headings.items():
        header.append('Subset {}'.format(n))
        info.append(h)

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

    print('Creating subset key terms table...')
    terms_to_get = []
    length = 0
    terms = []
    for s, t, o, p in skt:
        if length > 15000 - len(t['term']):
            terms.extend(client.get('terms/doc_counts', terms=terms_to_get, format='json'))
            
            terms_to_get = []
            length = 0
        terms_to_get.append(t['term'])
        length += len(t['term'])
    if length > 0:
        terms.extend(client.get('terms/doc_counts', terms=terms_to_get, format='json'))
        
    
    terms = {t['text']: t for t in terms}
    skt_table = []
    index = 0
    for s, t, o, p in skt:   
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
        skt_table.append({'term': t['text'],
                          'subset': s.partition(':')[0],
                          'value': s.partition(':')[2],
                          'odds_ratio': o,
                          'p_value': p,
                          'exact_matches': terms[t['text']]['num_exact_matches'],
                          'conceptual_matches': terms[t['text']]['num_related_matches'],
                          'Text 1': text_1,
                          'Text 2': text_2,
                          'Text 3': text_3,
                          'total_matches': terms[t['text']]['num_exact_matches'] + terms[t['text']]['num_related_matches']})
        index += 1
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
    time_waiting = 0
    while True:
        if time_waiting%30 == 0:
            if len(client.get()['running_jobs']) == 0:
                break
        sys.stderr.write('\r\tWaiting for driver training ({}sec)'.format(time_waiting))
        time.sleep(30)
        time_waiting += 30
    print('Done training.')

def create_terms_table(client, terms):
    print('Creating terms table...')
    table = []
    for t in terms:
        row = {}
        row['Term'] = t['text']
        search_result = client.get('docs/search', terms=[t['term']])
        row['Exact Matches'] = search_result['num_exact_matches']
        row['Related Matches'] = search_result['num_related_matches']
        table.append(row)
    return table
    
def create_themes_table(client, themes):
    print('Creating themes table...')
    for i, theme in enumerate(themes):
        search_terms = [t['text'] for t in theme['terms']]
        theme['name'] = ', '.join(search_terms)
        # Changed from numerical to "Theme #"
        theme['id'] = 'Theme {}'.format(i)
        theme['docs'] = sum([t['distinct_doc_count'] for t in theme['terms']])
        del theme['terms']
    return themes
                    
def create_terms_table(client, terms):
    print('Creating terms table...')
    table = []
    for t in terms:
        row = {}
        row['Term'] = t['text']
        search_result = client.get('docs/search', terms=[t['term']])
        row['Exact Matches'] = search_result['num_exact_matches']
        row['Related Matches'] = search_result['num_related_matches']
        table.append(row)
    return table
    
def create_drivers_table(client, drivers, topic_drive, average_score):
    driver_table = []
    for subset in drivers:
        if topic_drive:
            topic_drivers = client.put('prediction/drivers', predictor_name=subset)
            for driver in topic_drivers:
                row = {}
                row['driver'] = driver['text']
                row['type'] = 'user_defined'
                row['subset'] = subset
                row['impact'] = driver['regressor_dot']
                row['score'] = driver['driver_score']
                # ADDED RELATED TERMS
                related_terms = driver['terms']
                list_terms = client.get('terms', terms=related_terms)
                doc_count_terms_list = [related_terms[0]]
                related_text = []
                for term in list_terms:
                    related_text.append(term['text'])
                row['related_terms'] = related_text
                doc_count = client.get('terms/doc_counts', terms=doc_count_terms_list, use_json=True)
                count_sum = 0
                for doc_dict in doc_count:
                    count_sum += (doc_dict['num_exact_matches'] + doc_dict['num_related_matches'])
                row['doc_count'] = count_sum

                # Use the driver term to find related documents
                search_docs = client.get('docs/search', terms=driver['terms'], limit=500, exact_only=True)

                # Sort documents based on their association with the coefficient vector
                for doc in search_docs['search_results']:
                    document = doc[0]['document']
                    document['driver_as'] = get_as(driver['vector'],document['vector'])

                docs = sorted(search_docs['search_results'], key=lambda k: k[0]['document']['driver_as']) 
                # EXPERIMENTAL
                if average_score:
                    avg_score = 0
                    for score_doc in docs:
                        for category in score_doc[0]['document']['subsets']:
                            if subset in category:
                                avg_score += int(category.split(':')[1])
                                break
                    avg_score = float(avg_score/len(docs))
                    row['average_score'] = avg_score
                    #
                row['example_doc'] = ''
                row['example_doc2'] = ''
                row['example_doc3'] = ''
                if len(docs) >= 1:
                    row['example_doc'] = docs[0][0]['document']['text']
                if len(docs) >= 2:
                    row['example_doc2'] = docs[1][0]['document']['text']
                if len(docs) >= 3:
                    row['example_doc3'] = docs[2][0]['document']['text']
                driver_table.append(row)
        score_drivers = client.get('prediction/drivers', predictor_name=subset)
        for driver in score_drivers['negative']:
            row = {}
            row['driver'] = driver['text']
            row['type'] = 'auto_found'
            row['subset'] = subset
            row['impact'] = driver['regressor_dot']
            row['score'] = driver['driver_score']
            # ADDED RELATED TERMS
            related_terms = driver['similar_terms']
            list_terms = client.get('terms', terms=related_terms)
            doc_count_terms_list = [related_terms[0]]
            related_text = []
            for term in list_terms:
                related_text.append(term['text'])
            row['related_terms'] = related_text
            doc_count = client.get('terms/doc_counts', terms=doc_count_terms_list, use_json=True)
            count_sum = 0
            for doc_dict in doc_count:
                count_sum += (doc_dict['num_exact_matches'] + doc_dict['num_related_matches'])
            row['doc_count'] = count_sum


                # Use the driver term to find related documents
            search_docs = client.get('docs/search', terms=[driver['term']], limit=500, exact_only=True)

                # Sort documents based on their association with the coefficient vector
            for doc in search_docs['search_results']:
                document = doc[0]['document']
                document['driver_as'] = get_as(score_drivers['coefficient_vector'],document['vector'])

            docs = sorted(search_docs['search_results'], key=lambda k: k[0]['document']['driver_as']) 
            # EXPERIMENTAL
            if average_score:
                avg_score = 0
                for score_doc in docs:
                    for category in score_doc[0]['document']['subsets']:
                        if subset in category:
                            avg_score += int(category.split(':')[1])
                            break
                avg_score = float(avg_score/len(docs))
                row['average_score'] = avg_score
            #
            row['example_doc'] = ''
            row['example_doc2'] = ''
            row['example_doc3'] = ''
            if len(docs) >= 1:
                row['example_doc'] = docs[0][0]['document']['text']
            if len(docs) >= 2:
                row['example_doc2'] = docs[1][0]['document']['text']
            if len(docs) >= 3:
                row['example_doc3'] = docs[2][0]['document']['text']
            driver_table.append(row)
        for driver in score_drivers['positive']:
            row = {}
            row['driver'] = driver['text']
            row['type'] = 'auto_found'
            row['subset'] = subset
            row['impact'] = driver['regressor_dot']
            row['score'] = driver['driver_score']
            related_terms = driver['similar_terms']
            list_terms = client.get('terms', terms=related_terms)
            doc_count_terms_list = [related_terms[0]]
            related_text = []
            for term in list_terms:
                related_text.append(term['text'])
            row['related_terms'] = related_text
            doc_count = client.get('terms/doc_counts', terms=doc_count_terms_list, use_json=True)
            count_sum = 0
            for doc_dict in doc_count:
                count_sum += (doc_dict['num_exact_matches'] + doc_dict['num_related_matches'])
            row['doc_count'] = count_sum

                # Use the driver term to find related documents
            search_docs = client.get('docs/search', terms=[driver['term']], limit=500, exact_only=True)

                # Sort documents based on their association with the coefficient vector
            for doc in search_docs['search_results']:
                document = doc[0]['document']
                document['driver_as'] = get_as(score_drivers['coefficient_vector'],document['vector'])

            docs = sorted(search_docs['search_results'], key=lambda k: -k[0]['document']['driver_as'])
            # EXPERIMENTAL
            if average_score:
                avg_score = 0
                for score_doc in docs:
                    for category in score_doc[0]['document']['subsets']:
                        if subset in category:
                            avg_score += int(category.split(':')[1])
                            break
                avg_score = float(avg_score/len(docs))
                row['average_score'] = avg_score
            #
            row['example_doc'] = ''
            row['example_doc2'] = ''
            row['example_doc3'] = ''
            if len(docs) >= 1:
                row['example_doc'] = docs[0][0]['document']['text']
            if len(docs) >= 2:
                row['example_doc2'] = docs[1][0]['document']['text']
            if len(docs) >= 3:
                row['example_doc3'] = docs[2][0]['document']['text']
            driver_table.append(row)
    
    return driver_table


def create_trends_table(terms, topics, docs):
    term_list = []
    for t in terms:
        if t['vector'] != None:
            term_list.append(unpack64(t['vector']))
        else:
            term_list.append([0 for i in range(len(term_list[0]))])
    term_vecs = np.asarray(term_list)
    #term_vecs = np.asarray([unpack64(t['vector']) if t['vector'] != None for t in terms])
    concept_list = [t['text'] for t in terms]

    dated_docs = [d for d in docs if 'date' in d]
    dated_docs.sort(key = lambda k: k['date'])
    dates = np.asarray([[datetime.datetime.fromtimestamp(int(d['date'])).strftime('%Y-%m-%d %H:%M:%S')] for d in dated_docs])

    doc_vecs = np.asarray([unpack64(t['vector']) for t in dated_docs])
    
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
        #trends_table = [{key:value for key, value in zip(headers, r)} for r in results]
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
    
    
def write_table_to_csv(table, foldername, filename):

    print('Writing to file {}.'.format(filename))
    if len(table) == 0:
        print('Warning: No data to write to {}.'.format(filename))
        return
    filename = '../../' + foldername + '/' + filename
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    with open(filename, 'w') as file:
        writer = csv.DictWriter(file, fieldnames=table[0].keys())
        writer.writeheader()
        writer.writerows(table)

#def main():
#    parser = argparse.ArgumentParser(
#        description='Export data to Tableau compatible CSV files.'
#    )
#    parser.add_argument('account_id', help="The ID of the account that owns the project, such as 'demo'")
#    parser.add_argument('project_id', help="The ID of the project to analyze, such as '2jsnm'")
#    parser.add_argument('-t', '--term_count', default=100, help="The number of top terms to pull from the project")
#    parser.add_argument('-a', '--assoc_threshold', default=.3, help="The minimum association threshold to display")
#    parser.add_argument('-skt', '--skt_limit', default=20, help="The max number of subset key terms to display per subset")
#    parser.add_argument('-dterm', '--doc_term', default=False, action='store_true', help="Generate doc_term_table")
#    parser.add_argument('-dtopic', '--doc_topic', default=False, action='store_true', help="Generate doc_topic_table")
    #parser.add_argument('-dsubset', '--doc_subset', default=False, action='store_true', help="Generate doc_subset_table")
#    parser.add_argument('-trends', '--trend_tables', default=False, action='store_true', help="Generate trends_table and trendingterms_table")
#    parser.add_argument('-tdrive', '--topic_drive', default=False, action='store_true', help="Generate drivers_table with topics instead of drivers")
#    args = parser.parse_args()
#
#    client, docs, topics, terms, subsets, drivers, skt, themes = pull_lumi_data(args.account_id, args.project_id, skt_limit=args.skt_limit, term_count=args.term_count)
#    subsets = reorder_subsets(subsets)

#    doc_table, xref_table = create_doc_table(client, docs, subsets, themes, drivers)
#    write_table_to_csv(doc_table, 'doc_table.csv')
#    write_table_to_csv(xref_table, 'xref_table.csv')
    
#    if args.doc_term:
#        doc_term_table = create_doc_term_table(client, docs, terms, args.assoc_threshold)
#        write_table_to_csv(doc_term_table, 'doc_term_table.csv')
    
#    if args.doc_topic:
#        doc_topic_table = create_doc_topic_table(client, docs, topics)
#        write_table_to_csv(doc_topic_table, 'doc_topic_table.csv')
        
    #if args.doc_subset:
#    doc_subset_table = create_doc_subset_table(client, docs, subsets)
#    write_table_to_csv(doc_subset_table, 'doc_subset_table.csv')

#    themes_table = create_themes_table(client, themes)
#    write_table_to_csv(themes_table, 'themes_table.csv')

#    skt_table = create_skt_table(client, skt)
#    write_table_to_csv(skt_table, 'skt_table.csv')
    
#    driver_table = create_drivers_table(client, drivers, args.topic_drive)
#    write_table_to_csv(driver_table, 'drivers_table.csv')
    
#    if args.trend_tables:
#        trends_table, trendingterms_table = create_trends_table(terms, topics, docs)
#        write_table_to_csv(trends_table, 'trends_table.csv')
#        write_table_to_csv(trendingterms_table, 'trendingterms_table.csv')

#if __name__ == '__main__':
#    main()