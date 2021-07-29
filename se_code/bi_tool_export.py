import csv
import argparse
import numpy as np
import urllib

from collections import defaultdict 

from luminoso_api import V5LuminosoClient as LuminosoClient
from pack64 import unpack64
# from se_code.conjunctions_disjunctions import get_new_results
from se_code.subset_key_terms import subset_key_terms, create_skt_table
from se_code.score_drivers import get_all_docs, get_driver_fields, create_drivers_table, create_sdot_table, get_first_date_field, get_date_field_by_name, create_drivers_with_subsets_table


def parse_url(url):
    root_url = url.strip('/ ').split('/app')[0]
    api_url = root_url + '/api/v5'
    
    account_id = url.strip('/').split('/')[5]
    project_id = url.strip('/').split('/')[6]

    return root_url, api_url, account_id, project_id


def pull_lumi_data(project, api_url, skt_limit, concept_count=100, 
                   interval='day', themes=7, theme_terms=4, cln=None):

    '''
    Extract relevant data from Luminoso project
    :param project: Luminoso project id
    :param skt_limit: Number of terms per subset when creating subset key terms
    :param term_count: Number of top terms to include in the analysis
    :param interval: The appropriate time interval for trending ('day', 'week', 'month', 'year')
    :param themes: Number of themes to calculate
    :param theme_terms: Number of terms to represent each theme
    :param cln: Concept List Names a string of shared concept list names separated by |
    :return: Return lists of dictionaries containing project data
    '''
    print('Extracting Lumi data...')
    client = LuminosoClient.connect('{}/projects/{}'.format(api_url, project))
    
    if cln:
        concept_list_names = cln.split("|")
        concept_lists_raw = client.get("concept_lists/")
        concept_lists = [
            cl for cl in concept_lists_raw if cl["name"] in concept_list_names
        ]
        if not concept_lists:
            print(
                "Error: must specify at least one shared concept list. Lists available: {}".format(
                    [c["name"] for c in concept_lists_raw]
                )
            )
            concept_lists = None
    else:
        concept_lists = client.get("concept_lists/")

    docs = get_all_docs(client)
    
    metadata = client.get('metadata')['result']
    
    # saved_concepts = client.get('concepts/match_counts',
    #                            concept_selector={'type': 'saved'})['match_counts']
    # saved_concepts is now moving to a deeper dictionary with the saved_concept name as the key
    # and the data as the list of concepts.
    # For naming purposes scl = shared_concept_list
    scl_match_counts = {}
    for clist in concept_lists:
        concept_selector = {"type": "concept_list", "concept_list_id": clist['concept_list_id']}
        clist_match_counts = client.get('concepts/match_counts',concept_selector=concept_selector)
        clist_match_counts['concept_list_id'] = clist['concept_list_id']
        scl_match_counts[clist['name']] = clist_match_counts

    concepts = client.get('concepts/match_counts', 
                          concept_selector={'type': 'top', 
                                            'limit': concept_count})['match_counts']
    
    subset_counts = {}
    for m in metadata:
        if m['type'] == 'string':
            subset_counts[m['name']] = {}
            if len(m['values']) > 200:
                print("Subset {} has {} (too many) values. Reducing to first 200 values.".format(m['name'], len(m['values'])))
                for v in m['values'][:200]:
                    subset_counts[m['name']][v['value']] = v['count']
            else:
                for v in m['values']:
                    subset_counts[m['name']][v['value']] = v['count']

    skt = subset_key_terms(client, subset_counts, len(docs), skt_limit, threaded=True)
    
    driver_fields = get_driver_fields(client)
    
    themes = client.get('concepts', concept_selector={'type': 'suggested',
                                                      'num_clusters': themes,
                                                      'num_cluster_concepts': theme_terms})
    # set the theme_id values and unpack the vectors
    theme_id = ''
    cluster_labels = {}
    for r in themes['result']:
        if r['cluster_label'] not in cluster_labels:
            theme_id = 'Theme {}'.format(len(cluster_labels))
            cluster_labels[r['cluster_label']] = {'id': theme_id, 'name': []}
        # add the theme_id and unpack the vector
        r['theme_id'] = theme_id
        if ('vector' in r) and (len(r['vector']) > 0):
            r['fvector'] = [float(v) for v in unpack64(r['vector'])]

    return client, docs, scl_match_counts, concepts, metadata, driver_fields, skt, themes


def create_doc_term_table(docs, concepts, scl_match_counts):
    '''
    Creates a tabulated format for the relationships between docs & terms
    :param docs: List of document dictionaries
    :param concepts: List of concept dictionaries
    :param scl_match_counts: This list of matchcounts for each shared concept list (scl)
    :return: List of dicts containing doc_ids, related terms, score & whether an exact match was found
    '''

    doc_term_table = []

    concept_ids = defaultdict(list)
    for c in concepts:
        for t in c['exact_term_ids']:
            concept_ids[t].append((c['name'], 'top', None))

    for scl_name, saved_concepts in scl_match_counts.items():
        for s in saved_concepts['match_counts']:
            for t in s['exact_term_ids']:
                concept_ids[t].append((s['name'], 'saved', scl_name))
    
    doc_count = 0
    for doc in docs:
        concepts_in_doc = []
        if doc['vector']:
            for t in doc['terms']:
                if t['term_id'] in concept_ids:
                    for n in concept_ids[t['term_id']]:
                        if n not in concepts_in_doc:
                            concepts_in_doc.append(n)
                            doc_term_table.append({'doc_id': doc['doc_id'],
                                                   'term': n[0],
                                                   'exact_match': 1,
                                                   'concept_type': n[1],
                                                   'saved_concept_list': n[2]})
        doc_count += 1

    return doc_term_table


# def create_doc_topic_table(docs, saved_concepts):
#    '''
#    Create a tabulation of docs and topics they're related to
#    :param docs: List of document dictionaries
#    :param saved_concepts: List of saved concept dictionaries
#    :return: List of document ids associated topic and score
#    '''
#
#    doc_topic_table = []
#    for doc in docs:
#        if doc['vector']:
#            doc_vector = [float(v) for v in unpack64(doc['vector'])]
#            max_score = 0
#            max_topic = ''
#            for c in saved_concepts:
#                if c['vector']:
#                    saved_concept_vector = [float(v) for v in unpack64(c['vector'])]
#                    #if np.dot(doc_vector, topic_vector) >= .3:
#                    score = np.dot(doc_vector, saved_concept_vector)
#                    if score > max_score:
#                        max_score = score
#                        max_topic = c['name']
#            doc_topic_table.append({'doc_id': doc['doc_id'], 
#                                    'topic': max_topic,
#                                    'association': max_score})
#    return doc_topic_table


# def create_topic_topic_table(saved_concepts):
#    '''
#    Create a tabulation of topic to topic relationships
#    :param saved_concepts: List of saved concept dictionaries
#    :return: List of topic pairs and association score
#    '''
#
#    topic_topic_table = []
#    for concept in saved_concepts:
#        for c in saved_concepts:
#            if concept['vector'] and c['vector'] and concept['name'] != c['name']:
#                concept_vector = [float(v) for v in unpack64(concept['vector'])]
#                c_vector = [float(v) for v in unpack64(c['vector'])]
#                topic_topic_table.append({'topic': concept['name'],
#                                          'second topic': c['name'],
#                                          'association': np.dot(concept_vector, c_vector)})
#    return topic_topic_table


# def create_term_topic_table(concepts, saved_concepts):
#    '''
#    Create a tabulation of topic to term relationships
#    :param concepts: List of concept dictionaries
#    :param saved_concepts: List of saved concept dictionaries
#    :return: List of topics, terms and association score
#    '''
#
#    term_topic_table = []
#    for concept in concepts:
#        for saved_concept in saved_concepts:
#            if concept['vector'] and saved_concept['vector']:
#                concept_vector = [float(v) for v in unpack64(concept['vector'])]
#                saved_concept_vector = [float(v) for v in unpack64(saved_concept['vector'])]
#                term_topic_table.append({'term': concept['name'],
#                                         'topic': saved_concept['name'],
#                                         'association': np.dot(concept_vector, saved_concept_vector)})
#    return term_topic_table


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


def create_doc_table(client, docs, metadata, suggested_concepts, 
                     sentiment=False):
    '''
    Create a tabulation of documents and their related subsets & themes
    :param client: LuminosoClient object set to project path
    :param docs: List of document dictionaries
    :param metadata: List of metadata dictionaries
    :param suggested_concepts: The results from /concepts for suggested_concepts (same as themes)
    :param sentiment: Include doc level sentiment
    :return: List of documents with associated themes and list of cross-references between docs and subsets
    '''

    print('Creating doc table...')
    numeric_metadata = [m for m in metadata if m['type'] == 'number' or m['type'] == 'score']
    string_metadata = [m for m in metadata if m['type'] == 'string']
    date_metadata = [m for m in metadata if m['type'] == 'date']
    metadata_map = {}
    for i, m in enumerate(numeric_metadata):
        metadata_map[m['name']] = 'Subset %d' % i
    for i, m in enumerate(string_metadata):
        metadata_map[m['name']] = 'Subset %d' % (i + len(numeric_metadata))
    for i, m in enumerate(date_metadata):
        metadata_map[m['name']] = 'Subset %d' % (i + len(numeric_metadata) +
                                                 len(date_metadata))

    doc_table = []
        
    for doc in docs:
        row = {}
        row['doc_id'] = doc['doc_id']
        row['doc_text'] = doc['text']
        date_number = 0
        for m in doc['metadata']:
            if m['type'] == 'date':
                row['doc_date %d' % date_number] = '%s %s' % (
                    m['value'].split('T')[0], 
                    m['value'].split('T')[1].split('.')[0])
                date_number += 1
            row[metadata_map[m['name']]] = m['value']
        if date_number == 0:
            row['doc_date 0'] = 0
        if sentiment:
            row['sentiment'] = doc['sentiment']
            row['sentiment_confidence'] = doc['sentiment_confidence']
        
        # add the them (cluster) data
        doc['fvector'] = [float(v) for v in unpack64(doc['vector'])]

        max_score = 0
        max_id = ''
        for t in suggested_concepts['result']:
            if (len(t['vectors'][0])) > 0:
                t['fvector'] = [float(v) for v in unpack64(t['vectors'][0])]
                score = np.dot(doc['fvector'], t['fvector'])
                if score > max_score:
                    max_score = score
                    max_id = t['theme_id']

        row['theme_id'] = max_id
        row['theme_score'] = max_score

        doc_table.append(row)
        
    xref_table = [metadata_map]
    return doc_table, xref_table, metadata_map


def create_terms_table(concepts, scl_match_counts):
    '''
    Create a tabulation of top terms and their exact/total match counts
    :param concepts: List of concept dictionaries
    :param scl_match_counts: Dictionary of match_counts for each shared concept list to process
    :return: List of terms, and match counts
    '''

    print('Creating terms table...')
    table = []
    for c in concepts:
        row = {}
        row['term'] = c['name']
        row['exact_match_count'] = c['exact_match_count']
        row['related_match_count'] = c['match_count'] - c['exact_match_count']
        row['concept_type'] = 'top'
        table.append(row)
    for scl_name, saved_concepts in scl_match_counts.items():
        for s in saved_concepts['match_counts']:
            row = {}
            row['term'] = s['name']
            row['exact_match_count'] = s['exact_match_count']
            row['related_match_count'] = s['match_count'] - s['exact_match_count']
            row['concept_type'] = 'saved'
            row['saved_concept_list'] = scl_name
            table.append(row)
    return table


def create_themes_table(client, suggested_concepts):
    cluster_labels = {}
    themes = []

    # this is duplicating code done in get_lumi_data - may need refactor
    for r in suggested_concepts['result']:
        if r['cluster_label'] not in cluster_labels:
            cluster_labels[r['cluster_label']] = {'id': 'Theme %d' % len(cluster_labels),
                                                  'name': []}
        cluster_labels[r['cluster_label']]['name'].append(r['name'])
        
    for c in cluster_labels:
        # find related documents
        selector_docs = {'texts': cluster_labels[c]['name']}
        search_docs = client.get('docs', search=selector_docs, limit=3, match_type='exact')['result']
        
        selector = [{'texts': [t]} for t in cluster_labels[c]['name']]
        count = 0
        match_counts = client.get('concepts/match_counts', concept_selector={'type': 'specified', 'concepts': selector})['match_counts']
        for m in match_counts:
            count += m['exact_match_count']

        for sdoc in search_docs:
            row = {}
            row['cluster_label'] = c
            row['name'] = ', '.join(cluster_labels[c]['name'])
            row['id'] = cluster_labels[c]['id']
            row['docs'] = count
            row['doc_id'] = sdoc['doc_id']
            themes.append(row)
    return themes      


def create_sentiment_table(client, scl_match_counts, top_concepts, root_url=''):

    # first get the default sentiment output with sentiment suggestions
    results = client.get('/concepts/sentiment/')['match_counts']
    sentiment_match_counts = [{'texts': c['texts'],
                               'name': c['name'],
                               'concept_type': 'sentiment_suggestions',
                               'match_count': c['match_count'],
                               'exact_match_count': c['exact_match_count'],
                               'sentiment_share_positive': c['sentiment_share']['positive'],
                               'sentiment_share_neutral': c['sentiment_share']['neutral'],
                               'sentiment_share_negative': c['sentiment_share']['negative']
                               } for c in results]

    for scl_name, saved_concepts in scl_match_counts.items():
        concept_selector = {"type": "concept_list", "concept_list_id": saved_concepts['concept_list_id']}
        results_saved = client.get('/concepts/sentiment/',concept_selector=concept_selector)['match_counts']
        results_saved

        for c in results_saved:
            row = {'texts': c['texts'],
                   'name': c['name'],
                   'match_count': c['match_count'],
                   'concept_type': 'saved',
                   'saved_concept_list': scl_name,
                   'exact_match_count': c['exact_match_count'],
                   'sentiment_share_positive': c['sentiment_share']['positive'],
                   'sentiment_share_neutral': c['sentiment_share']['neutral'],
                   'sentiment_share_negative': c['sentiment_share']['negative']
                   }

            sentiment_match_counts.append(row)

    concept_selector = {"type": "top", 'limit': 100}
    results_top = client.get('/concepts/sentiment/',concept_selector=concept_selector)['match_counts']
    results_top

    for c in results_top:
        row = { 'texts': c['texts'],
                'name': c['name'],
                'match_count': c['match_count'],
                'concept_type': 'top',
                'exact_match_count': c['exact_match_count'],
                'sentiment_share_positive': c['sentiment_share']['positive'],
                'sentiment_share_neutral': c['sentiment_share']['neutral'],
                'sentiment_share_negative': c['sentiment_share']['negative']
                }

        sentiment_match_counts.append(row)

    # add three sample documents to each row
    for srow in sentiment_match_counts:

        if len(root_url)>0:
            srow['url'] = root_url+"/galaxy?suggesting=false&search="+urllib.parse.quote(" ".join(srow['texts']))

        # Use the driver term to find related documents
        search_docs = client.get('docs', search={'texts': srow['texts']}, limit=3, match_type='exact')['result']

        srow['example_doc'] = ''
        srow['example_doc2'] = ''
        srow['example_doc3'] = ''
        if len(search_docs) >= 1:
            srow['example_doc'] = search_docs[0]['text']
        if len(search_docs) >= 2:
            srow['example_doc2'] = search_docs[1]['text']
        if len(search_docs) >= 3:
            srow['example_doc3'] = search_docs[2]['text']

    return sentiment_match_counts

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
    
def write_table_to_csv(table, filename, calc_keys=False, encoding='utf-8'):
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
    with open(filename, 'w', newline='', encoding=encoding) as file:
        if calc_keys:
            fieldnames = {k for t_item in table for k in t_item.keys()}
        else:
            fieldnames = table[0].keys()

        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(table)


def main():
    parser = argparse.ArgumentParser(
        description='Export data to Business Intelligence compatible CSV files.'
    )
    parser.add_argument('project_url', help="The URL of the Daylight project to export from")
    parser.add_argument('-c', '--concept_count', default=20, help="The number of top concepts to pull from the project")
    parser.add_argument('-e', '--encoding', default='utf-8', help="Encoding of the file to write to")
    parser.add_argument("-l", "--concept_list_names", default=None, help="The names of this shared concept lists separated by |. Default = ALL lists")
    parser.add_argument('-sktl', '--skt_limit', default=20, help="The max number of subset key terms to display per subset")
    parser.add_argument('-docs', '--doc', default=False, action='store_true', help="Do not generate doc_table")
    parser.add_argument('-terms', '--terms', default=False, action='store_true', help="Do not generate terms_table")
    parser.add_argument('-theme', '--themes', default=False, action='store_true', help="Do not generate themes_table")
    parser.add_argument('-dterm', '--doc_term', default=False, action='store_true', help="Do not generate doc_term_table")
    parser.add_argument('-dsubset', '--doc_subset', default=False, action='store_true', help="Do not generate doc_subset_table")
    parser.add_argument('-skt', '--skt_table', default=False, action='store_true',help="Do not generate skt_tables")
    parser.add_argument('-drive', '--drive', default=False, action='store_true',help="Do not generate driver_table")
    parser.add_argument('-tdrive', '--topic_drive', default=False, action='store_true', help="If generating drivers_table do so with saved/top concepts as well as auto concepts")
    parser.add_argument('--driver_subset', default=False, action='store_true', help="Do not generate score drivers by subset")
    parser.add_argument('--driver_subset_fields', default=None, help='Which subsets to include in score driver by subset. Default = All with < 200 unique values. Samp: "field1,field2"')
    
    parser.add_argument('-sentiment', '--sentiment', default=False, action='store_true', help="Do not generate sentiment for top concepts")
    parser.add_argument('--sdot', action='store_true', help="Calculate over time")
    parser.add_argument('--sdot_end',default=None, help="Last date to calculate sdot MM/DD/YYYY - algorithm works moving backwards in time.")
    parser.add_argument('--sdot_iterations',default=7, help="Number of over time samples")
    parser.add_argument('--sdot_range',default=None, help="Size of each sample: M,W,D. If none given, range type will be calculated for best fit")
    parser.add_argument('--sdot_date_field',default=None,help="The name of the date field. If none, the first date field will be used")
    args = parser.parse_args()
    
    root_url, api_url, acct, proj = parse_url(args.project_url)
    print("starting subset drivers - topics={}".format(args.topic_drive))

    client, docs, scl_match_counts, concepts, metadata, driver_fields, skt, themes = pull_lumi_data(proj, api_url, skt_limit=int(args.skt_limit), concept_count=int(args.concept_count), cln=args.concept_list_names)

    # get the docs no matter what because later data needs the metadata_map
    doc_table, xref_table, metadata_map = create_doc_table(client, docs, metadata, themes, sentiment=not args.sentiment)

    ui_project_url = root_url + '/app/projects/' + acct + '/' + proj

    if not args.driver_subset:
        driver_table = create_drivers_with_subsets_table(client, driver_fields, args.topic_drive, ui_project_url, args.driver_subset_fields)
        write_table_to_csv(driver_table, 'subset_drivers_table.csv', encoding=args.encoding)
 
    if not args.doc:
        write_table_to_csv(doc_table, 'doc_table.csv', calc_keys=True, encoding=args.encoding)
        write_table_to_csv(xref_table, 'xref_table.csv', calc_keys=True, encoding=args.encoding)
    
    if not args.terms:
        terms_table = create_terms_table(concepts, scl_match_counts)
        write_table_to_csv(terms_table, 'terms_table.csv', calc_keys=True, encoding=args.encoding)
        
    if not args.themes:
        print('Creating themes table...')
        themes_table = create_themes_table(client, themes)
        write_table_to_csv(themes_table, 'themes_table.csv', encoding=args.encoding)

    # Combines list of concepts and shared concept lists
    if not args.doc_term:
        doc_term_table = create_doc_term_table(docs, concepts, scl_match_counts)
        write_table_to_csv(doc_term_table, 'doc_term_table.csv', encoding=args.encoding)
        
    if not args.doc_subset:
        doc_subset_table = create_doc_subset_table(docs, metadata_map)
        write_table_to_csv(doc_subset_table, 'doc_subset_table.csv', encoding=args.encoding)
    if not args.skt_table:
        skt_table = create_skt_table(client, skt)
        write_table_to_csv(skt_table, 'skt_table.csv', encoding=args.encoding)

    if not args.drive:
        driver_table = create_drivers_table(client, driver_fields, args.topic_drive, ui_project_url)
        write_table_to_csv(driver_table, 'drivers_table.csv', encoding=args.encoding)
    
    if not args.sentiment:
        print('Creating sentiment table...')
        sentiment_table = create_sentiment_table(client, scl_match_counts, concepts, root_url=ui_project_url)
        write_table_to_csv(sentiment_table, 'sentiment.csv', calc_keys=True, encoding=args.encoding)
    
    if bool(args.sdot):

        if args.sdot_date_field == None:
            date_field_info = get_first_date_field(client, True)
            if date_field_info == None:
                print("ERROR no date field in project")
                return
        else:
            date_field_info = get_date_field_by_name(client, args.sdot_date_field)
            if date_field_info == None:
                print("ERROR: no date field name: {}".format(args.sdot_date_field))
                return

        sdot_table = create_sdot_table(client, driver_fields, date_field_info, args.sdot_end, int(args.sdot_iterations), args.sdot_range, args.topic_drive, root_url=ui_project_url, docs=docs)
        write_table_to_csv(sdot_table, 'sdot_table.csv', encoding=args.encoding)

if __name__ == '__main__':
    main()
