import argparse
import numpy as np
import re
import sys
import urllib
from collections import defaultdict

from luminoso_api import V5LuminosoClient as LuminosoClient
from pack64 import unpack64
from se_code.subset_key_terms import subset_key_terms, create_skt_table
from se_code.score_drivers import (
    create_drivers_table, create_sdot_table,
    create_drivers_with_subsets_table, LuminosoData, write_table_to_csv
)


def parse_url(url):
    root_url = url.strip('/ ').split('/app')[0]
    api_url = root_url + '/api/v5'
    
    workspace_id = url.strip('/').split('/')[5]
    project_id = url.strip('/').split('/')[6]

    return root_url, api_url, workspace_id, project_id


def pull_lumi_data(project, api_url, skt_limit, concept_count=100,
                   num_themes=7, theme_concepts=4, cln=None):

    '''
    Extract relevant data from Luminoso project

    :param project: Luminoso project id
    :param skt_limit: Number of terms per subset when creating subset key terms
    :param concept_count: Number of top concepts to include in the analysis
    :param num_themes: Number of themes to calculate
    :param theme_concepts: Number of concepts to represent each theme
    :param cln: Concept List Names a string of shared concept list names
         separated by |
    :return: Return lists of dictionaries containing project data
    '''
    print('Extracting Lumi data...')
    client = LuminosoClient.connect('{}/projects/{}'.format(api_url, project))
    luminoso_data = LuminosoData(client)

    if cln:
        concept_list_names = cln.split("|")
        concept_lists_raw = client.get("concept_lists/")
        concept_lists = [
            cl for cl in concept_lists_raw if cl["name"] in concept_list_names
        ]
        if not concept_lists:
            print(
                "Error: must specify at least one shared concept list."
                " Lists available: {}".format(
                    [c["name"] for c in concept_lists_raw]
                )
            )
            sys.exit(1)
    else:
        concept_lists = client.get("concept_lists/")

    # For naming purposes scl = shared_concept_list
    scl_match_counts = {}
    for clist in concept_lists:
        concept_selector = {"type": "concept_list",
                            "concept_list_id": clist['concept_list_id']}
        clist_match_counts = client.get('concepts/match_counts',
                                        concept_selector=concept_selector)
        clist_match_counts['concept_list_id'] = clist['concept_list_id']
        scl_match_counts[clist['name']] = clist_match_counts

    concepts = client.get(
        'concepts/match_counts',
        concept_selector={'type': 'top', 'limit': concept_count}
    )['match_counts']
    
    subset_counts = {}
    for field in luminoso_data.metadata:
        if field['type'] == 'string':
            subset_counts[field['name']] = {}
            if len(field['values']) > 200:
                print(
                    "Subset {} has {} (too many) values. Reducing to first"
                    " 200 values.".format(field['name'], len(field['values']))
                )
            for value in field['values'][:200]:
                subset_counts[field['name']][value['value']] = value['count']

    skt = subset_key_terms(client, subset_counts, terms_per_subset=skt_limit)
    
    themes = client.get(
        'concepts',
        concept_selector={'type': 'suggested',
                          'num_clusters': num_themes,
                          'num_cluster_concepts': theme_concepts}
    )
    # set the theme_id values and unpack the vectors
    theme_id = ''
    cluster_labels = {}
    for concept in themes['result']:
        label = concept['cluster_label']
        if label not in cluster_labels:
            theme_id = 'Theme {}'.format(len(cluster_labels))
            cluster_labels[label] = {'id': theme_id, 'name': []}
        concept['theme_id'] = theme_id
        concept['fvector'] = unpack64(concept['vectors'][0]).tolist()

    return (luminoso_data, scl_match_counts, concepts, skt, themes)


def create_doc_term_table(docs, concepts, scl_match_counts):
    '''
    Creates a tabulated format for the relationships between docs & terms

    :param docs: List of document dictionaries
    :param concepts: List of concept dictionaries
    :param scl_match_counts: This list of matchcounts for each shared concept
         list (scl)
    :return: List of dicts containing doc_ids, related terms, score & whether
         an exact match was found
    '''

    doc_term_table = []

    concept_ids = defaultdict(list)
    for concept in concepts:
        for term_id in concept['exact_term_ids']:
            concept_ids[term_id].append((concept['name'], 'top', None))

    for scl_name, saved_concepts in scl_match_counts.items():
        for concept in saved_concepts['match_counts']:
            for term_id in concept['exact_term_ids']:
                concept_ids[term_id].append(
                    (concept['name'], 'shared', scl_name)
                )
    
    doc_count = 0
    for doc in docs:
        doc_count += 1
        if not doc['vector']:
            continue

        concepts_in_doc = set()
        for term in doc['terms']:
            term_id = term['term_id']
            if term_id in concept_ids:
                for triple in concept_ids[term_id]:
                    if triple in concepts_in_doc:
                        continue
                    concepts_in_doc.add(triple)
                    doc_term_table.append(
                        {'doc_id': doc['doc_id'],
                         'term': triple[0],
                         'exact_match': 1,
                         'concept_type': triple[1],
                         'saved_concept_list': triple[2],
                         'sentiment': term['sentiment'],
                         'sentiment_confidence': term['sentiment_confidence']}
                    )

    return doc_term_table


def create_doc_subset_table(docs, metadata_map):
    '''
    Create a tabulation of documents and associated subsets
    :param docs: List of document dictionaries
    :param metadata_map: Dictionary mapping metadata field names to subset names
    :return: List of document ids, subsets, subset names and subset values
    '''
    doc_subset_table = []
    for doc in docs:
        for field in doc['metadata']:
            doc_subset_table.append({'doc_id': doc['doc_id'],
                                     'subset': metadata_map[field['name']],
                                     'subset_name': field['name'],
                                     'value': field['value']})
    return doc_subset_table


def create_doc_table(luminoso_data, suggested_concepts):
    '''
    Create a tabulation of documents and their related subsets & themes
    :param luminoso_data: a LuminosoData object
    :param suggested_concepts: The results from /concepts for
         suggested_concepts (same as themes)
    :return: List of documents with associated themes and list of
         cross-references between docs and subsets
    '''

    print('Creating doc table...')
    sort_order = {'number': 0, 'score': 0, 'string': 1, 'date': 2}
    sorted_metadata = sorted(luminoso_data.metadata,
                             key=lambda x: sort_order[x['type']])
    metadata_map = {}
    for i, field in enumerate(sorted_metadata):
        metadata_map[field['name']] = 'Subset %d' % i

    doc_table = []

    for doc in luminoso_data.docs:
        row = {'doc_id': doc['doc_id'], 'doc_text': doc['text']}
        date_number = 0
        for field in doc['metadata']:
            if field['type'] == 'date':
                row['doc_date %d' % date_number] = '%s %s' % (
                    field['value'].split('T')[0],
                    field['value'].split('T')[1].split('.')[0])
                date_number += 1
            row[metadata_map[field['name']]] = field['value']
        if date_number == 0:
            row['doc_date 0'] = 0
        
        # add the them (cluster) data
        doc['fvector'] = unpack64(doc['vector']).tolist()

        max_score = 0
        max_id = ''
        for concept in suggested_concepts['result']:
            if len(concept['vectors'][0]) > 0:
                concept['fvector'] = unpack64(concept['vectors'][0]).tolist()
                score = np.dot(doc['fvector'], concept['fvector'])
                if score > max_score:
                    max_score = score
                    max_id = concept['theme_id']

        row['theme_id'] = max_id
        row['theme_score'] = max_score

        doc_table.append(row)
        
    xref_table = [metadata_map]
    return doc_table, xref_table, metadata_map


def create_doc_term_sentiment(docs):
    '''
    Create a tabluation of the term sentiment in the context of each document.

    :param docs: The document list that has include_sentiment_on_concepts flag
    :return List of term sentiments in context of documents
    '''

    # regex to remove the language from the term_id
    # "newest|en product|en"  ->  "newest product"
    _DELANGTAG_RE = re.compile(r'\|[a-z]+(?=\s|\Z)')

    table = []
    for doc in docs:
        for term in doc['terms']:
            if 'sentiment' in term:
                row = {**term,
                       'doc_id': doc['doc_id'],
                       'name': _DELANGTAG_RE.sub('', term['term_id'])}
                table.append(row)

    return table


def create_terms_table(concepts, scl_match_counts):
    '''
    Create a tabulation of top terms and their exact/total match counts

    :param concepts: List of concept dictionaries
    :param scl_match_counts: Dictionary of match_counts for each shared concept
         list to process
    :return: List of terms, and match counts
    '''
    print('Creating terms table...')
    table = []
    for concept in concepts:
        table.append(
            {'term': concept['name'],
             'exact_match_count': concept['exact_match_count'],
             'related_match_count': (concept['match_count']
                                     - concept['exact_match_count']),
             'concept_type': 'top'}
        )
    for scl_name, saved_concepts in scl_match_counts.items():
        for concept in saved_concepts['match_counts']:
            table.append(
                {'term': concept['name'],
                 'exact_match_count': concept['exact_match_count'],
                 'related_match_count': (concept['match_count']
                                         - concept['exact_match_count']),
                 'concept_type': 'shared',
                 'saved_concept_list': scl_name}
            )
    return table


def create_themes_table(client, suggested_concepts):
    cluster_labels = {}
    themes = []

    # this is duplicating code done in pull_lumi_data - may need refactor
    for concept in suggested_concepts['result']:
        if concept['cluster_label'] not in cluster_labels:
            cluster_labels[concept['cluster_label']] = {
                'id': 'Theme %d' % len(cluster_labels),
                'name': []
            }
        cluster_labels[concept['cluster_label']]['name'].append(concept['name'])
        
    for label, cluster in cluster_labels.items():
        name = cluster['name']
        # find related documents
        selector_docs = {'texts': name}
        search_docs = client.get('docs', search=selector_docs, limit=3,
                                 match_type='exact')['result']
        
        selector = [{'texts': [t]} for t in name]
        count = 0
        match_counts = client.get(
            'concepts/match_counts',
            concept_selector={'type': 'specified', 'concepts': selector}
        )['match_counts']
        for match_count in match_counts:
            count += match_count['exact_match_count']

        for sdoc in search_docs:
            themes.append(
                {'cluster_label': label,
                 'name': ', '.join(name),
                 'id': cluster['id'],
                 'docs': count,
                 'doc_id': sdoc['doc_id']})
    return themes      


def create_sentiment_table(client, scl_match_counts, root_url=''):

    # first get the default sentiment output with sentiment suggestions
    results = client.get('/concepts/sentiment/')['match_counts']
    sentiment_match_counts = [
        {'texts': concept['texts'],
         'name': concept['name'],
         'concept_type': 'sentiment_suggestions',
         'match_count': concept['match_count'],
         'exact_match_count': concept['exact_match_count'],
         'sentiment_share_positive': concept['sentiment_share']['positive'],
         'sentiment_share_neutral': concept['sentiment_share']['neutral'],
         'sentiment_share_negative': concept['sentiment_share']['negative']}
        for concept in results
    ]

    for scl_name, saved_concepts in scl_match_counts.items():
        results_saved = client.get(
            '/concepts/sentiment/',
            concept_selector={
                "type": "concept_list",
                "concept_list_id": saved_concepts['concept_list_id']
            }
        )['match_counts']

        sentiment_match_counts.extend([
            {'texts': concept['texts'],
             'name': concept['name'],
             'match_count': concept['match_count'],
             'concept_type': 'shared',
             'saved_concept_list': scl_name,
             'exact_match_count': concept['exact_match_count'],
             'sentiment_share_positive': concept['sentiment_share']['positive'],
             'sentiment_share_neutral': concept['sentiment_share']['neutral'],
             'sentiment_share_negative': concept['sentiment_share']['negative']}
            for concept in results_saved
        ])

    results_top = client.get(
        '/concepts/sentiment/',
        concept_selector={"type": "top", 'limit': 100}
    )['match_counts']

    sentiment_match_counts.extend([
        {'texts': concept['texts'],
         'name': concept['name'],
         'match_count': concept['match_count'],
         'concept_type': 'top',
         'exact_match_count': concept['exact_match_count'],
         'sentiment_share_positive': concept['sentiment_share']['positive'],
         'sentiment_share_neutral': concept['sentiment_share']['neutral'],
         'sentiment_share_negative': concept['sentiment_share']['negative']}
        for concept in results_top
    ])

    # add three sample documents to each row
    for srow in sentiment_match_counts:
        if len(root_url)>0:
            srow['url'] = (root_url
                           + "/galaxy?suggesting=false&search="
                           + urllib.parse.quote(" ".join(srow['texts'])))

        # Use the driver term to find related documents
        search_docs = client.get(
            'docs', search={'texts': srow['texts']}, limit=3,
            match_type='exact'
        )['result']

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


def main():
    parser = argparse.ArgumentParser(
        description='Export data to Business Intelligence compatible CSV files.'
    )
    parser.add_argument('project_url',
                        help="The URL of the Daylight project to export from")
    parser.add_argument('-c', '--concept_count', default=20,
                        help="The number of top concepts to pull from the"
                             " project")
    parser.add_argument('-e', '--encoding', default='utf-8',
                        help="Encoding of the file to write to")
    parser.add_argument("-l", "--concept_list_names", default=None,
                        help="The names of this shared concept lists separated"
                             " by |. Default = ALL lists")
    parser.add_argument('-sktl', '--skt_limit', default=20,
                        help="The max number of subset key terms to display"
                             " per subset")
    parser.add_argument('-docs', '--doc', default=False, action='store_true',
                        help="Do not generate doc_table")
    parser.add_argument('-doc_term_sentiment', '--doc_term_sentiment',
                        default=False,
                        action='store_true',
                        help="Do not generate doc_term_sentiment_table")
    parser.add_argument('-terms', '--terms', default=False,
                        action='store_true',
                        help="Do not generate terms_table")
    parser.add_argument('-theme', '--themes', default=False,
                        action='store_true',
                        help="Do not generate themes_table")
    parser.add_argument('-dterm', '--doc_term', default=False,
                        action='store_true',
                        help="Do not generate doc_term_table")
    parser.add_argument('-dsubset', '--doc_subset', default=False,
                        action='store_true',
                        help="Do not generate doc_subset_table")
    parser.add_argument('-skt', '--skt_table', default=False,
                        action='store_true', help="Do not generate skt_tables")
    parser.add_argument('-drive', '--drive', default=False,
                        action='store_true',
                        help="Do not generate driver_table")
    parser.add_argument('-tdrive', '--topic_drive', default=False,
                        action='store_true',
                        help="If generating drivers_table do so with"
                             " shared/top concepts as well as auto concepts")
    parser.add_argument('--driver_subset', default=False, action='store_true',
                        help="Do not generate score drivers by subset")
    parser.add_argument('--driver_subset_fields', default=None,
                        help='Which subsets to include in score driver by'
                             ' subset. Default = All with < 200 unique values.'
                             ' Samp: "field1,field2"')

    parser.add_argument('-sentiment', '--sentiment', default=False,
                        action='store_true',
                        help="Do not generate sentiment for top concepts")
    parser.add_argument('--sdot', action='store_true',
                        help="Calculate over time")
    parser.add_argument('--sdot_end', default=None,
                        help="Last date to calculate sdot MM/DD/YYYY -"
                             " algorithm works moving backwards in time.")
    parser.add_argument('--sdot_iterations', default=7,
                        help="Number of over time samples")
    parser.add_argument('--sdot_range', default=None,
                        help="Size of each sample: M,W,D. If none given, range"
                             " type will be calculated for best fit")
    parser.add_argument('--sdot_date_field', default=None,
                        help="The name of the date field. If none, the first"
                             " date field will be used")
    args = parser.parse_args()
    
    root_url, api_url, workspace, proj = parse_url(args.project_url)
    print("starting subset drivers - topics={}".format(args.topic_drive))

    lumi_data = pull_lumi_data(proj, api_url, skt_limit=int(args.skt_limit),
                               concept_count=int(args.concept_count),
                               cln=args.concept_list_names)
    (luminoso_data, scl_match_counts, concepts, skt, themes) = lumi_data
    client = luminoso_data.client
    docs = luminoso_data.docs

    # get the docs no matter what because later data needs the metadata_map
    doc_table, xref_table, metadata_map = create_doc_table(luminoso_data, themes)

    luminoso_data.set_root_url(
        root_url + '/app/projects/' + workspace + '/' + proj
    )

    if not args.driver_subset:
        driver_table = create_drivers_with_subsets_table(
            luminoso_data, args.topic_drive,
            subset_fields=args.driver_subset_fields
        )
        write_table_to_csv(driver_table, 'subset_drivers_table.csv',
                           encoding=args.encoding)

    if not args.doc:
        write_table_to_csv(doc_table, 'doc_table.csv', encoding=args.encoding)
        write_table_to_csv(xref_table, 'xref_table.csv',
                           encoding=args.encoding)

    if not args.doc_term_sentiment:
        doc_term_sentiment_table = create_doc_term_sentiment(docs)
        write_table_to_csv(doc_term_sentiment_table, 'doc_term_sentiment.csv',
                           encoding=args.encoding)

    if not args.terms:
        terms_table = create_terms_table(concepts, scl_match_counts)
        write_table_to_csv(terms_table, 'terms_table.csv',
                           encoding=args.encoding)

    if not args.themes:
        print('Creating themes table...')
        themes_table = create_themes_table(client, themes)
        write_table_to_csv(themes_table, 'themes_table.csv',
                           encoding=args.encoding)

    # Combines list of concepts and shared concept lists
    if not args.doc_term:
        doc_term_table = create_doc_term_table(docs, concepts, scl_match_counts)
        write_table_to_csv(doc_term_table, 'doc_term_table.csv',
                           encoding=args.encoding)
        
    if not args.doc_subset:
        doc_subset_table = create_doc_subset_table(docs, metadata_map)
        write_table_to_csv(doc_subset_table, 'doc_subset_table.csv',
                           encoding=args.encoding)
    if not args.skt_table:
        skt_table = create_skt_table(client, skt)
        write_table_to_csv(skt_table, 'skt_table.csv', encoding=args.encoding)

    if not args.drive:
        driver_table = create_drivers_table(luminoso_data, args.topic_drive)
        write_table_to_csv(driver_table, 'drivers_table.csv',
                           encoding=args.encoding)
    
    if not args.sentiment:
        print('Creating sentiment table...')
        sentiment_table = create_sentiment_table(client, scl_match_counts,
                                                 root_url=luminoso_data.root_url)
        write_table_to_csv(sentiment_table, 'sentiment.csv',
                           encoding=args.encoding)
    
    if bool(args.sdot):
        if args.sdot_date_field is None:
            date_field_info = luminoso_data.first_date_field
            if date_field_info is None:
                print("ERROR no date field in project")
                return
        else:
            date_field_info = luminoso_data.get_field_by_name(
                args.sdot_date_field
            )
            if date_field_info is None:
                print("ERROR: no date field name:"
                      " {}".format(args.sdot_date_field))
                return

        sdot_table = create_sdot_table(
            luminoso_data, date_field_info, args.sdot_end,
            int(args.sdot_iterations), args.sdot_range, args.topic_drive,
            root_url=luminoso_data.root_url
        )
        write_table_to_csv(sdot_table, 'sdot_table.csv', encoding=args.encoding)


if __name__ == '__main__':
    main()
