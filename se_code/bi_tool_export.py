import argparse
from collections import defaultdict
import numpy as np
import os
import psycopg2
import psycopg2.extras
from psycopg2 import Error
import re
import sys
from urllib.parse import urlparse

from luminoso_api import V5LuminosoClient as LuminosoClient
from pack64 import unpack64
from se_code.subset_key_terms import subset_key_terms, create_skt_table
from se_code.score_drivers import (
    create_drivers_table, create_sdot_table,
    create_drivers_with_subsets_table, LuminosoData, write_table_to_csv
)
from se_code.sentiment import (
    create_sentiment_table, create_sentiment_subset_table,
    create_sot_table
)


def db_create_sql_connection():

    db_connection_string = os.environ.get('DB_CONNECTION_STRING')
    if not db_connection_string:
        print("Need DB_CONNECTION_STRING environment var.")
        print("Format:")
        print("   postgresql://$DB_USER:$DB_PASS@$DB_HOST:$DB_PORT/$DB_NAME")
        exit(1)

    p = urlparse(db_connection_string)

    pg_connection_dict = {
        'dbname': p.path.strip('/'),
        'user': p.username,
        'password': p.password,
        'port': p.port,
        'host': p.hostname
    }

    try:
        conn = psycopg2.connect(**pg_connection_dict)
    except (Exception, Error) as error:
        print("Error while connecting to PostgreSQL", error)
        exit(1)

    return conn


def db_create_tables(conn):
    commands = (
        """
        CREATE TABLE IF NOT EXISTS docs (
            project_id varchar(16),
            doc_id varchar(40),
            doc_text text,
            theme_name varchar(64),
            theme_id varchar(16),
            theme_score numeric
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS doc_metadata (
            project_id varchar(16),
            doc_id varchar(40),
            metadata_name varchar(64),
            metadata_type varchar(16),
            metadata_value varchar(64)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS drivers (
            project_id varchar(16),
            concept varchar(128),
            driver_field varchar(128),
            list_type varchar(32),
            list_name varchar(64),
            relevance numeric,
            impact numeric,
            doc_count numeric,
            url varchar(256),
            example_doc1 text,
            example_doc2 text,
            example_doc3 text)
        """,
        """
        CREATE TABLE IF NOT EXISTS doc_term_sentiment (
            project_id varchar(16),
            name varchar(64),
            term_id varchar(64),
            doc_id varchar(40),
            loc_start numeric,
            loc_end numeric,
            sentiment varchar(16),
            sentiment_confidence numeric,
            share_concept_list varchar(64),
            shared_concept_name varchar(64)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS terms (
            project_id varchar(16),
            term varchar(64),
            exact_match_count numeric,
            related_match_count numeric,
            concept_type varchar(32),
            shared_concept_list varchar(64)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS themes (
            project_id varchar(16),
            cluster_label varchar(32),
            name varchar(64),
            id varchar(16),
            docs numeric,
            doc_id varchar(40)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS doc_term_summary (
            project_id varchar(16),
            doc_id varchar(40),
            term varchar(64),
            exact_match numeric,
            concept_type varchar(32),
            shared_concept_list varchar(64),
            sentiment varchar(16),
            sentiment_confidence numeric
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS doc_subset (
            project_id varchar(16),
            doc_id varchar(40),
            field_name varchar(64),
            field_type varchar(16),
            field_value varchar(64),
            value varchar(64)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS subset_key_terms (
            project_id varchar(16),
            term varchar(64),
            field_name varchar(64),
            field_value varchar(64),
            exact_matches numeric,
            conceptual_matches numeric,
            total_matches numeric,
            example_doc1 text,
            example_doc2 text,
            example_doc3 text
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS drivers (
            project_id varchar(16),
            concept varchar(128),
            driver_field varchar(128),
            list_type varchar(32),
            list_name varchar(64),
            relevance numeric,
            impact numeric,
            doc_count numeric,
            url varchar(256),
            example_doc1 text,
            example_doc2 text,
            example_doc3 text)
        """,
        """
        CREATE TABLE IF NOT EXISTS drivers_subset (
            project_id varchar(16),
            concept varchar(128),
            driver_field varchar(128),
            list_type varchar(32),
            list_name varchar(64),
            relevance numeric,
            impact numeric,
            doc_count numeric,
            url varchar(256),
            example_doc1 text,
            example_doc2 text,
            example_doc3 text,
            field_name varchar(64),
            field_value varchar(64))
        """,
        """
        CREATE TABLE IF NOT EXISTS drivers_over_time (
            project_id varchar(16),
            start_date timestamp,
            end_date timestamp,
            iteration_counter numeric,
            range_type varchar(16),
            concept varchar(128),
            driver_field varchar(128),
            list_type varchar(32),
            list_name varchar(64),
            relevance numeric,
            impact numeric,
            doc_count numeric,
            url varchar(256),
            example_doc1 text,
            example_doc2 text,
            example_doc3 text,
            field_name varchar(64),
            field_value varchar(64))
        """,
        """
        CREATE TABLE IF NOT EXISTS sentiment (
            project_id varchar(16),
            concept varchar(64),
            texts varchar(128),
            concept_type varchar(32),
            shared_concept_list varchar(64),
            match_count numeric,
            exact_match_count numeric,
            sentiment_share_positive numeric,
            sentiment_share_neutral numeric,
            sentiment_share_negative numeric,
            url varchar(256),
            example_doc1 text,
            example_doc2 text,
            example_doc3 text
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS sentiment_subsets (
            project_id varchar(16),
            list_type varchar(32),
            list_name varchar(64),
            relevance numeric,
            field_name varchar(64),
            field_value varchar(64),
            concept varchar(64),
            match_count numeric,
            exact_match_count numeric,
            conceptual_match_count numeric,
            sentiment_share_positive numeric,
            sentiment_share_neutral numeric,
            sentiment_share_negative numeric,
            sentiment_doc_count_positive numeric,
            sentiment_doc_count_neutral numeric,
            sentiment_doc_count_negative numeric,
            sentiment_doc_count_total numeric
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS sentiment_over_time (
            project_id varchar(16),
            start_date timestamp,
            end_date timestamp,
            iteration_counter numeric,
            range_type varchar(16),
            list_type varchar(32),
            list_name varchar(64),
            relevance numeric,
            field_name varchar(64),
            field_value varchar(64),
            concept varchar(64),
            concept_relevance numeric,
            match_count numeric,
            exact_match_count numeric,
            conceptual_match_count numeric,
            sentiment_share_positive numeric,
            sentiment_share_neutral numeric,
            sentiment_share_negative numeric,
            sentiment_doc_count_positive numeric,
            sentiment_doc_count_neutral numeric,
            sentiment_doc_count_negative numeric,
            sentiment_doc_count_total numeric
        )
        """
    )
    # THOUGHTS
    # I'm concerned with the doc_metadata that this should three separate tables based on type
    # date, string, numeric so deciding the graph type will be easier. We can have
    # separate views based on the values that are strings, numbers or dates. We can
    # still filter them out using this table, but I think dashboards will be easier if
    # all the values in the table are the same type.
    # Also concerned that doc_metadata and doc_subset are exactly the same

    try:
        cur = conn.cursor()

        # create tables one by one
        for command in commands:
            cur.execute(command)

        cur.close()
        conn.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
        return -1

    return 0


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
    luminoso_data.project_id = project

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
        #label = concept['cluster_label']
        label = concept['name']
        if label not in cluster_labels:
            theme_id = 'Theme {}'.format(len(cluster_labels))
            cluster_labels[label] = {'id': theme_id, 'name': []}
        concept['theme_id'] = theme_id
        concept['fvector'] = unpack64(concept['vectors'][0]).tolist()

    return (luminoso_data, scl_match_counts, concepts, skt, themes)


def create_doc_term_summary_table(docs, concepts, scl_match_counts):
    '''
    Creates a tabulated format for the relationships between docs & terms
    using the top terms and shared concept lists as a filter

    :param docs: List of document dictionaries
    :param concepts: List of concept dictionaries
    :param scl_match_counts: This list of matchcounts for each shared concept
         list (scl)
    :return: List of dicts containing doc_ids, related terms, score & whether
         an exact match was found
    '''

    doc_term_summary_table = []

    concept_ids = defaultdict(list)
    for concept in concepts:
        for term_id in concept['exact_term_ids']:
            concept_ids[term_id].append((concept['name'], 'top', None))

    for scl_name, shared_concepts in scl_match_counts.items():
        for concept in shared_concepts['match_counts']:
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
        for term in doc['terms'] + doc['fragments']:
            term_id = term['term_id']
            if term_id in concept_ids:
                for triple in concept_ids[term_id]:
                    if triple in concepts_in_doc:
                        continue
                    concepts_in_doc.add(triple)
                    doc_term_summary_table.append(
                        {'doc_id': doc['doc_id'],
                         'term': triple[0],
                         'exact_match': 1,
                         'concept_type': triple[1],
                         'shared_concept_list': triple[2],
                         'sentiment': term['sentiment'],
                         'sentiment_confidence': term['sentiment_confidence']}
                    )

    return doc_term_summary_table


def create_doc_subset_table(docs):
    '''
    Create a tabulation of documents and associated subsets
    :param docs: List of document dictionaries
    :return: List of document ids, subsets, subset names and subset values
    '''
    doc_subset_table = []
    for doc in docs:
        for field in doc['metadata']:
            doc_subset_table.append({'doc_id': doc['doc_id'],
                                     'field_name': field['name'],
                                     'field_type': field['type'],
                                     'field_value': field['value'],
                                     'value': field['value']})
    return doc_subset_table

def create_doc_metadata_table(luminoso_data):
    metadata_table = []
    for doc in luminoso_data.docs:
        mdrow = []
        for md in doc['metadata']:
            mdrow = {
                'metadata_name': md['name'],
                'metadata_type': md['type'],
                'metadata_value': md['value'],
                'doc_id': doc['doc_id']
            }
            metadata_table.append(mdrow)

    return metadata_table

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
    '''
    sort_order = {'number': 0, 'score': 0, 'string': 1, 'date': 2}
    sorted_metadata = sorted(luminoso_data.metadata,
                             key=lambda x: sort_order[x['type']])
    metadata_map = {}
    for i, field in enumerate(sorted_metadata):
        metadata_map[field['name']] = 'Subset %d' % i
    '''

    doc_table = []

    for doc in luminoso_data.docs:
        row = {'doc_id': doc['doc_id'], 'doc_text': doc['text']}
        '''
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
        '''
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
                    max_theme_name = concept['name']

        row['theme_id'] = max_id
        row['theme_score'] = max_score
        row['theme_name'] = max_theme_name

        doc_table.append(row)

    doc_metadata_table = create_doc_metadata_table(luminoso_data)

    # xref_table = [metadata_map]
    return doc_table, doc_metadata_table


def create_doc_term_sentiment(docs, include_shared_concept=False, concept_lists=None):
    '''
    Create a tabluation of the term sentiment in the context of each document.

    :param docs: The document list that has include_sentiment_on_concepts flag
    :return List of term sentiments in context of documents
    '''

    all_shared_concepts = []
    if include_shared_concept:
        # need to build up that list of shared concepts
        all_shared_concepts = [t for cl in concept_lists for c in cl['concepts'] for t in c['texts'] ]

    # regex to remove the language from the term_id
    # "newest|en product|en"  ->  "newest product"
    _DELANGTAG_RE = re.compile(r'\|[a-z]+(?=\s|\Z)')

    table = []
    for doc in docs:
        for term in doc['terms']:
            if 'sentiment' in term:
                name = _DELANGTAG_RE.sub('', term['term_id'])
                row = {**term,
                       'doc_id': doc['doc_id'],
                       'name': name}
                # sql doesn't allow 'end' so change names
                row['loc_end'] = row.pop('end')
                row['loc_start'] = row.pop('start')

                # check if this concept is in a shared concept list
                if name in all_shared_concepts:
                    for cl in concept_lists:
                        for c in cl['concepts']:
                            if name in c['texts']:
                                row2 = row.copy()
                                row2['share_concept_list'] = cl['name']
                                row2['shared_concept_name'] = c['name']
                                table.append(row2)

                else:
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
    for scl_name, shared_concepts in scl_match_counts.items():
        for concept in shared_concepts['match_counts']:
            table.append(
                {'term': concept['name'],
                 'exact_match_count': concept['exact_match_count'],
                 'related_match_count': (concept['match_count']
                                         - concept['exact_match_count']),
                 'concept_type': 'shared',
                 'shared_concept_list': scl_name}
            )
    return table

def create_themes_table(client, suggested_concepts):
    cluster_labels = {}
    themes = []

    # this is duplicating code done in pull_lumi_data - may need refactor
    for concept in suggested_concepts['result']:
        #if concept['cluster_label'] not in cluster_labels:
        if concept['name'] not in cluster_labels:
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


def write_to_sql(connection, table_name, project_id, data):
    # every table has a project_id, but most data doesn't have the column
    # just add it here. Aware that this modifies the data
    # for later calls, but this data is all transient and
    # only for output anyway
    for r in data:
        r['project_id'] = project_id

    if len(data) > 0:
        keys = list(set(val for dic in data for val in dic.keys())) 
        columns = ', '.join(keys)

        sql_data = []
        for row in data:
            tup = ()

            for k in keys:
                if k in row:
                    tup += (str(row[k]),)
                else:
                    tup += ("",)

            sql_data.append(tup)

        cursor = connection.cursor()
        insert_query = f"INSERT INTO {table_name} ({columns}) VALUES %s"
        psycopg2.extras.execute_values (
            cursor, insert_query, sql_data, template=None, page_size=100
        )

        connection.commit()
        cursor.close()


def output_data(data, format, filename, sql_connection, table_name, project_id, encoding):
    if format in 'sql':
        write_to_sql(sql_connection, table_name, project_id, data)
    else:
        write_table_to_csv(data, filename,encoding=encoding)

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
    parser.add_argument("-o", '--output_format', default='csv',
                        help="Output format, csv, sql")
    parser.add_argument('-sktl', '--skt_limit', default=20,
                        help="The max number of subset key terms to display"
                             " per subset")
    parser.add_argument('-docs', '--doc', default=False, action='store_true',
                        help="Do not generate doc_table")
    parser.add_argument('-doc_term_sentiment', '--doc_term_sentiment',
                        default=False,
                        action='store_true',
                        help="Do not generate doc_term_sentiment_table")
    parser.add_argument('-doc_term_sentiment_list', '--doc_term_sentiment_list',
                        default=False,
                        action='store_true',
                        help="Tag the term sentiment when concept is in a shared concept list")
    parser.add_argument('-terms', '--terms', default=False,
                        action='store_true',
                        help="Do not generate terms_table")
    parser.add_argument('-theme', '--themes', default=False,
                        action='store_true',
                        help="Do not generate themes_table")
    parser.add_argument('-dtermsum', '--doc_term_summary', default=False,
                        action='store_true',
                        help="Do not generate doc_term_summary_table")
    parser.add_argument('-dsubset', '--doc_subset', default=False,
                        action='store_true',
                        help="Do not generate doc_subset_table")
    parser.add_argument('-skt', '--skt_table', default=False,
                        action='store_true', help="Do not generate skt_tables")
    parser.add_argument('-drive', '--drive', default=False,
                        action='store_true',
                        help="Do not generate driver_table")
    parser.add_argument('-tdrive', '--topic_drive', default=True,
                        action='store_true',
                        help="If generating drivers_table do so with"
                             " top concepts, shared concept lists"
                             " and auto concepts")
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
                        help="Calculate score drivers over time")
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
    parser.add_argument('--sentiment_subsets',
                        action='store_true', default=False,
                        help="Do not generate sentiment subsets")
    parser.add_argument('--sentiment_subset_fields', default=None,
                        help='Which subsets to include in sentiments by'
                             ' subset. Default = All with < 200 unique values.'
                             ' Samp: "field1,field2"')
    parser.add_argument('--sot', action='store_true', default=False,
                        help="Calculate sentiment over time (SOT)")
    parser.add_argument('--sot_end', default=None,
                        help="Last date to calculate sot MM/DD/YYYY -"
                             " algorithm works moving backwards in time.")
    parser.add_argument('--sot_iterations', default=7,
                        help="Number of sentiment over time samples")
    parser.add_argument('--sot_range', default=None,
                        help="Size of each sample: M,W,D. If none given, range"
                             " type will be calculated for best fit")
    parser.add_argument('--sot_date_field', default=None,
                        help="The name of the date field for sot. If none, the first"
                             " date field will be used")
    args = parser.parse_args()

    root_url, api_url, workspace, project_id = parse_url(args.project_url)

    conn = None
    if args.output_format in 'sql':
        conn = db_create_sql_connection()

        print("creating sql tables")
        if db_create_tables(conn) != 0:
            exit(-1)

    print("starting subset drivers - topics={}".format(args.topic_drive))

    lumi_data = pull_lumi_data(project_id, api_url, skt_limit=int(args.skt_limit),
                               concept_count=int(args.concept_count),
                               cln=args.concept_list_names)
    (luminoso_data, scl_match_counts, concepts, skt, themes) = lumi_data
    client = luminoso_data.client
    docs = luminoso_data.docs

    # get the docs no matter what because later data needs the metadata_map
    doc_table, doc_metadata_table = create_doc_table(luminoso_data, themes)

    luminoso_data.set_root_url(
        root_url + '/app/projects/' + workspace + '/' + project_id
    )

    if not args.driver_subset:
        driver_subset_table = create_drivers_with_subsets_table(
            luminoso_data, args.topic_drive,
            subset_fields=args.driver_subset_fields
        )
        output_data(driver_subset_table, args.output_format,
            'drivers_subset_table.csv', conn,
            'drivers_subset', project_id, encoding=args.encoding)

    if not args.doc:
        output_data(doc_table, args.output_format,
            'doc_table.csv', conn,
            'docs', project_id, encoding=args.encoding)
        output_data(doc_metadata_table, args.output_format,
            'doc_metadata_table.csv', conn,
            'doc_metadata', project_id, encoding=args.encoding)

    if not args.doc_term_sentiment:
        concept_lists = None
        if args.doc_term_sentiment_list:
            concept_lists = client.get("concept_lists/")

        doc_term_sentiment_table = create_doc_term_sentiment(docs,
                                                             args.doc_term_sentiment_list,
                                                             concept_lists)
        output_data(doc_term_sentiment_table, args.output_format,
                    'doc_term_sentiment.csv', conn,
                    'doc_term_sentiment', project_id, encoding=args.encoding)

    if not args.terms:
        terms_table = create_terms_table(concepts, scl_match_counts)

        output_data(terms_table, args.output_format,
                    'terms_table.csv', conn,
                    'terms', project_id, encoding=args.encoding)

    if not args.themes:
        print('Creating themes table...')
        themes_table = create_themes_table(client, themes)
        output_data(themes_table, args.output_format,
                    'themes_table.csv', conn,
                    'themes', project_id, encoding=args.encoding)

    # Combines list of concepts and shared concept lists
    if not args.doc_term_summary:
        doc_term_summary_table = create_doc_term_summary_table(docs, concepts, scl_match_counts)
        output_data(doc_term_summary_table, args.output_format,
                    'doc_term_summary_table.csv', conn,
                    'doc_term_summary', project_id, encoding=args.encoding)

    if not args.doc_subset:
        doc_subset_table = create_doc_subset_table(docs)
        output_data(doc_subset_table, args.output_format,
                    'doc_subset_table.csv', conn,
                    'doc_subset', project_id, encoding=args.encoding)

    if not args.skt_table:
        skt_table = create_skt_table(client, skt)
        output_data(skt_table, args.output_format,
                    'skt_table.csv', conn,
                    'subset_key_terms', project_id, encoding=args.encoding)

    if not args.drive:
        print("Creating score drivers...")
        driver_table = create_drivers_table(luminoso_data, args.topic_drive)
        output_data(driver_table, args.output_format,
                    'drivers_table.csv', conn,
                    'drivers', project_id, encoding=args.encoding)

    if not args.sentiment:
        print('Creating sentiment table...')
        sentiment_table = create_sentiment_table(client, scl_match_counts,
                                                 root_url=luminoso_data.root_url)
        write_table_to_csv(sentiment_table, 'sentiment.csv',
                           encoding=args.encoding)
        output_data(sentiment_table, args.output_format,
                    'sentiment.csv', conn,
                    'sentiment', project_id, encoding=args.encoding)

    if not args.sentiment_subsets:
        print("Creating sentiment by subsets...")
        sentiment_subset_table = create_sentiment_subset_table(
            luminoso_data,
            args.sentiment_subset_fields)
        output_data(sentiment_subset_table, args.output_format,
                    'sentiment_subsets.csv', conn,
                    'sentiment_subsets', project_id, encoding=args.encoding)

    if bool(args.sot):
        print("Creating sentiment over time (sot)")

        if args.sot_date_field is None:
            date_field_info = luminoso_data.first_date_field
            if date_field_info is None:
                print("ERROR no date field in project for sot")
                return
        else:
            date_field_info = luminoso_data.get_field_by_name(
                args.sot_date_field
            )
            if date_field_info is None:
                print("ERROR: (sot) no date field name:"
                      " {}".format(args.sot_date_field))
                return

        sot_table = create_sot_table(
            luminoso_data, date_field_info, args.sot_end,
            int(args.sot_iterations), args.sot_range, args.sentiment_subset_fields
        )
        output_data(sot_table, args.output_format,
                    'sot_table.csv', conn,
                    'sentiment_over_time', project_id, encoding=args.encoding)

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
            int(args.sdot_iterations), args.sdot_range, args.topic_drive
        )
        output_data(sdot_table, args.output_format,
                    'sdot_table.csv', conn,
                    'drivers_over_time', project_id, encoding=args.encoding)


if __name__ == '__main__':
    main()
