import argparse
from collections import defaultdict
import json
import numpy as np
import os
import psycopg2
import psycopg2.extras
from psycopg2 import Error
import re
import sys
from urllib.parse import urlparse, unquote

from luminoso_api import V5LuminosoClient as LuminosoClient
from pack64 import unpack64
from se_code.unique_to_filter import (
    unique_to_filter, create_u2f_table, create_u2fot_table
)
from se_code.score_drivers import (
    parse_url, create_drivers_table, create_sdot_table,
    create_drivers_with_subsets_table, LuminosoData, write_table_to_csv
)
from se_code.sentiment import (
    create_sentiment_table, create_sentiment_subset_table,
    create_sot_table
)
from se_code.volume import (
    create_volume_table, create_volume_subset_table,
    create_vot_table
)
from se_code.outliers import (
    create_outlier_table, create_outlier_subset_table,
    create_outliersot_table
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
        'password': unquote(p.password),  # convert uri encoded strings back to strings, if pass has crazy chars
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
            average_score numeric,
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
            theme_name varchar(32),
            concepts text,
            id varchar(16),
            exact_matches numeric,
            example_doc1 varchar(40),
            example_doc2 varchar(40),
            example_doc3 varchar(40)
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
        CREATE TABLE IF NOT EXISTS doc_subsets (
            project_id varchar(16),
            doc_id varchar(40),
            field_name varchar(64),
            field_type varchar(16),
            field_value varchar(64),
            value varchar(64)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS unique_to_filter (
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
        CREATE TABLE IF NOT EXISTS unique_over_time (
            project_id varchar(16),
            start_date timestamp,
            end_date timestamp,
            iteration_counter numeric,
            range_type varchar(16),
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
        CREATE TABLE IF NOT EXISTS drivers_subsets (
            project_id varchar(16),
            concept varchar(128),
            driver_field varchar(128),
            list_type varchar(32),
            list_name varchar(64),
            relevance numeric,
            average_score numeric,
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
            average_score numeric,
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
            concept varchar(128),
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
            concept varchar(128),
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
            concept varchar(128),
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
        """,

        """
        CREATE TABLE IF NOT EXISTS volume (
            project_id varchar(16),
            concept varchar(128),
            texts varchar(128),
            concept_type varchar(32),
            shared_concept_list varchar(64),
            match_count numeric,
            exact_match_count numeric,
            conceptual_match_count numeric,
            url varchar(256),
            example_doc1 text,
            example_doc2 text,
            example_doc3 text
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS volume_subsets (
            project_id varchar(16),
            list_type varchar(32),
            list_name varchar(64),
            relevance numeric,
            field_name varchar(64),
            field_value varchar(64),
            concept varchar(128),
            match_count numeric,
            exact_match_count numeric,
            conceptual_match_count numeric
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS volume_over_time (
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
            concept varchar(128),
            concept_relevance numeric,
            match_count numeric,
            exact_match_count numeric,
            conceptual_match_count numeric
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS outliers (
            project_id varchar(16),
            list_type varchar(32),
            list_name varchar(64),
            concept varchar(128),
            relevance numeric,
            texts varchar(128),
            coverage numeric,
            match_type varchar(32),
            match_count numeric,
            exact_match_count numeric,
            conceptual_match_count numeric,
            url varchar(256),
            example_doc1 text,
            example_doc2 text,
            example_doc3 text
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS outlier_subsets (
            project_id varchar(16),
            list_type varchar(32),
            list_name varchar(64),
            field_name varchar(64),
            field_value varchar(64),
            concept varchar(128),
            relevance numeric,
            texts varchar(128),
            coverage numeric,
            match_type varchar(32),
            match_count numeric,
            exact_match_count numeric,
            conceptual_match_count numeric,
            example_doc1 text,
            example_doc2 text,
            example_doc3 text
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS outliers_over_time (
            project_id varchar(16),
            start_date timestamp,
            end_date timestamp,
            iteration_counter numeric,
            range_type varchar(16),
            list_type varchar(32),
            list_name varchar(64),
            field_name varchar(64),
            field_value varchar(64),
            concept varchar(128),
            relevance numeric,
            texts varchar(128),
            coverage numeric,
            match_type varchar(32),
            match_count numeric,
            exact_match_count numeric,
            conceptual_match_count numeric,
            example_doc1 text,
            example_doc2 text,
            example_doc3 text
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


def pull_lumi_data(project, api_url, u2f_limit, concept_count=100,
                   num_themes=7, theme_concepts=4, cln=None,
                   token=None):

    '''
    Extract relevant data from Luminoso project

    :param project: Luminoso project id
    :param u2f_limit: Number of terms per subset when creating unique terms
    :param concept_count: Number of top concepts to include in the analysis
    :param num_themes: Number of themes to calculate
    :param theme_concepts: Number of concepts to represent each theme
    :param cln: Concept List Names a string of shared concept list names
         separated by |
    :return: Return lists of dictionaries containing project data
    '''
    print('Extracting Lumi data...')
    if token:
        client = LuminosoClient.connect('{}/projects/{}'.format(api_url, project), token=token)
    else:
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

    subset_values_dict = {}
    for field in luminoso_data.metadata:
        if field['type'] == 'string':
            subset_values_dict[field['name']] = {}
            if len(field['values']) > 200:
                print(
                    "Subset {} has {} (too many) values. Reducing to first"
                    " 200 values.".format(field['name'], len(field['values']))
                )
            for value in field['values'][:200]:
                subset_values_dict[field['name']][value['value']] = value['count']

    u2f = unique_to_filter(client, subset_values_dict, terms_per_subset=u2f_limit)

    themes = client.get(
        'concepts',
        concept_selector={'type': 'suggested',
                          'num_clusters': num_themes,
                          'num_cluster_concepts': theme_concepts}
    )
    # set the theme_name values and unpack the vectors
    theme_name = ''
    cluster_labels = {}
    for concept in themes['result']:
        label = concept['cluster_label'].split('|')[0]
        # label = concept['name']
        if label not in cluster_labels:
            theme_name = label
            cluster_labels[label] = {'id': theme_name, 'name': []}
        concept['theme_name'] = theme_name
        concept['fvector'] = unpack64(concept['vectors'][0]).tolist()

    return (luminoso_data, scl_match_counts, concepts, u2f, themes, subset_values_dict)


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
                    max_id = concept['theme_name']
                    max_theme_name = concept['name']
                    max_cluster_name = concept['cluster_label'].split('|')[0]

        row['theme_name'] = max_id
        row['theme_score'] = max_score
        row['theme_name'] = max_cluster_name

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
        cluster_name = concept['cluster_label'].split("|")[0]
        if cluster_name not in cluster_labels:
            cluster_labels[cluster_name] = {
                'id': cluster_name,
                'concepts': []
            }
        cluster_labels[cluster_name]['concepts'].append(concept['name'])

    for label, cluster in cluster_labels.items():
        concepts = cluster['concepts']
        # find related documents
        selector_docs = {'texts': concepts}
        search_docs = client.get('docs', search=selector_docs, limit=3,
                                 match_type='exact')['result']

        selector = [{'texts': [t]} for t in concepts]
        count = 0
        match_counts = client.get(
            'concepts/match_counts',
            concept_selector={'type': 'specified', 'concepts': selector}
        )
        for match_count in match_counts['match_counts']:
            count += match_count['exact_match_count']

        row = {'theme_name': label,
               'concepts': '|'.join(concepts),
               'exact_matches': count}
        for idx, sdoc in enumerate(search_docs):
            row['example_doc{}'.format(idx+1)] = sdoc['doc_id']
        themes.append(row)
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

# convert metadat values to strings
def numbers_to_string(dict_list, key):
    a = 3
    a = a+1

def limit_string_length(dict_list, key, max_len):
    # need to limit the term len to max_len-1
    for t in dict_list:
        if isinstance(t[key], str) and len(t[key])>max_len:
            t[key] = t[key][:max_len-1]

def output_data(data, format, filename, sql_connection, table_name, project_id, encoding):
    if format in 'sql':
        write_to_sql(sql_connection, table_name, project_id, data)
    else:
        write_table_to_csv(data, filename,encoding=encoding)


def run_export(project_url=None,
               concept_count=20,
               encoding='utf-8',
               concept_list_names=None,
               output_format='csv',
               u2f_limit=20,
               skip_vol=False,
               skip_docs=False,
               skip_doc_term_sentiment=False,
               skip_doc_term_sentiment_list=False,
               skip_terms=False,
               skip_themes=False,
               skip_doc_term_summary=False,
               skip_doc_subset=False,
               skip_u2f_table=False,
               skip_drivers=False,
               volume_subset_fields=None,
               run_topic_drivers=False,
               skip_driver_subsets=False,
               driver_subset_fields=None,
               skip_sentiment=False,
               run_sdot=True,
               sdot_end=None,
               sdot_iterations=7,
               sdot_range=None,
               sdot_date_field=None,
               skip_sentiment_subsets=False,
               sentiment_subset_fields=None,
               run_vot=True,
               vot_end=None,
               vot_iterations=7,
               vot_range=None,
               vot_date_field=None,
               run_sot=True,
               sot_end=None,
               sot_iterations=7,
               sot_range=None,
               sot_date_field=None,
               run_u2fot=True,
               u2fot_end=None,
               u2fot_iterations=7,
               u2fot_range=None,
               u2fot_date_field=None,
               run_outliers=True,
               outlier_subset_fields=None,
               run_outliersot=True,
               outliersot_end=None,
               outliersot_iterations=7,
               outliersot_range=None,
               outliersot_date_field=None,
               token=None,
               db_connection=None
               ):

    root_url, api_url, workspace, project_id = parse_url(project_url)

    conn = None
    if output_format in 'sql':
        if not db_connection:
            conn = db_create_sql_connection()
        else:
            conn = db_connection

        print("creating sql tables")
        if db_create_tables(conn) != 0:
            exit(-1)

    lumi_data = pull_lumi_data(project_id, api_url, u2f_limit=int(u2f_limit),
                               concept_count=int(concept_count),
                               cln=concept_list_names, token=token)
    (luminoso_data, scl_match_counts, concepts, u2f, themes, subset_values_dict) = lumi_data
    client = luminoso_data.client
    docs = luminoso_data.docs

    # get the docs no matter what because later data needs the metadata_map
    doc_table, doc_metadata_table = create_doc_table(luminoso_data, themes)

    luminoso_data.set_root_url(
        root_url + '/app/projects/' + workspace + '/' + project_id
    )

    if not skip_vol:
        print('Creating volume table...')
        volume_table = create_volume_table(client, scl_match_counts,
                                           root_url=luminoso_data.root_url)
        output_data(volume_table, output_format,
                    'volume.csv', conn,
                    'volume', project_id, encoding=encoding)

        print("Creating volume by subsets...")
        volume_subset_table = create_volume_subset_table(
            luminoso_data,
            volume_subset_fields)
        if output_format in 'sql':
            limit_string_length(volume_subset_table, 'field_name', 63)
            numbers_to_string(volume_subset_table, 'field_value')
            limit_string_length(volume_subset_table, 'field_value', 63)

        output_data(volume_subset_table, output_format,
                    'volume_subsets.csv', conn,
                    'volume_subsets', project_id, encoding=encoding)

    if bool(run_vot):
        print("Creating volume over time (vot)")

        if vot_date_field is None:
            date_field_info = luminoso_data.first_date_field
            if date_field_info is None:
                print("ERROR no date field in project for vot")
                return
        else:
            date_field_info = luminoso_data.get_field_by_name(
                vot_date_field
            )
            if date_field_info is None:
                print("ERROR: (vot) no date field name:"
                      " {}".format(vot_date_field))
                return

        vot_table = create_vot_table(
            luminoso_data, date_field_info, sot_end,
            int(sot_iterations), sot_range, sentiment_subset_fields
        )
        if output_format in 'sql':
            limit_string_length(vot_table, 'field_name', 63)
            numbers_to_string(vot_table, 'field_value')
            limit_string_length(vot_table, 'field_value', 63)

        output_data(vot_table, output_format,
                    'volume_over_time.csv', conn,
                    'volume_over_time', project_id, encoding=encoding)

    if not skip_driver_subsets:
        print("starting subset drivers - topics={}".format(run_topic_drivers))

        driver_subset_table = create_drivers_with_subsets_table(
            luminoso_data, run_topic_drivers,
            subset_fields=driver_subset_fields
        )

        if output_format in 'sql':
            limit_string_length(driver_subset_table, 'field_name', 63)
            numbers_to_string(driver_subset_table, 'field_value')
            limit_string_length(driver_subset_table, 'field_value', 63)

        output_data(driver_subset_table, output_format,
            'drivers_subsets.csv', conn,
            'drivers_subsets', project_id, encoding=encoding)

    if not skip_docs:
        output_data(doc_table, output_format,
            'docs.csv', conn,
            'docs', project_id, encoding=encoding)

        if output_format in 'sql':
            limit_string_length(doc_metadata_table, 'metadata_name', 63)
            numbers_to_string(doc_metadata_table, 'metadata_value')
            limit_string_length(doc_metadata_table, 'metadata_value', 63)

        output_data(doc_metadata_table, output_format,
            'doc_metadata.csv', conn,
            'doc_metadata', project_id, encoding=encoding)

    if not skip_doc_term_sentiment:
        concept_lists = None
        if skip_doc_term_sentiment_list:
            concept_lists = client.get("concept_lists/")

        doc_term_sentiment_table = create_doc_term_sentiment(docs,
                                                             skip_doc_term_sentiment_list,
                                                             concept_lists)
        output_data(doc_term_sentiment_table, output_format,
                    'doc_term_sentiment.csv', conn,
                    'doc_term_sentiment', project_id, encoding=encoding)

    if not skip_terms:
        terms_table = create_terms_table(concepts, scl_match_counts)

        if output_format in 'sql':
            limit_string_length(terms_table, 'term', 63)

        output_data(terms_table, output_format,
                    'terms.csv', conn,
                    'terms', project_id, encoding=encoding)

    if not skip_themes:
        print('Creating themes table...')
        themes_table = create_themes_table(client, themes)
        output_data(themes_table, output_format,
                    'themes.csv', conn,
                    'themes', project_id, encoding=encoding)

    # Combines list of concepts and shared concept lists
    if not skip_doc_term_summary:
        doc_term_summary_table = create_doc_term_summary_table(docs, concepts, scl_match_counts)

        if output_format in 'sql':
            limit_string_length(doc_term_summary_table, 'term', 63)

        output_data(doc_term_summary_table, output_format,
                    'doc_term_summary.csv', conn,
                    'doc_term_summary', project_id, encoding=encoding)

    if not skip_doc_subset:
        doc_subset_table = create_doc_subset_table(docs)
        if output_format in 'sql':
            limit_string_length(doc_subset_table, 'field_name', 63)
            numbers_to_string(doc_subset_table, 'field_value')
            numbers_to_string(doc_subset_table, 'value')
            limit_string_length(doc_subset_table, 'field_value', 63)
            limit_string_length(doc_subset_table, 'value', 63)

        output_data(doc_subset_table, output_format,
                    'doc_subsets.csv', conn,
                    'doc_subsets', project_id, encoding=encoding)

    # unique to filter u2f was skt
    if not skip_u2f_table:
        print("Creating unique to filter...")

        u2f_table = create_u2f_table(client, u2f)

        if output_format in 'sql':
            limit_string_length(u2f_table, 'term', 63)
            limit_string_length(u2f_table, 'field_name', 63)
            numbers_to_string(u2f_table, 'field_value')
            limit_string_length(u2f_table, 'field_value', 63)

        output_data(u2f_table, output_format,
                    'unique_to_filter.csv', conn,
                    'unique_to_filter', project_id, encoding=encoding)

    # unique to filter over time (was skt)
    if bool(run_u2fot):
        print("Creating unique to filter over time...")

        if u2fot_date_field is None:
            date_field_info = luminoso_data.first_date_field
            if date_field_info is None:
                print("ERROR no date field in project")
                return
        else:
            date_field_info = luminoso_data.get_field_by_name(
                u2fot_date_field
            )
            if date_field_info is None:
                print("ERROR: no date field name:"
                      " {}".format(u2fot_date_field))
                return

        u2fot_table = create_u2fot_table(
            luminoso_data, date_field_info, u2fot_end,
            u2fot_iterations, u2fot_range, subset_values_dict,
            u2f_limit)
        if output_format in 'sql':
            limit_string_length(u2fot_table, 'field_name', 63)
            numbers_to_string(u2fot_table, 'field_value')
            limit_string_length(u2fot_table, 'field_value', 63)

        output_data(u2fot_table, output_format,
                    'unique_over_time.csv', conn,
                    'unique_over_time', project_id, encoding=encoding)

    if not skip_drivers:
        print("Creating score drivers...")
        driver_table = create_drivers_table(luminoso_data, run_topic_drivers)
        output_data(driver_table, output_format,
                    'drivers.csv', conn,
                    'drivers', project_id, encoding=encoding)

    if not skip_sentiment:
        print('Creating sentiment table...')
        sentiment_table = create_sentiment_table(client, scl_match_counts,
                                                 root_url=luminoso_data.root_url)
        output_data(sentiment_table, output_format,
                    'sentiment.csv', conn,
                    'sentiment', project_id, encoding=encoding)

    if not skip_sentiment_subsets:
        print("Creating sentiment by subsets...")
        sentiment_subset_table = create_sentiment_subset_table(
            luminoso_data,
            sentiment_subset_fields)
        if output_format in 'sql':
            limit_string_length(sentiment_subset_table, 'field_name', 63)
            numbers_to_string(sentiment_subset_table, 'field_value')
            limit_string_length(sentiment_subset_table, 'field_value', 63)

        output_data(sentiment_subset_table, output_format,
                    'sentiment_subsets.csv', conn,
                    'sentiment_subsets', project_id, encoding=encoding)

    if bool(run_sot):
        print("Creating sentiment over time (sot)")

        if sot_date_field is None:
            date_field_info = luminoso_data.first_date_field
            if date_field_info is None:
                print("ERROR no date field in project for sot")
                return
        else:
            date_field_info = luminoso_data.get_field_by_name(
                sot_date_field
            )
            if date_field_info is None:
                print("ERROR: (sot) no date field name:"
                      " {}".format(sot_date_field))
                return

        sot_table = create_sot_table(
            luminoso_data, date_field_info, sot_end,
            int(sot_iterations), sot_range, sentiment_subset_fields
        )
        if output_format in 'sql':
            limit_string_length(sot_table, 'field_name', 63)
            numbers_to_string(sot_table, 'field_value')
            limit_string_length(sot_table, 'field_value', 63)

        output_data(sot_table, output_format,
                    'sentiment_over_time.csv', conn,
                    'sentiment_over_time', project_id, encoding=encoding)

    if bool(run_sdot):
        if sdot_date_field is None:
            date_field_info = luminoso_data.first_date_field
            if date_field_info is None:
                print("ERROR no date field in project")
                return
        else:
            date_field_info = luminoso_data.get_field_by_name(
                sdot_date_field
            )
            if date_field_info is None:
                print("ERROR: no date field name:"
                      " {}".format(sdot_date_field))
                return

        sdot_table = create_sdot_table(
            luminoso_data, date_field_info, sdot_end,
            int(sdot_iterations), sdot_range, run_topic_drivers
        )
        if output_format in 'sql':
            limit_string_length(driver_subset_table, 'field_name', 63)
            numbers_to_string(driver_subset_table, 'field_value')
            limit_string_length(driver_subset_table, 'field_value', 63)

        output_data(sdot_table, output_format,
                    'drivers_over_time.csv', conn,
                    'drivers_over_time', project_id, encoding=encoding)

    if bool(run_outliers) or (bool(run_outliersot)):

        print('Getting outlier data...')

        concept_lists = client.get("concept_lists/")

        # get project info for calculating coverage
        proj_info = client.get("/")

        # For naming purposes scl = shared_concept_list
        scl_match_counts = {}
        for clist in concept_lists:
            concept_selector = {"type": "concept_list",
                                "concept_list_id": clist['concept_list_id']}
            clist_match_counts = client.get('concepts/match_counts',
                                            concept_selector=concept_selector)
            clist_match_counts['concept_list_id'] = clist['concept_list_id']
            scl_match_counts[clist['name']] = clist_match_counts

    if bool(run_outliers):
        print("Generating project outliers...")
        outlier_table = create_outlier_table(client, proj_info, scl_match_counts,
                                             "both", root_url=luminoso_data.root_url)
        outlier_table.extend(create_outlier_table(client, proj_info, scl_match_counts,
                                                  "exact", root_url=luminoso_data.root_url))
        output_data(outlier_table, output_format,
                    'outliers.csv', conn,
                    'outliers', project_id, encoding=encoding)

        print("Generating outliers by subsets...")
        outlier_subset_table = create_outlier_subset_table(
            luminoso_data,
            proj_info, 
            scl_match_counts, 
            "both",
            outlier_subset_fields)
        outlier_subset_table.extend(create_outlier_subset_table(
            luminoso_data,
            proj_info, 
            scl_match_counts, 
            "exact",
            outlier_subset_fields))
        output_data(outlier_subset_table, output_format,
                    'outlier_subsets.csv', conn,
                    'outlier_subsets', project_id, encoding=encoding)

    if bool(run_outliersot):
        print("Calculating outliers over time (outliersot)")

        if outliersot_date_field is None:
            date_field_info = luminoso_data.first_date_field
            if date_field_info is None:
                print("ERROR no date field in project for outliersot")
                return
        else:
            date_field_info = luminoso_data.get_field_by_name(
                sot_date_field
            )
            if date_field_info is None:
                print("ERROR: (outliersot) no date field name:"
                      " {}".format(sot_date_field))
                return

        outliersot_table = create_outliersot_table(
            luminoso_data, proj_info, scl_match_counts, "both",
            date_field_info, outliersot_end,
            int(outliersot_iterations), outliersot_range, 
            outlier_subset_fields
        )
        outliersot_table.extend(create_outliersot_table(
            luminoso_data, proj_info, scl_match_counts, "exact",
            date_field_info, outliersot_end,
            int(outliersot_iterations), outliersot_range, 
            outlier_subset_fields
        ))
        output_data(outliersot_table, output_format,
                    'outliers_over_time.csv', conn,
                    'outliers_over_time', project_id, encoding=encoding)

    print("Run export complete.")

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
    parser.add_argument('-u2fl', '--u2f_limit', default=20,
                        help="The max number of unique terms to display"
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
                        help="Do not generate doc_subsets_table")
    parser.add_argument('-u2f', '--u2f_table', default=False,
                        action='store_true', help="Do not generate u2f_tables")
    parser.add_argument('-drive', '--drive', default=False,
                        action='store_true',
                        help="Do not generate driver_table")
    parser.add_argument('-tdrive', '--topic_drive', default=True,
                        action='store_true',
                        help="If generating drivers_table do so with"
                             " top concepts, shared concept lists"
                             " and auto concepts")
    parser.add_argument('--driver_subsets', default=False, action='store_true',
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
    parser.add_argument('-skip_vol', '--skip_vol', default=False, action='store_true',
                        help="Do not generate doc_table")
    parser.add_argument('--volume_subset_fields', default=None,
                        help='Which subsets to include in volume by'
                             ' subset. Default = All with < 200 unique values.'
                             ' Samp: "field1,field2"')
    parser.add_argument('--vot', action='store_true', default=False,
                        help="Calculate volume over time (VOT)")
    parser.add_argument('--vot_end', default=None,
                        help="Last date to calculate vot MM/DD/YYYY -"
                             " algorithm works moving backwards in time.")
    parser.add_argument('--vot_iterations', default=7,
                        help="Number of volume over time samples")
    parser.add_argument('--vot_range', default=None,
                        help="Size of each sample: M,W,D. If none given, range"
                             " type will be calculated for best fit")
    parser.add_argument('--vot_date_field', default=None,
                        help="The name of the date field for vot. If none, the first"
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
    parser.add_argument('--u2fot', action='store_true', default=False,
                        help="Calculate unique to filter over time (U2FOT)")
    parser.add_argument('--u2fot_end', default=None,
                        help="Last date to calculate sot MM/DD/YYYY -"
                             " algorithm works moving backwards in time.")
    parser.add_argument('--u2fot_iterations', default=7,
                        help="Number of u2fot over time samples")
    parser.add_argument('--u2fot_range', default=None,
                        help="Size of each sample: M,W,D. If none given, range"
                             " type will be calculated for best fit")
    parser.add_argument('--u2fot_date_field', default=None,
                        help="The name of the date field for u2fot. If none, the first"
                             " date field will be used")
    parser.add_argument('--outliers',
                        action='store_true', default=True,
                        help="Do not generate outliers and outlier subsets")
    parser.add_argument('--outlier_subset_fields', default=None,
                        help='Which subsets to include in outlier by'
                             ' subset. Default = All with < 200 unique values.'
                             ' Samp: "field1,field2"')
    parser.add_argument('--outliersot', action='store_true', default=False,
                        help="Calculate outliers over time (SOT)")
    parser.add_argument('--outliersot_end', default=None,
                        help="Last date to calculate outlierot MM/DD/YYYY -"
                             " algorithm works moving backwards in time.")
    parser.add_argument('--outliersot_iterations', default=7,
                        help="Number of outliers over time samples")
    parser.add_argument('--outliersot_range', default=None,
                        help="Size of each sample: M,W,D. If none given, range"
                             " type will be calculated for best fit")
    parser.add_argument('--outliersot_date_field', default=None,
                        help="The name of the date field for sot. If none, the first"
                             " date field will be used")
    
    args = parser.parse_args()

    run_export(project_url=args.project_url,
               concept_count=args.concept_count,
               encoding=args.encoding,
               concept_list_names=args.concept_list_names,
               output_format=args.output_format,
               u2f_limit=args.u2f_limit,
               skip_vol=args.skip_vol,
               volume_subset_fields=args.volume_subset_fields,
               run_vot=args.vot,
               vot_end=args.vot_end,
               vot_iterations=args.vot_iterations,
               vot_range=args.vot_range,
               vot_date_field=args.vot_date_field,
               skip_docs=args.doc,
               skip_doc_term_sentiment=args.doc_term_sentiment,
               skip_doc_term_sentiment_list=args.doc_term_sentiment_list,
               skip_terms=args.terms,
               skip_themes=args.themes,
               skip_doc_term_summary=args.doc_term_summary,
               skip_doc_subset=args.doc_subset,
               skip_u2f_table=args.u2f_table,
               skip_drivers=args.drive,
               run_topic_drivers=args.topic_drive,
               skip_driver_subsets=args.driver_subsets,
               driver_subset_fields=args.driver_subset_fields,
               skip_sentiment=args.sentiment,
               run_sdot=args.sdot,
               sdot_end=args.sdot_end,
               sdot_iterations=args.sdot_iterations,
               sdot_range=args.sdot_range,
               sdot_date_field=args.sdot_date_field,
               skip_sentiment_subsets=args.sentiment_subsets,
               sentiment_subset_fields=args.sentiment_subset_fields,
               run_sot=args.sot,
               sot_end=args.sot_end,
               sot_iterations=args.sot_iterations,
               sot_range=args.sot_range,
               sot_date_field=args.sot_date_field,
               run_u2fot=args.u2fot,
               u2fot_end=args.u2fot_end,
               u2fot_iterations=args.u2fot_iterations,
               u2fot_range=args.u2fot_range,
               u2fot_date_field=args.u2fot_date_field,
               run_outliers=args.outliers,
               outlier_subset_fields=args.outlier_subset_fields,
               run_outliersot=args.sot,
               outliersot_end=args.outliersot_end,
               outliersot_iterations=args.outliersot_iterations,
               outliersot_range=args.outliersot_range,
               outliersot_date_field=args.outliersot_date_field,)


if __name__ == '__main__':
    main()
