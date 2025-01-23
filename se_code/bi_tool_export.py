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
from loguru import logger
from tqdm import tqdm
from pprint import pformat

from se_code.data_writer import (LumiDataWriter, LumiCsvWriter, LumiSqlWriter)
from se_code.unique_to_filter import (
    unique_to_filter, create_u2f_table, create_u2fot_table
)
from se_code.score_drivers import (
    parse_url, create_drivers_table, create_sdot_table,
    create_drivers_with_subsets_table, LuminosoData, lumi_date_to_epoch
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

DOC_BATCH_SIZE = 5000


def db_create_sql_connection():
    conn = None
    if 'DB_CONNECTION_STRING' in os.environ:
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
    else:
        print("Export type 'sql' requires DB_CONNECTION_STRING environment variable")
        exit(1)

    return conn


def get_sql_var_types(cmd):
    psplit = cmd.split("(", 1)
    command = psplit[0].strip()

    if command.upper().startswith("CREATE TABLE"):
        # the last on the line is the table name
        table_name = command.split(" ")[-1]
    else:
        # this isn't a create table command. Ignore
        return None

    cols = {}
    rest = psplit[1].strip()
    # trim the final ) off of it
    rest = ")".join(rest.split(")")[0:-1]).split(",")
    for col in rest:
        col = col.strip()
        col_name = col.split(" ")[0].strip()
        col_type = col.split(" ")[1].split("(")[0].split(",")[0].strip()
        max_col_len = -1
        if ("varchar" in col_type):
            max_col_len = int(col.split(" ")[1].split("(")[1].split(")")[0].strip())
        cols[col_name] = (col_name, col_type, max_col_len)

    return table_name, cols


def parse_schemas(commands):
    table_schemas = {}
    for c in commands:
        cdata = get_sql_var_types(c)
        table_schemas[cdata[0]] = cdata[1]

    return table_schemas


create_tables_commands = (
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
            average_score numeric,
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
table_schemas = parse_schemas(create_tables_commands)


def db_create_tables(conn):

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
        for command in create_tables_commands:
            cur.execute(command)

        cur.close()
        conn.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
        return -1

    return 0


def read_docs(client, date_field_name=None):
    docs = []
    docs_by_date = []
    while True:
        new_docs = client.get(
            'docs', limit=DOC_BATCH_SIZE, offset=len(docs),
            include_sentiment_on_concepts=True
        )['result']

        if date_field_name:
            for i, d in enumerate(new_docs):
                for m in d['metadata']:
                    if m['name'] == date_field_name:
                        date = lumi_date_to_epoch(m['value'])
                        if date is not None:
                            docs_by_date.append({'date': date, 'doc_id': d['doc_id'], 'i': 1})

        if new_docs:
            docs.extend(new_docs)
        else:
            break
    return docs, docs_by_date


def pull_lumi_data(project, api_url, concept_count=100,
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
    logger.info('Extracting Lumi data...')
    if token:
        client = LuminosoClient.connect('{}/projects/{}'.format(api_url, project), token=token)
    else:
        client = LuminosoClient.connect('{}/projects/{}'.format(api_url, project))
    luminoso_data = LuminosoData(client)
    luminoso_data.project_id = project
    
    logger.debug("connected to luminoso client")

    logger.debug("getting concept_lists")
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
    
    logger.debug("getting concept list match counts")
    # For naming purposes scl = shared_concept_list
    scl_match_counts = {}
    for clist in (t:=tqdm(concept_lists)):
        concept_selector = {"type": "concept_list",
                            "concept_list_id": clist['concept_list_id']}
        t.set_description(str(concept_selector))
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

    return (luminoso_data, scl_match_counts, concepts, themes, subset_values_dict)


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


def create_doc_table(client,
                     doc_table_writer,
                     doc_metadata_table_writer,
                     doc_term_sentiment_writer, concept_lists,
                     doc_term_summary_writer, concepts, scl_match_counts,
                     doc_subset_writer,
                     date_field_name, themes, tag_doc_term_sentiment_list):
    '''
    Read and write as it reads a tabulation of documents and their related subsets & themes
    :param client: a Luminoso client object
    :param doc_table_writer: a lumi_writer for the doc_table
    :param doc_metadata_table_writer: a lumi_writer for the doc metadata
    :param themes: The results from /concepts for
         suggested_concepts which are basically the themes

    :return: docs_by_date - the list of dates and doc_ids in the project 
    :                       used later for over time calculations
    '''

    logger.info('Creating doc table...')

    all_shared_concepts = []
    if tag_doc_term_sentiment_list:
        # need to build up that list of shared concepts
        all_shared_concepts = [t for cl in concept_lists for c in cl['concepts'] for t in c['texts'] ]

    # regex to remove the language from the term_id
    # "newest|en product|en"  ->  "newest product"
    _DELANGTAG_RE = re.compile(r'\|[a-z]+(?=\s|\Z)')

    doc_term_summary_table = []

    concept_ids = defaultdict(list)
    
    for concept in tqdm(concepts, desc="get concept names"):
        for term_id in concept['exact_term_ids']:
            concept_ids[term_id].append((concept['name'], 'top', None))

    for scl_name, shared_concepts in tqdm(scl_match_counts.items(), desc="get shared concept names"):
        for concept in shared_concepts['match_counts']:
            for term_id in concept['exact_term_ids']:
                concept_ids[term_id].append(
                    (concept['name'], 'shared', scl_name)
                )

    doc_table = []
    docs_by_date = []
    doc_metadata_table = []
    doc_term_sentiment_table = []
    doc_subset_table = []

    offset = 0
    while True:
        logger.debug("getting docs...")

        new_docs = client.get(
            'docs', limit=DOC_BATCH_SIZE, offset=offset,
            include_sentiment_on_concepts=True
        )['result']
        logger.debug("getting docs... done.")

        if date_field_name:
            for i, d in tqdm(enumerate(new_docs), desc="process date_field_name"):
                for m in d['metadata']:
                    if m['name'] == date_field_name:
                        date = lumi_date_to_epoch(m['value'])
                        if date is not None:
                            docs_by_date.append({'date': date, 'doc_id': d['doc_id'], 'i': 1})

        for doc in tqdm(new_docs, desc="process docs"):
            row = {'doc_id': doc['doc_id'], 'doc_text': doc['text']}
            if (len(doc['text']) > 0):
                # add the theme (cluster) data
                doc['fvector'] = unpack64(doc['vector']).tolist()

                max_score = 0
                max_id = ''
                max_cluster_name = None
                for concept in themes['result']:
                    if len(concept['vectors'][0]) > 0:
                        concept['fvector'] = unpack64(concept['vectors'][0]).tolist()
                        score = np.dot(doc['fvector'], concept['fvector'])
                        if score > max_score:
                            max_score = score
                            max_id = concept['theme_name']
                            max_cluster_name = concept['cluster_label'].split('|')[0]

                if max_cluster_name:
                    row['theme_name'] = max_id
                    row['theme_score'] = max_score
                    row['theme_name'] = max_cluster_name

            doc_table.append(row)
            offset += 1

            # output doc_table before it get's too big and takes up too much memory
            if len(doc_table) > DOC_BATCH_SIZE:
                if doc_table_writer:
                    doc_table_writer.output_data(doc_table)
                doc_table = []

            mdrow = []
            for md in doc['metadata']:
                mdrow = {
                    'metadata_name': md['name'],
                    'metadata_type': md['type'],
                    'metadata_value': md['value'],
                    'doc_id': doc['doc_id']
                }
                doc_metadata_table.append(mdrow)

            # output doc_metadata table before it gets too big
            if len(doc_metadata_table) > DOC_BATCH_SIZE:
                if doc_metadata_table_writer:
                    doc_metadata_table_writer.output_data(doc_metadata_table)
                doc_metadata_table = []

            # build the doc term sentiment table
            for term in tqdm(doc['terms'], desc="process terms", leave=False):
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
                                    doc_term_sentiment_table.append(row2)

                    else:
                        doc_term_sentiment_table.append(row)

            if len(doc_term_sentiment_table) > DOC_BATCH_SIZE:
                if doc_term_sentiment_writer:
                    doc_term_sentiment_writer.output_data(doc_term_sentiment_table)
                doc_term_sentiment_table = []

            if doc['vector']:
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

            if len(doc_term_summary_table) > DOC_BATCH_SIZE:
                if doc_term_summary_writer:
                    doc_term_summary_writer.output_data(doc_term_summary_table)
                doc_term_summary_table = []

            for field in doc['metadata']:
                doc_subset_table.append({'doc_id': doc['doc_id'],
                                         'field_name': field['name'],
                                         'field_type': field['type'],
                                         'field_value': field['value']})
            if len(doc_subset_table) > DOC_BATCH_SIZE:
                if doc_subset_writer:
                    doc_subset_writer.output_data(doc_subset_table)
                doc_subset_table = []

        # if we are done reading and got zero docs
        # first write out any remaining data, then break the loop
        if len(new_docs) <= 0:
            if doc_table_writer and len(doc_table) > 0:
                doc_table_writer.output_data(doc_table)
            if doc_metadata_table_writer and len(doc_metadata_table) > 0:
                doc_metadata_table_writer.output_data(doc_metadata_table)
            if doc_term_sentiment_writer and len(doc_term_sentiment_table) > 0:
                doc_term_sentiment_writer.output_data(doc_term_sentiment_table)
            if doc_term_summary_writer and len(doc_term_summary_table) > 0:
                doc_term_summary_writer.output_data(doc_term_summary_table)
            if doc_subset_writer and len(doc_subset_table) > 0:
                doc_subset_writer.output_data(doc_subset_table)
            break

    return docs_by_date


def create_terms_table(lumi_writer, concepts, scl_match_counts):
    '''
    Create a tabulation of top terms and their exact/total match counts

    :param lumi_writer: A LumiDataWriter that sends to either csv or sql in write as it goes to save memory
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
        if len(table) > DOC_BATCH_SIZE:
            lumi_writer.output_data(table)
            table = []

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
        if len(table) > DOC_BATCH_SIZE:
            lumi_writer.output_data(table)
            table = []

    # one final write to empty the data to storage
    lumi_writer.output_data(table)
    table = []


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


def create_writer(format, filename, sql_connection, table_name, project_id, encoding):
    if format in 'sql':
        return LumiSqlWriter(sql_connection, table_schemas, table_name, project_id)
    else:
        return LumiCsvWriter(filename, table_name, project_id, encoding)


def run_export(export_config):
    root_url, api_url, workspace, project_id = parse_url(export_config['project_url'])

    logger.info(f"export_config: \n{pformat(export_config)}")
    conn = None
    if export_config['output_format'] in 'sql':
        if not export_config['db_connection']:
            conn = db_create_sql_connection()
        else:
            conn = export_config['db_connection']

        print("creating sql tables")
        if db_create_tables(conn) != 0:
            exit(-1)

    lumi_data = pull_lumi_data(project_id, api_url,
                               concept_count=int(export_config['concept_count']),
                               cln=export_config['concept_list_names'],
                               token=export_config['token'])
    (luminoso_data, scl_match_counts, concepts, themes, subset_values_dict) = lumi_data
    client = luminoso_data.client

    # this was implemented to reduce the time of the export on the dashboard system
    # it's a quick fix and needs a better solution
    luminoso_data.reduce_field_list = export_config['reduce_field_list']

    luminoso_data.set_root_url(
        root_url + '/app/projects/' + workspace + '/' + project_id
    )

    date_field_info = luminoso_data.first_date_field
    date_field_name = date_field_info['name']

    # luminoso_data used to read all the docs, it was taking too much memory
    # so now we are going to write the docs to storage as they are written,
    # but also get the list of docs by date for over time calculations
    # docs, docs_by_date = read_docs(luminoso_data.client, date_field_name)
    # luminoso_data.docs_by_date = docs_by_date
    
    logger.debug("starting exports.")

    if not export_config['skip_docs']:
    
        # if we are not a csv file then do it
        # if we are a csv file and the file already exists, skip this part of the export
        if (export_config['output_format']!='csv') or ((not export_config['skip_if_csv_exists'] or not os.path.isfile('docs.csv'))):
            doc_table_writer = create_writer(export_config['output_format'],
                                            'docs.csv', conn,
                                            'docs', project_id, 
                                            encoding=export_config['encoding'])
            doc_metadata_table_writer = create_writer(export_config['output_format'],
                                                    'doc_metadata.csv', conn,
                                                    'doc_metadata', project_id, 
                                                    encoding=export_config['encoding'])
        else:
            print("skipping doc export - csv exists")
    else:
        doc_table_writer = None
        doc_metadata_table_writer = None
    
    if not export_config['skip_doc_term_sentiment']:
        if (export_config['output_format']!='csv') or ((not export_config['skip_if_csv_exists'] or not os.path.isfile('doc_term_sentiment.csv'))):
    
            doc_term_sentiment_writer = create_writer(export_config['output_format'],
                                                    'doc_term_sentiment.csv', conn,
                                                    'doc_term_sentiment', project_id,
                                                    encoding=export_config['encoding'])
            concept_lists = client.get("concept_lists/")
        else:
            print("skipping doc_term_sentiment.csv - file exists")
    
    else:
        doc_term_sentiment_writer = None
        concept_lists = None
    
    if not export_config['skip_doc_term_summary']:
        if (export_config['output_format']!='csv') or ((not export_config['skip_if_csv_exists'] or not os.path.isfile('doc_term_summary.csv'))):
    
            doc_term_summary_writer = create_writer(export_config['output_format'],
                                        'doc_term_summary.csv', conn,
                                        'doc_term_summary', project_id,
                                        encoding=export_config['encoding'])
        else:
            print("Skipping doc_term_summary.csv - file exists")
    else:
        doc_term_summary_writer = None
    
    if not export_config['skip_doc_subset']:
        if (export_config['output_format']!='csv') or ((not export_config['skip_if_csv_exists'] or not os.path.isfile('doc_subsets.csv'))):
    
            doc_subset_writer = create_writer(export_config['output_format'],
                                        'doc_subsets.csv', conn,
                                        'doc_subsets', project_id,
                                        encoding=export_config['encoding'])
            docs_by_date = create_doc_table(luminoso_data.client, 
                                            doc_table_writer,
                                            doc_metadata_table_writer,
                                            doc_term_sentiment_writer, concept_lists,
                                            doc_term_summary_writer, concepts, scl_match_counts,
                                            doc_subset_writer,
                                            date_field_name, themes, export_config['tag_doc_term_sentiment_list'])
            luminoso_data.docs_by_date = docs_by_date
        else:
            print("Skipping doc_subsets.csv - file exists")
    else:
        doc_subset_writer = None
    
    if not export_config['skip_terms']:
        if (export_config['output_format']!='csv') or ((not export_config['skip_if_csv_exists'] or not os.path.isfile('terms.csv'))):
    
            lumi_writer = create_writer(export_config['output_format'],
                                        'terms.csv', conn,
                                        'terms', project_id,
                                        encoding=export_config['encoding'])
            create_terms_table(lumi_writer, concepts, scl_match_counts)
        else:
            print("Skipping terms.csv - file exists")
    
    
    if not export_config['skip_themes']:
    
        print('Creating themes table...')
        themes_table = create_themes_table(client, themes)
    
        if (export_config['output_format']!='csv') or ((not export_config['skip_if_csv_exists'] or not os.path.isfile('themes.csv'))):
    
            lumi_writer = create_writer(export_config['output_format'],
                                        'themes.csv', conn,
                                        'themes', project_id,
                                        encoding=export_config['encoding'])
            lumi_writer.output_data(themes_table)
        else:
            print("Skipping themes.csv - file exists")
    
    
    if not export_config['skip_vol']:
        print('Creating volume table...')
    
        if (export_config['output_format']!='csv') or ((not export_config['skip_if_csv_exists'] or not os.path.isfile('volume.csv'))):
    
            lumi_writer = create_writer(export_config['output_format'],
                                        'volume.csv', conn,
                                        'volume', project_id,
                                        encoding=export_config['encoding'])
            volume_table = create_volume_table(client, scl_match_counts,
                                            root_url=luminoso_data.root_url)
    
            lumi_writer.output_data(volume_table)
        else:
            print("Skipping volume.csv - file exists")
    
        if (export_config['output_format']!='csv') or ((not export_config['skip_if_csv_exists'] or not os.path.isfile('volume_subsets.csv'))):
    
            print("Creating volume by subsets...")
            lumi_writer = create_writer(export_config['output_format'],
                                        'volume_subsets.csv', conn,
                                        'volume_subsets', project_id, 
                                        encoding=export_config['encoding'])
            create_volume_subset_table(
                lumi_writer, luminoso_data,
                export_config['volume_subset_fields'])
        else:
            print("Skipping volumne_subsets.csv - file exists")
    
    if bool(export_config['run_vot']):
    
        if (export_config['output_format']!='csv') or ((not export_config['skip_if_csv_exists'] or not os.path.isfile('volume_over_time.csv'))):
    
            print("Creating volume over time (vot)")
    
            if export_config['vot_date_field'] is None:
                date_field_info = luminoso_data.first_date_field
                if date_field_info is None:
                    print("ERROR no date field in project for vot")
                    return
            else:
                date_field_info = luminoso_data.get_field_by_name(
                    export_config['vot_date_field']
                )
                if date_field_info is None:
                    print("ERROR: (vot) no date field name:"
                        " {}".format(export_config['vot_date_field']))
                    return
    
            lumi_writer = create_writer(export_config['output_format'],
                                        'volume_over_time.csv', conn,
                                        'volume_over_time', project_id,
                                        encoding=export_config['encoding'])
            create_vot_table(
                lumi_writer, luminoso_data,
                date_field_info, export_config['sot_end'],
                int(export_config['sot_iterations']), export_config['sot_range'],
                export_config['volume_subset_fields']
            )
            # lumi_writer.output_data(vot_table)
        else:
            print("Skipping volume_over_time.csv - file exits")
    
    # unique to filter u2f was skt
    if not export_config['skip_u2f_table']:
        if (export_config['output_format']!='csv') or ((not export_config['skip_if_csv_exists'] or not os.path.isfile('unique_to_filter.csv'))):
    
            print("Creating unique to filter...")
    
            u2f = unique_to_filter(client, subset_values_dict, terms_per_subset=int(export_config['u2f_limit']))
    
            u2f_table = create_u2f_table(client, u2f)
            lumi_writer = create_writer(export_config['output_format'],
                                        'unique_to_filter.csv', conn,
                                        'unique_to_filter', project_id,
                                        encoding=export_config['encoding'])
            lumi_writer.output_data(u2f_table)
        else:
            print("Skipping unique_to_filter.csv - file exists")
    
    # unique to filter over time (was skt)
    if bool(export_config['run_u2fot']):
        if (export_config['output_format']!='csv') or ((not export_config['skip_if_csv_exists'] or not os.path.isfile('unique_over_time.csv'))):
    
            print("Creating unique to filter over time...")
    
            if export_config['u2fot_date_field'] is None:
                date_field_info = luminoso_data.first_date_field
                if date_field_info is None:
                    print("ERROR no date field in project")
                    return
            else:
                date_field_info = luminoso_data.get_field_by_name(
                    export_config['u2fot_date_field']
                )
                if date_field_info is None:
                    print("ERROR: no date field name:"
                        " {}".format(export_config['u2fot_date_field']))
                    return
    
            lumi_writer = create_writer(export_config['output_format'],
                                        'unique_over_time.csv', conn,
                                        'unique_over_time', project_id,
                                        encoding=export_config['encoding'])
            create_u2fot_table(
                lumi_writer, luminoso_data, date_field_info,
                export_config['u2fot_end'], export_config['u2fot_iterations'],
                export_config['u2fot_range'], subset_values_dict,
                export_config['u2f_limit'])
        else:
            print("Skipping unique_over_time.csv - file exists")
    
    if not export_config['skip_drivers']:
        if (export_config['output_format']!='csv') or ((not export_config['skip_if_csv_exists'] or not os.path.isfile('drivers.csv'))):
    
            print("Creating score drivers...")
            driver_table = create_drivers_table(luminoso_data, export_config['run_topic_drivers'])
            lumi_writer = create_writer(export_config['output_format'],
                                        'drivers.csv', conn,
                                        'drivers', project_id,
                                        encoding=export_config['encoding'])
            lumi_writer.output_data(driver_table)
        else:
            print("Skipping drivers.csv - file exists")
    
    if not export_config['skip_driver_subsets']:
        if (export_config['output_format']!='csv') or ((not export_config['skip_if_csv_exists'] or not os.path.isfile('drivers_subsets.csv'))):
    
            print("starting subset drivers - topics={}".format(export_config['run_topic_drivers']))
    
            lumi_writer = create_writer(export_config['output_format'],
                                        'drivers_subsets.csv', conn,
                                        'drivers_subsets', project_id, 
                                        encoding=export_config['encoding'])
    
            create_drivers_with_subsets_table(
                lumi_writer, luminoso_data, export_config['run_topic_drivers'],
                subset_fields=export_config['driver_subset_fields']
            )
        else:
            print("Skipping drivers_subsets.csv - file exists")
    
    if bool(export_config['run_sdot']):
        if (export_config['output_format']!='csv') or ((not export_config['skip_if_csv_exists'] or not os.path.isfile('drivers_over_time.csv'))):
    
            if export_config['sdot_date_field'] is None:
                date_field_info = luminoso_data.first_date_field
                if date_field_info is None:
                    print("ERROR no date field in project")
                    return
            else:
                date_field_info = luminoso_data.get_field_by_name(
                    export_config['sdot_date_field']
                )
                if date_field_info is None:
                    print("ERROR: no date field name:"
                        " {}".format(export_config['sdot_date_field']))
                    return
    
            lumi_writer = create_writer(export_config['output_format'],
                        'drivers_over_time.csv', conn,
                        'drivers_over_time', project_id, encoding=export_config['encoding'])
            create_sdot_table(
                lumi_writer, luminoso_data, date_field_info, export_config['sdot_end'],
                int(export_config['sdot_iterations']),
                export_config['sdot_range'], export_config['run_topic_drivers']
            )
        else:
            print("Skipping drivers_over_time.csv - file exists")
    
    if not export_config['skip_sentiment']:
        if (export_config['output_format']!='csv') or ((not export_config['skip_if_csv_exists'] or not os.path.isfile('sentiment.csv'))):
    
            print('Creating sentiment table...')
            sentiment_table = create_sentiment_table(client, scl_match_counts,
                                                    root_url=luminoso_data.root_url)
            lumi_writer = create_writer(export_config['output_format'],
                                        'sentiment.csv', conn,
                                        'sentiment', project_id, 
                                        encoding=export_config['encoding'])
            lumi_writer.output_data(sentiment_table)
        else:
            print("skipping sentiment.csv - file exists")

    logger.info("start.")
    
    if not export_config['skip_sentiment_subsets']:
        if (export_config['output_format']!='csv') or ((not export_config['skip_if_csv_exists'] or not os.path.isfile('sentiment_subsets.csv'))):
    
            print("Creating sentiment by subsets...")
            lumi_writer = create_writer(export_config['output_format'],
                                        'sentiment_subsets.csv', conn,
                                        'sentiment_subsets', project_id,
                                        encoding=export_config['encoding'])
            create_sentiment_subset_table(
                lumi_writer, luminoso_data,
                export_config['sentiment_subset_fields'])
        else:
            print("Skipping sentiment_subsets.csv - file exists")
            
                
                
    logger.info("done.")

    if bool(export_config['run_sot']):
        if (export_config['output_format']!='csv') or ((not export_config['skip_if_csv_exists'] or not os.path.isfile('sentiment_over_time.csv'))):
    
            print("Creating sentiment over time (sot)")
    
            if export_config['sot_date_field'] is None:
                date_field_info = luminoso_data.first_date_field
                if date_field_info is None:
                    print("ERROR no date field in project for sot")
                    return
            else:
                date_field_info = luminoso_data.get_field_by_name(
                    export_config['sot_date_field']
                )
                if date_field_info is None:
                    print("ERROR: (sot) no date field name:"
                        " {}".format(export_config['sot_date_field']))
                    return
    
            lumi_writer = create_writer(export_config['output_format'],
                                        'sentiment_over_time.csv', conn,
                                        'sentiment_over_time', project_id,
                                        encoding=export_config['encoding'])
            create_sot_table(
                lumi_writer, luminoso_data, date_field_info, export_config['sot_end'],
                int(export_config['sot_iterations']), export_config['sot_range'],
                export_config['sentiment_subset_fields']
            )
        else:
            print("Skipping sentiment_over_time.csv - file exists")
    
    if not bool(export_config['skip_outliers']) or (bool(export_config['run_outliersot'])):
    
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

    if not bool(export_config['skip_outliers']):
        if (export_config['output_format']!='csv') or ((not export_config['skip_if_csv_exists'] or not os.path.isfile('outliers.csv'))):
    
            print("Generating outlier concepts...")
            outlier_table = create_outlier_table(client, proj_info,
                                                scl_match_counts,
                                                "both",
                                                root_url=luminoso_data.root_url)
            outlier_table.extend(create_outlier_table(client, proj_info,
                                                    scl_match_counts,
                                                    "exact",
                                                    root_url=luminoso_data.root_url))
            lumi_writer = create_writer(export_config['output_format'],
                                        'outliers.csv', conn,
                                        'outliers', project_id,
                                        encoding=export_config['encoding'])
            lumi_writer.output_data(outlier_table)
        else:
            print("Skipping outliers.csv - file exitst")
    
        print("Generating outliers by subsets...")
        outlier_subset_table = create_outlier_subset_table(
            luminoso_data,
            proj_info, 
            scl_match_counts, 
            "both",
            export_config['outlier_subset_fields'])
        outlier_subset_table.extend(create_outlier_subset_table(
            luminoso_data,
            proj_info, 
            scl_match_counts, 
            "exact",
            export_config['outlier_subset_fields']))
    
        if (export_config['output_format']!='csv') or ((not export_config['skip_if_csv_exists'] or not os.path.isfile('outlier_subsets.csv'))):
    
            lumi_writer = create_writer(export_config['output_format'],
                                        'outlier_subsets.csv', conn,
                                        'outlier_subsets', project_id,
                                        encoding=export_config['encoding'])
            lumi_writer.output_data(outlier_subset_table)
        else:
            print("Skipping outlier_subsets.csv - file exists")
    
    if bool(export_config['run_outliersot']):
        if (export_config['output_format']!='csv') or ((not export_config['skip_if_csv_exists'] or not os.path.isfile('outliers_over_time.csv'))):
    
            print("Calculating outlier concepts over time (outliersot)")
    
            if export_config['outliersot_date_field'] is None:
                date_field_info = luminoso_data.first_date_field
                if date_field_info is None:
                    print("ERROR no date field in project for outliersot")
                    return
            else:
                date_field_info = luminoso_data.get_field_by_name(
                    export_config['outliersot_date_field']
                )
                if date_field_info is None:
                    print("ERROR: (outliersot) no date field name:"
                        " {}".format(export_config['outliersot_date_field']))
                    return
    
            outliersot_table = create_outliersot_table(
                luminoso_data, proj_info, scl_match_counts, "both",
                date_field_info, export_config['outliersot_end'],
                int(export_config['outliersot_iterations']),
                export_config['outliersot_range'], 
                export_config['outlier_subset_fields']
            )
            outliersot_table.extend(create_outliersot_table(
                luminoso_data, proj_info, scl_match_counts, "exact",
                date_field_info, export_config['outliersot_end'],
                int(export_config['outliersot_iterations']),
                export_config['outliersot_range'], 
                export_config['outlier_subset_fields']
            ))
            lumi_writer = create_writer(export_config['output_format'],
                                        'outliers_over_time.csv', conn,
                                        'outliers_over_time', project_id,
                                        encoding=export_config['encoding'])
            lumi_writer.output_data(outliersot_table)
        else:
            print("Skipping outliers_over_time.csv - file exists")

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
    parser.add_argument('-s', '--skip_if_csv_exists', default=False,
                        action='store_true',
                        help="Skip if the csv already exists, for run again speedup" )
    parser.add_argument('-skip_docs', '--skip_docs',
                        default=False,
                        action='store_true',
                        help="Skip generating doc_table")
    parser.add_argument('-skip_doc_term_sentiment', '--skip_doc_term_sentiment',
                        default=False,
                        action='store_true',
                        help="Skip generating doc_term_sentiment_table")
    parser.add_argument('-tag_doc_term_sentiment_list', '--tag_doc_term_sentiment_list',
                        default=False,
                        action='store_true',
                        help="Tag the term sentiment when concept is in a shared concept list - default=false")
    parser.add_argument('-skip_terms', '--skip_terms',
                        default=False,
                        action='store_true',
                        help="Skip generating the terms_table")
    parser.add_argument('-skip_themes', '--skip_themes',
                        default=False,
                        action='store_true',
                        help="Skip generating the themes_table")
    parser.add_argument('-skip_dtermsum', '--skip_doc_term_summary',
                        default=False,
                        action='store_true',
                        help="Skip generating doc_term_summary_table")
    parser.add_argument('-skip_doc_subset', '--skip_doc_subset', default=False,
                        action='store_true',
                        help="Skip generating the doc_subsets_table")
    parser.add_argument('-u2f', '--skip_u2f_table', default=False,
                        action='store_true', help="Skip generating the u2f_tables")
    parser.add_argument('-skip_drivers', '--skip_drivers', default=False,
                        action='store_true',
                        help="Skip generating the driver_table")
    parser.add_argument('-tdrive', '--topic_drive', default=True,
                        action='store_true',
                        help="If generating drivers_table do so with"
                             " top concepts, shared concept lists"
                             " and auto concepts")
    parser.add_argument('-skip_driver_subsets', '--skip_driver_subsets', default=False,
                        action='store_true',
                        help="Skip generating score drivers by subset")
    parser.add_argument('--driver_subset_fields', default=None,
                        help='Which subsets to include in score driver by'
                             ' subset. Default = All with < 200 unique values.'
                             ' Samp: "field1,field2"')

    parser.add_argument('-skip_sentiment', '--skip_sentiment', default=False,
                        action='store_true',
                        help="Skip generating sentiment for top concepts")
    parser.add_argument('--sdot', action='store_true',
                        default=False,
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
                        help="Skip generating the volume table vol_table")
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
    parser.add_argument('--skip_sentiment_subsets',
                        action='store_true', default=False,
                        help="Skip generating sentiment subsets")
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
    parser.add_argument('-skip_outliers', '--skip_outliers',
                        action='store_true', default=False,
                        help="Skip generating outliers and outlier subsets")
    parser.add_argument('--outlier_subset_fields', default=None,
                        help='Which subsets to include in outlier by'
                             ' subset. Default = All with < 200 unique values.'
                             ' Samp: "field1,field2"')
    parser.add_argument('-outliersot', '--outliersot', action='store_true',
                        default=False,
                        help="Calculate outlier concepts over time (outliersot)")
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

    export_config = {
        'project_url':args.project_url,
        'concept_count': args.concept_count,
        'encoding': args.encoding,
        'concept_list_names': args.concept_list_names,
        'output_format': args.output_format,
        'skip_if_csv_exists': args.skip_if_csv_exists,
        'u2f_limit': args.u2f_limit,
        'skip_vol': args.skip_vol,
        'volume_subset_fields': args.volume_subset_fields,
        'run_vot': args.vot,
        'vot_end': args.vot_end,
        'vot_iterations': args.vot_iterations,
        'vot_range': args.vot_range,
        'vot_date_field': args.vot_date_field,
        'skip_docs': args.skip_docs,
        'skip_doc_term_sentiment': args.skip_doc_term_sentiment,
        'tag_doc_term_sentiment_list': args.tag_doc_term_sentiment_list,
        'skip_terms': args.skip_terms,
        'skip_themes': args.skip_themes,
        'skip_doc_term_summary': args.skip_doc_term_summary,
        'skip_doc_subset': args.skip_doc_subset,
        'skip_u2f_table': args.skip_u2f_table,
        'skip_drivers': args.skip_drivers,
        'run_topic_drivers': args.topic_drive,
        'skip_driver_subsets': args.skip_driver_subsets,
        'driver_subset_fields': args.driver_subset_fields,
        'skip_sentiment': args.skip_sentiment,
        'run_sdot': args.sdot,
        'sdot_end': args.sdot_end,
        'sdot_iterations': args.sdot_iterations,
        'sdot_range': args.sdot_range,
        'sdot_date_field': args.sdot_date_field,
        'skip_sentiment_subsets': args.skip_sentiment_subsets,
        'sentiment_subset_fields': args.sentiment_subset_fields,
        'run_sot': args.sot,
        'sot_end': args.sot_end,
        'sot_iterations': args.sot_iterations,
        'sot_range': args.sot_range,
        'sot_date_field': args.sot_date_field,
        'run_u2fot': args.u2fot,
        'u2fot_end': args.u2fot_end,
        'u2fot_iterations': args.u2fot_iterations,
        'u2fot_range': args.u2fot_range,
        'u2fot_date_field': args.u2fot_date_field,
        'skip_outliers': args.skip_outliers,
        'outlier_subset_fields': args.outlier_subset_fields,
        'run_outliersot': args.outliersot,
        'outliersot_end': args.outliersot_end,
        'outliersot_iterations': args.outliersot_iterations,
        'outliersot_range': args.outliersot_range,
        'outliersot_date_field': args.outliersot_date_field,
        'token': None,
        'db_connection': None,
        'reduce_field_list': False
    }
    run_export(export_config)


if __name__ == '__main__':
    main()
