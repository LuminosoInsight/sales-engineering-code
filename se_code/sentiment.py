import argparse
import csv
from datetime import datetime, timedelta
#import urllib.parse
#import numpy as np
#import pandas as pd

from luminoso_api import V5LuminosoClient as LuminosoClient
#from pack64 import unpack64
from se_code.score_drivers import (
     LuminosoData, write_table_to_csv
)

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

    for scl_name, shared_concepts in scl_match_counts.items():
        results_saved = client.get(
            '/concepts/sentiment/',
            concept_selector={
                "type": "concept_list",
                "concept_list_id": shared_concepts['concept_list_id']
            }
        )['match_counts']

        sentiment_match_counts.extend([
            {'texts': concept['texts'],
             'name': concept['name'],
             'match_count': concept['match_count'],
             'concept_type': 'shared',
             'shared_concept_list': scl_name,
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

    # FIXME: very similar to _create_rows_from_drivers(), except that it only
    #  needs the top 3 (and therefore doesn't need to sort them)
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


def _create_row_for_sentiment_subsets(luminoso_data, api_params, subset_name, subset_value, list_type, list_name):
    """
    Helper function for create_sentiment_subset_table().
    """
    rows = []

    sentiments = luminoso_data.client.get(
        'concepts/sentiment', **api_params
    )

    for c in sentiments['match_counts']:
        rows.append({'type': list_type,
                     'name': list_name,
                     'subset_name': subset_name,
                     'subset_value': subset_value,
                     'concept': c['name'],
                     'relevance': c['relevance'],
                     'match_count': c['match_count'],
                     'exact_match_count': c['exact_match_count'],
                     'conceptual_match_count': c['match_count'] - c['exact_match_count'],
                     'sentiment_share_positive': c['sentiment_share']['positive'],
                     'sentiment_share_neutral': c['sentiment_share']['neutral'],
                     'sentiment_share_negative': c['sentiment_share']['negative'],
                     'sentiment_doc_count_positive': c['sentiment_counts']['positive'],
                     'sentiment_doc_count_neutral': c['sentiment_counts']['neutral'],
                     'sentiment_doc_count_negative': c['sentiment_counts']['negative'],
                     'sentiment_doc_count_total': c['sentiment_counts']['total']
                     })

    return rows


def create_sentiment_subset_table(luminoso_data, subset_fields):
    '''
    Create tabulation of sentiment output
    :param luminoso_data: a LuminosoData object
    :param filter_list: document filter (as a list of dicts)
    :return: List of sentiments
    '''
    print("Generating sentiment by subsets...")
    sentiment_table = []

    # if the user specifies the list of subsets to process
    if not subset_fields:
        subset_fields = luminoso_data.get_best_subset_fields()
    else:
        subset_fields = subset_fields.split(",")

    # process sentiments by subset
    sentiment_table = []

    for field_name in subset_fields:
        field_values = luminoso_data.get_fieldvalues_for_fieldname(field_name)
        print("{}: field_values = {}".format(field_name, field_values))
        for field_value in field_values:
            filter_list = [{"name": field_name, "values": field_value}]
            print("sentiment filter={}".format(filter_list))

            api_params = {'filter': filter_list}

            for list_name in luminoso_data.concept_lists:
                concept_list_params = dict(api_params,
                                        concept_selector={'type': 'concept_list', 'name': list_name})
                sentiment_table.extend(_create_row_for_sentiment_subsets(
                    luminoso_data, concept_list_params, field_name, field_value[0], 'shared_concept_list', list_name
                ))

            top_params = dict(api_params, concept_selector={'type': 'top'})
            sentiment_table.extend(_create_row_for_sentiment_subsets(
                luminoso_data, top_params, field_name, field_value[0], 'auto', 'top'
            ))

            sentiment_params = dict(api_params, concept_selector={'type': 'sentiment_suggested'})
            sentiment_table.extend(_create_row_for_sentiment_subsets(
                luminoso_data, sentiment_params, field_name, field_value[0], 'auto', 'sentiment_suggested'
            ))

    return sentiment_table


def main():

    parser = argparse.ArgumentParser(
        description='Export score drivers and write to a file'
    )
    parser.add_argument('project_url',
                        help="The complete URL of the Daylight project")
    parser.add_argument('--encoding', default='utf-8',
                        help="Encoding type of the files to write to")
    parser.add_argument('--sentiment_subset_fields', default=None,
                        help='Which subsets to include in sentiments by'
                             ' subset. Default = All with < 200 unique values.'
                             ' Samp: "field1,field2"')
    args = parser.parse_args()

    project_url = args.project_url.strip('/')
    api_url = project_url.split('/app')[0].strip() + '/api/v5'
    project_id = project_url.split('/')[6].strip()

    client = LuminosoClient.connect(
            url='%s/projects/%s' % (api_url.strip('/'), project_id),
            user_agent_suffix='se_code:sentiment'
        )
    luminoso_data = LuminosoData(client)
    print('Getting Sentiment...')

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

    sentiment_table = create_sentiment_table(client, scl_match_counts,
                                             root_url=luminoso_data.root_url)
    write_table_to_csv(sentiment_table, 'sentiment.csv',
                        encoding=args.encoding)

    sentiment_subset_table = create_sentiment_subset_table(
        luminoso_data,
        args.sentiment_subset_fields)
    write_table_to_csv(sentiment_subset_table, 'sentiment_subsets.csv',
                        encoding=args.encoding)

if __name__ == '__main__':
    main()
