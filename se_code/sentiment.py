import argparse
import csv
from datetime import datetime, timedelta
import urllib.parse

from luminoso_api import V5LuminosoClient as LuminosoClient
from se_code.score_drivers import (
     LuminosoData, write_table_to_csv
)

def create_sentiment_table(client, scl_match_counts, root_url=''):

    # first get the default sentiment output with sentiment suggestions
    results = client.get('/concepts/sentiment/')['match_counts']
    sentiment_match_counts = [
        {'texts': concept['texts'],
         'concept': concept['name'],
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
             'concept': concept['name'],
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
         'concept': concept['name'],
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

        srow['example_doc1'] = ''
        srow['example_doc2'] = ''
        srow['example_doc3'] = ''
        if len(search_docs) >= 1:
            srow['example_doc1'] = search_docs[0]['text']
        if len(search_docs) >= 2:
            srow['example_doc2'] = search_docs[1]['text']
        if len(search_docs) >= 3:
            srow['example_doc3'] = search_docs[2]['text']

    return sentiment_match_counts


def _create_row_for_sentiment_subsets(luminoso_data, api_params, subset_name, subset_value, list_type, list_name, prepend_to_rows=None):
    """
    Helper function for create_sentiment_subset_table().
    """
    rows = []

    sentiments = luminoso_data.client.get(
        'concepts/sentiment', **api_params
    )

    for c in sentiments['match_counts']:
        row = {'list_type': list_type,
               'list_name': list_name,
               'field_name': subset_name,
               'field_value': subset_value,
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
               }
        if prepend_to_rows:
            row = {**prepend_to_rows, **row}
        rows.append(row)
    return rows


def create_sentiment_subset_table(luminoso_data, subset_fields=None, filter_list=None, prepend_to_rows=None, add_overall_values=False):
    '''
    Create tabulation of sentiment output
    :param luminoso_data: a LuminosoData object
    :param filter_list: document filter (as a list of dicts)
    :return: List of sentiments
    '''
    sentiment_table = []

    # if the user specifies the list of subsets to process
    if not subset_fields:
        subset_fields = luminoso_data.get_best_subset_fields()
    else:
        subset_fields = subset_fields.split(",")

    # process sentiments by subset
    sentiment_table = []

    orig_filter_list = filter_list

    api_params = {'filter': filter_list}

    # this is typically only for over-time output since the project
    # wide values are available in the standard output
    if add_overall_values:
        # time slice for project wide overall top and sentiment suggested concepts
        concept_list_params = dict(api_params,
                                   concept_selector={'type': 'top', 'limit': 100})
        sentiment_table.extend(_create_row_for_sentiment_subsets(
            luminoso_data, concept_list_params, '', '', 
            'overall', 'Top Concepts', prepend_to_rows
        ))
    
        concept_list_params = dict(api_params,
                                   concept_selector={"type": "suggested", "num_clusters": 7, "num_cluster_concepts": 4})
        sentiment_table.extend(_create_row_for_sentiment_subsets(
            luminoso_data, concept_list_params, '', '', 
            'overall', 'Suggested Clusters', prepend_to_rows
        ))

        concept_list_params = dict(api_params,
                                   concept_selector={'type': 'sentiment_suggested'})
        sentiment_table.extend(_create_row_for_sentiment_subsets(
            luminoso_data, concept_list_params, '', '', 
            'overall', 'Suggested Sentiment', prepend_to_rows
        ))

    for field_name in subset_fields:
        field_values = luminoso_data.get_fieldvalues_for_fieldname(field_name)
        print("{}: field_values = {}".format(field_name, field_values))
        if not field_values:
            print("  {}: skipping".format(field_name))
        else:
            for field_value in field_values:
                if (not isinstance(field_value[0], str)) or len(field_value[0])<64:
                    filter_list = []
                    if orig_filter_list:
                        filter_list.extend(orig_filter_list)
                    filter_list.append({"name": field_name, "values": field_value})
                    print("sentiment filter={}".format(filter_list))

                    api_params = {'filter': filter_list}

                    for list_name in luminoso_data.concept_lists:
                        concept_list_params = dict(api_params,
                                                concept_selector={'type': 'concept_list', 'name': list_name})
                        sentiment_table.extend(_create_row_for_sentiment_subsets(
                            luminoso_data, concept_list_params, field_name, field_value[0], 
                            'shared_concept_list', list_name, prepend_to_rows
                        ))

                    top_params = dict(api_params, concept_selector={'type': 'top'})
                    sentiment_table.extend(_create_row_for_sentiment_subsets(
                        luminoso_data, top_params, field_name, field_value[0], 
                        'auto', 'Top', prepend_to_rows
                    ))
        
                    suggested_params = dict(api_params,
                                            concept_selector={"type": "suggested", "num_clusters": 7, "num_cluster_concepts": 4})
                    sentiment_table.extend(_create_row_for_sentiment_subsets(
                        luminoso_data, suggested_params, field_name, field_value[0], 
                        'auto', 'Suggested Clusters', prepend_to_rows
                        ))

                    suggested_params = dict(api_params,
                                            concept_selector={"type": "suggested", "num_clusters": 7, "num_cluster_concepts": 4})
                    sentiment_table.extend(_create_row_for_sentiment_subsets(
                        luminoso_data, suggested_params, field_name, field_value[0], 
                        'auto', 'Suggested Clusters', prepend_to_rows
                    ))

                    sentiment_params = dict(api_params, concept_selector={'type': 'sentiment_suggested'})
                    sentiment_table.extend(_create_row_for_sentiment_subsets(
                        luminoso_data, sentiment_params, field_name, field_value[0], 
                        'auto', 'sentiment_suggested', prepend_to_rows
                    ))

    return sentiment_table


def create_sot_table(luminoso_data, date_field_info, end_date, iterations,
                     range_type, subset_fields):
    sot_data_raw = []

    if end_date is None or len(end_date) == 0:
        end_date = date_field_info['maximum']

    date_field_name = date_field_info['name']
    try:
        end_date_dt = datetime.strptime(end_date, '%m/%d/%Y')
    except ValueError:
        end_date_dt = datetime.strptime(end_date, '%Y-%m-%dT%H:%M:%SZ')

    end_date_epoch = end_date_dt.timestamp()
    start_date_dt = None

    if range_type is None or range_type not in ['M', 'W', 'D']:
        range_type = luminoso_data.find_best_interval(date_field_name,
                                                      iterations)
    range_descriptions = {'M': 'Month', 'W': 'Week', 'D': 'Day'}
    range_description = range_descriptions[range_type]

    print("sot starting. Date Field: {}, Iterations: {},"
          " Range Type: {}".format(date_field_name, iterations, range_type))

    # run the number of iterations
    for count in range(iterations):
        if range_type == "M":
            if not start_date_dt:
                if end_date_dt.day == 1:
                    print("error, cannot start with the beginning of a"
                          " month. Starting with previous month")
                    end_date_dt = end_date_dt - timedelta(days=1)
                start_date_dt = end_date_dt.replace(day=1)
            else:
                end_date_dt = start_date_dt.replace(day=1) - timedelta(days=1)
                start_date_dt = end_date_dt.replace(day=1)

            end_date_epoch = end_date_dt.timestamp()
            start_date_epoch = start_date_dt.timestamp()

        elif range_type == "W":  # week
            start_date_epoch = end_date_epoch - 60*60*24*7
        else:  # day
            start_date_epoch = end_date_epoch - 60*60*24
        start_date_dt = datetime.fromtimestamp(start_date_epoch)

        prepend_to_rows = {
            "start_date": start_date_dt.isoformat(),
            "end_date":  end_date_dt.isoformat(),
            "iteration_counter": count,
            "range_type": range_description
        }
        filter_list = [{"name": date_field_name,
                        "minimum": int(start_date_epoch),
                        "maximum": int(end_date_epoch)}]

        sd_data = create_sentiment_subset_table(luminoso_data, subset_fields,
                                                filter_list, prepend_to_rows, True)
        sot_data_raw.extend(sd_data)

        # move to the nextdate
        end_date_epoch = start_date_epoch
        end_date_dt = datetime.fromtimestamp(end_date_epoch)

    return sot_data_raw


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

    project_url = args.project_url.strip('/')
    api_url = project_url.split('/app')[0].strip() + '/api/v5'
    project_id = project_url.split('/')[6].strip()

    client = LuminosoClient.connect(
            url='%s/projects/%s' % (api_url.strip('/'), project_id),
            user_agent_suffix='se_code:sentiment'
        )
    luminoso_data = LuminosoData(client)
    print('Getting sentiment data...')

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

    if bool(args.sot):
        print("Calculating sentiment over time (sot)")

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
        write_table_to_csv(sot_table, 'sot_table.csv',
                           encoding=args.encoding)

    print("Generating project sentiment...")
    sentiment_table = create_sentiment_table(client, scl_match_counts,
                                             root_url=luminoso_data.root_url)
    write_table_to_csv(sentiment_table, 'sentiment.csv',
                       encoding=args.encoding)

    print("Generating sentiment by subsets...")
    sentiment_subset_table = create_sentiment_subset_table(
        luminoso_data,
        args.sentiment_subset_fields)
    write_table_to_csv(sentiment_subset_table, 'sentiment_subsets.csv',
                       encoding=args.encoding)


if __name__ == '__main__':
    main()
