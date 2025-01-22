import argparse
from datetime import datetime, timedelta
import requests
import urllib.parse

from luminoso_api import V5LuminosoClient as LuminosoClient
from luminoso_api import LuminosoServerError
from se_code.data_writer import (LumiCsvWriter)
from se_code.score_drivers import (
     LuminosoData, write_table_to_csv
)

from tqdm import tqdm
from loguru import logger
from pprint import pformat

WRITER_BATCH_SIZE = 5000

def create_volume_table(client, scl_match_counts, root_url=''):

    # first get the default volume output
    results_top = client.get(
        '/concepts/match_counts/',
        concept_selector={"type": "top", 'limit': 100}
    )['match_counts']
    volume_match_counts = [
        {'texts': concept['texts'],
         'concept': concept['name'],
         'concept_type': 'top',
         'match_count': concept['match_count'],
         'exact_match_count': concept['exact_match_count'],
         'conceptual_match_count': concept['match_count'] - concept['exact_match_count']
         }
        for concept in results_top
    ]
    
    logger.info("for scl_name, shared_concepts in tqdm(scl_match_counts.items()):")
    for scl_name, shared_concepts in tqdm(scl_match_counts.items()):
        results_saved = client.get(
            '/concepts/match_counts/',
            concept_selector={
                "type": "concept_list",
                "concept_list_id": shared_concepts['concept_list_id']
            }
        )['match_counts']

        volume_match_counts.extend([
            {'texts': concept['texts'],
             'concept': concept['name'],
             'concept_type': 'shared',
             'shared_concept_list': scl_name,
             'match_count': concept['match_count'],
             'exact_match_count': concept['exact_match_count'],
             'conceptual_match_count': concept['match_count'] - concept['exact_match_count']
             }
            for concept in results_saved
        ])

    # add three sample documents to each row
    logger.info("add three sample documents to each row")
    for srow in tqdm(volume_match_counts):
        if len(root_url) > 0:
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

    return volume_match_counts


def _create_row_for_volume_subsets(luminoso_data, api_params, subset_name, subset_value, list_type, list_name, prepend_to_rows=None):
    """
    Helper function for create_volume_subset_table().
    """
    rows = []

    try:
        volumes = luminoso_data.client.get(
            'concepts/match_counts', **api_params
        )
    except requests.exceptions.HTTPError as e:
        print(f"HTTPError {e}, volume - url too long? - skipping this subset {str(api_params['filter'])}")
        return rows
    except LuminosoServerError as e2:
        print(f"LuminosoServerError {e2}, volume - url too long? - skipping this subset {str(api_params['filter'])}")
        return rows
    except Exception as e3:
        print(f"Exception {e3}, volume - url too long? - skipping this subset {str(api_params['filter'])}")
        return rows

    for c in tqdm(volumes['match_counts'], desc="_create_row_for_volume_subsets", leave=False):
        row = {'list_type': list_type,
               'list_name': list_name,
               'field_name': subset_name,
               'field_value': subset_value,
               'concept': c['name'],
               'relevance': c['relevance'],
               'match_count': c['match_count'],
               'exact_match_count': c['exact_match_count'],
               'conceptual_match_count': c['match_count'] - c['exact_match_count']
               }
        if prepend_to_rows:
            row = {**prepend_to_rows, **row}
        rows.append(row)
    return rows


def create_volume_subset_table(lumi_writer, luminoso_data, subset_fields=None, filter_list=None, prepend_to_rows=None, add_overall_values=False):
    '''
    Create tabulation of volume output
    :param luminoso_data: a LuminosoData object
    :param filter_list: document filter (as a list of dicts)
    :return: List of volumes
    '''
    
    volume_table = []

    # if the user specifies the list of subsets to process
    if not subset_fields:
        subset_fields = luminoso_data.get_best_subset_fields()
    else:
        subset_fields = subset_fields.split(",")

    # process volme by subset
    volume_table = []

    orig_filter_list = filter_list

    api_params = {'filter': filter_list}

    # this is typically only for over-time output since the project
    # wide values are available in the standard output
    if add_overall_values:
        # time slice for project wide overall top, clusters, unique and sentiment suggested concepts
        concept_list_params = dict(api_params,
                                   concept_selector={'type': 'top', 'limit': 100})
        volume_table.extend(_create_row_for_volume_subsets(
            luminoso_data, concept_list_params, '', '', 
            'overall', 'Top Concepts', prepend_to_rows
        ))

        concept_list_params = dict(api_params,
                                   concept_selector={"type": "suggested", "num_clusters": 7, "num_cluster_concepts": 4})
        volume_table.extend(_create_row_for_volume_subsets(
            luminoso_data, concept_list_params, '', '',
            'overall', 'Suggested Clusters', prepend_to_rows
        ))

        concept_list_params = dict(api_params,
                                   concept_selector={'type': 'sentiment_suggested'})
        volume_table.extend(_create_row_for_volume_subsets(
            luminoso_data, concept_list_params, '', '',
            'overall', 'Suggested Sentiment', prepend_to_rows
        ))

    for field_name in tqdm(subset_fields, desc="for field_name"):
        field_values = luminoso_data.get_fieldvalue_lists_for_fieldname(field_name)
        print("{}: volume field_values = {}".format(field_name, field_values))
        if not field_values:
            print("  {}: skipping".format(field_name))
        else:
            for field_value in tqdm(field_values, desc="for field_value", leave=False):
                if (not isinstance(field_value[0], str)) or len(field_value[0])<64:
                    filter_list = []
                    if orig_filter_list:
                        filter_list.extend(orig_filter_list)
                    filter_list.append({"name": field_name, "values": field_value})
                    # print("volume filter={}".format(filter_list))

                    api_params = {'filter': filter_list}

                    for list_name in tqdm(luminoso_data.concept_lists, desc="for list_name", leave=False):
                        concept_list_params = dict(api_params,
                                                concept_selector={'type': 'concept_list', 'name': list_name})
                        volume_table.extend(_create_row_for_volume_subsets(
                            luminoso_data, concept_list_params, field_name, field_value[0], 
                            'shared_concept_list', list_name, prepend_to_rows
                        ))

                    top_params = dict(api_params, concept_selector={'type': 'top'})
                    volume_table.extend(_create_row_for_volume_subsets(
                        luminoso_data, top_params, field_name, field_value[0],
                        'auto', 'Top', prepend_to_rows
                    ))

                    suggested_params = dict(api_params,
                                            concept_selector={"type": "suggested", "num_clusters": 7, "num_cluster_concepts": 4})
                    volume_table.extend(_create_row_for_volume_subsets(
                        luminoso_data, suggested_params, field_name, field_value[0],
                        'auto', 'Suggested Clusters', prepend_to_rows
                        ))

                    suggested_params = dict(api_params,
                                            concept_selector={"type": "suggested", "num_clusters": 7, "num_cluster_concepts": 4})
                    volume_table.extend(_create_row_for_volume_subsets(
                        luminoso_data, suggested_params, field_name, field_value[0],
                        'auto', 'Suggested Clusters', prepend_to_rows
                    ))

                    sentiment_params = dict(api_params, concept_selector={'type': 'sentiment_suggested'})
                    volume_table.extend(_create_row_for_volume_subsets(
                        luminoso_data, sentiment_params, field_name, field_value[0],
                        'auto', 'sentiment_suggested', prepend_to_rows
                    ))

                    unique_params = dict(api_params, concept_selector={'type': 'unique_to_filter'})
                    volume_table.extend(_create_row_for_volume_subsets(
                        luminoso_data, unique_params, field_name, field_value[0],
                        'auto', 'unique_to_filter', prepend_to_rows
                    ))

                if len(volume_table) > WRITER_BATCH_SIZE:
                    if lumi_writer:
                        lumi_writer.output_data(volume_table)
                    volume_table = []

            # Need to add one entry that covers this entire field
            # only use this field value list if all the field values are less than len 64
            # because there are issues with overly long field value names.
            field_values = luminoso_data.get_all_fieldvalues_for_fieldname(field_name)
            field_values_oversize = [fv for fv in field_values if isinstance(field_value,str) and len(field_value)>63]
            if len(field_values_oversize) == 0:
                filter_list = []
                if orig_filter_list:
                    filter_list.extend(orig_filter_list)
                filter_list.append({"name": field_name, "values": field_values})
                # print("volume _all_ filter={}".format(filter_list))

                api_params = {'filter': filter_list}

                for list_name in luminoso_data.concept_lists:
                    concept_list_params = dict(api_params,
                                               concept_selector={'type': 'concept_list', 'name': list_name})
                    volume_table.extend(_create_row_for_volume_subsets(
                        luminoso_data, concept_list_params, field_name, "_all_",
                        'shared_concept_list', list_name, prepend_to_rows
                    ))

                top_params = dict(api_params, concept_selector={'type': 'top'})
                volume_table.extend(_create_row_for_volume_subsets(
                    luminoso_data, top_params, field_name, "_all_",
                    'auto', 'Top', prepend_to_rows
                ))

                suggested_params = dict(api_params,
                                        concept_selector={"type": "suggested", "num_clusters": 7, "num_cluster_concepts": 4})
                volume_table.extend(_create_row_for_volume_subsets(
                    luminoso_data, suggested_params, field_name, "_all_",
                    'auto', 'Suggested Clusters', prepend_to_rows
                    ))

                suggested_params = dict(api_params,
                                        concept_selector={"type": "suggested", "num_clusters": 7, "num_cluster_concepts": 4})
                volume_table.extend(_create_row_for_volume_subsets(
                    luminoso_data, suggested_params, field_name, "_all_",
                    'auto', 'Suggested Clusters', prepend_to_rows
                ))

                sentiment_params = dict(api_params, concept_selector={'type': 'sentiment_suggested'})
                volume_table.extend(_create_row_for_volume_subsets(
                    luminoso_data, sentiment_params, field_name, "_all_",
                    'auto', 'sentiment_suggested', prepend_to_rows
                ))

                unique_params = dict(api_params, concept_selector={'type': 'unique_to_filter'})
                volume_table.extend(_create_row_for_volume_subsets(
                    luminoso_data, unique_params, field_name, "_all_",
                    'auto', 'unique_to_filter', prepend_to_rows
                ))

            if len(volume_table) > WRITER_BATCH_SIZE:
                if lumi_writer:
                    lumi_writer.output_data(volume_table)
                volume_table = []

    # write any excess data
    if len(volume_table) > 0:
        if lumi_writer:
            lumi_writer.output_data(volume_table)
        volume_table = []

    return


def create_vot_table(lumi_writer, luminoso_data, date_field_info,
                     end_date, iterations,
                     range_type, subset_fields):
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
        range_type = luminoso_data.find_best_interval(iterations)
    range_descriptions = {'M': 'Month', 'W': 'Week', 'D': 'Day'}
    range_description = range_descriptions[range_type]

    print("vot starting. Date Field: {}, Iterations: {},"
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

        print(f"vot starting. Iteration: {range_description}-{count}, Date: {start_date_dt.isoformat()},")

        create_volume_subset_table(lumi_writer, luminoso_data, subset_fields,
                                                filter_list, prepend_to_rows, True)

        # move to the nextdate
        end_date_epoch = start_date_epoch
        end_date_dt = datetime.fromtimestamp(end_date_epoch)

    return 


def main():

    parser = argparse.ArgumentParser(
        description='Export volume data and write to a file'
    )
    parser.add_argument('project_url',
                        help="The complete URL of the Daylight project")
    parser.add_argument('--encoding', default='utf-8',
                        help="Encoding type of the files to write to")
    parser.add_argument('--volume_subset_fields', default=None,
                        help='Which subsets to include in volumes by'
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
    args = parser.parse_args()

    project_url = args.project_url.strip('/')
    api_url = project_url.split('/app')[0].strip() + '/api/v5'
    project_id = project_url.split('/')[6].strip()

    client = LuminosoClient.connect(
            url='%s/projects/%s' % (api_url.strip('/'), project_id),
            user_agent_suffix='se_code:volume'
        )
    luminoso_data = LuminosoData(client)

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

    if bool(args.vot):
        print("Calculating volume over time (vot)")

        if args.vot_date_field is None:
            date_field_info = luminoso_data.first_date_field
            if date_field_info is None:
                print("ERROR no date field in project for vot")
                return
        else:
            date_field_info = luminoso_data.get_field_by_name(
                args.vot_date_field
            )
            if date_field_info is None:
                print("ERROR: (vot) no date field name:"
                      " {}".format(args.vot_date_field))
                return

        vot_table = create_vot_table(
            luminoso_data, date_field_info, args.vot_end,
            int(args.vot_iterations), args.vot_range, args.volume_subset_fields
        )
        write_table_to_csv(vot_table, 'vot_table.csv',
                           encoding=args.encoding)

    print("Generating project volume data...")
    volume_table = create_volume_table(client, scl_match_counts,
                                       root_url=luminoso_data.root_url)
    write_table_to_csv(volume_table, 'volume.csv',
                       encoding=args.encoding)

    print("Generating volume by subsets...")
    lumi_writer = LumiCsvWriter('volume_subsets.csv', 'volume_subsets', project_id, args.encoding)
    create_volume_subset_table(
        lumi_writer, luminoso_data,
        args.volume_subset_fields)


if __name__ == '__main__':
    main()
