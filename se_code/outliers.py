import argparse
import csv
from datetime import datetime, timedelta
import itertools 
import urllib.parse

from luminoso_api import V5LuminosoClient as LuminosoClient
from se_code.score_drivers import (
     LuminosoData, write_table_to_csv
)

def create_outlier_table(client, proj_info, scl_match_counts, match_type, root_url=''):

    outliers = []
    for scl_name, cl in scl_match_counts.items():

        outlier_cs = {"type": "concept_list", "concept_list_id": cl['concept_list_id']}

        setup_outlier_both_results = client.post("concepts/outliers/", 
                                                 concept_selector=outlier_cs, 
                                                 match_type=match_type)

        coverage_pct = 100-((setup_outlier_both_results['filter_count'] / proj_info['document_count']) * 100)

        # get the list of outlier concepts
        outlier_filter = [{"special": "outliers"}]
        unique_cs = {"type": "unique_to_filter", "limit": 20}
        outlier_concepts = client.get("concepts/match_counts", concept_selector=unique_cs, filter=outlier_filter)

        outliers.extend([
            {'list_type': 'shared_concept_list',
             'list_name': scl_name,
             'concept': concept["name"],
             'relevance': concept['relevance'],
             'texts': concept['texts'],
             'coverage': coverage_pct,
             'match_type': match_type,
             'match_count': concept['match_count'],
             'exact_match_count': concept['exact_match_count'],
             'conceptual_match_count': concept['match_count'] - concept['exact_match_count']}
            for concept in outlier_concepts['match_counts']
        ])

    # add three sample documents to each row
    for srow in outliers:
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

    return outliers


def _create_row_for_outlier_subsets(luminoso_data, api_params, proj_info, subset_name, subset_value, list_type, list_name, prepend_to_rows=None):
    """
    Helper function for create_outlier_subset_table().
    """
    rows = []

    outliers = luminoso_data.client.get(
        'concepts/match_counts', **api_params
    )

    for c in outliers['match_counts']:

        row = [
            {'list_type': list_type,
             'list_name': list_name,
             'field_name': subset_name,
             'field_value': subset_value,
             'concept': concept["name"],
             'relevance': concept['relevance'],
             'texts': concept['texts'],
             'coverage': coverage_pct,
             'match_type': match_type,
             'match_count': concept['match_count'],
             'exact_match_count': concept['exact_match_count'],
             'conceptual_match_count': concept['match_count'] - concept['exact_match_count']}
            for concept in outliers['match_counts']
        ]

        if prepend_to_rows:
            row = {**prepend_to_rows, **row}
        rows.append(row)
    return rows


def create_outlier_subset_table(luminoso_data, proj_info, scl_match_counts, match_type, subset_fields=None, filter_list=None, prepend_to_rows=None):
    '''
    Create tabulation of outlier output
    :param luminoso_data: a LuminosoData object
    :param filter_list: document filter (as a list of dicts)
    :return: List of outliers
    '''

    # if the user specifies the list of subsets to process
    if not subset_fields:
        subset_fields = luminoso_data.get_best_subset_fields()
    else:
        subset_fields = subset_fields.split(",")

    # process outliers by subset
    outliers_table = []

    orig_filter_list = filter_list

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
                    print("outlier filter={}".format(filter_list))

                    for scl_name, cl in scl_match_counts.items():

                        outlier_cs = {"type": "concept_list", "concept_list_id": cl['concept_list_id']}

                        setup_outlier_both_results = luminoso_data.client.post("concepts/outliers/", 
                                                                concept_selector=outlier_cs, 
                                                                match_type=match_type)

                        coverage_pct = 100-((setup_outlier_both_results['filter_count'] / proj_info['document_count']) * 100)

                        # get the list of outlier concepts
                        outlier_filter = [{"special": "outliers"}]
                        outlier_filter.extend(filter_list)
                        unique_cs = {"type": "unique_to_filter", "limit": 20}
                        outlier_concepts = luminoso_data.client.get("concepts/match_counts", concept_selector=unique_cs, filter=outlier_filter)

                        for concept in outlier_concepts['match_counts']:
                            row = {'list_type': 'shared_concept_list',
                                   'list_name': scl_name,
                                   'field_name': field_name,
                                   'field_value': field_value,
                                   'concept': concept["name"],
                                   'relevance': concept['relevance'],
                                   'texts': concept['texts'],
                                   'coverage': coverage_pct,
                                   'match_type': match_type,
                                   'match_count': concept['match_count'],
                                   'exact_match_count': concept['exact_match_count'],
                                   'conceptual_match_count': concept['match_count'] - concept['exact_match_count']}
                            if prepend_to_rows:
                                row = {**prepend_to_rows, **row}                            
                            outliers_table.append(row)

    return outliers_table


def create_outliersot_table(luminoso_data, proj_info, scl_match_counts, match_type,
                            date_field_info, end_date, iterations,
                            range_type, subset_fields):
    outliersot_data_raw = []

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

    print("Outliersot starting. Date Field: {}, Iterations: {},"
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

        od_data = create_outlier_subset_table(luminoso_data, proj_info, scl_match_counts, 
                                              match_type, subset_fields,
                                              filter_list, prepend_to_rows)
        outliersot_data_raw.extend(od_data)

        # move to the nextdate
        end_date_epoch = start_date_epoch
        end_date_dt = datetime.fromtimestamp(end_date_epoch)

    return outliersot_data_raw


def main():

    parser = argparse.ArgumentParser(
        description='Export outliers and write to a file'
    )
    parser.add_argument('project_url',
                        help="The complete URL of the Daylight project")
    parser.add_argument('--encoding', default='utf-8',
                        help="Encoding type of the files to write to")
    parser.add_argument('--outlier_subset_fields', default=None,
                        help='Which subsets to include in outliers by'
                             ' subset. Default = All with < 200 unique values.'
                             ' Samp: "field1,field2"')
    parser.add_argument('--outliersot', action='store_true', default=False,
                        help="Calculate outliers over time (OUTLIERSOT)")
    parser.add_argument('--outliersot_end', default=None,
                        help="Last date to calculate outliersot MM/DD/YYYY -"
                             " algorithm works moving backwards in time.")
    parser.add_argument('--outliersot_iterations', default=7,
                        help="Number of sentiment over time samples")
    parser.add_argument('--outliersot_range', default=None,
                        help="Size of each sample: M,W,D. If none given, range"
                             " type will be calculated for best fit")
    parser.add_argument('--outliersot_date_field', default=None,
                        help="The name of the date field for outliersot. If none, the first"
                             " date field will be used")
    args = parser.parse_args()

    project_url = args.project_url.strip('/')
    api_url = project_url.split('/app')[0].strip() + '/api/v5'
    project_id = project_url.split('/')[6].strip()

    client = LuminosoClient.connect(
            url='%s/projects/%s' % (api_url.strip('/'), project_id),
            user_agent_suffix='se_code:outliers'
        )
    luminoso_data = LuminosoData(client)
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

    print("Generating project outliers...")
    outlier_table = create_outlier_table(client, proj_info, scl_match_counts,
                                         "both", root_url=luminoso_data.root_url)
    outlier_table.extend(create_outlier_table(client, proj_info, scl_match_counts,
                                              "exact", root_url=luminoso_data.root_url))
    write_table_to_csv(outlier_table, 'outliers.csv',
                       encoding=args.encoding)

    print("Generating outliers by subsets...")
    outlier_subset_table = create_outlier_subset_table(
        luminoso_data,
        proj_info, 
        scl_match_counts, 
        "both",
        args.outlier_subset_fields)
    outlier_subset_table.extend(create_outlier_subset_table(
        luminoso_data,
        proj_info, 
        scl_match_counts, 
        "exact",
        args.outlier_subset_fields))
    write_table_to_csv(outlier_subset_table, 'outlier_subsets.csv',
                       encoding=args.encoding)


    if bool(args.outliersot):
        print("Calculating outliers over time (outliersot)")

        if args.outliersot_date_field is None:
            date_field_info = luminoso_data.first_date_field
            if date_field_info is None:
                print("ERROR no date field in project for outliersot")
                return
        else:
            date_field_info = luminoso_data.get_field_by_name(
                args.sot_date_field
            )
            if date_field_info is None:
                print("ERROR: (outliersot) no date field name:"
                      " {}".format(args.sot_date_field))
                return

        outliersot_table = create_outliersot_table(
            luminoso_data, proj_info, scl_match_counts, "both",
            date_field_info, args.outliersot_end,
            int(args.outliersot_iterations), args.outliersot_range, 
            args.outlier_subset_fields
        )
        outliersot_table.extend(create_outliersot_table(
            luminoso_data, proj_info, scl_match_counts, "exact",
            date_field_info, args.outliersot_end,
            int(args.outliersot_iterations), args.outliersot_range, 
            args.outlier_subset_fields
        ))
        write_table_to_csv(outliersot_table, 'outliersot_table.csv',
                           encoding=args.encoding)

if __name__ == '__main__':
    main()
