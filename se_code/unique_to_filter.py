import argparse
from datetime import datetime, timedelta

from luminoso_api import V5LuminosoClient as LuminosoClient
from se_code.data_writer import (LumiCsvWriter)
from se_code.score_drivers import (
     parse_url, LuminosoData, write_table_to_csv
)

WRITER_BATCH_SIZE = 5000


def unique_to_filter(client, subset_values_dict, terms_per_subset=10, filter_list=None):
    """
    Find 'unique terms' for a subset, those that appear disproportionately more
    inside a subset than outside of it.

    :param client: Luminoso api client
    :param subset_values_dict: dict mapping each metadata field name to an iterable
        of the field's values
    :param terms_per_subset: number of terms to get for each field value
    :return: List of triples of the form (field_name, value, <concept_dict>),
        where <concept_dict> is the API's concept object from the match
        counts endpoint
    """
    results = []

    for name in sorted(subset_values_dict):
        # make sure all the field values are <64 characters
        # they need to fit on a url and some projects use
        # full documents as metadata fields so just exclude
        # those.
        l64 = [s for s in subset_values_dict[name] if ((isinstance(s, str)) and len(s)>64)]
        if len(l64) == 0:
            for subset in sorted(subset_values_dict[name]):
                combined_filter = [{'name': name, 'values': [subset]}]
                if filter_list:
                    combined_filter.extend(filter_list)
                unique_to_filter = client.get(
                    'concepts/match_counts',
                    filter=combined_filter,
                    concept_selector={'type': 'unique_to_filter',
                                      'limit': terms_per_subset}
                )['match_counts']
                results.extend(
                    [(name, subset, concept) for concept in unique_to_filter]
                )

    return results


def create_u2f_table(client, u2f_tuples, prepend_to_rows=None):
    '''
    Create tabulation of unique to filter terms analysis (terms distinctive within a subset)
    :param client: LuminosoClient object pointed to project path
    :param u2f_tuples: List of unique to filter terms triples
    :return: List of unique to filter output with example documents & match counts
    '''

    u2f_table = []
    for name, subset, concept in u2f_tuples:
        docs = client.get('docs', limit=3, search={'texts': [concept['name']]},
                                        filter=[{'name': name, 'values': [subset]}])
        # excel has a max doc length of 32k; pad the list with two additional
        # values, and then pull out the first three
        doc_texts = [doc['text'][:32766] for doc in docs['result']]
        if len(doc_texts)>0:
            text_1, text_2, text_3, *_ = (doc_texts + ['', ''])
        else:
            text_1, text_2, text_3 = (['','',''])
        row = {'term': concept['name'],
               'field_name': name,
               'field_value': subset,
               'exact_matches': concept['exact_match_count'],
               'conceptual_matches': (concept['match_count']
                                      - concept['exact_match_count']),
               'total_matches': concept['match_count'],
               'example_doc1': text_1,
               'example_doc2': text_2,
               'example_doc3': text_3}
        if prepend_to_rows:
            row = {**prepend_to_rows, **row}      
        u2f_table.append(row)

    return u2f_table


def create_u2fot_table(lumi_writer, luminoso_data, date_field_info, end_date, iterations,
                       range_type, subset_values_dict, terms_per_subset):
    u2fot_data_raw = []

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

    print("u2fot starting. Date Field: {}, Iterations: {},"
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

        print(f"u2fot starting. Iteration: {range_description}-{count}, Date: {start_date_dt.isoformat()},")

        result = unique_to_filter(luminoso_data.client, subset_values_dict,
                                  terms_per_subset=terms_per_subset, 
                                  filter_list=filter_list)
        u2f_data = create_u2f_table(luminoso_data.client, result,
                                    prepend_to_rows=prepend_to_rows)

        u2fot_data_raw.extend(u2f_data)

        if len(u2fot_data_raw) > WRITER_BATCH_SIZE:
            if lumi_writer:
                lumi_writer.output_data(u2fot_data_raw)
            u2fot_data_raw = []

        # move to the nextdate
        end_date_epoch = start_date_epoch
        end_date_dt = datetime.fromtimestamp(end_date_epoch)

    if len(u2fot_data_raw) > 0:
        if lumi_writer:
            lumi_writer.output_data(u2fot_data_raw)
        u2fot_data_raw = []
    return


def main():
    parser = argparse.ArgumentParser(
        description='Export unique to filter and write to a file'
    )
    parser.add_argument('project_url',
                        help="The complete URL of the Daylight project")
    parser.add_argument('-u2f', '--u2f_limit', default=20,
                        help="The max number of unique terms to display"
                             " per subset, default 20")
    parser.add_argument('-e', '--encoding', default='utf-8',
                        help="Encoding type of the file to write to")
    parser.add_argument('--ot', action='store_true', default=False,
                        help="Calculate u2f over time (U2FOT)")
    parser.add_argument('--ot_end', default=None,
                        help="Last date to calculate ot MM/DD/YYYY -"
                             " algorithm works moving backwards in time.")
    parser.add_argument('--ot_iterations', default=7,
                        help="Number of sentiment over time samples")
    parser.add_argument('--ot_range', default=None,
                        help="Size of each sample: M,W,D. If none given, range"
                             " type will be calculated for best fit")
    parser.add_argument('--ot_date_field', default=None,
                        help="The name of the date field for ot. If none, the first"
                             " date field will be used")
    args = parser.parse_args()

    root_url, api_url, workspace, project_id = parse_url(args.project_url)

    client = LuminosoClient.connect(
        url='%s/projects/%s' % (api_url.strip('/ '), project_id),
        user_agent_suffix='se_code:unique_to_filter'
    )

    luminoso_data = LuminosoData(client)

    subset_values_dict = {}
    metadata = client.get('metadata')['result']
    for field in metadata:
        if field['type'] in ('date', 'number'):
            continue
        subset_values_dict[field['name']] = [v['value'] for v in field['values']]

    print('Retrieving Unique Terms...')
    result = unique_to_filter(client, subset_values_dict,
                              terms_per_subset=int(args.u2f_limit))
    table = create_u2f_table(client, result)
    write_table_to_csv(table, 'unique.csv', encoding=args.encoding)

    if bool(args.ot):
        print("Calculating unique terms over time (u2fot)")

        if args.ot_date_field is None:
            date_field_info = luminoso_data.first_date_field
            if date_field_info is None:
                print("ERROR no date field in project for u2fot")
                return
        else:
            date_field_info = luminoso_data.get_field_by_name(
                args.ot_date_field
            )
            if date_field_info is None:
                print("ERROR: (u2fot) no date field name:"
                      " {}".format(args.ot_date_field))
                return

        lumi_writer = LumiCsvWriter('unique_over_time.csv', 'unique_over_time', project_id, args.encoding)

        create_u2fot_table(
            lumi_writer, luminoso_data, date_field_info, args.ot_end,
            int(args.ot_iterations), args.ot_range, subset_values_dict,
            int(args.u2f_limit)
        )

if __name__ == '__main__':
    main()
