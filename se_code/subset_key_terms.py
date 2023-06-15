import argparse
import csv

from luminoso_api import V5LuminosoClient as LuminosoClient


def subset_key_terms(client, subset_counts, terms_per_subset=10):
    """
    Find 'key terms' for a subset, those that appear disproportionately more
    inside a subset than outside of it.

    :param client: LuminosoClient object pointed to project path
    :param subset_counts: dict mapping each metadata field name to an iterable
        of the field's values
    :param terms_per_subset: number of terms to get for each field value
    :return: List of triples of the form (field_name, value, <concept_dict>),
        where <concept_dict> is the API's concept object from the match
        counts endpoint
    """
    results = []

    for name in sorted(subset_counts):
        for subset in sorted(subset_counts[name]):
            unique_to_filter = client.get(
                'concepts/match_counts',
                filter=[{'name': name, 'values': [subset]}],
                concept_selector={'type': 'unique_to_filter',
                                  'limit': terms_per_subset}
            )['match_counts']
            results.extend(
                [(name, subset, concept) for concept in unique_to_filter]
            )

    return results


def create_skt_table(client, skt_tuples):
    '''
    Create tabulation of subset key terms analysis (terms distinctive within a subset)
    :param client: LuminosoClient object pointed to project path
    :param skt_tuples: List of subset key terms triples
    :return: List of subset key terms output with example documents & match counts
    '''

    print('Creating subset key terms table...')
    skt_table = []
    for name, subset, concept in skt_tuples:
        docs = client.get('docs', limit=3, search={'texts': [concept['name']]},
                          filter=[{'name': name, 'values': [subset]}])
        # excel has a max doc length of 32k; pad the list with two additional
        # values, and then pull out the first three
        doc_texts = [doc['text'][:32766] for doc in docs['result']]
        text_1, text_2, text_3, *_ = (doc_texts + ['', ''])
        skt_table.append(
            {'term': concept['name'],
             'field_name': name,
             'field_value': subset,
             'exact_matches': concept['exact_match_count'],
             'conceptual_matches': (concept['match_count']
                                    - concept['exact_match_count']),
             'total_matches': concept['match_count'],
             'example_doc1': text_1,
             'example_doc2': text_2,
             'example_doc3': text_3}
        )
    return skt_table


def write_table_to_csv(table, filename, encoding='utf-8'):
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
        writer = csv.DictWriter(file, fieldnames=table[0].keys())
        writer.writeheader()
        writer.writerows(table)


def main():
    parser = argparse.ArgumentParser(
        description='Export Subset Key Terms and write to a file'
    )
    parser.add_argument('project_url',
                        help="The complete URL of the Daylight project")
    parser.add_argument('-skt', '--skt_limit', default=20,
                        help="The max number of subset key terms to display"
                             " per subset, default 20")
    parser.add_argument('-e', '--encoding', default='utf-8',
                        help="Encoding type of the file to write to")
    args = parser.parse_args()

    project_url = args.project_url.strip('/')
    api_url = project_url.split('/app')[0].strip() + '/api/v5'
    project_id = project_url.split('/')[-1].strip()
    client = LuminosoClient.connect(
        url='%s/projects/%s' % (api_url.strip('/ '), project_id),
        user_agent_suffix='se_code:subset_key_terms'
    )
    subset_counts = {}
    metadata = client.get('metadata')['result']
    for field in metadata:
        if field['type'] in ('date', 'number'):
            continue
        subset_counts[field['name']] = [v['value'] for v in field['values']]

    print('Retrieving Subset Key Terms...')
    result = subset_key_terms(client, subset_counts,
                              terms_per_subset=int(args.skt_limit))
    table = create_skt_table(client, result)
    write_table_to_csv(table, 'skt_table.csv', encoding=args.encoding)


if __name__ == '__main__':
    main()
