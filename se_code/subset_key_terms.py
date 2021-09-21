from __future__ import division
from luminoso_api import V5LuminosoClient as LuminosoClient
import argparse
import csv


def subset_key_terms(client, subset_counts, terms_per_subset=10):
    """
    Find 'key terms' for a subset, those that appear disproportionately more
    inside a subset than outside of it.

    Parameters:

    - client: a LuminosoClient pointing to the appropriate project
    - terms_per_subset: how many key terms to find in each subset
    - scan_terms: how many relevant terms to consider from each subset
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
            scores = [
                (name, subset, concept, 0, 0) for concept in unique_to_filter
            ]
            results.extend(scores)

    return results


def create_skt_table(client, skt_tuples):
    '''
    Create tabulation of subset key terms analysis (terms distinctive within a subset)
    :param client: LuminosoClient object pointed to project path
    :param skt_tuples: List of subset key terms 5-tuples
    :return: List of subset key terms output with example documents & match counts
    '''

    print('Creating subset key terms table...')
    skt_table = []
    for name, subset, term, odds_ratio, pvalue in skt_tuples:
        docs = client.get('docs', limit=3, search={'texts': [term['name']]},
                          filter=[{'name': name, 'values': [subset]}])
        doc_texts = [doc['text'] for doc in docs['result']]
        text_length = len(doc_texts)
        text_1 = ''
        text_2 = ''
        text_3 = ''
        # excel has a max doc length of 32k
        if text_length == 1:
            text_1 = doc_texts[0]
        elif text_length == 2:
            text_1 = doc_texts[0]
            text_2 = doc_texts[1]
        elif text_length > 2:
            text_1 = doc_texts[0]
            text_2 = doc_texts[1]
            text_3 = doc_texts[2]
        skt_table.append(
            {'term': term['name'],
             'subset': name,
             'value': subset,
             'odds_ratio': odds_ratio,
             'p_value': pvalue,
             'exact_matches': term['exact_match_count'],
             'conceptual_matches': (term['match_count']
                                    - term['exact_match_count']),
             'Text 1': text_1[:32766],
             'Text 2': text_2[:32766],
             'Text 3': text_3[:32766],
             'total_matches': term['match_count']}
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


def get_all_docs(client):
    docs = []
    while True:
        new_docs = client.get('docs', limit=25000, offset=len(docs))
        if new_docs['result']:
            docs.extend(new_docs['result'])
        else:
            return docs


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
    docs = get_all_docs(client)
    subset_counts = {}
    for d in docs:
        for m in d['metadata']:
            if m['type'] != 'date' and m['type'] != 'number':
                if m['name'] not in subset_counts:
                    subset_counts[m['name']] = {}
                if m['value'] not in subset_counts[m['name']]:
                    subset_counts[m['name']][m['value']] = 0
                subset_counts[m['name']][m['value']] += 1
    print('Retrieving Subset Key Terms...')
    result = subset_key_terms(client, subset_counts,
                              terms_per_subset=int(args.skt_limit))
    table = create_skt_table(client, result)
    write_table_to_csv(table, 'skt_table.csv', encoding=args.encoding)


if __name__ == '__main__':
    main()
