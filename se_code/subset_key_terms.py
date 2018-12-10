from __future__ import division
from luminoso_api import V5LuminosoClient as LuminosoClient
import numpy as np
from scipy.stats import fisher_exact
import argparse, csv, sys, getpass, json


def subset_key_terms(client, subset_counts, total_count, terms_per_subset=10, scan_terms=1000):
    """
    Find 'key terms' for a subset, those that appear disproportionately more
    inside a subset than outside of it. We determine this using Fisher's
    exact test, choosing the terms that have statistically significant
    differences with the largest odds ratio.

    Parameters:

    - client: a LuminosoClient pointing to the appropriate project
    - terms_per_subset: how many key terms to find in each subset
    - scan_terms: how many relevant terms to consider from each subset

    To avoid spurious results, we filter results by their statistical p-value.
    Ideally, a subset that is an arbitrary sample from the project as a whole
    will have no key terms.

    The cutoff p-value is adjusted according to `scan_terms`, to somewhat
    compensate for running so many statistical tests. The expected number of
    spurious results is .05 per subset.
    """
    pvalue_cutoff = 1 / scan_terms / 20
    results = []
    index = 0
    for name in sorted(subset_counts):
        for subset in sorted(subset_counts[name]):
            index += 1
            subset_terms = client.get('concepts/match_counts', 
                                      filter=[{'name':name,'values':[subset]}], 
                                      concept_selector={"type": "top",
                                                        "limit": scan_terms})['match_counts']
            length = 0
            termlist = []
            all_terms = []
            for term in subset_terms:
                if length + len(term['exact_term_ids'][0]) > 1000:
                    all_terms.extend(client.get('terms', term_ids=termlist))
                    termlist = []
                    length = 0
                termlist.append(term['exact_term_ids'][0])
                length += len(term['exact_term_ids'][0])
            if len(termlist) > 0:
                all_terms.extend(client.get('terms', term_ids=termlist))
            all_term_dict = {term['term_id']: term['total_doc_count'] for term in all_terms}

            subset_scores = []
            for term in subset_terms:
                term_in_subset = term['exact_match_count']
                term_outside_subset = all_term_dict[term['exact_term_ids'][0]] - term_in_subset + 1
                docs_in_subset = subset_counts[name][subset]
                docs_outside_subset = total_count - docs_in_subset + 1
                if term_in_subset < 0 or term_outside_subset < 0 or docs_in_subset < 0 or docs_outside_subset < 0:
                    print('term: %s' % term)
                    print('term in subset: %d' % term_in_subset)
                    print('term outside subset: %d' % term_outside_subset)
                    print('docs in subset: %d' % docs_in_subset)
                    print('docs outside subset: %d' % docs_outside_subset)
                table = np.array([
                    [term_in_subset, term_outside_subset],
                    [docs_in_subset, docs_outside_subset]
                ])
                odds_ratio, pvalue = fisher_exact(table, alternative='greater')
                if pvalue < pvalue_cutoff:
                    subset_scores.append((name, subset, term, odds_ratio, pvalue))

            if len(subset_scores) > 0:
                subset_scores.sort(key=lambda x: ('%s: %s' % (x[0], x[1]), -x[3]))
            results.extend(subset_scores[:terms_per_subset])

    return results

def create_skt_table(client, skt):
    '''
    Create tabulation of subset key terms analysis (terms distinctive within a subset)
    :param client: LuminosoClient object pointed to project path
    :param skt: List of subset key terms dictionaries
    :return: List of subset key terms output with example documents & match counts
    '''

    print('Creating subset key terms table...')
    skt_table = []
    for n, s, t, o, p in skt:   
        docs = client.get('docs', limit=3, search={'texts':[t['name']]}, filter=[{'name':n,'values':[s]}])
        doc_texts = [doc['text'] for doc in docs['result']]
        text_length = len(doc_texts)
        text_1 = ''
        text_2 = ''
        text_3 = ''
        if text_length == 1:
            text_1 = doc_texts[0]
        elif text_length == 2:
            text_1 = doc_texts[0]
            text_2 = doc_texts[1]
        elif text_length > 2:
            text_1 = doc_texts[0]
            text_2 = doc_texts[1]
            text_3 = doc_texts[2]
        skt_table.append({'term': t['name'],
                          'subset': n,
                          'value': s,
                          'odds_ratio': o,
                          'p_value': p,
                          'exact_matches': t['exact_match_count'],
                          'conceptual_matches': t['match_count'] - t['exact_match_count'],
                          'Text 1': text_1,
                          'Text 2': text_2,
                          'Text 3': text_3,
                          'total_matches': t['match_count']})
    return skt_table

def write_table_to_csv(table, filename):
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
    with open(filename, 'w') as file:
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
    parser.add_argument('project_url', help="The complete URL of the Analytics project")
    parser.add_argument('-skt', '--skt_limit', default=20, help="The max number of subset key terms to display per subset, default 20")
    parser.add_argument('-t', '--token', default=None, help="Authentication token for Daylight")
    args = parser.parse_args()
    
    project_url = args.project_url.strip('/')
    api_url = project_url.split('/app')[0].strip() + '/api/v5'
    project_id = project_url.split('/')[-1].strip()
    if args.token:
        client = LuminosoClient.connect(url='%s/projects/%s' % (api_url.strip('/ '), project_id), token=args.token)
    else:
        client = LuminosoClient.connect(url='%s/projects/%s' % (api_url.strip('/ '), project_id))
    docs = get_all_docs(client)
    subset_counts = {}
    for d in docs:
        for m in d['metadata']:
            if m['type'] != 'date':
                if m['name'] not in subset_counts:
                    subset_counts[m['name']] = {}
                if m['value'] not in subset_counts[m['name']]:
                    subset_counts[m['name']][m['value']] = 0
                subset_counts[m['name']][m['value']] += 1
    total_count = len(docs)
    print('Retrieving Subset Key Terms...')
    skt = subset_key_terms(client, subset_counts, total_count, terms_per_subset=int(args.skt_limit))
    table = create_skt_table(client, skt)
    write_table_to_csv(table, 'skt_table.csv')
    
if __name__ == '__main__':
    main()