from __future__ import division
from luminoso_api import LuminosoClient
import numpy as np
from scipy.stats import fisher_exact
import argparse, csv, sys, getpass
import concurrent.futures


def subset_key_terms(client, api_url, account, project, terms_per_subset=10, scan_terms=1000):
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
    subset_counts = client.get()['counts']
    pvalue_cutoff = 1 / scan_terms / 20
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:

        futures = {executor.submit(skt, subset, scan_terms, subset_counts, pvalue_cutoff, api_url, account, project): subset for subset in sorted(subset_counts)}
        for future in concurrent.futures.as_completed(futures):
            subset_scores = future.result()

            results.extend(subset_scores[:terms_per_subset])

    return results


def skt(subset, scan_terms, subset_counts, pvalue_cutoff, api_url, account, project):
    client = LuminosoClient.connect(url='{}/projects/{}/{}'.format(api_url, account, project))
    subset_terms = client.get('terms', subset=subset, limit=scan_terms)
    length = 0
    termlist = []
    all_terms = []
    for term in subset_terms:
        if length + len(term['term']) > 100:
            all_terms.extend(client.get('terms', terms=termlist))
            termlist = []
            length = 0
        termlist.append(term['term'])
        length += len(term['term'])
    if len(termlist) > 0:
        all_terms.extend(client.get('terms', terms=termlist))
    all_term_dict = {term['term']: term['distinct_doc_count'] for term in all_terms}

    subset_scores = []
    for term in subset_terms:
        term_in_subset = term['distinct_doc_count']
        term_outside_subset = all_term_dict[term['term']] - term_in_subset + 1
        docs_in_subset = subset_counts[subset]
        docs_outside_subset = subset_counts['__all__'] - subset_counts[subset] + 1
        table = np.array([
            [term_in_subset, term_outside_subset],
            [docs_in_subset, docs_outside_subset]
        ])
        odds_ratio, pvalue = fisher_exact(table, alternative='greater')
        if pvalue < pvalue_cutoff:
            subset_scores.append((subset, term, odds_ratio, pvalue))

    if len(subset_scores) > 0:
        subset_scores.sort(key=lambda x: (x[0], -x[2]))

    return subset_scores

def create_skt_table(client, skt):
    '''
    Create tabulation of subset key terms analysis (terms distinctive within a subset)
    :param client: LuminosoClient object pointed to project path
    :param skt: List of subset key terms dictionaries
    :return: List of subset key terms output with example documents & match counts
    '''

    print('Creating subset key terms table...')
    terms = []
    for s, t, o, p in skt:
        terms.extend(client.get('terms/doc_counts', terms=[t['term']], subsets=[s], format='json'))
    
    terms = {t['text']: t for t in terms}
    skt_table = []
    index = 0
    for s, t, o, p in skt:   
        docs = client.get('docs/search', limit=3, text=t['text'], subset=s)
        doc_texts = [ids[0]['document']['text'] for ids in docs['search_results']]
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
        skt_table.append({'term': t['text'],
                          'subset': s.partition(':')[0],
                          'value': s.partition(':')[2],
                          'odds_ratio': o,
                          'p_value': p,
                          'exact_matches': terms[t['text']]['num_exact_matches'],
                          'conceptual_matches': terms[t['text']]['num_related_matches'],
                          'Text 1': text_1,
                          'Text 2': text_2,
                          'Text 3': text_3,
                          'total_matches': terms[t['text']]['num_exact_matches'] + terms[t['text']]['num_related_matches']})
        index += 1
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

def wait_for_jobs(client, text):
    '''
    Repeatedly test for project recalculation, display text when complete
    :param client: LuminosoClient object pointed to project path
    :param text: String to print when complete
    :return: None
    '''

    time_waiting = 0
    while len(client.get()['running_jobs']) != 0:
        sys.stderr.write('\r\tWaiting for {} ({}sec)'.format(text, time_waiting))
        time.sleep(30)
        time_waiting += 30
        
        
def main():
    parser = argparse.ArgumentParser(
        description='Export Subset Key Terms and write to a file'
    )
    parser.add_argument('project_url', help="The complete URL of the Analytics project")
    parser.add_argument('-skt', '--skt_limit', default=20, help="The max number of subset key terms to display per subset, default 20")
    args = parser.parse_args()
    
    project_url = args.project_url.strip('/')
    api_url = project_url.split('/app')[0].strip() + '/api/v4'
    account_id = project_url.split('/')[-2].strip()
    project_id = project_url.split('/')[-1].strip()
    
    count = 0
    while count < 3:
        username = input('Username: ')
        password = getpass.getpass()
        try:
            client = LuminosoClient.connect(url='%s/projects/%s/%s' % (api_url.strip('/'), account_id, project_id), username=username, password=password)
            break
        except:
            print('Incorrect credentials, please re-enter username and password')
            count += 1
            continue
    if count >= 3:
        print('Username and password incorrect')
            
    print('Retrieving Subset Key Terms...')
    skt = subset_key_terms(client, terms_per_subset=int(args.skt_limit))
    table = create_skt_table(client, skt)
    write_table_to_csv(table, 'skt_table.csv')
    
if __name__ == '__main__':
    main()