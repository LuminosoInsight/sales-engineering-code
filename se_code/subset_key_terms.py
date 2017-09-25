from __future__ import division
from luminoso_api import LuminosoClient
import numpy as np
from scipy.stats import fisher_exact
import argparse


def subset_key_terms(client, terms_per_subset=10, scan_terms=1000):
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

    for subset in sorted(subset_counts):
        subset_terms = client.get('terms', subset=subset, limit=scan_terms)
        termlist = [term['term'] for term in subset_terms]
        all_terms = client.get('terms', terms=termlist)
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
                subset_scores.append((subset, term['text'], odds_ratio, pvalue))

        subset_scores.sort(key=lambda x: (x[0], -x[2], x[1]))
        results.extend(subset_scores[:terms_per_subset])

    return results


def run(account_id, project_id, username, terms_per_subset, scan_terms,
        api_url='https://analytics.luminoso.com/api/v4'):
    """
    Find key terms in all the subsets of a project, and print the results
    to standard output in tab-separated form.
    """
    client = LuminosoClient.connect(
        '%s/projects/%s/%s/' % (api_url, account_id, project_id),
        username=username
    )
    key_terms = subset_key_terms(client, terms_per_subset=terms_per_subset,
                                 scan_terms)

    print('Subset\tText\tOdds ratio\tUncorrected p-value')
    for subset, text, fisher, pvalue in key_terms:
        print('%s\t%s\t%6.5g\t%6.5g' % (subset, text, fisher, pvalue))



def main():
    parser = argparse.ArgumentParser(
        description='Automatically find representative topics for a Luminoso project.'
    )
    parser.add_argument('account_id', help="The ID of the account that owns the project, such as 'demo'")
    parser.add_argument('project_id', help="The ID of the project to analyze, such as '2jsnm'")
    parser.add_argument('username', help="A Luminoso username with access to the project")
    parser.add_argument('-scan', '--scan_terms', type=int, default=1000, help="Get number of top terms")
    parser.add_argument('-t', '--terms-per-subset', type=int, default=10, help="Maximum key terms to find in each subset")
    parser.add_argument('-a', '--api-url', default='https://analytics.luminoso.com/api/v4', help="The base URL for the Luminoso API (defaults to the production API, https://analytics.luminoso.com/api/v4)")
    args = parser.parse_args()
    run(args.account_id, args.project_id, args.username,
        args.terms_per_subset, args.scan_terms, args.api_url)


if __name__ == '__main__':
    main()
