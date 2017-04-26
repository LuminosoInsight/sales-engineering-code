"""
Example use: to get the document search results for new disjunction of the terms: 'highly
recommend' and 'not disappointed' in the Central Foods project, and hide the documents with the
exact matches of either term, run:

python conjunctions_disjunctions.py zoo vfzct 'highly recommend' 'not disappointed' --new
--disjunction --docs --hide-exact
"""

from collections import defaultdict

import click
import numpy as np

from luminoso_api import LuminosoClient

from se_code import fuzzy_logic


def connect(account_id, project_id):
    client = LuminosoClient.connect(
        'https://analytics.luminoso.com/api/v4/projects/{}/{}'.format(account_id, project_id))
    return client


def get_current_results(client, search_terms, unit, n, hide_exact):
    """
    Given a list of search terms, return the n documents or terms (unit) that our current
    solution would return when supplied with these terms. It has an option to hide the documents
    containing exact matches of the search terms (hide_exact=True).
    """
    search_terms = ' '.join(search_terms)
    search_results = client.get(unit + '/search', text=search_terms, limit=10000)['search_results']
    display_count = 1
    for result, matching_strength in search_results:

        if display_count > int(n):
            break

        to_display = ''
        if unit == 'docs':
            if hide_exact:
                if not result['exact_indices']:
                    to_display = result['document']['text']
            else:
                to_display = result['document']['text']
        else:  # unit == 'terms'
            to_display = result['text']

        if to_display:
            print('{}.\t{}\t{}'.format(display_count, round(matching_strength, 2), to_display))
            print()
            display_count += 1


def get_new_results(client, terms, unit, n, function, hide_exact):
    """
    Given a list of search terms, return the n documents or terms (unit) that the new solution
    would return when supplied with these terms. It has an option to hide the documents
    containing exact matches of the search terms (hide_exact=True).

    Depending on which function is supplied, get_new_results() will return either a conjunction
    or a disjunction. Functions resulting in a conjunction are ex. fuzzy_logic.fuzzy_and or
    fuzzy_logic.min_and. Functions resulting in a disjunction are ex. fuzzy_logic.fuzzy_or and
    fuzzy_logic.max_and.
    """
    scores = defaultdict(lambda: len(terms) * [0])

    # Get matching scores for top term, document pairs
    for i, term in enumerate(terms):
        search_results = client.get(unit + '/search', text=term, limit=10000)['search_results']
        for result, matching_strength in search_results:
            if unit == 'docs':
                doc_id = result['document']['_id']
            else:
                doc_id = result['text']
            scores[doc_id][i] = matching_strength

    # Compute combined scores
    final_scores = []
    for doc_id, doc_scores in scores.items():
        score = function(doc_scores)
        final_scores.append((doc_id, score))

    final_scores = sorted(final_scores, key=lambda x: x[1] if not np.isnan(x[1]) else 0,
                          reverse=True)

    # Print scores
    display_count = 1
    for doc_id, score in final_scores:

        if display_count > int(n):
            break

        to_display = ''
        if unit == 'docs':
            document = client.get('/docs', id=doc_id)
            if hide_exact:
                doc_terms = [triple[0] for triple in document['terms']]
                if not any(term in doc_terms for term in terms):
                    to_display = document['text']
            else:
                to_display = document['text']
        else:
            to_display = doc_id

        if to_display:
            print('{}.\t{}\t{}'.format(display_count, round(score, 2), to_display))
            print()
            display_count += 1


@click.command()
@click.argument('account_id')
@click.argument('project_id')
@click.argument('search_terms', nargs=-1)
@click.option('--current/--new', default=False, help='Get current or new results. Default: new')
@click.option('--conjunction/--disjunction', default=True, help='Get conjunctions or disjunctions. '
                                                                'Default: conjunction')
@click.option('--docs', 'unit', flag_value='docs', default=True, help='Get documents')
@click.option('--terms', 'unit', flag_value='terms', help='Get terms')
@click.option('--n', default=10, help='Number of results to show. Default: 10.')
@click.option('--hide-exact', is_flag=True, help='Hide the documents including exact matches of '
                                                 'either one of the search terms. Default: False.')
def main(account_id, project_id, search_terms, current, conjunction, unit, n, hide_exact):
    client = connect(account_id, project_id)

    if current and conjunction:
        get_current_results(client, search_terms, unit, n, hide_exact)
    elif not current and conjunction:
        get_new_results(client, search_terms, unit, n, fuzzy_logic.fuzzy_and, hide_exact)
    elif current and not conjunction:
        get_current_results(client, search_terms, unit, n, hide_exact)
    elif not current and not conjunction:
        get_new_results(client, search_terms, unit, n, fuzzy_logic.fuzzy_or, hide_exact)

if __name__ == '__main__':
    main()
