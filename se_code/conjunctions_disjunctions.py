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

from se_code.fuzzy_logic import clamp, fuzzy_and, fuzzy_or, fuzzy_not, tanh_clamp


def connect(account_id, project_id):
    client = LuminosoClient.connect(
        'https://analytics.luminoso.com/api/v4/projects/{}/{}'.format(account_id, project_id))
    return client


def get_current_results(client, search_terms, neg_terms, zero, unit, n, hide_exact):
    """
    Given a list of search terms, return the n documents or terms (unit) that our current
    solution would return when supplied with these terms. It has an option to hide the documents
    containing exact matches of the search terms (hide_exact=True).
    """
    neg_terms = {'text': neg_term for neg_term in neg_terms}
    search_terms = ' '.join(search_terms)

    if zero:
        search_results = client.get(unit + '/search', text=search_terms, zero=neg_terms,
                                    limit=10000)['search_results']
    else:
        search_results = client.get(unit + '/search', text=search_terms, negative=neg_terms,
                                    limit=10000)['search_results']

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


def get_new_results(client, search_terms, neg_terms, unit, n, operation, hide_exact):
    """
    Given a list of search terms, return the n documents or terms (unit) that the new solution
    would return when supplied with these terms. It has an option to hide the documents
    containing exact matches of the search terms (hide_exact=True).
    """
    scores = defaultdict(lambda: len(search_terms + neg_terms) * [0])
    exact_matches = defaultdict(lambda: False)
    display_texts = {}

    # Get matching scores for top term, document pairs
    for i, term in enumerate(search_terms + neg_terms):
        search_results = client.get(unit + '/search', text=term, limit=10000)['search_results']

        for result, matching_strength in search_results:
            if unit == 'docs':
                _id = result['document']['_id']
                display_texts[_id] = result['document']['text']
            else:
                _id = result['text']
                display_texts[_id] = _id

            if i >= len(search_terms):
                scores[_id][i] = fuzzy_not(normalize_score(matching_strength, unit))
            else:
                scores[_id][i] = normalize_score(matching_strength, unit)
                if hide_exact and result['exact_indices']:
                    exact_matches[_id] = True

    # Compute combined scores
    final_scores = []
    for _id, doc_scores in scores.items():
        score = compute_score(doc_scores, operation, search_terms, neg_terms)
        final_scores.append((_id, score))
    final_scores = sorted(final_scores, key=lambda x: x[1] if not np.isnan(x[1]) else 0,
                          reverse=True)

    # Print scores
    display_count = 1
    for _id, score in final_scores:

        if display_count > int(n):
            break

        # to_display = get_result_to_display(client, doc_id, exact_matches, hide_exact, unit)
        if not (hide_exact and exact_matches[_id]):
            print('{}.\t{}\t{}'.format(display_count, round(score, 2), display_texts[_id]))
            print()
            display_count += 1


def compute_score(doc_scores, operation, search_terms, neg_terms):
    """
    Compute a score for a document. This score consists of scores for each search term,
    as well as the score for neg_terms. First, search term scores are combined using either
    fuzzy_and or fuzzy_or. If any neg_terms are supplied, they will be combined with the search
    terms score using fuzzy_and.
    """
    if operation == 'conjunction':
        score = fuzzy_and(doc_scores)
    else:
        score = fuzzy_or(doc_scores[:len(search_terms)])
        if neg_terms:
            score = fuzzy_and([score] + doc_scores[len(search_terms):])
    return score


def normalize_score(score, unit):
    """
    Normalize document scores using tanh_clamp. Normalize terms using clamp, since their max
    value is already 1.
    """
    if unit == 'terms':
        return clamp(score)
    else:
        return tanh_clamp(score)


@click.command()
@click.argument('account_id')
@click.argument('project_id')
@click.argument('search_terms', nargs=-1)
@click.option('--neg-terms', multiple=True, help='Specify terms that should be negated.')
@click.option('--zero/--negative', default=True, help='Negative means the opposite of the topic '
                                                      'and zero means irrelevant to the topic. '
                                                      'Default: zero.')
@click.option('--current/--new', default=False, help='Get current or new results. Default: new')
@click.option('--conjunction/--disjunction', default=True, help='Get conjunctions or disjunctions. '
                                                                'Default: conjunction')
@click.option('--docs', 'unit', flag_value='docs', default=True, help='Get documents')
@click.option('--terms', 'unit', flag_value='terms', help='Get terms')
@click.option('--n', default=10, help='Number of results to show. Default: 10.')
@click.option('--hide-exact', is_flag=True, help='Hide the documents including exact matches of '
                                                 'either one of the search terms. Default: False.')
def main(account_id, project_id, search_terms, neg_terms, zero, current, conjunction, unit, n,
         hide_exact):
    client = connect(account_id, project_id)

    if current and conjunction:
        get_current_results(client, search_terms, neg_terms, zero, unit, n, hide_exact)
    elif not current and conjunction:
        get_new_results(client, search_terms, neg_terms, unit, n, 'conjunction', hide_exact)
    elif current and not conjunction:
        get_current_results(client, search_terms, neg_terms, zero, unit, n, hide_exact)
    elif not current and not conjunction:
        get_new_results(client, search_terms, neg_terms, unit, n, 'disjunction',
                        hide_exact)


if __name__ == '__main__':
    main()
