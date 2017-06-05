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
                                    limit=n)['search_results']
    else:
        search_results = client.get(unit + '/search', text=search_terms, negative=neg_terms,
                                    limit=n)['search_results']

    # Save results
    results = []
    for doc, matching_strength in search_results:
        results.append({'text':doc['document']['text'],
                  'doc_id':doc['document']['_id'],
                  'score':matching_strength})
    
    return results


def get_new_results(client, search_terms, neg_terms, unit, n, operation, hide_exact):
    """
    Given a list of search terms, return the n documents or terms (unit) that the new solution
    would return when supplied with these terms. It has an option to hide the documents
    containing exact matches of the search terms (hide_exact=True).
    """
    scores = defaultdict(lambda: len(search_terms + neg_terms) * [0])
    exact_matches = defaultdict(lambda: False)

    # Get matching scores for top term, document pairs
    for i, term in enumerate(search_terms + neg_terms):
        search_results = client.get(unit + '/search', text=term, limit=10000)['search_results']

        for result, matching_strength in search_results:
            doc_id = get_doc_id(result, unit)

            if i >= len(search_terms):
                scores[doc_id][i] = fuzzy_not(normalize_score(matching_strength, unit))
            else:
                scores[doc_id][i] = normalize_score(matching_strength, unit)
                if hide_exact and result['exact_indices']:
                    exact_matches[doc_id] = True

    # Compute combined scores
    final_scores = []
    for doc_id, doc_scores in scores.items():
        score = compute_score(doc_scores, operation, search_terms, neg_terms)
        final_scores.append((doc_id, score))
    final_scores = sorted(final_scores, key=lambda x: x[1] if not np.isnan(x[1]) else 0,
                          reverse=True)

    # Save results
    results = []
    for doc_id, score in final_scores[:n]:
        results.append({'text':get_result_to_display(client, doc_id, exact_matches, hide_exact, unit),
                  'doc_id':doc_id,
                  'score':score})
    
    return results


def get_result_to_display(client, doc_id, exact_matches, hide_exact, unit):
    to_display = ''
    if unit == 'docs':
        document = client.get('/docs', id=doc_id)
        if hide_exact:
            if not exact_matches[doc_id]:
                to_display = document['text']
        else:
            to_display = document['text']
    else:
        to_display = doc_id
    return to_display


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


def get_doc_id(result, unit):
    if unit == 'terms':
        return result['text']
    else:
        return result['document']['_id']

