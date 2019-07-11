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
import concurrent.futures

from luminoso_api import V5LuminosoClient as LuminosoClient

from se_code.fuzzy_logic import clamp, fuzzy_and, fuzzy_or, fuzzy_not, tanh_clamp


def connect(project_url):
    api_root = project_url.split('/app')[0]
    project_id = project_url.strip('/ ').split('/')[-1]
    client = LuminosoClient.connect(
        '{}/api/v5/projects/{}'.format(api_root, project_id))
    return client


def get_current_results(client, search_terms, unit, n):
    """
    Given a list of search terms, return the n documents or terms (unit) that our current
    solution would return when supplied with these terms.
    """
    search_terms = ' '.join(search_terms)
    
    if unit == 'docs':
        search_results = client.get(unit, search={'texts':search_terms}, limit=10000)['result']
    else:
        search_results = search_results = client.get('concepts', concept_selector={'type':'related', 
                                                             'search_concept':{'texts':search_terms},
                                                             'limit':10000})['result']
    start_idx = 0

    # Save results
    results = []
    for result in search_results[start_idx:start_idx+n]:
        if unit == 'docs':
            results.append({'text': result['text'],
                            'doc_id': result['doc_id'],
                            'score': result['match_score']})
        else:
            results.append({'text': result['texts'][0],
                            'score': result['match_score']})

    return results


def get_new_results(client, search_terms, unit, n, operation):
    """
    Given a list of search terms, return the n documents or terms (unit) that the new solution
    would return when supplied with these terms.
    """
    scores = defaultdict(lambda: len(search_terms) * [0])
    display_texts = {}
    if unit == 'docs': 
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(search_docs, client, unit, term, i)
                       for i, term in enumerate(search_terms)]

            for future in concurrent.futures.as_completed(futures):
                (i, search_results) = future.result()
                
                for result in search_results:
                    _id = result['doc_id']
                    display_texts[_id] = result['text']
                    matching_strength = result['match_score']
                    if i >= len(search_terms):
                        scores[_id][i] = fuzzy_not(normalize_score(matching_strength, unit))
                    else:
                        scores[_id][i] = normalize_score(matching_strength, unit)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(search_concepts, client, term, i)
                       for i, term in enumerate(search_terms)]

            for future in concurrent.futures.as_completed(futures):
                (i, search_results) = future.result()
                
                for result in search_results:
                    _id = result['doc_id']
                    display_texts[_id] = result['text']
                    matching_strength = result['match_score']
                    if i >= len(search_terms):
                        scores[_id][i] = fuzzy_not(normalize_score(matching_strength, unit))
                    else:
                        scores[_id][i] = normalize_score(matching_strength, unit)

    # Compute combined scores
    final_scores = []
    for _id, doc_scores in scores.items():
         if len(doc_scores) == len(search_terms):
            score = compute_score(doc_scores, operation, search_terms, neg_terms)
            final_scores.append((_id, score))
    final_scores = sorted(final_scores, key=lambda x: x[1] if not np.isnan(x[1]) else 0,
                          reverse=True)

    # Save results
    results = []
    for _id, score in final_scores[:n]:
        results.append({'text': display_texts[_id],
                        '_id': _id,
                        'score': score})

    return results


def search_docs(client, unit, term, i):
    return i, client.get(unit, search={'texts':[term]}, limit=10000)['result']
    #return i, client.get(unit + '/search', text=term, limit=10000)['search_results']
    
def search_concepts(client, term, i):
    return i, client.get('concepts', concept_selector={'type': 'related',
                                                       'search_concept': {'texts':[term]},
                                                       'limit':10000})['result']


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
@click.argument('project_url')
@click.argument('search_terms', nargs=-1)
@click.option('--current/--new', default=False, help='Get current or new results. Default: new')
@click.option('--conjunction/--disjunction', default=True, help='Get conjunctions or disjunctions. '
                                                                'Default: conjunction')
@click.option('--docs', 'unit', flag_value='docs', default=True, help='Get documents')
@click.option('--concepts', 'unit', flag_value='concepts', help='Get concepts')
@click.option('--n', default=10, help='Number of results to show. Default: 10.')
def main(project_url, search_terms, current, conjunction, unit, n):
    client = connect(project_url)

    if current and conjunction:
        results = get_current_results(client, search_terms, unit, n)
    elif not current and conjunction:
        results = get_new_results(client, search_terms, unit, n, 'conjunction')
    elif current and not conjunction:
        results = get_current_results(client, search_terms, unit, n)
    elif not current and not conjunction:
        results = get_new_results(client, search_terms, unit, n, 'disjunction')

    display_count = 1
    for result in results:
        print('{}.\t{}\t{}'.format(display_count, round(result['score'], 2), result['text']))
        print()
        display_count += 1

if __name__ == '__main__':
    main()
