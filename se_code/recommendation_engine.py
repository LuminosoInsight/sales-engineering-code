from __future__ import division
from pack64 import unpack64
from scipy.stats import fisher_exact
from scipy.optimize import basinhopping

from luminoso_api import LuminosoClient

import numpy as np
import json
import csv
import pickle
import math
import time
from scipy.optimize._differentialevolution import differential_evolution


def get_all_docs(client):
    '''
    Gets all documents from a given Analytics project
    '''
    docs = []
    while True:
        newdocs = client.get('docs', limit=25000, offset=len(docs))
        if newdocs:
            docs.extend(newdocs)
        else:
            return docs


def get_subset_term_info(client, subset_input, term_count=500):
    '''
    Return a dictionary for each subset containing the top X terms &
    the number of documents within the given subset.
    '''
    subset_term_info = {}
    subset_stats = client.get('subsets/stats')
    for s in subset_stats:
        if s['subset'] != '__all__':
            subset_name = s['subset'].split(':')[0].strip()
            if subset_input == subset_name:
                terms = client.get('terms',
                                   subset=s['subset'],
                                   limit=term_count)
                subset_term_info[s['subset']] = {'terms': terms,
                                                 'count': s['count']}
    return subset_term_info


def create_subset_details_v3(client, sst_list, skt_list, subset_term_info,
                             sst_cutoff, skt_cutoff, sst_weight, skt_weight,
                             debug=False):
    '''
    Creates a set of subset vectors
    skt = Subset key terms
    sst = Subset shared terms
    '''
    subset_details = []

    # Create a list of terms common across subsets
    shared_text = [text for text in sst_list
                   if sst_list[text] > sst_cutoff]

    # Create a dictionary of terms distinguishing across subsets
    skt_text = {}
    for subset in skt_list:
        skt_text[subset] = {d['term']: d['oddsratio'] for d in skt_list[subset]
                            if d['p-value'] < skt_cutoff}

    # Weight each vector based on how distinguishing or common it is
    for subset_value in subset_term_info:
        subset_terms = subset_term_info[subset_value]['terms']
        term_vecs = [unpack64(term['vector'])
                     for term in subset_terms]
        term_weights = []
        for term in subset_terms:
            if term['term'] in shared_text:
                term_weights.append(term['score'] * 0)
            elif (subset_value in skt_text and
                  term['term'] in skt_text[subset_value]):
                term_weights.append(np.log(term['score'] *
                                    skt_text[subset_value][term['term']] *
                                    skt_weight + 1))
            else:
                term_weights.append(term['score'])
        if debug:
            terms = [{'term': t['term'],
                      'weight': w,
                      'relevance': t['score']}
                     for t, w in zip(subset_terms, term_weights)]
            writer = csv.DictWriter(open('{}.csv'.format(
                                        subset_value.replace('/', '-')), 'w'),
                                    fieldnames=terms[0].keys())
            writer.writeheader()
            writer.writerows(terms)
        terms_vector = np.average(term_vecs, weights=term_weights, axis=0)
        subset_details.append({
                    'name': subset_value.split(':')[1].strip(),
                    'doc_count': subset_term_info[subset_value]['count'],
                    'vector': terms_vector,
                    'top_term': subset_terms[np.argmax(term_weights)]['text']
                    })
    return subset_details


def subset_shared_terms(client, terms_per_subset=50, scan_terms=1000,
                        min_score=30):
    '''
    Returns terms that are well represented across multiple subsets in the
    entire project
    '''
    subset_counts = client.get()['counts']
    subset_scores = {}
    index = 0
    for subset in sorted(subset_counts):
        index += 1
        subset_terms = client.get('terms', subset=subset, limit=scan_terms)
        length = 0
        termlist = []
        all_terms = []
        for term in subset_terms:
            if length + len(term['term']) > 15000:
                all_terms.extend(client.get('terms', terms=termlist))
                termlist = []
                length = 0
            termlist.append(term['term'])
            length += len(term['term'])
        if len(termlist) > 0:
            all_terms.extend(client.get('terms', terms=termlist))
        all_term_dict = {term['term']: term['distinct_doc_count']
                         for term in all_terms}

        for term in subset_terms:
            term_in_subset = term['distinct_doc_count']
            term_outside_subset = all_term_dict[term['term']] - term_in_subset
            docs_in_subset = subset_counts[subset]
            docs_outside_subset = (subset_counts['__all__'] -
                                   subset_counts[subset])
            table = np.array([
                [term_in_subset, term_outside_subset],
                [docs_in_subset, docs_outside_subset]
            ])
            _, pvalue = fisher_exact(table, alternative='greater')
            if term['score'] > min_score:
                if term['term'] in subset_scores:
                    subset_scores[term['term']] += pvalue
                else:
                    subset_scores[term['term']] = pvalue

    return {k: v/len(subset_counts) for k, v in subset_scores.items()}


def subset_key_terms(client, terms_per_subset=10, scan_terms=1000,
                     min_score=30):
    """
    Find 'key terms' for a subset, those that appear disproportionately more
    inside a subset than outside of it.
    """
    subset_counts = client.get()['counts']
    subset_scores = {subset: [] for subset in subset_counts}
    index = 0
    for subset in sorted(subset_counts):
        index += 1
        subset_terms = client.get('terms', subset=subset, limit=scan_terms)
        length = 0
        termlist = []
        all_terms = []
        for term in subset_terms:
            if length + len(term['term']) > 1000:
                all_terms.extend(client.get('terms', terms=termlist))
                termlist = []
                length = 0
            termlist.append(term['term'])
            length += len(term['term'])
        if len(termlist) > 0:
            all_terms.extend(client.get('terms', terms=termlist))
        all_term_dict = {term['term']: term['distinct_doc_count']
                         for term in all_terms}

        for term in subset_terms:
            term_in_subset = term['distinct_doc_count']
            term_outside_subset = (all_term_dict[term['term']] -
                                   term_in_subset + 1)
            docs_in_subset = subset_counts[subset]
            docs_outside_subset = (subset_counts['__all__'] -
                                   subset_counts[subset] + 1)
            table = np.array([
                [term_in_subset, term_outside_subset],
                [docs_in_subset, docs_outside_subset]
            ])
            oddsratio, pvalue = fisher_exact(table, alternative='greater')
            if term['score'] > min_score:
                if subset in subset_scores:
                    subset_scores[subset].append({'term': term['term'],
                                                  'p-value': pvalue,
                                                  'oddsratio': oddsratio})
                else:
                    subset_scores[subset] = [{'term': term['term'],
                                              'p-value': pvalue,
                                              'oddsratio': oddsratio}]

    return subset_scores


def vectorize_query(query_terms, client, sst_list, sst_weight):
    '''
    Create a search vector based on the search query input by the user and
    weighting of the resulting match score based on the search query's length
    '''

    texts = []
    term_vectors = []
    term_weights = []
    for term in query_terms:
        if term['vector']:
            texts.append(term['text'])
            term_vectors.append(unpack64(term['vector']))
            if term['term'] in sst_list:#and sst_list[term['term']] > sst_weight:
                term_weights.append((1 - sst_list[term['term']]) * sst_weight)
            else:
                term_weights.append(1)
    question_vec = np.average(term_vectors, weights=term_weights, axis=0)

    if len(texts) > 1:
        match_score_weight = np.mean(np.dot(term_vectors,
                                            np.transpose(term_vectors)))
    else:
        match_score_weight = 1
    return question_vec, match_score_weight


def recommend_subset(question_vec, subset_details,
                     num_results=3, min_count=50):
    '''
    Output recommended subsets based on the user input search query
    '''
    subset_vecs = [c['vector'] for c in subset_details]
    match_scores = np.dot(subset_vecs, question_vec)
    match_indices = np.argsort(match_scores)[::-1]

    results = []
    for idx in match_indices:
        if subset_details[idx]['doc_count'] > min_count:
            subset_details[idx]['match_score'] = match_scores[idx]
            results.append(subset_details[idx])
    return results[:num_results]


def find_example_docs(client, subset, query_vec, n_docs=1, source_field=None):
    '''
    Find documents from the selected subset that best match the search query.
    If the subset is specified in a source field, a broad search must be
    performed, as opposed to traditional subsets that can be filtered
    directly using the docs/search endpoint.
    '''

    # Get sentiment vectors
    topics = client.get('topics')
    pos = [unpack64(t['vector']) for t in topics if t['name'] == 'Positive'][0]
    neg = [unpack64(t['vector']) for t in topics if t['name'] == 'Negative'][0]

    if source_field:
        example_doc_pool = client.get('docs/search',
                                      vector=query_vec,
                                      limit=20000,
                                      doc_fields=['text', 'source', 'vector']
                                      )['search_results']

        example_doc_pool = [e[0]['document'] for e in example_doc_pool
                        if e[0]['document']['source'][source_field] == subset]

        if len(example_doc_pool) == 0:
            # If no docs are found, return a set of blank docs with 0 scores
            return [('', 0) for i in range(n_docs)]
    else:
        example_doc_pool = client.get('docs/search',
                                      vector=query_vec,
                                      subset=subset,
                                      limit=100)['search_results']
        example_doc_pool = [e[0]['document'] for e in example_doc_pool]

    vector_matches = []
    for doc in example_doc_pool:
        score = 0
        score += np.dot(unpack64(doc['vector']), unpack64(query_vec))
        score += np.dot(unpack64(doc['vector']), pos)
        score -= np.dot(unpack64(doc), neg)
        vector_matches.append(score)

    doc_indexes = np.argsort(vector_matches)[::-1]

    example_docs = []
    for i in range(n_docs):
        example_docs.append((example_doc_pool[doc_indexes[i]]['text'],
                             vector_matches[doc_indexes[i]]))

    return example_docs


def test_queries(client, queries, subset_details, sst_list, sst_weight,
                 results_filename=None, save_file=False):
    '''
    Reads a set of queries from a CSV file and outputs a CSV file with
    recommendation results.

    CSV file needs the following headers:
    query - the search query
    result - the expected result for the query
    score - score of result against query
    new_result - the new recommendation

    Scoring:
    0 - Not enough data for query to perform well
    1 - Irrelevant result
    2 - Reasonable result, but not ideal
    3 - Ideal result
    '''
    times = {}
    times['vectorize_query'] = []
    times['recommend_subset'] = []

    query_docs = client.post_data('docs/vectors',
                                  json.dumps([{'text': q['query']}
                                              for q in queries]),
                                  content_type='application/json')
    queries_docs = [[t for t, _, _ in d['terms']] for d in query_docs]
    queries_terms = client.get('/terms', terms=list(set([item for sublist in queries_docs for item in sublist])))
    for i, query in enumerate(queries):
        start_time = time.time()
        query_terms = [q for q in queries_terms
                       if q['term'] in queries_docs[i] and
                       q['vector'] is not None]
        if query_terms:
            query_vector, _ = vectorize_query(query_terms,
                                              client,
                                              sst_list,
                                              sst_weight)
            end_time = time.time()
            times['vectorize_query'].append(end_time-start_time)
            start_time = time.time()
            recommendations = recommend_subset(query_vector,
                                               subset_details,
                                               num_results=1)
            end_time = time.time()
            times['recommend_subset'].append(end_time-start_time)
            query['new_result'] = recommendations[0]['name']
    print('Vectorize Query: {}'.format(np.average(times['vectorize_query'])))
    print('Recommend Subset: {}'.format(np.average(times['recommend_subset'])))
    if save_file:
        writer = csv.DictWriter(open(results_filename, 'w'),
                                fieldnames=queries[0].keys())
        writer.writeheader()
        writer.writerows(queries)
    return queries


def score_test_queries(query_results):
    '''
    Score the test queries output from the test_queries function.
    '''
    unique_queries = len(set([q['query'] for q in query_results]))
    score = 0
    scored_queries = 0
    for query in query_results:
        if query['new_result'] == query['result']:
            score += int(query['score'])
            scored_queries += 1
    print('Total Queries: {}'.format(unique_queries))
    print('Scored Queries: {}'.format(scored_queries))
    print('Score: {}/{} = {}'.format(score,
                                     scored_queries*3,
                                     score/(scored_queries*3)))
    print('Optimization Score: {}'.format(
            sigmoid(scored_queries/unique_queries) * score/(scored_queries*3)))
    return -sigmoid(scored_queries/unique_queries) * score/(scored_queries*3)


def sigmoid(x):
    return 1 / (1 + math.exp(-x*3))


def optimize_function(weights, data):
    '''
    Optimization function called by optimize_weights
    '''
    print(weights)
    start_time = time.time()
    subset_details = create_subset_details_v3(client,
                                              data['sst_list'],
                                              data['skt_list'],
                                              data['subset_term_info'],
                                              weights[0],
                                              weights[1],
                                              weights[2],
                                              weights[3])
    end_time = time.time()
    print('Subset details: {:2}s'.format(end_time-start_time))
    start_time = time.time()
    query_results = test_queries(client,
                                 data['queries'],
                                 subset_details,
                                 data['sst_list'],
                                 weights[2],
                                 results_filename='intermediate_results.csv',
                                 save_file=False)
    end_time = time.time()
    print('Query results: {:2}s'.format(end_time-start_time))
    return score_test_queries(query_results)


def optimize_weights(weights, data):
    '''
    Take a set of graded queries, output optimal weights & final score
    '''
    results = differential_evolution(optimize_function,
                                     bounds=[(0.00001, 1.0),
                                             (0, 1.0),
                                             (0.00001, 1.0),
                                             (1.0, 20.0)],
                                     args=(data,),
                                     maxiter=100)
    print(results)
    return results

if __name__ == '__main__':
    client = LuminosoClient.connect('/projects/a53y655v/prtcgdw7')

    rebuild = False
    optimize = True

    if rebuild:
        print('Rebuilding data')
        data = {}
        data['sst_list'] = subset_shared_terms(client)
        data['skt_list'] = subset_key_terms(client)
        data['subset_term_info'] = get_subset_term_info(client, 'Category',
                                                        term_count=1000)
        pickle.dump(data, open('optimization_dataV2.p', 'wb'))
    else:
        print('Loading data from file')
        data = pickle.load(open('optimization_dataV2.p', 'rb'))

    queries = []
    queries_reader = csv.DictReader(open('intermediate_result_scores.csv', 'r'))
    for row in queries_reader:
        queries.append(row)
    data['queries'] = queries

    if optimize:
        results = optimize_weights(np.asarray([0.10381204,
                                               0.14558712,
                                               0.001,
                                               1]),
                                   data)
        optimal_weights = results.x
    else:
        optimal_weights = [  9.25061528e-03,   2.96993695e-01,   1.80252751e-01,
         1.39775613e+01]
        #[  9.25061528e-03,   2.96993695e-01,   5.80252751e-01, 1.39775613e+01]
        #[  4.19304125e-03, 1.22478451e-01, 4.59817417e-01, 6.20227358e+00]
        #[ 0.01113418,  0.36183964,  0.37692194,  9.99969457]
        #[ 0.02463569,  0.6435483 ,  0.71295812,  8.03028856]
        #[ 0.69229096,  0.06517147,  0.52024399,  1.17443286]

    subset_details = create_subset_details_v3(client,
                                              data['sst_list'],
                                              data['skt_list'],
                                              data['subset_term_info'],
                                              optimal_weights[0],
                                              optimal_weights[1],
                                              optimal_weights[2],
                                              optimal_weights[3])
    writer = csv.DictWriter(open('subset_best_terms.csv', 'w'),
                            fieldnames=subset_details[0].keys())
    writer.writeheader()
    writer.writerows(subset_details)
    query_results = test_queries(client,
                                 data['queries'],
                                 subset_details,
                                 data['sst_list'],
                                 optimal_weights[2],
                                 results_filename='intermediate_results.csv',
                                 save_file=True)
    score_test_queries(query_results)
    '''
    weight[0] = sst_cutoff
    weight[1] = skt_cutoff
    weight[2] = sst_weight
    weight[3] = skt_weight
    '''
