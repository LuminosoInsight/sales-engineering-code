from __future__ import division

from pack64 import unpack64
from luminoso_api import LuminosoClient

import argparse
from argparse import RawTextHelpFormatter
import numpy as np
import json
import csv
import pickle
import math

from scipy.optimize._differentialevolution import differential_evolution
from scipy.stats import fisher_exact


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
                             skt_cutoff, skt_weight, sst_cutoff, sst_weight,
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
            if (subset_value in skt_text and
                term['term'] in skt_text[subset_value]):
                term_weights.append(np.log(term['score'] *
                                    skt_text[subset_value][term['term']] *
                                    skt_weight + 1))
            elif term['term'] in shared_text:
                term_weights.append(term['score'] * sst_weight)
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

def create_subset_info(subset_details):
    subset_info = {}
    for detail in subset_details:
        subset_info[detail['name']] = {'doc_count': detail['doc_count'],
                                       'vector': detail['vector'],
                                       'top_term': detail['top_term']}
    return subset_info


def subset_terms(client, scan_terms=1000, min_score=30):
    '''
    Find the most relevant terms per subset and return their p-value
    and odds ratios
    '''
    subset_counts = client.get()['counts']
    subset_key_scores = {subset: [] for subset in subset_counts}
    subset_shared_scores = {}
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
                                   term_in_subset)
            docs_in_subset = subset_counts[subset]
            docs_outside_subset = (subset_counts['__all__'] -
                                   subset_counts[subset])
            table = np.array([
                [term_in_subset, term_outside_subset],
                [docs_in_subset, docs_outside_subset]
            ])
            oddsratio, pvalue = fisher_exact(table, alternative='greater')
            _, shared_pvalue = fisher_exact(table, alternative='two-sided')

            if term['score'] > min_score:
                if term['term'] in subset_shared_scores:
                    subset_shared_scores[term['term']] += shared_pvalue
                else:
                    subset_shared_scores[term['term']] = shared_pvalue
                if subset in subset_key_scores:
                    subset_key_scores[subset].append({'term': term['term'],
                                                      'p-value': pvalue,
                                                      'oddsratio': oddsratio})
                else:
                    subset_key_scores[subset] = [{'term': term['term'],
                                                  'p-value': pvalue,
                                                  'oddsratio': oddsratio}]
    subset_shared_scores = {k: v/len(subset_counts)
                            for k, v in subset_shared_scores.items()}

    return subset_shared_scores, subset_key_scores


def get_query_terms(query, client):
    query_doc = client.post_data('docs/vectors',
                                 json.dumps([{'text': query}]),
                                 content_type='application/json')[0]
    query_doc_terms = [t for t, _, _ in query_doc['terms']]
    query_terms = client.get('/terms',
                             terms=list(set([term for term in query_doc_terms])))
    query_terms = [q for q in query_terms
                   if q['term'] in query_doc_terms and
                   q['vector'] is not None]
    return query_terms


def vectorize_preferences(preference_file, subset_info, client, #user,
                          sst_list, sst_cutoff):
    preference_list = []
    #preference_vectors = []
    #preference_weights = []
    with open(preference_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            score = int(row['Score'])
#             PLACEHOLDER FOR PERSONALIZATION
#             score = row[user]
            preference = row['Preference']
            if row['Type'] == 'Category':
                #if score >= 0:
                preference_list.append({
                    'name': preference,
                    'vector': subset_info[preference]['vector'],
                    'score': score
                })
                #else:
                #    preference_vectors.append(-1 * subset_info[preference]['vector'])
                #    preference_weights.append(-1 * score)
            else:
                query_terms = get_query_terms(preference, client)
                pref_vec = vectorize_query(query_terms,
                                           client,
                                           sst_list,
                                           sst_cutoff)
                #if score >= 0:
                preference_list.append({
                    'name': preference,
                    'vector': subset_info[preference]['vector'],
                    'score': score
                })
                    #preference_vectors.append(pref_vec)
                    #preference_weights.append(score)
                #else:
                #    preference_vectors.append(-1 * pref_vec)
                #    preference_weights.append(-1 * score)
    #preference_vec = np.average(preference_vectors,
    #                            weights=preference_weights,
    #                            axis=0)
    return preference_list

def vectorize_query(query_terms, client, sst_list, sst_cutoff, sst_weight,
                    preference_list=None, pref_weight=100, dist_cutoff=.5):
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
            if term['term'] in sst_list and sst_list[term['term']] > sst_cutoff:
                term_weights.append(sst_weight)
            else:
                term_weights.append(1)
    question_vec = np.average(term_vectors, weights=term_weights, axis=0)

    if preference_list:
        pref_list = []
        for preference in preference_list:
            if np.linalg.norm(preference['vector'] - question_vec) < dist_cutoff:
                if preference['score'] >= 0:
                    pref_list.append({
                    'name': preference['name'],
                    'vector': preference['vector'],
                    'score': preference['score']
                    })
                else:
                    pref_list.append({
                    'name': preference['name'],
                    'vector': -1 * preference['vector'],
                    'score': -1 * preference['score']
                    })
        #dist_list = sorted(dist_list, key=lambda pref: pref['distance'], reverse=True)
        #personal_vecs = [question_vec, preference_vec]
        #question_vec = np.average(personal_vecs, weights=[1, pref_weight], axis=0)
        for pref in pref_list:
            question_vec = np.average([question_vec, pref['vector']], 
                                      weights=[pref_weight, pref['score'] ** 2], 
                                      axis=0)
    return question_vec


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


def test_queries(client, queries, subset_details, pref_list, sst_list, 
                 sst_cutoff, sst_weight, pref_cutoff, pref_weight,
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

    query_docs = client.post_data('docs/vectors',
                                  json.dumps([{'text': q['query']}
                                              for q in queries]),
                                  content_type='application/json')
    queries_docs = [[t for t, _, _ in d['terms']] for d in query_docs]
    queries_terms = client.get('/terms',
                               terms=list(set([item
                                               for sublist in queries_docs
                                               for item in sublist])))
    for i, query in enumerate(queries):
        query_terms = [q for q in queries_terms
                       if q['term'] in queries_docs[i] and
                       q['vector'] is not None]
        if query_terms:
            query_vector = vectorize_query(query_terms,
                                           client,
                                           sst_list,
                                           sst_cutoff,
                                           sst_weight,
                                           pref_list,
                                           pref_cutoff,
                                           pref_weight)

            recommendations = recommend_subset(query_vector,
                                               subset_details,
                                               num_results=1)
            query['new_result'] = recommendations[0]['name']

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
        if query['new_result'] == query['scored_result']:
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


def optimize_function(weights, data, client):
    '''
    Optimization function called by optimize_weights
    '''
    subset_details = create_subset_details_v3(client,
                                              data['sst_list'],
                                              data['skt_list'],
                                              data['subset_term_info'],
                                              weights[0],
                                              weights[1],
                                              weights[2],
                                              weights[3])

    query_results = test_queries(client,
                                 data['queries'],
                                 subset_details,
                                 data['pref_list'],
                                 data['sst_list'],
                                 weights[4],
                                 weights[5],
                                 weights[6],
                                 weights[7],
                                 save_file=False)

    return score_test_queries(query_results)


def optimize_weights(data, client):
    '''
    Take a set of graded queries, output optimal weights & final score
    '''
    results = differential_evolution(optimize_function,
                                     bounds=[(0, 0.5),
                                             (1, 20.0),
                                             (0, 0.5),
                                             (0, 1.0),
                                             (0, 0.5),
                                             (0, 1.0),
                                             (0, 20.0),
                                             (25, 200)],
                                     args=(data, client),
                                     maxiter=100)
    print(results)
    return results


def run(account_id, project_id, username, query_file, api_url, rebuild=False,
        optimize=False, prefix='Category', personalization_file=None,
        rebuild_personalization=False):

    client = LuminosoClient.connect('{}/projects/{}/{}'.format(api_url,
                                                               account_id,
                                                               project_id),
                                    username=username)

    if rebuild:
        print('Rebuilding subset_vector_file.p')
        data = {}
        #data['sst_list'] = subset_shared_terms(client)
        #data['skt_list'] = subset_key_terms(client)
        data['sst_list'], data['skt_list'] = subset_terms(client)
        data['subset_term_info'] = get_subset_term_info(client, prefix,
                                                        term_count=1000)
        pickle.dump(data, open('subset_vector_file.p', 'wb'))
    else:
        print('Loading data from subset_vector_file.p')
        data = pickle.load(open('subset_vector_file.p', 'rb'))

    queries = []
    queries_reader = csv.DictReader(open(query_file, 'r'))
    for row in queries_reader:
        queries.append(row)
    data['queries'] = queries

    if optimize:
        print('Rebuilding optimized_weights.p')
        optimal_weights = optimize_weights(data, client).x
        pickle.dump(optimal_weights, open('optimized_weights.p', 'wb'))
    else:
        print('Loading data from optimized_weights.p')
        optimal_weights = pickle.load(open('optimized_weights.p', 'rb'))

    
    subset_details = create_subset_details_v3(client,
                                              data['sst_list'],
                                              data['skt_list'],
                                              data['subset_term_info'],
                                              optimal_weights[0],
                                              optimal_weights[1],
                                              optimal_weights[2],
                                              optimal_weights[3],)

    if personalization_file:
        if rebuild_personalization:
            subset_info = create_subset_info(subset_details)
            data['pref_list'] = vectorize_preferences(personalization_file, 
                                                      subset_info, 
                                                      client, #user,
                                                      data['sst_list'], 
                                                      optimal_weights[4])
            pickle.dump(data['pref_list'], open('preferences_list.p', 'wb'))
        else:
            data['pref_list'] = pickle.load(open('preferences_list.p', 'rb'))
        query_results = test_queries(client,
                                     data['queries'],
                                     subset_details,
                                     data['pref_list'],
                                     data['sst_list'],
                                     optimal_weights[4],
                                     optimal_weights[5],
                                     optimal_weights[6],
                                     optimal_weights[7],
                                     results_filename=query_file,
                                     save_file=True)
    else:
        query_results = test_queries(client,
                                     data['queries'],
                                     subset_details,
                                     None,
                                     data['sst_list'],
                                     optimal_weights[4],
                                     optimal_weights[5],
                                     None,
                                     None,
                                     results_filename=query_file,
                                     safe_file=True)
    score_test_queries(query_results)


def main():
    parser = argparse.ArgumentParser(
        description='Recommendation Engine\n\n'
        'The recommendation engine creates a set of subset vectors'
        ' which are representative of the key features of each subset.'
        ' These subset vectors can be compared to the vector of an incoming'
        ' query in order to produce a recommended subset.\n\n'
        'query_file needs to be a CSV file with the following headers:\n'
        'query: phrase to be queries against recommendation engine\n'
        'new_result: placeholder column for recommendation results\n'
        'scored_result: one or more scored results per query\n'
        'score: score for query when "scored_result" is returned (1-3 scale)',
        formatter_class=RawTextHelpFormatter)

    parser.add_argument('account_id',
                        help="The ID of the account that owns the"
                        "project, such as 'demo'")
    parser.add_argument('project_id',
                        help="The ID of the project to analyze,"
                        "such as 'pr2jsnm'")
    parser.add_argument('username',
                        help="A Luminoso username with access to"
                        "the project")
    parser.add_argument('query_file',
                        help="Name of the file containing a set"
                        "of queries")
    parser.add_argument('-r',
                        '--rebuild',
                        default=False, action='store_true',
                        help="Rebuild the subset vector file:"
                        "subset_vector_file.p"
                        "each subset")
    parser.add_argument('-o',
                        '--optimize',
                        default=False, action='store_true',
                        help="Re-optimize the function weights, and store in:"
                        "optimized_weights.p")
    parser.add_argument('-u',
                        '--url',
                        default='https://analytics.luminoso.com/api/v4',
                        help="The base URL for the Luminoso API (defaults to"
                        "the production API.")
    parser.add_argument('-p',
                        '--prefix',
                        default='Category',
                        help="Prefix of subsets to be used, for example:"
                        "'Category' is the prefix for 'Category: subset'")
    parser.add_argument('-f',
                        '--personalization_file',
                        default=None,
                        help="File name containing the personalization data")

    args = parser.parse_args()
    run(args.account_id, args.project_id, args.username, args.query_file, args.url,
        rebuild=args.rebuild, optimize=args.optimize, prefix=args.prefix,
        personalization_file=args.personalization_file)


if __name__ == '__main__':
    main()
