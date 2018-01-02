from __future__ import division
from pack64 import unpack64
from scipy.stats import fisher_exact
from luminoso_api import LuminosoClient

import numpy as np
import json
import csv
import pickle


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


def create_subset_details_v1(client, shared_text, field, subset_input):
    '''
    Creates a list of dictionaries holding each subset's name, doc count, and
    average vector based on the subset's top terms, with the terms shared
    between subsets down-weighted
    '''
    docs = get_all_docs(client)
    subset_details = []
    category_list = {}
    if field == 'subsets':
        subset_stats = client.get('subsets/stats')
        for s in subset_stats:
            if s['subset'] != '__all__':
                subset_name = s['subset'].split(':')[0].strip()
                subset_value = s['subset'].split(':')[1].strip()
                if subset_input == subset_name:
                    terms = client.get('terms', subset=s['subset'], limit=500)
                    term_vecs = []
                    term_weights = []
                    for term in terms:
                        term_vecs.append(unpack64(term['vector']))
                        if term['text'] in shared_text:
                            term_weights.append(term['score'] * .1)
                        else:
                            term_weights.append(term['score'])
                    terms_vector = np.average(term_vecs, weights=term_weights, axis=0)
                    subset_details.append({'name': subset_value,
                                           'doc_count': s['count'],
                                           'vector': terms_vector})
    else:
        for doc in docs:
            doc_vec = unpack64(doc['vector'])
            if subset_input in doc['source']:
                if doc['source'][subset_input] in category_list:
                    category = category_list[doc['source'][subset_input]]
                    subset_details[category]['doc_count'] += 1
                    subset_details[category]['vector'].append(doc_vec)
                else:
                    subset_details.append({'name': doc['source'][subset_input],
                                           'doc_count': 1,
                                           'vector': [doc_vec]})
        for s in subset_details:
            s['vector'] = np.mean(s['vector'], axis=0)

    return subset_details


### OPTIMIZATION ###
def get_subset_term_info(client, subset_input):
    subset_term_info = {}
    subset_stats = client.get('subsets/stats')
    for s in subset_stats:
        if s['subset'] != '__all__':
            subset_name = s['subset'].split(':')[0].strip()
            subset_value = s['subset'].split(':')[1].strip()
            if subset_input == subset_name:
                terms = client.get('terms', subset=s['subset'], limit=500)
                subset_term_info[s['subset']] = {'terms': terms, 'count': s['count']}
    return subset_term_info

def create_subset_details_v3(client, shared_text, key_text, subset_term_info, 
                             shared_cutoff, key_cutoff, shared_weight, key_weight):
    subset_details = []
    shared_text = [text for text in shared_text if shared_text[text] > shared_cutoff]
    for subset in key_text:
        key_text[subset] = [d for d in key_text[subset] if d['p-value'] < key_cutoff]
    for subset_value in subset_term_info:
        term_vecs = []
        term_weights = []
        for term in subset_term_info[subset_value]['terms']:
            term_vecs.append(unpack64(term['vector']))
            if term['text'] in shared_text:
                term_weights.append(term['score'] * shared_weight)
            elif subset_value in key_text and term['text'] in key_text[subset_value]['text']:
                term_weights.append(term['score'] * key_weight)
            else:
                term_weights.append(term['score'])
        terms_vector = np.average(term_vecs, weights=term_weights, axis=0)
        subset_details.append({'name': subset_value.split(':')[1].strip(),
                               'doc_count': subset_term_info[subset_value]['count'],
                               'vector': terms_vector})
        return subset_details
    
def create_source_details_v3(client, subset_input,
                             shared_cutoff, key_cutoff, shared_weight, key_weight):
    docs = get_all_docs(client)
    source_details = []
    category_list = {}
    for doc in docs:
        doc_vec = unpack64(doc['vector'])
        if subset_input in doc['source']:
            if doc['source'][subset_input] in category_list:
                category = category_list[doc['source'][subset_input]]
                source_details[category]['doc_count'] += 1
                source_details[category]['vector'].append(doc_vec)
                source_details[category]['doc_terms'].append(len(doc['terms']))
            else:
                category_list[doc['source'][subset_input]] = len(source_details)
                source_details.append({'name': doc['source'][subset_input],
                                       'doc_count': 1,
                                       'vector': [doc_vec],
                                       'doc_terms': [len(doc['terms'])]})
    for s in source_details:
        s['doc_terms'] /= np.max(s['doc_terms'])
        s['vector'] = np.mean(s['vector'], weights=doc['doc_terms'], axis=0)
    return source_details

#def create_subset_details_v3(client, shared_text, key_text, field, subset_input):
#    '''
#    Creates a list of dictionaries holding each subset's name, doc count, and
#    average vector based on the subset's top terms, with the terms shared
#    between subsets down-weighted
#    '''
#    subset_details = []
#    category_list = {}
#    if field == 'subsets':
#        subset_stats = client.get('subsets/stats')
#        for s in subset_stats:
#            if s['subset'] != '__all__':
#                subset_name = s['subset'].split(':')[0].strip()
#                subset_value = s['subset'].split(':')[1].strip()
#                if subset_input == subset_name:
#                    terms = client.get('terms', subset=s['subset'], limit=500)
#                    term_vecs = []
#                    term_weights = []
#                    for term in terms:
#                        term_vecs.append(unpack64(term['vector']))
#                        if term['text'] in shared_text:
#                            term_weights.append(term['score'] * .1)
#                        elif term['text'] in key_text[s['subset']]:
#                            term_weights.append(term['score'] * 2)
#                        else:
#                            term_weights.append(term['score'])
#                    terms_vector = np.average(term_vecs, weights=term_weights, axis=0)
#                    subset_details.append({'name': subset_value,
#                                           'doc_count': s['count'],
#                                           'vector': terms_vector})
#    else:
#        docs = get_all_docs(client)
#        for doc in docs:
#            doc_vec = unpack64(doc['vector'])
#            if subset_input in doc['source']:
#                if doc['source'][subset_input] in category_list:
#                    category = category_list[doc['source'][subset_input]]
#                    subset_details[category]['doc_count'] += 1
#                    subset_details[category]['vector'].append(doc_vec)
#                    subset_details[category]['doc_terms'].append(len(doc['terms']))
#                else:
#                    subset_details.append({'name': doc['source'][subset_input],
#                                           'doc_count': 1,
#                                           'vector': [doc_vec],
#                                           'doc_terms': [len(doc['terms'])]})
#        for s in subset_details:
#            s['doc_terms'] /= np.max(s['doc_terms'])
#            s['vector'] = np.mean(s['vector'], weights=doc['doc_terms'], axis=0)
#    return subset_details

###

def subset_shared_terms(client, terms_per_subset=50, scan_terms=1000):
    '''
    Returns terms that are well represented across multiple subsets in the
    entire project
    '''
    subset_counts = client.get()['counts']
    #pvalue_cutoff = .95
    results = []
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

        subset_scores = []
        for term in subset_terms:
            term_in_subset = term['distinct_doc_count']
            term_outside_subset = all_term_dict[term['term']] - term_in_subset
            docs_in_subset = subset_counts[subset]
            docs_outside_subset = subset_counts['__all__'] - subset_counts[subset]
            table = np.array([
                [term_in_subset, term_outside_subset],
                [docs_in_subset, docs_outside_subset]
            ])
            odds_ratio, pvalue = fisher_exact(table, alternative='greater')
            #if pvalue > pvalue_cutoff:
            subset_scores.append((subset, term, odds_ratio, pvalue))

        if len(subset_scores) > 0:
            subset_scores.sort(key=lambda x: (x[0], -x[2]))
        #results.extend(subset_scores[:terms_per_subset])
        results.extend(subset_scores)

    shared_text = {}
    for _, term, _, p_value in results:
        for text in term['all_texts']:
            shared_text[text] = p_value

    return shared_text

def subset_key_terms(client, terms_per_subset=10, scan_terms=1000):
    """
    Find 'key terms' for a subset, those that appear disproportionately more
    inside a subset than outside of it.
    """
    subset_counts = client.get()['counts']
    #pvalue_cutoff = 1 / scan_terms / 20
    results = []
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
            #if pvalue < pvalue_cutoff:
            subset_scores.append((subset, term, odds_ratio, pvalue))

        if len(subset_scores) > 0:
            subset_scores.sort(key=lambda x: (x[0], -x[2]))
        #results.extend(subset_scores[:terms_per_subset])
        results.extend(subset_scores)
        
        
        key_text = {}
        for subset, term, _, p_value in results:
            if subset not in key_text:
                key_text[subset] = []
            key_text[subset].append({'text': term['all_texts'],
                                     'p-value': p_value})

    return key_text


def vectorize_query(description, client, shared_text):
    '''
    Create a search vector based on the search query input by the user and
    weighting of the resulting match score based on the search query's length
    '''

    question_doc = client.post_data('docs/vectors',
                                    json.dumps([{'text': description}]),
                                    content_type='application/json')[0]
    description_words = [t[0] for t in question_doc['terms']]
    texts = []
    term_vectors = []
    term_weights = []
    for word in description_words:
        search_result = client.get('terms/search', terms=[word], limit=1)['search_results']
        if len(search_result) > 0:
            term = search_result[0][0]
            if term['vector']:
                texts.append(term['text'])
                term_vectors.append(unpack64(term['vector']))
                if word in shared_text:
                    term_weights.append(term['score'] * .1)
                else:
                    term_weights.append(term['score'])
    question_vec = np.average(term_vectors, weights=term_weights, axis=0)

    if len(texts) > 1:
        match_score_weight = np.mean(np.dot(term_vectors,
                                            np.transpose(term_vectors)))
    else:
        match_score_weight = 1
    return question_vec, match_score_weight


def recommend_subset(question_vec, subset_details, num_results=3, min_count=50):
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


def test_queries(client, queries_filename, details_filename, sst_filename, skt_filename, results_filename):
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
    queries = []
    queries_reader = csv.DictReader(open(queries_filename, 'r'))
    for row in queries_reader:
        queries.append(row)
    subset_details = pickle.load(open(details_filename, 'rb'))
    shared_text = pickle.load(open(sst_filename, 'rb'))
    key_text = pickle.load(open(skt_filename, 'rb'))
    for query in queries:
        query_vector, _ = vectorize_query(query['query'],
                                          client,
                                          shared_text)
        recommendations = recommend_subset(query_vector,
                                           subset_details,
                                           num_results=1)
        query['new_result'] = recommendations[0]['name']
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
    return score/(scored_queries*3)

if __name__ == '__main__':
    client = LuminosoClient.connect('/projects/x86x624r/prj5n6zx')
    test_queries(client,
                 'testqueries.csv',
                 'V1_1228.p',
                 'V1_1228sst.p',
                 'V1_1228_results.csv')
