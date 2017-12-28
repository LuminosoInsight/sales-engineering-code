from __future__ import division
from pack64 import unpack64
from scipy.stats import fisher_exact

import numpy as np
import json
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


def create_subset_vectors_v1(client, docs, field, subset_input):
    '''
    Creates a list of dictionaries holding each subset's name, doc count, and
    average vector
    '''
    categories = []
    category_list = {}
    if field == 'subsets':
        subset_stats = client.get('subsets/stats')
        for s in subset_stats:
            if s['subset'] != '__all__':
                subset_name = s['subset'].split(':')[0].strip()
                subset_value = s['subset'].split(':')[1].strip()
                if subset_input == subset_name:
                    categories.append({'name': subset_value,
                                       'doc_count': s['count'],
                                       'vector': unpack64(s['mean'])})
    else:
        for doc in docs:
            if subset_input in doc['source']:
                if doc['source'][subset_input] in category_list:
                    cat = category_list[doc['source'][subset_input]]
                    categories[cat]['doc_count'] += 1
                    categories[cat]['vector'].append(unpack64(doc['vector']))
                else:
                    categories.append({'name': doc['source'][subset_input],
                                       'doc_count': 1,
                                       'vector': [unpack64(doc['vector'])]})
        for category in categories:
            category['vector'] = np.mean(category['vector'], axis=0)

    return categories


def create_subset_vectors_v3(client, docs, shared_text, field, subset_input):
    '''
    Creates a list of dictionaries holding each subset's name, doc count, and
    average vector based on the subset's top terms, with the terms shared
    between subsets down-weighted
    '''
    categories = []
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
                    terms_vector = np.average(term_vecs, weights=term_weights)
                    categories.append({'name': subset_value,
                                       'doc_count': s['count'],
                                       'vector': terms_vector})
    else:
        for doc in docs:
            if subset_input in doc['source']:
                if doc['source'][subset_input] in category_list:
                    cat = category_list[doc['source'][subset_input]]
                    categories[cat]['doc_count'] += 1
                    categories[cat]['vector'].append(unpack64(doc['vector']))
                else:
                    categories.append({'name': doc['source'][subset_input],
                                       'doc_count': 1,
                                       'vector': [unpack64(doc['vector'])]})
        for category in categories:
            category['vector'] = np.mean(category['vector'], axis=0)

    return categories


def subset_shared_terms(client, terms_per_subset=50, scan_terms=1000):
    '''
    Returns terms that are well represented across multiple subsets in the
    entire project
    '''
    subset_counts = client.get()['counts']
    pvalue_cutoff = .95
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
            if pvalue > pvalue_cutoff:
                subset_scores.append((subset, term, odds_ratio, pvalue))

        if len(subset_scores) > 0:
            subset_scores.sort(key=lambda x: (x[0], -x[2]))
        results.extend(subset_scores[:terms_per_subset])

    shared_text = []
    for _, term, _, _ in results:
        for text in term['all_texts']:
            shared_text.append(text)

    return shared_text


def vectorize_query(description, client, min_count=0):
    '''
    Create a search vector based on the search query input by the user and
    weighting of the resulting match score based on the search query's length
    '''
    shared_text = subset_shared_terms(client)
    question_doc = client.post_data('docs/vectors',
                                    json.dumps([{'text': description}]),
                                    content_type='application/json')[0]
    description_words = [t[0] for t in question_doc['terms']]
    texts = []
    term_vectors = []
    term_weights = []
    for word in description_words:
        term = client.get('terms/search', terms=[word], limit=1)['search_results'][0][0]
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
    return question_vec, match_score_weight, len(texts)


def recommend_subset(client, description, field, subset_input, display=3, min_count=50):
    '''
    Output recommended subsets based on the user input search query
    '''
    docs = get_all_docs(client)
    categories = create_subset_vectors_v1(client, docs, field, subset_input)
    subset_vecs = [c['vector'] for c in categories]
    question_vec, match_score_weight, doc_term_count = vectorize_query(description, client)
    match_scores = np.dot(subset_vecs, question_vec) / doc_term_count
    match_indices = np.argsort(match_scores)[::-1]

    count = 0
    for idx in match_indices:
        if categories[idx]['doc_count'] > min_count:
            print(categories[idx]['name'])
            print(match_scores[idx] * match_score_weight)
            print(categories[idx]['doc_count'])
            print()
            count += 1
            if count > display - 1:
                break


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


def save_subset_vectors(filename, subset_vectors):
    '''
    Saves subset vector object as a pickled file for reuse.
    '''

    file = open(filename, 'w')
    pickle.dump(subset_vectors, file)


def load_subset_vectors(filename):
    '''
    Loads subset vector object from a pickled file.
    '''

    file = open(filename, 'r')
    subset_vectors = pickle.load(file)
    return subset_vectors
