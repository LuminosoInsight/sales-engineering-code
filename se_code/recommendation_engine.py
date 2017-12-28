from __future__ import division
from luminoso_api import LuminosoClient
from pack64 import pack64, unpack64
from scipy.stats import fisher_exact

import numpy as np
import csv
import json
import argparse


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
    Creates a list of dictionaries holding each subset's name, doc count, and average vector
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
                    categories[category_list[doc['source'][subset_input]]]['doc_count'] += 1
                    categories[category_list[doc['source'][subset_input]]]['vector'].append(unpack64(doc['vector']))
                else:
                    category_list[doc['source'][subset_input]] = len(categories)
                    categories.append({'name': doc['source'][subset_input],
                                       'doc_count': 1,
                                       'vector': [unpack64(doc['vector'])]})
        for category in categories:
            category['vector'] = np.mean(category['vector'], axis=0)
           
    return categories

def create_subset_vectors_v3(client, docs, shared_text, field, subset_input):
    '''
    Creates a list of dictionaries holding each subset's name, doc count, and average vector based
    on the subset's top terms, with the terms shared between subsets downweighted
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
                    categories[category_list[doc['source'][subset_input]]]['doc_count'] += 1
                    categories[category_list[doc['source'][subset_input]]]['vector'].append(unpack64(doc['vector']))
                else:
                    category_list[doc['source'][subset_input]] = len(categories)
                    categories.append({'name': doc['source'][subset_input],
                                       'doc_count': 1,
                                       'vector': [unpack64(doc['vector'])]})
        for category in categories:
            category['vector'] = np.mean(category['vector'], axis=0)
           
    return categories

def subset_shared_terms(client, terms_per_subset=50, scan_terms=1000):
    '''
    Returns terms that are well represented across multiple subsets in the entire project
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
        all_term_dict = {term['term']: term['distinct_doc_count'] for term in all_terms}

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
    Create a search vector based on the search query inputted by the user and the weighting
    of the resulting match score based on the search query's length
    '''
    shared_text = subset_shared_terms(client)
    question_doc = client.post_data('docs/vectors', json.dumps([{'text': description}]), content_type='application/json')[0]
    description_words = [t[0] for t in question_doc['terms']]
    texts = []
    term_vectors = []
    term_weights = []
    for word in description_words:
        term = client.get('terms/search', terms=[word], limit=1)
        texts.append(term['search_results'][0][0]['text'])
        term_vectors.append(unpack64(term['vector']))
        if word in shared_text:
            term_weights.append(term['score'] * .1)
        else:
            term_weights.append(term['score'])
    question_vec = np.average(term_vectors, weights=term_weights, axis=0)
    
    if len(texts) > 1:
        match_score_weight = np.mean(np.dot(term_vectors, np.transpose(term_vectors)))
    else:
        match_score_weight = 1
    return question_vec, match_score_weight, len(texts)
    
def recommend_subset(client, description, field, subset_input, display=3):
    '''
    Output recommended subsets based on the user inputted search query
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
