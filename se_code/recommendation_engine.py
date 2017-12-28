from __future__ import division
from luminoso_api import LuminosoClient
from pack64 import pack64, unpack64
from scipy.stats import fisher_exact

import numpy as np
import csv
import json
import argparse

### TODO:
###       - Change all mentions of 'Title' to generic input-defined subset/source
###       - Define second layer? e.g. genre, type of food, etc.
###       - Pickle 'subset_vectors' and personalized_selections

#def parse_response(lambda_response, genres, topics, titles):
    #filters = {}
    #if lambda_response['Movie'] == 'Movie':
    #    filters['isMovie'] = True
        
    #if lambda_response['Genre'] in genres:
    #    filters['Genre'] = lambda_response['Genre']
        
    #if lambda_response['Topics'] in topics:
    #    filters['Topics'] = lambda_response['Topics']
        
    #if lambda_response['ProgramTitle'] in titles:
    #    filters['ProgramTitle'] = lambda_response['ProgramTitle']
    
    #return filters
###
def subset_shared_terms(client, terms_per_subset=50, scan_terms=1000):
    subset_counts = client.get()['counts']
    #pvalue_cutoff = 1 / scan_terms / 20
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
                #actual_subset_terms.append(term)
            length += len(term['term'])
        if len(termlist) > 0:
            all_terms.extend(client.get('terms', terms=termlist))
        all_term_dict = {term['term']: term['distinct_doc_count'] for term in all_terms}

        subset_scores = []
        for term in subset_terms:
            term_in_subset = term['distinct_doc_count']
            term_outside_subset = all_term_dict[term['term']] - term_in_subset# + 1
            docs_in_subset = subset_counts[subset]
            docs_outside_subset = subset_counts['__all__'] - subset_counts[subset]# + 1
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

    return results
###


def personalized_search_exact(previous_selections, match_scores, categories, second):
    previous_length = len(previous_selections)
    selections_weight = {}
    for idx, selection in enumerate(previous_selections):
        selections_weight.update({selection: 1 + (float(((idx + 1) / previous_length)) / 2)})
    for idx, score in enumerate(match_scores):
        if categories[idx][second] in selections_weight:
            match_scores[idx] = score * selections_weight[categories[idx][second]]
    return match_scores

def personalized_search_conceptual(previous_selections, categories, q_vec, weight=.25):
    vectors = []
    previous_selections = [t.lower() for t in previous_selections]
    for category in categories:
        if category['name'].lower() in previous_selections:
            category_to_print = category
            target_vec = category['vector']
            vectors.append(target_vec)
    if len(vectors) == 0:
        return q_vec
    sum_vec = np.copy(vectors[0])
    if len(vectors) > 1:
        for vector in vectors[1:]:
            # weight based on recency?
            sum_vec += vector
    avg_vec = sum_vec / len(vectors)
    #weight = .25
    avg_vec = avg_vec * weight
    # Average instead of sum?
    #return (q_vec * 4 + avg_vec) / 5?
    return q_vec + avg_vec
    
def search_content(description, client, categories, filters, history, second, personalize=False, display=3, min_count=0):
    shared = subset_shared_terms(client)
    shared_text = []
    for _, term, _, _ in shared:
        for text in term['all_texts']:
            shared_text.append(text)
    
    # TODO: Get term vectors of the words and downweight subset shared terms instead of removing them
    description_words = description.split(' ')
    for word in description_words:
        if word in shared_text:
            description_words.remove(word)
            
    description = ' '.join(description_words)
    # Get question vector
    q_doc = client.post_data('docs/vectors',json.dumps([{'text':description}]),content_type='application/json')[0]
    q_vec = q_doc['vector']
    q_vec = unpack64(q_vec)
    #if personalize == 'concept':
    if personalize:
        q_vec = personalized_search_conceptual(history, categories, q_vec)
    doc_term_count = len(q_doc['terms'])
    q_term_dot = 1
    
    if doc_term_count > 1:
        q_terms = [t for t,_,_ in q_doc['terms']]
        q_terms_vecs = [unpack64(client.get('terms/search',terms=[term], limit=1)['search_results'][0][0]['vector'])
                        for term in q_terms]
        q_term_dot = np.mean(np.dot(q_terms_vecs,np.transpose(q_terms_vecs)))
    
    # Get best subset
    # PICKLE based on client information?
    subset_vecs = [t['vector'] for t in categories]
    match_scores = np.dot(subset_vecs,q_vec)/doc_term_count
    #if personalize == 'exact':
    #    match_scores = personalized_search_exact(history, match_scores, categories, second)
    match_indexes = np.argsort(match_scores)[::-1]
    
    count = 0
    for idx in match_indexes:
        if categories[idx]['doc_count'] > min_count:
        #if titles[idx]['doc_count'] == 1:
            print(categories[idx]['name'])
            print(match_scores[idx]*q_term_dot)
            print(categories[idx]['doc_count'])
            if second:
                print(categories[idx][second])
            print()
            count += 1
            if count > display - 1:
                break

def add_previous_search(selections, result):
    selections.append(result)
    if len(selections) > 5:
        del selections[0]
    return selections            

def clear_previous_search(selections):
    return []

def get_docs(client):
    docs = []
    while True:
        newdocs = client.get('docs', limit=25000, offset=len(docs))
        if newdocs:
            docs.extend(newdocs)
        else:
            return docs
        
def main():
    parser = argparse.ArgumentParser(
        description='Create a classification model based on an existing project using subsets as labels.'
    )
    parser.add_argument(
        'account_id',
        help="The ID of the account that owns the project, such as 'demo'"
        )
    parser.add_argument(
        'project_id',
        help="The ID of the project"
        )
    parser.add_argument(
        'subset',
        help="Subset Category that the search result is under (Title, Name, etc.)"
        )
    parser.add_argument(
        'search',
        help="Text to search project on"
        )
    parser.add_argument('-u', '--source', default=False, action='store_true', help="Relevant data stored in Source field instead of subset")
    parser.add_argument('-s', '--second', help="\'Category\' of selection e.g. genre of film, type of food. Only applies if information is stored in the source field")
    parser.add_argument('-p', '--personalize', default=False, action='store_true', help="Assign weighting onto previous personalized searches")
    parser.add_argument('-d', '--display_count', default=3, help="Number of choices to display")
    parser.add_argument('-m', '--min_doc_count', default=1, help="Only display choices with more than this number of documents")
    args = parser.parse_args()
    
    client = LuminosoClient.connect('/projects/%s/%s' % (args.account_id, args.project_id))
    docs = get_docs(client)
    
    second = None
    if args.second:
        second = args.second
        
    field = 'subsets'
    if args.source:
        field = 'source'
                
    categories = []
    category_list = {}
    if field == 'subsets':
        subset_stats = client.get('subsets/stats')
        #subsets = {}
        for s in subset_stats:
            if s['subset'] != '__all__':
                subset_name = s['subset'].split(':')[0].strip()
                subset_value = s['subset'].split(':')[1].strip()
                if args.subset == subset_name:
                    #terms = client.get('terms', subset=s['subset'], limit=500)
                    #term_vecs = []
                    #term_weights = []
                    #for term in terms:
                        #term_vecs.append(unpack64(term['vector']))
                        #term_weights.append(term['score'])
                    #terms_vector = np.average(term_vecs, weights=term_weights)
                    categories.append({'name': subset_value,
                                       'doc_count': s['count'],
                                       #'vector': terms_vector,
                                       'vector': unpack64(s['mean'])})
                #subsets[subset_name] = subset_value
    else:
        for doc in docs:
            if args.subset in doc['source']:
                if doc['source'][args.subset] in category_list:
                    categories[category_list[doc['source'][args.subset]]]['doc_count'] += 1
                    #categories[category_list[doc['source'][args.subset]]]['vector'] = np.sum([unpack64(doc['vector']), categories[category_list[doc['source'][args.subset]]]['vector']], axis=0)
                    categories[category_list[doc['source'][args.subset]]]['vector'].append(unpack64(doc['vector']))
                else:
                    category_list[doc['source'][args.subset]] = len(categories)
                    # HOW DO WE GET AVERAGE TERM VECTORS OF SOURCES?
                    if second and second in doc['source']:
                        categories.append({'name': doc['source'][args.subset],
                                           'doc_count': 1,
                                           'vector': [unpack64(doc['vector'])],
                                           second: doc['source'][second]})
                    else:
                        categories.append({'name': doc['source'][args.subset],
                                           'doc_count': 1,
                                           'vector': [unpack64(doc['vector'])]})
        for category in categories:
            category['vector'] = np.mean(category['vector'], axis=0)
                    
    
    #for doc in docs:
    #    if field == 'subsets':
    #        subsets = {}
    #        for subset in doc['subsets']:
    #            if subset != '__all__':
    #                subset_name = subset.split(':')[0].strip()
    #                subset_value = subset.split(':')[1].strip()
    #                subsets[subset_name] = subset_value
    #        if args.subset in subsets:
    #            subset_value = subsets[args.subset]
    #            if subset_value in category_list:
    #                categories[category_list[subset_value]]['doc_count'] += 1
    #                categories[category_list[subset_value]]['vector'] = np.sum([unpack64(doc['vector']), categories[category_list[subset_value]]['vector']], axis=0)
    #            else:
    #                category_list[subset_value] = len(categories)
    #                if second and second in subsets:
    #                    categories.append({'name': subset_value,
    #                                       'doc_count': 1,
    #                                       'vector': unpack64(doc['vector']),
    #                                       second: subsets[second]})
    #                else:
    #                    categories.append({'name': subset_value,
    #                                       'doc_count': 1,
    #                                       'vector': unpack64(doc['vector'])})
    #    else:
    #        #source_fields = [f.lower() for f in doc['source']]
    #        if args.subset in doc['source']:
    #            if doc['source'][args.subset] in category_list:
    #                categories[category_list[doc['source'][args.subset]]]['doc_count'] += 1
    #                categories[category_list[doc['source'][args.subset]]]['vector'] = np.sum([unpack64(doc['vector']), categories[category_list[doc['source'][args.subset]]]['vector']], axis=0)
    #            else:
    #                category_list[doc['source'][args.subset]] = len(categories)
    #                if second and second in doc['source']:
    #                    categories.append({'name': doc['source'][args.subset],
    #                                       'doc_count': 1,
    #                                       'vector': unpack64(doc['vector']),
    #                                       second: doc['source'][second]})
    #                else:
    #                    categories.append({'name': doc['source'][args.subset],
    #                                       'doc_count': 1,
    #                                       'vector': unpack64(doc['vector'])})
                
    
    #for category in categories:
    #    category['vector'] = category['vector']/category['doc_count']
        
    # PICKLE?
    previous_selections = []
    #selection = 'Documentary'
    #previous_selections = add_previous_search(previous_selections, selection)
    #previous_selections = clear_previous_search(previous_selections)
    qtr = search_content(args.search, client, categories, [], previous_selections, second, args.personalize, int(args.display_count), int(args.min_doc_count))
    
if __name__ == '__main__':
    main()


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
