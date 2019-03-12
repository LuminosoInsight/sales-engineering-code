import json
import numpy as np

from luminoso_api import V5LuminosoClient as LuminosoClient
from pack64 import unpack64


def search_subsets(client, question, subset_vecs, subset_list, min_docs=50, top_subsets=5, top_reviews=2, field=None):
    
    # Get question vector
    q_doc = client.post('vectorize', texts=question.split(' '))
    q_text = [d['text'][d['terms'][0][2][0]:d['terms'][0][2][1]] for d in q_doc]
    q_vec = q_doc['vector']
    
    # Get sentiment vectors
    topics = client.get('concepts/saved')
    pos = [[float(v) for v in unpack64(t['vector'])] for t in topics if t['name'] == 'Positive'][0]
    neg = [[float(v) for v in unpack64(t['vector'])] for t in topics if t['name'] == 'Negative'][0]
    
    # Get best subset
    match_scores = np.dot(subset_vecs,[float(v) for v in unpack64(q_vec)])
    match_indexes = np.argsort(match_scores)[::-1]
    matches = 0
    results = []
    
    for idx in match_indexes:

        if subset_list[idx]['count'] >= min_docs:
            matching_subset = subset_list[idx]['subset']
            if field:
                example_docs = client.get('docs', search=q_text, 
                                          filter={'name': field, 'values': [matching_subset]}, 
                                          limit=5000)['result']
            else:
                example_docs = client.get('docs', search=q_text, limit=5000)['result']
            print(example_docs[0])
            # Get example verbatim
            vector_matches = []
            for doc in example_docs:
                score = 0
                score += np.dot([float(v) for v in unpack64(doc[0]['document']['vector'])],
                                [float(v) for v in unpack64(q_vec)])
                score += np.dot([float(v) for v in unpack64(doc[0]['document']['vector'])],pos)
                score -= np.dot([float(v) for v in unpack64(doc[0]['document']['vector'])],neg)
                vector_matches.append(score)

            doc_indexes = np.argsort(vector_matches)[::-1]
                
            for i in range(top_reviews):
                results.append({'car':matching_subset,
                                'score':match_scores[idx],#vector_matches[doc_indexes[i]],
                                'example_review':example_docs[doc_indexes[i]][0]['document']['text']})
            matches += 1
            if matches >= top_subsets:
                break
    #print(results)
    return question, results
