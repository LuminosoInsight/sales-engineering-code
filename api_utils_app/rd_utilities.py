import json
import numpy as np

from luminoso_api import LuminosoClient
from pack64 import unpack64


def search_subsets(client, question, subset_vecs, subset_stats, min_docs=20, top_subsets=5, top_reviews=5):
    
    # Get question vector
    q_doc = client.post_data('docs/vectors',json.dumps([{'text':question}]),content_type='application/json')[0]
    q_vec = q_doc['vector']
    
    # Get sentiment vectors
    topics = client.get('topics')
    pos = [unpack64(t['vector']) for t in topics if t['name'] == 'Positive'][0]
    neg = [unpack64(t['vector']) for t in topics if t['name'] == 'Negative'][0]
    
    # Get best subset
    match_scores = np.dot(subset_vecs,unpack64(q_vec))
    match_indexes = np.argsort(match_scores)[::-1]
    matches = 0
    results = []
    
    for idx in match_indexes:
        #print('{}:{} -- {}'.format(subset_stats[idx]['subset'],match_scores[idx],subset_stats[idx]['count']))
        if subset_stats[idx]['count'] >= min_docs:
            matching_subset = subset_stats[idx]['subset']
            print(matching_subset)
            example_docs = client.get('docs/search', vector=q_vec, subset=matching_subset)
            
            # Get example verbatim
            vector_matches = []
            for doc in example_docs['search_results']:
                score = 0
                score += np.dot(unpack64(doc[0]['document']['vector']),unpack64(q_vec))
                score += np.dot(unpack64(doc[0]['document']['vector']),pos)
                score -= np.dot(unpack64(doc[0]['document']['vector']),neg)
                
                vector_matches.append(score)
            doc_indexes = np.argsort(vector_matches)[::-1]
            
            for i in range(top_reviews):
                results.append({'car':matching_subset,
                                'score':vector_matches[doc_indexes[i]],
                                'example_review':example_docs['search_results'][doc_indexes[i]][0]['document']['text']})
            matches += 1
            if matches >= top_subsets:
                break
            
    return question, results
