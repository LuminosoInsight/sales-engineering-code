import json
import numpy as np
from doc_utilities import read_documents,search_documents
from luminoso_api import V5LuminosoClient
from pack64 import unpack64

def calc_metadata_vectors(client,metadata,only_field=None):
    metadata = [m for m in metadata if m['type']!='date']
    
    # get all docs/vectors for each metadata
    for m in metadata:
        # may only need to calculate vectors for one field
        if not only_field or m['name']==only_field:
            # different calculation if there are no 'values'
            if 'values' in m and len(m['values'])>0:
                
                # calculate the overall count for the field
                m['count'] = sum([v['count'] for v in m['values']])

                # get the vector for each field value
                for v in m['values']:
                    if (m['type']=='string'):
                        filter = [{"name": m['name'],"values":[v['value']]}]                    
                    else:
                        filter = [{"name": m['name'],"minimum": v['value'],
                             "maximum": v['value']}]
                    docs = read_documents(client,filter)

                    if (len(docs)>1):
                        v['vector'] = np.mean([np.array([float(v) for v in unpack64(d['vector'])]) for d in docs],axis=0) 
                    else:
                        v['vector'] = np.array([float(v) for v in unpack64(docs[0]['vector'])])

                # if there is more than one document, calculate the weighted mean
                if len(m['values'])>1:
                    m['vector'] = np.average([v['vector'] for v in m['values']],weights=[v['count'] for v in m['values']],axis=0)
                else:
                    m['vector'] = m['values'][0]['vector']

            elif ('minimum' in m):

                # create the filter and add the documents
                filter = [{"name": m['name'],"minimum": m['minimum'],
                 "maximum": m['maximum']}]
                docs = read_documents(client,filter)

                # top level metadata doesn't include number of documents. add it here
                m['count'] = len(docs)

                # get all the vectors and calculate mean
                vects = [[float(v) for v in unpack64(d['vector'])] for d in docs]
                if (len(vects)>1):
                    m['vector'] = np.mean(vects,axis=0)
                else:
                    m['vector'] = vects
                m['values'] = []

    return metadata

def search_subsets(client, questions, metadata_with_vectors = None, field = None, sample_docs = False):

    texts_concept = client.post('vectorize/',
                     texts=questions)
    for t in texts_concept:
        t['unp_vector'] = np.array([float(v) for v in unpack64(t['vector'])])
        
    if not metadata_with_vectors:
        metadata = client.get('/metadata')['result']
        if field:
            metadata = calc_metadata_vectors(client,metadata,only_field=field)
        else:
            metadata = calc_metadata_vectors(client,metadata)
    else:
        metadata = metadata_with_vectors
        
    if not field:
        subset_vects = [m['vector'] for m in metadata]
        subset_names = [m['name'] for m in metadata]
        subset_values = subset_names
        metadata_by_field = {m['name']:m for m in metadata}
        
    else:
        subset_vects = [v['vector'] for m in metadata for v in m['values'] if m['name']==field]
        subset_names = ["{}:{}".format(m['name'],v['value']) for m in metadata if m['name']==field for v in m['values']]
        subset_values = [v['value'] for m in metadata for v in m['values'] if m['name']==field]
    
    if sample_docs:
        subset_docs = []
        for m in subset_values:
            if not field:
                filter = [{"name": m,
                 "values":[m2['value'] for m2 in metadata_by_field[m]['values']],
                 }]
            else:
                filter = [{"name":field, "values":[m]}]
                
            # get all the saved concepts
            concept_selector = {'type': 'top'}
            concepts = client.get('concepts/match_counts', concept_selector=concept_selector,filter=filter)['match_counts']
            # sort out the top concepts
            concepts = sorted(concepts, key = lambda c: (c['match_count']), reverse=True)

            # get list of top 3 concepts to search on
            texts = []
            for c in concepts[:3]:
                texts.extend(c['texts'])
            search_selector = {"texts": texts}
                              
            doc_list = [d['text'] for d in search_documents(client,search_selector,max_docs=3)]
            if (len(doc_list)<3):
                for i in range(len(doc_list),3):
                    doc_list.append("")
            subset_docs.append(doc_list)
            
    result = {}
    for t in texts_concept:
        match_scores = [np.dot(t['unp_vector'],ss_vect,) for ss_vect in subset_vects]
        if sample_docs:
            result[t['text']] = [{'field':name,'score':assoc_score,'docs':samp_docs} for name, assoc_score, samp_docs in zip(subset_names, match_scores, subset_docs)]
        else:
            result[t['text']] = [{'field':name,'score':assoc_score,'docs':['','','']} for name, assoc_score in zip(subset_names, match_scores)]
    
    # sort the results and rank
    result_x = {}
    for k,r in result.items():
        result_x[k] = [{'field':x['field'],'score':x['score'],'doc_0':x['docs'][0],'doc_1':x['docs'][1],'doc_2':x['docs'][2]} for i,x in enumerate(sorted(r, key = lambda c: (c['score']), reverse=True),1)]

    return result_x
                                                                                                     
''' V4 VERSION
def search_subsets(client, question, subset_vecs, subset_list, min_docs=50, top_subsets=5, top_reviews=2, field=None):
    
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

        if subset_list[idx]['count'] >= min_docs:
            matching_subset = subset_list[idx]['subset']
            if field:
                example_docs = client.get('docs/search', vector=q_vec, limit=5000)['search_results']
                example_docs = [e for e in example_docs if e[0]['document']['source'][field] == matching_subset]
            else:
                example_docs = client.get('docs/search', vector=q_vec, subset=matching_subset)['search_results']
            print(example_docs[0])
            # Get example verbatim
            vector_matches = []
            for doc in example_docs:
                score = 0
                score += np.dot(unpack64(doc[0]['document']['vector']),unpack64(q_vec))
                score += np.dot(unpack64(doc[0]['document']['vector']),pos)
                score -= np.dot(unpack64(doc[0]['document']['vector']),neg)
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
'''