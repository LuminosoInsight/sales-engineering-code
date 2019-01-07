import json
import csv
import numpy as np
import argparse

from luminoso_api import LuminosoClient
from pack64 import unpack64


def get_all_docs(client, subset=None):

    '''Pull all docs from project, filtered by subset if specified'''

    docs = []
    offset = 0
    while True:
        if subset:
            new_docs = client.get('docs',
                                  offset=offset,
                                  limit=25000,
                                  subset=subset)
        else:
            new_docs = client.get('docs',
                                  offset=offset,
                                  limit=25000)

        if not new_docs:
            return docs

        docs.extend(new_docs)
        offset += 25000


def top_term_stats(client, num_terms=1000):

    '''Get top terms, measure distribution of related terms'''

    top_terms = client.get('terms', limit=num_terms)
    top_terms = [term for term in top_terms if term['vector'] is not None]

    term_vects = [unpack64(term['vector']) for term in top_terms]

    all_scores = np.dot(term_vects, np.transpose(term_vects))
    dispersion_list = {term['term']: (np.std(scores > .6), term['vector'])
                       for term, scores in zip(top_terms, all_scores)}

    return dispersion_list, top_terms


def get_collocations(term_list):

    '''Get list of collocated terms'''

    collocated_terms = []
    for term in term_list:
        if len(term['term'].split(' ')) > 1:
            collocated_terms.extend(term['term'].split(' '))

    return term_list


def generate_intent_score(term_list, dispersion_list, collocation_list):

    '''Generate heuristic for intent score'''

    for term in term_list:
        term['collocations'] = len([t for t in collocation_list
                                    if t == term['term']])

        term['collocation'] = (len(term['term'].split(' ')) > 1) * 1
        term['averageness'] = term['score']/term['total_doc_count']
        term['dispersion'] = dispersion_list[term['term']][0]
        term['intent_score'] = np.product([(1 + term['collocation']),
                                           term['averageness'],
                                           (1 - term['dispersion']),
                                           (term['collocations'] + 1)])

    term_list.sort(key=lambda k: -k['intent_score'])

    return term_list


def create_intent_pairs(term_list, num_intent_topics=75, intent_threshold=1, add_generic=False):

    '''Generate a list of intent topic definitions'''

    intent_list = []

    term_vects = [unpack64(t['vector'])
                  for t in term_list if t['intent_score'] > intent_threshold]

    for term in term_list[:num_intent_topics]:
        term_similarity = np.dot(unpack64(term['vector']),
                                 np.transpose(term_vects))
        second_terms = [term2
                        for term2, similarity in zip(term_list, term_similarity)
                        if similarity > .6 and similarity < .98]

        # Add generic intents
        if add_generic:
            intent_list.append({'name': 'general-{}'.format(term['text']),
                                        'topic_def': [{'text': term['text']}],
                                        'text': 'general-{}'.format(term['text'])})

        for term2 in second_terms:
                topic_text = '{} {}'.format(term['text'], term2['text'])
                intent_list.append({'name': '{}-{}'.format(
                                     term['text'], term2['text']),
                                    'topic_def': [],
                                    'text': topic_text})
                intent_list[-1]['topic_def'].extend([{'text': term['text']},
                                                     {'text': term2['text']}])
    print('Intent Topics Created:{}'.format(len(intent_list)))
    
    return intent_list


def remove_duplicate_terms(term_list, threshold=.85):

    '''Remove Duplicate Terms'''

    duplicate_terms = []
    term_vects = [unpack64(term['vector']) for term in term_list]
    all_scores = np.dot(term_vects, np.transpose(term_vects))
    num_largest = 3000 + len(term_list)
    indices = (-all_scores).argpartition(num_largest, axis=None)[:num_largest]

    x, y = np.unravel_index(indices, all_scores.shape)
    count = 0
    for x, y in zip(x, y):
        if x < y and all_scores[x, y] >= threshold:
            print('Duplicate Found: {}=={}'.format(term_list[x]['text'],term_list[y]['text']))
            duplicate_terms.append(term_list[y]['term'])
            count += 1

    return [t for t in term_list if t['term'] not in duplicate_terms]


def remove_duplicates(client, intent_list):

    '''Remove Duplicate Intents'''

    intent_topics = [{key_name: doc[key_name]
                      for key_name in ['name', 'topic_def', 'text']}
                     for doc in intent_list]

    intent_docs = client.post_data('docs/vectors',
                                   json.dumps(intent_list),
                                   content_type='application/json')

    doc_vects = [unpack64(doc['vector']) for doc in intent_docs]
    all_scores = np.dot(doc_vects, np.transpose(doc_vects))
    num_largest = 3000 + len(intent_docs)
    indices = (-all_scores).argpartition(num_largest, axis=None)[:num_largest]

    x, y = np.unravel_index(indices, all_scores.shape)
    count = 0
    for x, y in zip(x, y):
        if x < y and all_scores[x, y] >= .998:
            intent_docs[y]['text'] = None
            count += 1

    intent_topics = [t for t, d in zip(intent_topics, intent_docs)
                     if d['text'] is not None]

    return intent_topics


def set_intent_vectors(client, intent_list, threshold=.9):

    '''Create vectors for each intent, return full list'''

    for label in intent_list:
        intent_vectors = []
        for topic_def in label['topic_def']:
            topic = client.get('terms/search', text=topic_def['text'])
            intent_vectors.append(unpack64(topic['search_vector']))

        intent_similarity = min([np.dot(a, b)
                                 for a in intent_vectors
                                 for b in intent_vectors])

        if intent_similarity < threshold or len(label['topic_def']) == 1:
            label['topic_vectors'] = intent_vectors
    intent_list = [a for a in intent_list if 'topic_vectors' in a]

    return intent_list


def doc_search(client, intent_list, all_terms):

    '''Search all documents for examples matching each intent'''

    docs = get_all_docs(client)
    for doc in docs:
        doc_fragments = [t for t, _, _ in doc['fragments']]
        doc_fragments.extend([t for t, _, _ in doc['terms']])
        doc_fragments = set(doc_fragments)
        doc_terms = [term for term in all_terms
                     if term['term'] in doc_fragments]

        doc_term_vects = [unpack64(term['vector']) for term in doc_terms]

        if len(doc_terms) > 0:
            doc['classification'] = []
            for label in intent_list:
                match_terms = np.dot(doc_term_vects,
                                     np.transpose(label['topic_vectors']))
                score_matrix = np.transpose(np.power(match_terms, 3))

                doc['classification'].append(np.mean(np.max(score_matrix,
                                                            axis=1)))

    return docs


def save_doc_search_results(docs, intent_list, threshold=.5):

    '''Save document search results to a file'''

    labels = [intent['name'] for intent in intent_list]
    intents = []
    auto_intents = []

    with open('results.csv', 'w') as file:
        writer = csv.writer(file)
        headers = ['_id', 'text', 'intent', 'score', 'subsets']
        headers.extend(labels)
        writer.writerow(headers)
        count = 0
        for doc in [doc for doc in docs
                    if 'classification' in doc and
                    np.max(doc['classification']) > threshold]:

            row = [doc['_id'], doc['text'],
                   labels[np.argmax(doc['classification'])],
                   np.max(doc['classification']), doc['subsets']]
            row.extend(doc['classification'])
            writer.writerow(row)
            count += 1

            intents.append(doc['subsets'])
            auto_intents.append(np.argmax(doc['classification']))

    return intents, auto_intents


def main(args):

    
    root_url = '/'.join(args.project_url.split('/')[:-4]) + '/api/v4'
    account_id = args.project_url.split('/')[-2]
    project_id = args.project_url.split('/')[-1]
    client = LuminosoClient.connect(url=root_url, username=args.username)
    client = client.change_path('/projects/{}/{}'.format(account_id,
                                                         project_id))

    print('Finding terms...')
    dispersion_list, term_list = top_term_stats(client, args.num_terms)
    collocation_list = get_collocations(term_list)
    term_list = generate_intent_score(term_list,
                                      dispersion_list,
                                      collocation_list)
    term_list = remove_duplicate_terms(term_list, threshold=.85)

    print('Generating intents...')
    intent_list = create_intent_pairs(term_list, num_intent_topics=args.pair_terms, add_generic=args.generic)
    intent_list = remove_duplicates(client, intent_list)
    intent_list = set_intent_vectors(client, intent_list)

    print('Searching documents for intents...')
    doc_search_results = doc_search(client, intent_list, term_list)

    print('Saving results to "results.csv"...')
    save_doc_search_results(doc_search_results, intent_list, threshold=.5)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Create a set of intents (term-pairings) and '
        'associated documents for the purposes of bootstrapping a classifier'
    )
    parser.add_argument(
        'project_url',
        help="The URL of the project to run Intent Discovery on"
        )
    parser.add_argument(
        '-u', '--username',
        help='Username (email) of Luminoso account'
        )
    parser.add_argument(
        '-n', '--num_terms', default=1000,
        help="Number of terms to consider for intent pairing"
        )
    parser.add_argument(
        '-p', '--pair_terms', default=75,
        help="Top terms ranked by intentness to use for pairing"
        )
    parser.add_argument(
        '-t', '--intent_threshold', default=1,
        help='Threshold for determining a term\'s "intentness"'
        )
    parser.add_argument(
        '-g', '--generic', default=False, action='store_true',
        help='Flag to include "generic" intents (single term intents)'
        )
    args = parser.parse_args()
    main(args)
