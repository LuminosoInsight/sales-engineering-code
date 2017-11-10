"""
Example use:
python sentiment_informed_topics.py zoo vfzct --list-terms --pos --n-results=10 --n-terms=1000 -v
"""

import json

import click
from lumi_science.sentiment import SentimentScorer
from lumi_science.tree_clustering import ClusterTree
from luminoso_api import LuminosoClient
from pack64 import unpack64, pack64
from scipy.stats import hmean


def connect(account_id, project_id):
    # TODO move to v5
    client = LuminosoClient.connect(
        'https://analytics.luminoso.com/api/v4/projects/{}/{}'.format(account_id, project_id))
    return client


def get_project_terms(client, language, n_terms):
    """
    Get the top n terms in the project. 'vector' and 'score' fields of each term will be
    overwritten, so save the original values in 'orig-vector' and 'orig-score'. Assign a
    sentiment to each term.
    """
    sentiment_scorer = SentimentScorer(language)
    terms = client.get('terms', limit=n_terms)
    terms = [term for term in terms if term['vector']]
    for term in terms:
        term['vector'] = term['orig-vector'] = unpack64(term['vector'])
        term['orig-score'] = term['score']
        term['sentiment-score'] = sentiment_scorer.term_sentiment(term['term'])
    return terms


def get_terms_to_include(terms, emotion):
    if emotion == 'pos':
        return [term for term in terms if term['sentiment-score'] > 0]
    elif emotion == 'neg':
        return [term for term in terms if term['sentiment-score'] < 0]
    elif emotion == 'love':
        vocab = ['wonderful|en', 'amaze|en', 'awesome|en', 'fabulous|en', 'fantastic|en']
        return [term for term in terms if term['term'] in vocab]
    elif emotion == 'sadness':
        vocab = ['sad|en', 'upset|en', 'depress|en', 'miserable|en', 'sadly|en', 'cry|en',
                 'sorry|en', 'unhappy|en', 'unfortunate|en', 'depress|en']
        return [term for term in terms if term['term'] in vocab]
    elif emotion == 'anger':
        vocab = ['frustrate|en', 'angry|en', 'mad|en', 'annoy|en', 'bother|en', 'hate|en']
        return [term for term in terms if term['term'] in vocab]
    elif emotion == 'interest':
        # This one is weak
        vocab = ['interest|en', 'intrigue|en', 'curious|en', 'anticipation|en']
        return [term for term in terms if term['term'] in vocab]
    elif emotion == 'fear':
        vocab = ['anxiety|en', 'worry|en', 'nervous|en', 'fear|en', 'dread|en', 'bother|en',
                 'anxious|en', 'scare|en', 'apprehensive|en']
        return [term for term in terms if term['term'] in vocab]


def get_sent_axis(proj_terms, sent):
    """
    Compute a sentiment or emotion axis.
    """
    sent_terms = get_terms_to_include(proj_terms, sent)

    try:
        axis_sum = sum(abs(term['sentiment-score']) for term in sent_terms)
        axis = sent_terms[0]['vector'] * 0
        for term in sent_terms:
            axis += term['orig-vector'] * abs(term['sentiment-score']) * term['score'] / axis_sum
        axis /= (axis.dot(axis)) ** 0.5
        return axis
    except IndexError:
        # TODO handle this more gracefully
        print('Can\'t create the axis')


def list_sorted_sentiment_terms(client, terms, pos_or_neg, n_results):
    """
    Get the project terms that best match the specified sentiment axis (positive or negative). Each
    term is assigned a sentiment score (how well it matched the sentiment axis) and a relevance
    score. The terms are then sorted using these two scores.
    """
    sentiment_terms = get_sentiment_terms(client, terms, pos_or_neg, n_results)

    sentiment_scores = [term['sentiment-matching-score'] for term in sentiment_terms]
    relevance_scores = [term['score'] for term in sentiment_terms]
    norm_scores = []
    for term in sentiment_terms:
        norm_sent_match_score = normalize_score(term['sentiment-matching-score'], sentiment_scores)
        norm_relevance_score = normalize_score(term['score'], relevance_scores)
        norm_scores.append((term['text'], norm_sent_match_score, norm_relevance_score))
    return sort_scores(norm_scores, n_results)


def get_sentiment_terms(client, terms, sent, n_results):
    axis = get_sent_axis(terms, sent)
    sentiment_terms = client.get(
            'terms/search', vectors=json.dumps([pack64(axis)]),
            limit=n_results if n_results > 100 else 100
        )['search_results']

    for term, sentiment_matching_score in sentiment_terms:
        term['sentiment-matching-score'] = sentiment_matching_score
        term['sentiment'] = sent

    return [term[0] for term in sentiment_terms]


def normalize_score(current_score, other_scores):
    """
    Normalize the scores to be between 0 and 1
    """
    return round((current_score - min(other_scores)) / (max(other_scores) - min(other_scores)), 3)


def sort_scores(scores, n_results):
    """
    Sort using the harmonic mean of the sentiment and relevance scores. If either of a term's
    scores is not positive, the score for the term is zero. The secondary sorting key is the sentiment
    score.
    """
    return sorted(scores,
                  key=lambda scores: (hmean([scores[1], scores[2]])
                                      if all([scores[1], scores[2]]) else 0, scores[1]),
                  reverse=True)[:n_results]


def print_sentiment_terms(scores, verbose=False):
    """
    Print the top results. If verbose=True, also print the sentiment score and the relevance score.
    """
    for term, sent_score, relevance_score in scores:
        output = '* {}'.format(term)
        if verbose:
            output += ' [sentiment: {}, relevance: {}]'.format(sent_score, relevance_score)
        print(output)


def get_sentiment_informed_clusters(client, terms, limit=100):
    """
    Cluster the project terms taking the sentiment information into consideration. For each term
    that is similar to either a positive or negative sentiment axis, make it more similar to that
    sentiment axis and increase its score to ensure that it's more important during the
    clustering process.
    """
    pos_terms = get_sentiment_terms(client, terms, 'pos', limit)
    neg_terms = get_sentiment_terms(client, terms, 'neg', limit)
    pos_terms_text = [term['text'] for term in pos_terms]
    neg_terms_text = [term['text'] for term in neg_terms]

    pos_axis = get_sent_axis(proj_terms=terms, sent='pos')
    neg_axis = get_sent_axis(proj_terms=terms, sent='neg')

    # I need to take a
    for term in terms:
        if term['text'] in pos_terms_text or term['sentiment-score'] > 0:
            # term['vector'] = stretch_axis(term['orig-vector'], pos_axis, 1.2)
            term['sentiment'] = 'pos'
            term['score'] = compute_new_relevance_score(term, pos_axis)
        elif term['text'] in neg_terms_text or term['sentiment-score'] < 0:
            # term['vector'] = stretch_axis(term['orig-vector'], neg_axis, 1.2)
            term['score'] = compute_new_relevance_score(term, neg_axis)
            term['sentiment'] = 'neg'

    tree = ClusterTree.from_term_list(terms)
    return tree


def get_baseline_clusters(client, language):
    terms = get_project_terms(client, language, 500)
    return ClusterTree.from_term_list(terms)


def stretch_axis(vec, axis, magnitude=1.):
    """
    Stretch the term vector along the sentiment axis.
    """
    return vec + vec.dot(axis) * axis * magnitude


def compute_new_relevance_score(term, axis):
    """
    Increase the relevance score of the terms that match the sentiment axis.
    """
    return term['orig-score'] * (1. + term['vector'].dot(axis))


def get_cluster_label(subtree_terms):
    score = 0

    for term in subtree_terms:
        try:
            if term['sentiment'] == 'neg':
                score -= 1
            else:
                score += 1
        except KeyError:
            continue
    score = score/len(subtree_terms)
    if score >= 0.6:
        return 'POS'
    elif score <= -0.6:
        return 'NEG'
    return None


def print_clusters(tree, n=7, verbose=False):
    """
    Print the sentiment-informed clusters. If verbose-True, include the change in their relevance
    scores.
    """
    for subtree in tree.flat_cluster_list(n):
        subtree_terms = [term for term in subtree.filtered_termlist[:5]]
        cluster_label = get_cluster_label(subtree_terms)
        if cluster_label:
            print('***{}***'.format(cluster_label))
        print(subtree.term['text'])
        for term in subtree_terms:
            output = '* {} '.format(term['text'])
            if verbose:
                output += '{} ==> {}'.format(round(term['orig-score'], 2), round(term['score'], 2))
            print(output)
        print()


def cluster_sentiment_terms(client, terms):
    # These will be sorted by how well they match a sentiment_axis
    pos_terms = get_sentiment_terms(client, terms, 'pos', 200)
    neg_terms = get_sentiment_terms(client, terms, 'neg', 200)
    sent_terms = pos_terms + neg_terms
    for term in sent_terms:
        term['vector'] = unpack64(term['vector'])
    tree = ClusterTree.from_term_list(sent_terms)
    return tree


@click.command()
@click.argument('account_id')
@click.argument('project_id')
@click.option('--emotion', default='pos')
@click.option('--language', '-l', default='en')
@click.option('--sentiment-terms', is_flag=True, help='Use --list-terms to get a list of'
                                                                 'sentiment terms. Use '
                                                                 '--cluster-terms to get '
                                                                 'sentiment-informed term clusters')
@click.option('--sentiment-informed-clusters', is_flag=True, help='Use to get 28 suggested topics, '
                                                    'with sentiment topic being more prominent')
@click.option('--sentiment-clusters', is_flag=True, help='Cluster only the sentiment terms')
@click.option('--n-terms', default=500, help='Number of project terms to operate on. More terms '
                                             'usually means more accurate results.')
@click.option('--n-results', default=30, help='Number of results to show')
@click.option('--verbose', '-v', is_flag=True, help='Show details about the results')
@click.option('--baseline', is_flag=True, help='Show baseline clusters.')
def main(account_id, project_id, emotion, language, sentiment_terms, sentiment_informed_clusters,
         sentiment_clusters, n_terms, n_results, verbose, baseline):
    client = connect(account_id, project_id)
    terms = get_project_terms(client, language, n_terms)

    if sentiment_terms:
        sentiment_terms = list_sorted_sentiment_terms(client, terms, emotion, n_results)
        print_sentiment_terms(sentiment_terms, verbose=verbose)

    if sentiment_informed_clusters:
        sentiment_informed_clusters = get_sentiment_informed_clusters(client, terms)
        print_clusters(sentiment_informed_clusters, verbose=verbose)

        if baseline:
            print('\n=== BASELINE ===')
            baseline_clusters = get_baseline_clusters(client, language)
            print_clusters(baseline_clusters, verbose=verbose)

    if sentiment_clusters:
        sentiment_terms_clusters = cluster_sentiment_terms(client, terms)
        print_clusters(sentiment_terms_clusters, verbose=False)


if __name__ == '__main__':
    main()
