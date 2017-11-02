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
        term['sentiment'] = sentiment_scorer.term_sentiment(term['term'])
    return terms


def get_sent_axis(terms, pos):
    """
    Compute either a positive or a negative sentiment axis.
    """

    if pos:
        def is_right_sentiment(term):
            return term['sentiment'] > 0
    else:
        def is_right_sentiment(term):
            return term['sentiment'] < 0

    axis_sum = sum(abs(term['sentiment']) for term in terms if is_right_sentiment(term))
    axis = terms[0]['vector'] * 0
    for term in terms:
        if is_right_sentiment(term):
            axis += term['orig-vector'] * abs(term['sentiment']) * term['score'] / axis_sum
    axis /= (axis.dot(axis)) ** 0.5
    return axis


def get_sentiment_terms(client, terms, pos_or_neg, n_results):
    """
    Get the project terms that best match the specified sentiment axis (positive or negative). Each
    term is assigned a sentiment score (how well it matched the sentiment axis) and a relevance
    score. The terms are then sorted using these two scores.
    """
    axis = get_sent_axis(terms, pos_or_neg)

    sentiment_terms = [
        (term[0]['text'], term[1], term[0]['score'])
        for term in client.get(
            'terms/search', vectors=json.dumps([pack64(axis)]),
            limit=n_results if n_results > 100 else 100
        )['search_results']
        ]
    sentiment_scores = [term[1] for term in sentiment_terms]
    relevance_scores = [term[2] for term in sentiment_terms]
    norm_scores = []
    for text, sent_match_score, relevance_score in sentiment_terms:
        norm_sent_match_score = normalize_score(sent_match_score, sentiment_scores)
        norm_relevance_score = normalize_score(relevance_score, relevance_scores)
        norm_scores.append((text, norm_sent_match_score, norm_relevance_score))
    return sort_scores(norm_scores, n_results)


def normalize_score(current_score, other_scores):
    """
    Normalize the scores to be between 0 and 1
    """
    return round((current_score - min(other_scores)) / (max(other_scores) - min(other_scores)), 3)


def sort_scores(scores, n_results):
    """
    Sort using the harmonic mean of the sentiment and relevance scores. If either of a term's scores
    is not positive, the score for the term is zero. The secondary sorting key is the sentiment
    score.
    """
    return sorted(scores,
                  key=lambda sentiment, relevance: (hmean([sentiment, relevance])
                                                    if all([sentiment, relevance]) else 0,
                                                    sentiment),
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


def get_sentiment_clusters(client, terms, limit=100):
    """
    Cluster the project terms taking the sentiment information into consideration. For each term
    that is similar to either a positive or negative sentiment axis, make it more similar to that
    sentiment axis and increase its score to ensure that it's more important during the
    clustering process.
    """
    pos_terms = [term[0] for term in get_sentiment_terms(client, terms, True, limit)]
    neg_terms = [term[0] for term in get_sentiment_terms(client, terms, False, limit)]

    pos_axis = get_sent_axis(terms=terms, pos=True)
    neg_axis = get_sent_axis(terms=terms, pos=False)

    for term in terms:
        if term['text'] in pos_terms or term['sentiment'] > 0:
            term['vector'] = stretch_axis(term['orig-vector'], pos_axis, 2.)
            term['score'] = compute_new_relevance_score(term, pos_axis)
        elif term['text'] in neg_terms or term['sentiment'] < 0:
            term['vector'] = stretch_axis(term['orig-vector'], neg_axis, 2.)
            term['score'] = compute_new_relevance_score(term, neg_axis)

    tree = ClusterTree.from_term_list(terms)
    return tree


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


def print_clusters(tree, n=7, verbose=False):
    """
    Print the sentiment-informed clusters. If verbose-True, include the change in their relevance
    scores.
    """
    for subtree in tree.flat_cluster_list(n):
        subtree_terms = [term for term in subtree.filtered_termlist[:5]]
        print(subtree.term['text'])
        for term in subtree_terms:
            output = '* {} '.format(term['text'])
            if verbose:
                output += '{} ==> {}'.format(round(term['orig-score'], 2), round(term['score'], 2))
            print(output)


@click.command()
@click.argument('account_id')
@click.argument('project_id')
@click.option('--language', '-l', default='en')
@click.option('--list-terms/--cluster-terms', default=True, help='Use --list-terms to get a list of'
                                                                 'sentiment terms. Use '
                                                                 '--cluster-terms to get '
                                                                 'sentiment-informed term clusters')
@click.option('--pos/--neg', default=True, help='Use --pos for a list of positive sentiment terms '
                                                'and --neg for a list of negative sentiment terms')
@click.option('--n-terms', default=500, help='Number of project terms to operate on. More terms '
                                             'usually means more accurate results.')
@click.option('--n-results', default=30, help='Number of results to show')
@click.option('--verbose', '-v', is_flag=True, help='Show details about the results')
def main(account_id, project_id, language, list_terms, pos, n_terms, n_results, verbose):
    client = connect(account_id, project_id)
    terms = get_project_terms(client, language, n_terms)

    if list_terms:
        sentiment_terms = get_sentiment_terms(client, terms, pos, n_results)
        print_sentiment_terms(sentiment_terms, verbose=verbose)
    else:
        sentiment_clusters = get_sentiment_clusters(client, terms)
        print_clusters(sentiment_clusters, verbose=verbose)


if __name__ == '__main__':
    main()
