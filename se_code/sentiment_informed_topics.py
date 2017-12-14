"""
Example uses:
python sentiment_informed_topics.py zoo rjcfz --sentiment pos --terms --verbose
python sentiment_informed_topics.py zoo rjcfz --clusters --n-terms=1000
"""

import json

import click
from lumi_science.sentiment import SentimentScorer
from lumi_science.tree_clustering import ClusterTree
from luminoso_api import LuminosoClient
from pack64 import unpack64, pack64
from scipy.stats import hmean


class SentimentTopics:
    def __init__(self, account_id, project_id, language, n_terms):
        self.client = self._connect(account_id, project_id)
        self.sentiment_scorer = SentimentScorer(language)
        self.n_terms = n_terms
        self.project_terms = self._get_project_terms()
        self.axes = {}
        self.domain_sentiment_terms = {}

    @staticmethod
    def _connect(account_id, project_id):
        client = LuminosoClient.connect(
            'https://analytics.luminoso.com/api/v4/projects/{}/{}'.format(account_id, project_id))
        return client

    def _get_project_terms(self):
        """
        Get the top n terms in the project. Assign a sentiment score (orig-sentiment-score) to each
        term according to the sentiment scorer. Unpack each term's vector and overwrite its
        'vector' field.
        """
        terms = self.client.get('terms', limit=self.n_terms)
        terms = [term for term in terms if term['vector']]
        for term in terms:
            term['vector'] = unpack64(term['vector'])
            term['orig-sentiment-score'] = self.sentiment_scorer.term_sentiment(term['term'])
        return terms

    def _get_known_sentiment_terms(self, sentiment):
        """
        Get the terms from which a sentiment axis will be built.
        """
        if sentiment == 'pos':
            return [term for term in self.project_terms if term['orig-sentiment-score'] > 0]
        elif sentiment == 'neg':
            return [term for term in self.project_terms if term['orig-sentiment-score'] < 0]

    def _get_sent_axis(self, sentiment):
        """
        Compute the sentiment axis. #TODO more documentation
        """
        if sentiment in self.axes:
            return self.axes[sentiment]

        sent_terms = self._get_known_sentiment_terms(sentiment)
        axis = sent_terms[0]['vector'] * 0
        axis_sum = sum(abs(term['orig-sentiment-score']) for term in sent_terms)
        for term in sent_terms:
            axis += term['vector'] * abs(term['orig-sentiment-score']) * term['score'] / axis_sum
        axis /= (axis.dot(axis)) ** 0.5
        self.axes[sentiment] = axis
        return self.axes[sentiment]

    def get_domain_sentiment_terms(self, sentiment, n_results):
        """
        Get the terms in the project that best match the sentiment axis. Each term's new
        sentiment score (as opposed to the original sentiment score, which is assigned by
        SentimentScorer) is the matching strength returned by the search terms endpoint.
        """
        if (sentiment, n_results) in self.domain_sentiment_terms:
            return self.domain_sentiment_terms[(sentiment, n_results)]

        axis = self._get_sent_axis(sentiment)
        terms = self.client.get(
            'terms/search', vectors=json.dumps([pack64(axis)]),
            limit=n_results if n_results > 100 else 100)['search_results']

        sentiment_terms = []
        for term, matching_strength in terms:
            term['new-sentiment-score'] = matching_strength
            term['vector'] = unpack64(term['vector'])
            sentiment_terms.append(term)

        self.domain_sentiment_terms[(sentiment, n_results)] = sentiment_terms
        return self.domain_sentiment_terms[(sentiment, n_results)]

    def sorted_sentiment_terms(self, sentiment, n_results):
        """
        Get the project terms that best match the specified sentiment axis. Each
        term is assigned a sentiment score (how well it matched the axis) and a relevance score. The
        terms are then sorted using these two scores.
        """
        sentiment_terms = self.get_domain_sentiment_terms(sentiment, n_results)

        all_sentiment_scores = []
        all_relevance_scores = []
        for term in sentiment_terms:
            all_sentiment_scores.append(term['new-sentiment-score'])
            all_relevance_scores.append(term['score'])

        for term in sentiment_terms:
            term['norm-new-sent-score'] = self._normalize(term['new-sentiment-score'],
                                                          all_sentiment_scores)
            term['norm-relevance-score'] = self._normalize(term['score'], all_relevance_scores)

        return sorted(sentiment_terms, key=self._harmonic_mean, reverse=True)[:n_results]

    @staticmethod
    def _normalize(score, all_scores):
        """
        Normalize the scores to be between 0 and 1
        """
        return round((score - min(all_scores)) / (max(all_scores) - min(all_scores)), 3)

    @staticmethod
    def _harmonic_mean(term):
        """
        If the norm-new-sent-score and the norm-relevance-score are positive, return their
        harmonic mean. Otherwise, return 0. The secondary sorting key is norm-new-sent-score.
        """
        scores = (term['norm-new-sent-score'], term['norm-relevance-score'])
        if all(scores):  # harmonic mean is defined
            return hmean(scores), term['norm-new-sent-score']
        else:
            return 0, term['norm-new-sent-score']

    def cluster_sentiment_terms(self):
        """
        Get 200 positive terms and 200 negative terms, and cluster them.
        """
        pos_terms = self.get_domain_sentiment_terms('pos', 200)
        neg_terms = self.get_domain_sentiment_terms('neg', 200)
        sent_terms = pos_terms + neg_terms
        tree = ClusterTree.from_term_list(sent_terms)
        return tree


def print_clusters(tree, n_clusters=7):
    """
    Print the sentiment clusters.
    """
    for subtree in tree.flat_cluster_list(n_clusters):
        subtree_terms = [term for term in subtree.filtered_termlist[:3]]
        print(subtree.term['text'])
        for term in subtree_terms:
            print('{} '.format(term['text']))
        print()


def print_sentiment_terms(terms, verbose=False):
    """
    Print the top sentiment terms. If verbose=True, also print the sentiment score and the
    relevance score.
    """
    for term in terms:
        output = '{}'.format(term['text'])
        if verbose:
            output += '\t{}\t{}'.format(term['norm-new-sent-score'], term['norm-relevance-score'])
        print(output)


@click.command()
@click.argument('account_id')
@click.argument('project_id')
@click.option('--sentiment', default='pos', help='pos, neg')
@click.option('--language', '-l', default='en')
@click.option('--terms', is_flag=True, help='Use to get a list of sentiment terms')
@click.option('--clusters', is_flag=True, help='Cluster only the sentiment terms')
@click.option('--n-terms', default=1000, help='Number of project terms to operate on. More terms '
                                             'usually means more accurate results.')
@click.option('--n-results', default=30, help='Number of results to show')
@click.option('--n-clusters', default=7, help='Show clusters')
@click.option('--verbose', '-v', is_flag=True, help='Show details about the results')
def main(account_id, project_id, sentiment, language, terms, clusters, n_terms, n_results,
         n_clusters, verbose):

    sentiment_topics = SentimentTopics(account_id, project_id, language, n_terms)

    if terms:
        print_sentiment_terms(sentiment_topics.sorted_sentiment_terms(sentiment, n_results),
                              verbose=verbose)

    if clusters:
        print_clusters(sentiment_topics.cluster_sentiment_terms(), n_clusters)

if __name__ == '__main__':
    main()
