"""
Example uses:
python sentiment_topics.py zoo rjcfz --sentiment pos --terms --verbose
python sentiment_topics.py zoo rjcfz --clusters --n-terms=1000

Approach:
Get top terms in the project equal to n_terms. Average the vectors for the terms which have a
sentiment score assigned to them by SentimentScorer, taking into account that sentiment score and
the term's relevance. Use a sentiment axis created in this way to query the project for similar
terms. Sort the terms using their new sentiment score (how well they match the sentiment axis)
and their relevance. Return the number of top domain-specific terms equal to n_results.

This code does not depend on lumi_science, but instead uses copies of SentimentScorer and
ClusterTree.
"""
import click
from luminoso_api import LuminosoClient
from pack64 import unpack64, pack64
from scipy.stats import hmean
from sentiment import SentimentScorer
from tree_clustering import ClusterTree


class SentimentTopics:
    def __init__(self, client, n_terms, n_results):
        self.client = client
        self.project_terms = self._get_project_terms(n_terms)
        self.axes = {}
        self.domain_sentiment_terms = {}
        self.n_results = n_results

    def _get_project_terms(self, n_terms):
        """
        Get the top n terms in the project. Assign a sentiment score (orig-sentiment-score) to each
        term according to the sentiment scorer. Unpack each term's vector and overwrite its
        'vector' field.
        """
        language = self.client.get(fields=['language'])['language']
        sentiment_scorer = SentimentScorer(language)
        terms = self.client.get('terms', limit=n_terms)
        terms = [term for term in terms if term['vector']]
        for term in terms:
            term['vector'] = unpack64(term['vector'])
            term['orig-sentiment-score'] = sentiment_scorer.term_sentiment(term['term'])
        return terms

    def _get_known_sentiment_terms(self, sentiment):
        """
        Get the terms from which a sentiment axis will be built. For positive sentiment,
        these are the terms with the original sentiment score (assigned by SentimentScorer) of
        over 0. For negative sentiment, these are terms with the original sentiment score of less
        than 0.
        """
        if sentiment == 'pos':
            return [term for term in self.project_terms if term['orig-sentiment-score'] > 0]
        elif sentiment == 'neg':
            return [term for term in self.project_terms if term['orig-sentiment-score'] < 0]

    def _get_sent_axis(self, sentiment):
        """
        Return the sentiment axis for a specified sentiment. The sentiment axis is computed using
        the terms in the project that have a sentiment score assigned to them by SentimentScorer
        (orig-sentiment-score). Their vectors are averaged, weighted by the orig-sentiment-score
        as well as the relevance to the project (score).
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

    def get_domain_sentiment_terms(self, sentiment):
        """
        Get the terms in the project that best match the sentiment axis. Each term's new
        sentiment score (as opposed to the original sentiment score, which is assigned by
        SentimentScorer) is the matching strength returned by the search terms endpoint.
        """
        if sentiment in self.domain_sentiment_terms:
            return self.domain_sentiment_terms[sentiment]

        axis = self._get_sent_axis(sentiment)
        terms = self.client.get(
            'terms/search', vector=pack64(axis),
            limit=self.n_results if self.n_results > 200 else 200)['search_results']

        sentiment_terms = []
        for term, matching_strength in terms:
            term['new-sentiment-score'] = matching_strength
            term['vector'] = unpack64(term['vector'])
            sentiment_terms.append(term)

        self.domain_sentiment_terms[sentiment] = sentiment_terms
        return self.domain_sentiment_terms[sentiment]

    def sorted_sentiment_terms(self, sentiment):
        """
        Get the project terms that best match the specified sentiment axis. Normalize their
        sentiment and relevance scores. Return the list of top n_results domain-specific terms,
        sorted by the harmonic mean of their sentiment and relevance scores.
        """
        sentiment_terms = self.get_domain_sentiment_terms(sentiment)
        self._normalize_scores(sentiment_terms)
        return sorted(sentiment_terms, key=self._harmonic_mean, reverse=True)[:self.n_results]

    def _normalize_scores(self, terms):
        """
        Normalize sentiment score (new-sentiment-score) and relevance score (score) of each term
        to a value between 0 and 1.
        """
        all_sentiment_scores = [term['new-sentiment-score'] for term in terms]
        all_relevance_scores = [term['score'] for term in terms]
        for term in terms:
            term['norm-new-sent-score'] = self._normalize_score(term['new-sentiment-score'],
                                                                all_sentiment_scores)
            term['norm-relevance-score'] = self._normalize_score(term['score'], all_relevance_scores)

    @staticmethod
    def _normalize_score(score, all_scores):
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
        pos_terms = self.get_domain_sentiment_terms('pos')[:200]
        neg_terms = self.get_domain_sentiment_terms('neg')[:200]
        tree = ClusterTree.from_term_list(pos_terms + neg_terms)
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
@click.option('--api', default='https://analytics.luminoso.com/api/v4/projects')
@click.option('--sentiment', default='pos', help='pos, neg')
@click.option('--terms', is_flag=True, help='Use to get a list of sentiment terms')
@click.option('--clusters', is_flag=True, help='Cluster only the sentiment terms')
@click.option('--n-terms', default=500, help='Number of project terms among which to find terms '
                                              'with original sentiment score. This list will be '
                                              'used to create a sentiment axis.')
@click.option('--n-results', default=30, help='Number of results to show')
@click.option('--n-clusters', default=7, help='Show clusters')
@click.option('--verbose', '-v', is_flag=True, help='Show details about the results')
def main(account_id, project_id, api, sentiment, terms, clusters, n_terms, n_results, n_clusters,
         verbose):
    client = LuminosoClient.connect('{}/{}/{}'.format(api, account_id, project_id))
    sentiment_topics = SentimentTopics(client, n_terms, n_results)

    if terms:
        print_sentiment_terms(sentiment_topics.sorted_sentiment_terms(sentiment), verbose=verbose)

    if clusters:
        print_clusters(sentiment_topics.cluster_sentiment_terms(), n_clusters)

if __name__ == '__main__':
    main()
