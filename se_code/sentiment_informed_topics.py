"""
Example uses:
python sentiment_informed_topics.py zoo rjcfz --emotion pos --sentiment-terms --verbose
python sentiment_informed_topics.py zoo rjcfz --sentiment-informed-clusters --baseline --verbose
python sentiment_informed_topics.py zoo rjcfz --sentiment-clusters --n-terms=1000

"""

import json
import random
from collections import defaultdict

import click
from lumi_science.sentiment import SentimentScorer
from lumi_science.tree_clustering import ClusterTree
from luminoso_api import LuminosoClient
from pack64 import unpack64, pack64
from scipy.stats import hmean
from sklearn.linear_model import LogisticRegression


class SentimentTopics:
    def __init__(self, account_id, project_id, language, n_terms):
        self.client = self._connect(account_id, project_id)
        self.sentiment_scorer = SentimentScorer(language)
        self.n_terms = n_terms
        self.project_terms = self._get_project_terms(self.n_terms)
        self.axes = {}
        self.sentiment_terms = defaultdict(lambda: None)

    @staticmethod
    def _connect(account_id, project_id):
        # TODO move to v5
        client = LuminosoClient.connect(
            'https://analytics.luminoso.com/api/v4/projects/{}/{}'.format(account_id, project_id))
        return client

    def _get_project_terms(self, limit):
        """
        Get the top n terms in the project. 'vector' and 'score' fields of each term will be
        overwritten, so save the original values in 'orig-vector' and 'orig-score'. Assign a
        sentiment to each term.
        """
        terms = self.client.get('terms', limit=limit)
        terms = [term for term in terms if term['vector']]
        for term in terms:
            term['vector'] = term['orig-vector'] = unpack64(term['vector'])
            term['orig-score'] = term['score']
            term['sentiment-score'] = self.sentiment_scorer.term_sentiment(term['term'])
        return terms

    def _get_sent_axis(self, emotion):
        """
        Compute a sentiment or emotion axis.
        """
        if emotion in self.axes:
            return self.axes[emotion]
        else:
            sent_terms = self._get_terms_to_include_in_axis(emotion)
            axis_sum = sum(abs(term['sentiment-score']) for term in sent_terms)
            axis = sent_terms[0]['vector'] * 0
            for term in sent_terms:
                axis += term['orig-vector'] * abs(term['sentiment-score']) * term[
                    'score'] / axis_sum
            axis /= (axis.dot(axis)) ** 0.5
            self.axes[emotion] = axis
            return self.axes[emotion]

    def get_sent_classifier(self):
        SEED = 217
        random.seed(SEED)
        pos_term = self._get_terms_to_include_in_axis('pos')
        neg_term = self._get_terms_to_include_in_axis('neg')

        # get 100 random neither positive nor negative terms
        neither_terms = [term for term in self.project_terms if term['sentiment-score'] == 0]
        random.shuffle(neither_terms,)
        neither_terms = neither_terms[:100]

        X = [term['orig-vector'] for term in pos_term + neg_term + neither_terms]
        y = ['pos' for term in pos_term]
        y.extend(['neg' for term in neg_term])
        y.extend(['neither' for term in neither_terms])
        cls = LogisticRegression()
        cls.fit(X, y)
        print(cls.classes_)

        neg_terms = []
        pos_terms = []
        for term in self.project_terms:
            pred = cls.predict_proba(term['orig-vector'].reshape(1, -1))
            if pred[0][0] > 0.8:
                term['classifier-score'] = pred[0][0]
                neg_terms.append(term)
            if pred[0][2] > 0.8:
                term['classifier-score'] = pred[0][2]
                pos_terms.append(term)

        neg_terms = sorted(neg_terms, key= lambda x: x['classifier-score'], reverse=True)
        pos_terms = sorted(pos_terms, key=lambda x: x['classifier-score'], reverse=True)
        return pos_terms, neg_terms

    @staticmethod
    def _classifier_sorting_function(term):
        """
        If the norm-axis-score and the norm-relevance-score are positive, return their harmonic
        mean. Otherwise, return 0. The secondary sorting key is norm-axis-score.
        """
        scores = (term['norm-classifier-score'], term['norm-relevance-score'])
        if all(scores):  # harmonic mean is defined
            return hmean(scores), term['norm-classifier-score']
        else:
            return 0, term['norm-classifier-score']


    def sorted_classifier_terms(self):
        pos_terms, neg_terms = self.get_sent_classifier()

        pos_relevance_scores = [term['score'] for term in pos_terms]
        neg_relevance_scores = [term['score'] for term in neg_terms]
        pos_classifier_scores = [term['classifier-score'] for term in pos_terms]
        neg_classifier_scores = [term['classifier-score'] for term in neg_terms]
        for term in pos_terms:
            term['norm-relevance-score'] = self._normalize_score(term['score'],
                                                                 pos_relevance_scores)
            term['norm-classifier-score'] = self._normalize_score(term['classifier-score'], pos_classifier_scores)

        for term in neg_terms:
            term['norm-relevance-score'] = self._normalize_score(term['score'],
                                                                 neg_relevance_scores)
            term['norm-classifier-score'] = self._normalize_score(term['classifier-score'], neg_classifier_scores)

        sorted_pos = sorted(pos_terms,
                            key = self._classifier_sorting_function,
                            reverse=True)
        sorted_neg = sorted(neg_terms,
                            key = self._classifier_sorting_function,
                            reverse=True)
        for term in sorted_pos[:30]:
            print(term['text'], term['norm-classifier-score'], term['norm-relevance-score'])

        for term in sorted_neg[:30]:
            print(term['text'], term['norm-classifier-score'], term['norm-relevance-score'])


    def _get_terms_to_include_in_axis(self, emotion):
        """
        Get the terms from which a sentiment/emotion axis will be built. Experimentally, return seed
        terms for emotion axes.
        """
        vocab = []
        if emotion == 'pos':
            return [term for term in self.project_terms if term['sentiment-score'] > 0]
        elif emotion == 'neg':
            return [term for term in self.project_terms if term['sentiment-score'] < 0]

        # Not in the original Plutchik wheel, but present in our data
        elif emotion == 'love':
            vocab = ['wonderful|en', 'amaze|en', 'awesome|en', 'fabulous|en', 'fantastic|en']

        elif emotion == 'sadness':
            vocab = ['sad|en', 'upset|en', 'depress|en', 'miserable|en', 'sadly|en', 'cry|en',
                     'sorry|en', 'unhappy|en', 'unfortunate|en', 'depress|en']

        elif emotion == 'anger':
            vocab = ['frustrate|en', 'angry|en', 'mad|en', 'annoy|en', 'bother|en', 'hate|en']

        elif emotion == 'anticipation':
            vocab = ['interest|en', 'intrigue|en', 'curious|en', 'anticipation|en']

        elif emotion == 'fear':
            vocab = ['anxiety|en', 'worry|en', 'nervous|en', 'fear|en', 'dread|en', 'bother|en',
                     'anxious|en', 'scare|en', 'apprehensive|en']

        elif emotion == 'surprise':
            vocab = []
        return [term for term in self.project_terms if term['term'] in vocab]


    def _get_sentiment_terms(self, emotion, n_results):
        """
        Get the terms in the project that best match the axis of sentiment/emotion passed in sent.
        """
        if self.sentiment_terms[(emotion, n_results)] is not None:
            return self.sentiment_terms[(emotion, n_results)]

        axis = self._get_sent_axis(emotion)
        sentiment_terms = self.client.get(
            'terms/search', vectors=json.dumps([pack64(axis)]),
            limit=n_results if n_results > 100 else 100)['search_results']

        for term, axis_score in sentiment_terms:
            term['axis-score'] = axis_score
            term['sent'] = emotion

        sentiment_terms = [term[0] for term in sentiment_terms]  # drop the axis matching score
        self.sentiment_terms[(emotion, n_results)] = sentiment_terms
        return self.sentiment_terms[(emotion, n_results)]

    def sorted_sentiment_terms(self, emotion, n_results):
        """
        Get the project terms that best match the specified sentiment or emotion axis. Each
        term is assigned a sentiment score (how well it matched the axis) and a relevance score. The
        terms are then sorted using these two scores.
        """
        sentiment_terms = self._get_sentiment_terms(emotion, n_results)

        sentiment_scores = [term['axis-score'] for term in sentiment_terms]
        relevance_scores = [term['score'] for term in sentiment_terms]
        for term in sentiment_terms:
            term['norm-axis-score'] = self._normalize_score(term['axis-score'],
                                                            sentiment_scores)
            term['norm-relevance-score'] = self._normalize_score(term['score'], relevance_scores)
        return self._sort_scores(emotion, n_results)

    def _sort_scores(self, emotion, n_results):
        """
        Sort using the harmonic mean of the sentiment and relevance scores. If either of a term's
        scores is not positive, the score for the term is zero. The secondary sorting key is the
        sentiment score.
        """
        return sorted(self.sentiment_terms[(emotion, n_results)],
                      key=self._sorting_function,
                      reverse=True)[:n_results]

    def sentiment_informed_clusters(self, limit=100, stretch=False):
        """
        Cluster the project terms taking the sentiment information into consideration. For each term
        that is similar to either a positive or negative sentiment axis, make it more similar to
        that sentiment axis and increase its score to ensure that it's more important during the
        clustering process.
        """
        pos_terms = self._get_sentiment_terms('pos', limit)
        neg_terms = self._get_sentiment_terms('neg', limit)
        pos_terms_text = [term['text'] for term in pos_terms]
        neg_terms_text = [term['text'] for term in neg_terms]

        pos_axis = self._get_sent_axis(emotion='pos')
        neg_axis = self._get_sent_axis(emotion='neg')

        for term in self.project_terms:
            if term['text'] in pos_terms_text or term['sentiment-score'] > 0:
                term['sentiment'] = 'pos'
                term['score'] = self._compute_new_relevance_score(term, pos_axis)
                if stretch:
                    term['vector'] = self._stretch_axis(term['orig-vector'], pos_axis, 1.2)
            elif term['text'] in neg_terms_text or term['sentiment-score'] < 0:
                term['score'] = self._compute_new_relevance_score(term, neg_axis)
                term['sentiment'] = 'neg'
                if stretch:
                    term['vector'] = self._stretch_axis(term['orig-vector'], neg_axis, 1.2)

        tree = ClusterTree.from_term_list(self.project_terms)
        return tree

    def baseline_clusters(self):
        """
        Cluster the top 500 terms in the project, for comparison with the new method.
        """
        terms = self._get_project_terms(500)
        return ClusterTree.from_term_list(terms)

    def cluster_sentiment_terms(self):
        """
        Get 200 positive terms and 200 negative terms, and cluster them.
        """
        pos_terms = self._get_sentiment_terms('pos', 200)
        neg_terms = self._get_sentiment_terms('neg', 200)
        sent_terms = pos_terms + neg_terms
        for term in sent_terms:
            term['vector'] = unpack64(term['vector'])
        tree = ClusterTree.from_term_list(sent_terms)
        return tree

    @staticmethod
    def _normalize_score(current_score, other_scores):
        """
        Normalize the scores to be between 0 and 1
        """
        return round((current_score - min(other_scores)) / (max(other_scores) - min(other_scores)),
                     3)

    @staticmethod
    def _stretch_axis(vec, axis, magnitude=1.):
        """
        Stretch the term vector along the sentiment axis.
        """
        return vec + vec.dot(axis) * axis * magnitude

    @staticmethod
    def _compute_new_relevance_score(term, axis):
        """
        Increase the relevance score of the terms that match the sentiment axis.
        """
        return term['orig-score'] * (1. + term['vector'].dot(axis))

    @staticmethod
    def _sorting_function(term):
        """
        If the norm-axis-score and the norm-relevance-score are positive, return their harmonic
        mean. Otherwise, return 0. The secondary sorting key is norm-axis-score.
        """
        scores = (term['norm-axis-score'], term['norm-relevance-score'])
        if all(scores):  # harmonic mean is defined
            return hmean(scores), term['norm-axis-score']
        else:
            return 0, term['norm-axis-score']


def print_clusters(tree, n=7, verbose=False):
    """
    Print the sentiment-informed clusters. If verbose=True, include the change in their relevance
    scores.
    """
    for subtree in tree.flat_cluster_list(n):
        subtree_terms = [term for term in subtree.filtered_termlist[:3]]
        print(subtree.term['text'])
        for term in subtree_terms:
            output = '{} '.format(term['text'])
            if verbose:
                output += '{} ==> {}'.format(round(term['orig-score'], 2), round(term['score'], 2))
            print(output)
        print()


def print_sentiment_terms(terms, verbose=False):
    """
    Print the top sentiment terms. If verbose=True, also print the sentiment score and the
    relevance score.
    """
    for term in terms:
        output = '{}'.format(term['text'])
        if verbose:
            output += '\t{}\t{}'.format(term['norm-axis-score'], term['norm-relevance-score'])
        print(output)


@click.command()
@click.argument('account_id')
@click.argument('project_id')
@click.option('--emotion', default='pos', help='pos, neg, love, sadness, anger, interest, or fear')
@click.option('--language', '-l', default='en')
@click.option('--sentiment-terms', is_flag=True, help='Use to get a list of sentiment terms')
@click.option('--sentiment-informed-clusters', is_flag=True, help='Use to get 28 suggested topics, '
                                                                  'with sentiment topic being more '
                                                                  'prominent')
@click.option('--sentiment-clusters', is_flag=True, help='Cluster only the sentiment terms')
@click.option('--classifier', is_flag=True)
@click.option('--n-terms', default=500, help='Number of project terms to operate on. More terms '
                                             'usually means more accurate results.')
@click.option('--n-results', default=30, help='Number of results to show')
@click.option('--verbose', '-v', is_flag=True, help='Show details about the results')
@click.option('--baseline', is_flag=True, help='Show baseline clusters.')
def main(account_id, project_id, emotion, language, sentiment_terms,
         sentiment_informed_clusters, sentiment_clusters, classifier, n_terms, n_results, verbose,
         baseline):
    sentiment_topics = SentimentTopics(account_id, project_id, language, n_terms)

    if sentiment_terms:
        print_sentiment_terms(sentiment_topics.sorted_sentiment_terms(emotion, n_results),
                              verbose=verbose)

    if sentiment_informed_clusters:
        print_clusters(sentiment_topics.sentiment_informed_clusters(), verbose=verbose)

        if baseline:
            print('\n= BASELINE =')
            print_clusters(sentiment_topics.baseline_clusters(), verbose=False)

    if sentiment_clusters:
        print_clusters(sentiment_topics.cluster_sentiment_terms(), verbose=False)

    if classifier:
        sentiment_topics.sorted_classifier_terms()


if __name__ == '__main__':
    main()
