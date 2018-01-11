import logging
import math
import re

logger = logging.getLogger(__name__)

TAGGED_STEMS_RE = re.compile(r'(\S+)(?:\|[a-z]+)(?:\s|\Z)')


def read_sentiments(filename):
    """
    Read a sentiment file in our approximately-CSV format.
    """
    sentiment = {}
    for line in open(filename, encoding='utf-8'):
        line = line.rstrip()
        if not line or line.startswith('#'):
            continue
        word, sentiment_str = line.split(',', 1)

        # Words in the list can contain spaces between their stems (or even
        # hyphens, if they came from the original AFINN input). We'd like to
        # leave them that way for legibility, but ignore spaces and hyphens
        # when looking up terms. This lets us have the same sentiment for
        # "absent minded" and "absentminded", for example, and lets us handle
        # Japanese the same way as English.
        word = word.replace('-', '').replace(' ', '')
        value = float(sentiment_str)

        if value != 0:
            # If we're adding the same entry twice, keep only the one with the
            # stronger value
            if word not in sentiment or abs(sentiment[word]) < abs(value):
                sentiment[word] = value
    return sentiment


class SentimentScorer:
    """
    An object that looks up sentiment values for a given language.
    """
    def __init__(self, language=None):
        self.language = language
        self.sentiment = {}

        # Read sentiments from the multilingual (emoji) file as well as the
        # language-specific file
        for file_lang in ('mul', self.language):
            filename = 'data/sentiment.%s.txt' % file_lang
            try:
                self.sentiment.update(read_sentiments(filename))
            except FileNotFoundError:
                logger.warn("No sentiment dictionary for language %r", file_lang)

    def term_sentiment(self, term):
        """
        Get the sentiment of a term on a -5 to 5 scale.

        Negated terms will get -1/2 times the sentiment of the term they're
        negating.
        """
        text = ' '.join(TAGGED_STEMS_RE.findall(term))
        words = text.split()
        if not words:
            return 0.

        multiplier = 1
        if words[0] == 'not':
            text = ''.join(words[1:])
            multiplier = -0.5
        else:
            text = ''.join(words)

        text = text.replace('-', '')
        text = text.replace('#', '')
        return self.sentiment.get(text, 0.) * multiplier

    def sentiment_curve(self, values):
        """
        Take the weighted average of the values, favoring stronger sentiments.
        Then use tanh to put the result on a -1 to 1 scale.

        The denominator of the weighted average starts at 1 as a sort of prior
        that biases it against very short documents. A document containing just
        the word "Awesome!" should not be the best possible document.

        This method is public so it can be used by tests.
        """
        weights = [abs(v) for v in values]
        weighted = [v * abs(v) for v in values]
        weighted_avg = sum(weighted) / (1 + sum(weights))
        return math.tanh(weighted_avg)

    def _avg_term_sentiment(self, stems):
        """
        Given a list of terms in Luminoso stem form (such as
        'good|en luck|en'), get an overall sentiment for the list, on a scale
        from -1 to 1.
        """
        values = [self.term_sentiment(stem) for stem in stems]
        return self.sentiment_curve(values)

    def doc_sentiment(self, doc):
        """
        Get the overall sentiment of an analyzed document, on a -1 to 1 scale.
        """
        stems = [triple[0] for triple in doc['terms'] + doc['fragments']]
        return self._avg_term_sentiment(stems)

