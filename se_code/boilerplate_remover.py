from lumi_science.text_readers import get_reader
from luminoso_api.json_stream import open_json_or_csv_somehow
from collections import defaultdict
from operator import itemgetter
import argparse
import time
import itertools
import codecs
import json
import sys
import re


SEPARATOR = '¶'
GAP = '___'
DEFAULT_TOKENS_TO_SCAN = 1000000


class BPDetector(object):
    def __init__(self, reader='ssr', window_size=7, bp_replacement=SEPARATOR,
                 threshold=6, use_gaps=True):
        """
        A BPDetector is an object designed to go through large amounts of text
        and replace the repeated phrases with a single word indicating a
        stock phrase. Its initialization arguments are:

            * reader

                The text reader used to break the sentence apart into words. By
                default, it uses the SpaceSplittingReader defined for exactly
                this purpose; the name of any other reader can be specified
                instead.

            * window_size

                The minimum number of words a phrase has to have in order to be
                boilerplate. Defaults to 7. If this is set too high, it may miss
                things, but if it's set too low, it'll catch common phrases
                like "this is" or "he did not" or the like.

            * bp_replacement

                The text to use for the stock phrase replacement, which may
                contain "%d" where the number for the phrase should go. By
                default, this is the reader's hard punctuation marker.

            * use_gaps

                Whether to allow boilerplate phrases to contain gaps: a single
                word out of each sequence of `window_size` boilerplate words
                will be allowed to vary instead of matching exactly, but phrases
                with gaps will have to match more often.

            * threshold

                The number of times a phrase has to appear in the tokens that are
                scanned to be considered boilerplate.

        Another way to create a BPDetector is to load it from a JSON file with
        `BPDetector.load_data(filename)`.
        """
        self.reader_name = reader
        self.reader = get_reader(reader)
        self.counts = defaultdict(float)
        self.gap_fillers = defaultdict(set)
        self.boilerplate = set()
        self.window_size = window_size
        self.use_gaps = use_gaps
        if bp_replacement is None:
            try:
                bp_replacement = reader.HARD_PUNCT[0]
            except (AttributeError, IndexError):
                bp_replacement = '¶'
        self.bp_replacement = bp_replacement
        self.threshold = threshold

    def train(self, docs, tokens_to_scan=DEFAULT_TOKENS_TO_SCAN, verbose=False):
        """
        Scan through a sequence of documents, counting their n-grams of length
        `self.window_size`, including versions with gaps in them if appropriate.
        The ones that occur often enough will go into the boilerplate set.
        """
        n_tokens = 0
        prev_proportion = 0
        for doc in docs:
            n_tokens += self.collect_ngrams_from_doc(doc)
            if n_tokens >= tokens_to_scan:
                if verbose:
                    print('[100%] Collecting ngrams.')
                break
            if verbose:
                proportion = n_tokens * 100 // tokens_to_scan
                if proportion > prev_proportion:
                    print('[%d%%] Collecting ngrams.' % proportion, end='\r')
                    prev_proportion = proportion

        self._find_bp_in_ngrams()

    def collect_ngrams_from_doc(self, doc):
        """
        Count the ngrams from a single document. Return the number of tokens
        that were read.
        """
        doc['bp_tokens'] = (doc.get('tokens') or
                            self.reader.text_to_token_triples(doc['text']))
        token_triples = doc['bp_tokens']

        if len(token_triples) < self.window_size:
            # We can't read any n-grams from this document.
            return 0

        for left in range(len(token_triples)):
            right = left + self.window_size
            if right > len(token_triples):
                continue

            # Make a tuple of the normalized forms of the words in this n-gram.
            stems = tuple(token_triples[i][0] for i in range(left, right))
            self.counts[stems] += 1.

            # Also add versions with 'gaps' if requested, but at half the
            # weight.
            if self.use_gaps:
                for gapped, filler in add_gap(stems):
                    self.counts[gapped] += 0.5
                    self.gap_fillers[gapped].add(filler)

        return len(token_triples)

    def _find_bp_in_ngrams(self):
        """
        Scan through the counted n-grams to make a set of the ones that may
        be boilerplate.
        """
        self.boilerplate = set(bp for bp, count in self.counts.items()
                               if count >= self.threshold)

    def boilerplate_match(self, bset, words):
        """
        Determine whether a sequence of words matches a known boilerplate
        n-gram, taking versions with gaps into account. Return the sequence
        that it matched.
        """
        if words in bset:
            return words
        if self.use_gaps:
            for gap_slot in range(len(words)):
                gapped = words[:gap_slot] + (GAP,) + words[gap_slot + 1:]
                if gapped in bset:
                    return gapped
        return False

    def remove_boilerplate(self, doc):
        """
        Transform a document in place, removing sequences of boilerplate from
        its text.

        Return the spans of character indices that were removed.
        """
        # Remember what the original text was
        doc['original_text'] = doc['text']

        # Make some attributes into locals for efficiency
        boilerplate_set = self.boilerplate
        window_size = self.window_size
        overlapping_ngrams = []
        token_triples = (doc.get('bp_tokens') or doc.get('tokens') or
                         self.reader.text_to_token_triples(doc['text']))
        tokenwords = tuple(triple[0] for triple in token_triples)
        startpoint = 0

        # boilerplate_spans is a list of pairs, each containing:
        #   - An inner pair of (start location, end location)
        #   - A tuple of the words or gaps that were matched
        boilerplate_spans = []
        prev_endpoint = -1
        prev_startpoint = 0

        # Look for all the n-grams that match something in the boilerplate_set,
        # and merge them into spans in boilerplate_spans.
        for startpoint in range(len(token_triples) - window_size + 1):
            endpoint = startpoint + window_size
            these_tokens = tokenwords[startpoint:endpoint]
            match_seq = self.boilerplate_match(boilerplate_set, these_tokens)
            if match_seq:
                if startpoint > prev_endpoint:
                    # If it's separate from the previous span, add it as a new span.
                    boilerplate_spans.append(((startpoint, endpoint), match_seq))
                else:
                    # If it overlaps with or touches the previous span, extend that
                    # span.
                    prev_span, prev_seq = boilerplate_spans[-1]
                    # 'overhang' is the amount we need to extend the span by, which
                    # tells us how many words to add from the newly matched sequence.
                    overhang = startpoint - prev_startpoint
                    updated_seq = prev_seq + match_seq[-overhang:]

                    # Replace the most recent span in the list with the merged span.
                    boilerplate_spans[-1] = ((prev_span[0], endpoint), updated_seq)
                prev_startpoint = startpoint
                prev_endpoint = endpoint

        # Iterate through the spans backwards (to preserve indices), removing
        # text that meets the criteria for boilerplate.
        removed_spans = []
        for span, token_seq in boilerplate_spans[::-1]:
            startpoint, endpoint = span
            # We'll trim the boilerplate span according to various criteria.
            # We repeat this loop for as long as something is being trimmed.
            while startpoint + window_size < endpoint:
                # If the sequence starts or ends with a gap, shorten the
                # sequence to exclude the gap.
                if token_seq[0] in GAP:
                    token_seq = token_seq[1:]
                    startpoint += 1
                elif token_seq[-1] == GAP:
                    token_seq = token_seq[:-1]
                    endpoint -= 1
                # We'd prefer not to absorb unrelated text that appears after
                # the boilerplate. One case where this would happen is when
                # we have a lot of boilerplate, then hard punctuation
                # (the SEPARATOR), then the start of a new sentence that
                # uses common words.
                #
                # The way we correct for this is to ensure that SEPARATOR
                # does not occur too close to the beginning or end of the
                # token sequence, unless it is unavoidable because the token
                # sequence is at the beginning or end of the text overall.
                elif SEPARATOR in token_seq[-window_size:-1] and endpoint < len(token_triples):
                    # There's a separator too close to the end. Look for the
                    # innermost separator and trim the sequence to end there.
                    trim_loc = -window_size
                    while token_seq[trim_loc] != SEPARATOR:
                        trim_loc += 1
                        assert trim_loc < 0
                    token_seq = token_seq[:trim_loc]
                    endpoint += trim_loc
                elif SEPARATOR in token_seq[1:window_size] and startpoint > 0:
                    # There's a separator too close to the end. Look for the
                    # innermost separator and trim the sequence to start there.
                    trim_loc = window_size
                    while token_seq[trim_loc - 1] != SEPARATOR:
                        trim_loc -= 1
                        assert trim_loc > 0
                    token_seq = token_seq[trim_loc:]
                    startpoint += trim_loc
                else:
                    # None of these criteria matched, so we're done trimming.
                    break

            # If the sequence is too short after trimming, it's not actually
            # boilerplate. Otherwise, it's ready to be removed.
            if startpoint + window_size < endpoint:
                tstart = token_triples[startpoint][2][0]
                tend = token_triples[endpoint - 1][2][1]
                removed = doc['text'][tstart:tend]
                removed_spans.append((tstart, tend))
                doc['text'] = doc['text'][:tstart] + self.bp_replacement + doc['text'][tend:]

        if 'bp_tokens' in doc:
            del doc['bp_tokens']
        return removed_spans

    def save_data(self, output_filename, ngram_threshold=3):
        """
        Store the n-grams and the configuration of this BPDetector in a JSON
        file, which can be loaded later to re-use it on more text.
        """
        ngram_threshold = min(ngram_threshold, self.threshold)
        options = {
            'reader': self.reader_name,
            'window_size': self.window_size,
            'threshold': self.threshold,
            'use_gaps': self.use_gaps
        }
        ngrams = [
            [list(key), value]
            for (key, value) in self.counts.items()
            if value >= ngram_threshold
        ]
        data = dict(options=options, ngrams=ngrams)
        with open(output_filename, 'w', encoding='utf-8') as out:
            json.dump(data, out, ensure_ascii=False)

    @classmethod
    def load_data(cls, filename):
        """
        Create a BPDetector by loading its data from a JSON file.
        """
        obj = cls()
        with open(filename) as file:
            data = json.load(file)
            obj = cls(
                reader=data['options']['reader'],
                window_size=data['options']['window_size'],
                threshold=data['options']['threshold'],
                use_gaps=data['options']['use_gaps']
            )
            obj.counts = {
                tuple(key): value
                for (key, value) in data['ngrams']
            }
        obj._find_bp_in_ngrams()
        return obj

    def handle_docs(self, docs, output, verbose=False):
        """
        Remove boilerplate from a sequence of documents, modifying them
        in place. If verbose=True, every 10000th document will be displayed
        with boilerplate highlighted.
        """
        count = 0
        with open(output, 'w', encoding='utf-8') as out:
            for doc in docs:
                count += 1
                removed_spans = self.remove_boilerplate(doc)
                print(json.dumps(doc, ensure_ascii=False), file=out)
                if verbose and count % 10000 == 0:
                    text_to_show = doc['original_text']
                    for span in removed_spans:
                        start, end = span
                        text_to_show = (
                            text_to_show[:start]
                            + highlight(text_to_show[start:end])
                            + text_to_show[end:]
                        )
                    print('Document %d: %s' % (count, text_to_show))

    def run(self, input, output, train=False, output_ngrams=None, verbose=False,
            tokens_to_scan=DEFAULT_TOKENS_TO_SCAN):
        """
        Run a sequence of operations for fixing boilerplate in a file.

        - If `train` is True, learn boilerplate by reading `tokens_to_scan` tokens
          from the input. (Otherwise, the boilerplate ngrams must be set some
          other way, such as by loading them from a file.)
        - If `output_ngrams` is set, save the ngrams as the given filename.
        - Iterate through the input file, removing boilerplate and writing the
          results to an output file.
        """
        docs = open_json_or_csv_somehow(input)
        if train:
            docs, train_docs = itertools.tee(docs)
            self.train(train_docs, tokens_to_scan=tokens_to_scan, verbose=verbose)
        if output_ngrams:
            self.save_data(output_ngrams)

        if len(self.boilerplate) == 0 and len(self.counts) == 0:
            raise RuntimeError("No boilerplate data has been loaded.")

        self.handle_docs(docs, output, verbose=verbose)


def add_gap(words):
    """
    Given a sequence of words, iterate through all the possibilities of
    replacing one of those words with the GAP value.
    """
    for gap_slot in range(len(words)):
        gapped = words[:gap_slot] + (GAP,) + words[gap_slot + 1:]
        yield gapped, words[gap_slot]


def highlight(text):
    """
    Wrap text in an "ANSI escape" that makes it display in red.

    Future work might involve outputting results in HTML so we can show them
    on a Web page.
    """
    return '\x1b[91m{%s}\x1b[39m' % text


def main():
    """
    Handle options for using this boilerplate detector at the command line.
    """
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-i', '--input-ngrams', type=str, metavar='FILENAME',
                       help="An existing JSON file of boilerplate ngrams to apply")
    group.add_argument('-o', '--output-ngrams', type=str, metavar='FILENAME',
                       help="A file to write boilerplate ngrams to, so they can be reused")
    parser.add_argument('-t', '--threshold', type=int,
                        help="The minimum number of occurrences of an n-gram to remove")
    parser.add_argument('-e', '--exact', action='store_true',
                        help="Boilerplate matches must be exact (no gaps that may vary)")
    parser.add_argument('-s', '--scan', type=int, default=1000000,
                        help="Number of tokens to learn ngrams from (default 1,000,000)")
    parser.add_argument('-q', '--quiet', action='store_true',
                        help="Don't print progress and examples while running")
    parser.add_argument('input', help='A JSON stream or CSV file of input documents')
    parser.add_argument('output', help='A JSON stream of output documents')
    args = parser.parse_args()

    # Create a BPDetector with either the defaults or its saved values
    if args.input_ngrams:
        bp = BPDetector.load_data(args.input_ngrams)
    else:
        bp = BPDetector()

    # Override the threshold if asked
    if args.threshold:
        bp.threshold = args.threshold

    # Set use_gaps based on the --exact parameter
    bp.use_gaps = (not args.exact)

    docs = open_json_or_csv_somehow(sys.argv[1])
    bp.run(input=args.input, output=args.output,
           output_ngrams=args.output_ngrams,
           train=(not args.input_ngrams),
           tokens_to_scan=args.scan,
           verbose=(not args.quiet))


if __name__ == '__main__':
    main()

