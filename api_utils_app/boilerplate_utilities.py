from luminoso_api.json_stream import open_json_or_csv_somehow
from luminoso_api import LuminosoClient
from collections import defaultdict
from math import floor, ceil
import argparse
import time
import itertools
import json
import re


SEPARATOR = '¶'
GAP = '___'
DEFAULT_TOKENS_TO_SCAN = 1000000
FILE_PATH = './boilerplate_download/'

def all_docs(cli, fields, redis):
    """ Fetches all documents from a project """
    num_docs = len(cli.get('docs/ids'))
    limit = 10000
    offset = 0
    batch_num = 1
    batches = str(ceil(num_docs/limit))
    docs = []
    while True:
        print('[Batch {0} of {1}] Downloading documents'.format(str(batch_num), batches))
        redis.publish('boilerplate', '[Batch {0} of {1}] Downloading documents'.format(str(batch_num), batches))
        batch = cli.get('docs', limit=limit, offset=offset, doc_fields=fields)
        docs.extend(batch)
        if len(batch) < limit:
            return docs
        batch_num += 1
        offset += limit

def download_docs_and_save_to_file(acct, proj, user, passwd, filepath, redis):
    """
    Downloads all the documents from a project and saves them to a .jsons 
    file to be used as input to the method `run`
    """
    cli = LuminosoClient.connect('/projects/'+acct+'/'+proj, username=user, password=passwd)
    name = cli.get('/')['name']
    docs = all_docs(cli, fields=['text', 'date', 'source', 'subsets', 'language', 'predict'], redis=redis)
    with open(filepath, 'w') as f:
        json.dump(docs, f)
    return filepath, name

def chunks(l, n):
    n = max(1, n)
    return [l[i:i + n] for i in range(0, len(l), n)]

def boilerplate_create_proj(docs_path, name, acct, recalc, username, password):
    with open(FILE_PATH + docs_path, 'r') as f:
        new_docs = [json.loads(line) for line in f]
    lang = new_docs[0]['language']
    cli = LuminosoClient.connect('/projects/'+acct,
                    username=username, password=password)
    cli = cli.change_path(cli.post('/', name='(Boilerplate cleaned) '+name)['project_id'])
    for docs in chunks(new_docs, 10000):
        cli.upload('docs', docs)
    job_num = cli.post('docs/recalculate', language=lang)
    if recalc:
        cli.wait_for(job_num)
    proj = cli.get('/')['project_id']
    return acct, proj

def get_ngrams(seq, window_size):
    """
    Get all ngrams of the given sequence with the given size. The items in
    the list returned will be of the form ((startpos, endpos), ngram).
    """
    return [((i, i + window_size), seq[i:(i + window_size)])
            for i in range(len(seq) - window_size + 1)]

class SpaceSplittingReader:
    '''
    A class that tokenizes text in a very simple way for boilerplate detection.
    For backwards compatibility, this emulates the TextReader interface from
    lumi_science, although it only provides the text_to_token_triples() method.
    '''
    # This is the stem used for newlines.
    HARDLINE_REPLACEMENT = '^^^'
    # This regular expression matches any single newline character or any
    # sequence of one or more non-whitespace characters.
    WORD_RE = re.compile(r'\S+|\n')

    def normalize(self, word):
        '''
        A method to normalize a word for the purposes of boilerplate detection.
        If the word is a newline, it returns "^^^"; otherwise it merely case-
        folds the word.
        '''
        return self.HARDLINE_REPLACEMENT if word == '\n' else word.casefold()

    def text_to_token_triples(self, text):
        '''
        Split the text at spaces and return the words and newlines, normalized.
        For compatibility, this returns a list of (stem, tag, endpoints)
        triples, but because no actual stemmer is run the tag is always None.
        '''
        return [(self.normalize(m.group()), None, (m.start(), m.end()))
                for m in self.WORD_RE.finditer(text)]

class BPDetector(object):
    def __init__(self, window_size=7, bp_replacement=SEPARATOR,
                 threshold=6, use_gaps=True):
        """
        A BPDetector is an object designed to go through large amounts of text
        and replace the repeated phrases with a single word indicating a
        stock phrase. Its initialization arguments are:
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
        self.reader_name = 'ssr'
        self.reader = SpaceSplittingReader()
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

    def train(self, docs, redis, tokens_to_scan=DEFAULT_TOKENS_TO_SCAN, verbose=False):
        """
        Scan through a sequence of documents, counting their n-grams of length
        `self.window_size`, including versions with gaps in them if appropriate.
        The ones that occur often enough will go into the boilerplate set.
        """
        ###NOTE: If there are fewer than DEFAULT_TOKENS_TO_SCAN tokens in the corpus,
        #        then the progress meter will be off.
        n_tokens = 0
        prev_proportion = 0
        for doc in docs:
            n_tokens += self.collect_ngrams_from_doc(doc)
            if n_tokens >= tokens_to_scan:
                break
            if verbose:
                proportion = n_tokens * 100 // tokens_to_scan
                if proportion > prev_proportion:
                    print('[%d%%] Collecting ngrams' % proportion, end='\r')
                    redis.publish('boilerplate', '[%d%%] Collecting ngrams' % proportion)
                    prev_proportion = proportion

        if verbose:
            print('[100%] Collecting ngrams')
            redis.publish('boilerplate', '[100%] Collecting ngrams')
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

        for (startpos, endpos), ngram in get_ngrams(token_triples, self.window_size):
            # Make a tuple of the normalized forms of the words in this n-gram.
            stems = tuple(token[0] for token in ngram)

            # Don't count most n-grams that have a SEPARATOR in the middle.
            # The separator should appear on the edges of n-grams, so that it
            # separates.
            #
            # However, we'd like to be able to match sequences of separated
            # pieces that form a larger mass of boilerplate. So if an edge of
            # the n-gram is already a separator, or the start or end of the
            # text, then we allow other separators as well.
            separator_ok = (
                startpos == 0 or endpos == len(token_triples)
                or stems[0] == SEPARATOR or stems[-1] == SEPARATOR
            )
            if separator_ok or SEPARATOR not in stems[1:-1]:
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

    def boilerplate_match(self, words):
        """
        Determine whether a sequence of words matches a known boilerplate
        n-gram, taking versions with gaps into account. Return the sequence
        that it matched.
        """
        if words in self.boilerplate:
            return words
        if self.use_gaps:
            for gap_slot in range(1, len(words) - 1):
                gapped = words[:gap_slot] + (GAP,) + words[gap_slot + 1:]
                if gapped in self.boilerplate:
                    return gapped
        return False

    def merge_boilerplate_spans(self, words):
        """
        Find n-grams of this list of tokenized words that match known
        boilerplate sequences, and combine them into possibly longer sequences.
        Returns tuples of ((start, end), seq), indicating the sequences that
        were matched (with gaps indicated) and their indices in the list.
        """
        prev_endpoint = -1
        prev_startpoint = 0
        window_size = self.window_size
        boilerplate_spans = []

        # Look for all the n-grams that match something in the boilerplate_set,
        # and merge them into spans in boilerplate_spans.
        for (startpoint, endpoint), these_tokens in get_ngrams(words, window_size):
            match_seq = self.boilerplate_match(these_tokens)
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

        return boilerplate_spans

    def remove_boilerplate(self, doc):
        """
        Transform a document in place, removing sequences of boilerplate from
        its text.
        Return the spans of character indices that were removed.
        """
        # Remember what the original text was
        doc['original_text'] = doc['text']

        # Make some attributes into locals for efficiency
        token_triples = (doc.get('bp_tokens') or doc.get('tokens') or
                         self.reader.text_to_token_triples(doc['text']))
        tokenwords = tuple(triple[0] for triple in token_triples)
        startpoint = 0

        # boilerplate_spans is a list of pairs, each containing:
        #   - An inner pair of (start location, end location)
        #   - A tuple of the words or gaps that were matched
        boilerplate_spans = self.merge_boilerplate_spans(tokenwords)

        # Iterate through the spans backwards (to preserve indices), removing
        # text that meets the criteria for boilerplate.
        removed_spans = []
        for (startpoint, endpoint), token_seq in boilerplate_spans[::-1]:
            tstart = token_triples[startpoint][2][0]
            tend = token_triples[endpoint - 1][2][1]
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

    def handle_docs(self, docs, output, redis, print_every_x, verbose=False):
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
                redis.publish('boilerplate', '[Printing sample documents]')
                print(json.dumps(doc, ensure_ascii=False), file=out)
                if verbose and count % print_every_x == 0:
                    text_to_show = doc['original_text']
                    for start, end in removed_spans:
                        text_to_show = (
                            text_to_show[:start]
                            + highlight(text_to_show[start:end])
                            + text_to_show[end:]
                        )
                    redis.publish('boilerplate', 'Document %d: %s <br><br>' % (count, text_to_show))

    def run(self, acct, proj, user, passwd, redis,
            sample_docs=10, train=False, output_ngrams=None,
            verbose=False, tokens_to_scan=DEFAULT_TOKENS_TO_SCAN):
        """
        Run a sequence of operations for fixing boilerplate in a file.
        - If `train` is True, learn boilerplate by reading `tokens_to_scan` tokens
          from the input. (Otherwise, the boilerplate ngrams must be set some
          other way, such as by loading them from a file.)
        - If `output_ngrams` is set, save the ngrams as the given filename.
        - Iterate through the input file, removing boilerplate and writing the
          results to an output file.
        """
        time_ms = str(time.time()).replace('.','')
        input_f = FILE_PATH+time_ms+'_input.json'
        output_f = time_ms+'_output.json'
        input, name = download_docs_and_save_to_file(acct, proj, user, passwd, input_f, redis)
        docs = list(open_json_or_csv_somehow(input))
        print_every_x = floor(len(docs)/sample_docs)
        if train:
            docs, train_docs = itertools.tee(docs)
            self.train(train_docs, redis=redis, tokens_to_scan=tokens_to_scan, verbose=verbose)
        if output_ngrams:
            self.save_data(output_ngrams)

        if not self.counts:
            raise RuntimeError("No boilerplate data has been loaded.")

        self.handle_docs(docs, FILE_PATH+output_f, print_every_x=print_every_x, redis=redis, verbose=verbose)
        return output_f, name, acct

def add_gap(words):
    """
    Given a sequence of words, iterate through all the possibilities of
    replacing one of those words with the GAP value.
    """
    for gap_slot in range(1, len(words) - 1):
        gapped = words[:gap_slot] + (GAP,) + words[gap_slot + 1:]
        yield gapped, words[gap_slot]

def highlight(text):
    """
    Wrap text in an "ANSI escape" that makes it display in red.
    Future work might involve outputting results in HTML so we can show them
    on a Web page.
    """
    return '<span style="color:red">{%s}</span>' % text