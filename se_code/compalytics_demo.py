"""
Script that performs the following in an infinite loop:
- POSTs messages to a Compass project for classification, at random intervals,
  in batches of random size
- Collects messages that are classified with a certain topic
  (default: Unclassified, Other)
- When enough messages are collected:
  - Purges old messages on the Analytics project
  - POSTs collected messages to the project
  - Triggers a rebuild of the project

Requires an input file as its first argument. The input file may be in jsons,
jsons.gz, or csv format, as long as it has one document per line with the text
of the message in a "text" field.

Optionally, the details (API URL, project ID, user name) of both of the Compass
and Analytics projects can be provided.

If either the Compass user or Analytics user's password is not stored in the
COMPASS_PASSWORD or ANALYTICS_PASSWORD environment variable, respectively, the
user will be prompted for it.

The random selection of both batch and interval sizes may be influenced or
eliminated altogether by passing an integer range to the "-b" or "-i" switches.
If, for example, you want to post 1 message with every request, use "-b 1-2".
If you want to space requests evenly at 5-second intervals, use "-i 5-6".

The -r and -w switches control the behavior of documents being posted to the
Analytics project. The -r switch specifies the amount of documents that should
be collected before attempting to modify the Analytics project. The -w switch
specifies how many documents to retain on the Analytics project in total.

The -t switch specifies a list of topics that determines whether a document is
passed to the Analytics project; if any of the classifiers returns a topic on
this list, the document is sent. It requires a list of strings, separated by
spaces. If the topic contains a space, it can be enclosed in quotation marks.
For example, '-t "topic 1" topic2' is valid.

Because the script must be runnable on any machine, it uses only packages
available in Python 2.6 or 2.7.
"""
import codecs
from datetime import datetime
from functools import partial
from getpass import getpass
import gzip
import json
import os
import random
import requests
import sys
import time


# Place to store (and refresh) the auth header for POST requests
COMPASS_HEADERS = None
ANALYTICS_HEADERS = None

# Default number of messages to post per request (1-15, inclusive)
BATCH_SIZES = [i for i in range(1, 16)]
# Default number of seconds to wait between requests (0-10, inclusive)
INTERVALS = [i for i in range(11)]

# Analytics expects this date format. Python's datetime.utcnow() is not ISO
# 8601 compliant since it excludes the Z for UTC time, so we specify our own
# format for strftime() that does.
ANALYTICS_DATE_FMT = '%Y-%m-%dT%H:%M:%SZ'

# Classifications to keep an eye out for
TOPICS = ('UNCLASSIFIED', 'Other')

# How many messages to collect before rebuilding the project
REBUILD_THRESHOLD = 201

# How many messages to keep on the Analytics project
WINDOW_SIZE = 997

# See http://wiki.bash-hackers.org/scripting/terminalcodes for a more indepth
# explanation of these terminal codes
CURSOR_UP_ONE = '\x1b[1A'
ERASE_LINE = '\x1b[2K'


class AnalyticsNotReady(Exception):
    pass


# Define some lightweight logging functions to keep terminal footprint short
def _log(s, writer=sys.stdout):
    writer.write('\n' + ERASE_LINE + s)
    writer.write(CURSOR_UP_ONE + '\r')
    writer.flush()


def _log_error(s):
    _log(s, writer=sys.stderr)


def _parse_range(integer_pair):
    """
    Parse the `integer_pair` (of the form "m - n", with or without spaces)
    into a lower and upper bound and return a list of integers over that range,
    suitable for use with random.choice(). Raise a RuntimeError if the pair of
    integers passed in does not consitute a valid range.
    """
    try:
        bounds = [int(i.strip()) for i in integer_pair.split('-', 1)]
    except ValueError:
        raise RuntimeError('Invalid integers in range %r' % integer_pair)

    if bounds[1] < 1:
        raise RuntimeError('Minimum value for second integer in range %r is 1'
                           % integer_pair)
    if bounds[0] >= bounds[1]:
        raise RuntimeError(
            'First integer in range %r must be smaller than second'
            % integer_pair
        )

    return [i for i in range(bounds[0], bounds[1])]


def _validate_window_size(window_size):
    if window_size < 50:
        raise RuntimeError('Minimum window size is 50')
    elif window_size > 25000:
        raise RuntimeError('Maximum window size is 25000')


def _validate_rebuild_threshold(threshold):
    """
    Makes sure the rebuild threshold is a sane number; that is, at least twice
    the minimum number of documents on an Analytics project (50) and crucially,
    not more than the set window size on the project.

    This relies on WINDOW_SIZE having been validated and set beforehand!
    """
    if threshold < 50:
        raise RuntimeError('Minimum value for rebuild threshold is 50')
    if threshold > WINDOW_SIZE:
        raise RuntimeError('Maximum value for rebuild threshold is the window'
                           ' size (currently %d)' % WINDOW_SIZE)


def _load_messages(infile):
    """
    Read the messages from the `infile` and return a list of dicts with just a
    "text" field.
    """
    if infile.endswith('.csv'):
        from csv import DictReader
        with codecs.open(infile, encoding='utf-8') as f:
            reader = DictReader(f)
            return [{'text': row['text']} for row in reader]

    if infile.endswith('.jsons.gz'):
        ifp = gzip.open(infile, 'rt')
    elif infile.endswith('.jsons'):
        ifp = codecs.open(infile, encoding='utf-8')

    docs = [json.loads(line.strip()) for line in ifp]
    ifp.close()
    return docs


def _with_date_metadata(docs):
    """
    Add a timestamp of the current time (UTC) as metadata to each element in
    `docs`, a list of documents being passed to an Analytics project.
    """
    for d in docs:
        d['metadata'] = [{
            'name': 'Creation Date',
            'value': datetime.utcnow().strftime(ANALYTICS_DATE_FMT),
            'type': 'date'
        }]
    return docs


def _get_compass_password():
    return os.environ.get('COMPASS_PASSWORD') or getpass()


def _get_analytics_password():
    return os.environ.get('ANALYTICS_PASSWORD') or getpass()


def _login(url, username, password):
    """
    Return an Authorization header given the 'url', 'username', and `password`.
    """
    user_data = {'username': username, 'password': password}
    resp = requests.post(url, data=user_data)
    if not resp.ok:
        raise RuntimeError('Cannot log in as "%s": %s' % (username, resp.text))

    # Analytics and Compass pass tokens differently
    token = (resp.json().get('token') or
             resp.json().get('result', {}).get('token'))

    return {'Authorization': 'Token %s' % token,
            'Content-Type': 'application/json'}


def _get_compass_token(username, login_url):
    """
    Given a username and login URL, gets and sets the Compass project token.
    """
    global COMPASS_HEADERS
    COMPASS_HEADERS = _login(login_url,
                             username,
                             _get_compass_password())
    _log('Got new Compass token')


def _get_analytics_token(username, login_url):
    """
    Given a username and login URL, gets and sets the Analytics project token.
    """
    global ANALYTICS_HEADERS
    ANALYTICS_HEADERS = _login(login_url,
                               username,
                               _get_analytics_password())
    _log('Got new Analytics token')


def _check_resp_base(resp, token_getter=None):
    """
    This is a base function, meant to be used with functools.partial() to
    preload token_getter, which is a partial derived from _get_compass_token or
    _get_analytics_token (which are themselves preloaded with a username and
    URL).
    """
    if resp.ok:
        return True
    elif resp.status_code == 401:
        token_getter()
    else:
        _log_error('ERROR: %s - %s' % (resp.status_code, resp.reason))
    return False


def _validate_and_set_params(args):
    if args.intervals:
        global INTERVALS
        INTERVALS = _parse_range(args.intervals)

    if args.batches:
        global BATCH_SIZES
        BATCH_SIZES = _parse_range(args.batches)

    if args.window_size:
        global WINDOW_SIZE
        _validate_window_size(args.window_size)
        WINDOW_SIZE = args.window_size

    if args.rebuild_threshold:
        global REBUILD_THRESHOLD
        _validate_rebuild_threshold(args.rebuild_threshold)
        REBUILD_THRESHOLD = args.rebuild_threshold

    # Verification already handled by argparse
    global TOPICS
    TOPICS = args.topics


def main(args):
    _validate_and_set_params(args)

    # For convenience's sake, define partial functions to preload relevant
    # parameters into response checking, which can potentially re-log in as
    # necessary
    compass_login_url = args.compass_url + 'login/'
    analytics_login_url = args.analytics_url + 'v4/user/login/'

    analytics_token_getter = partial(
        _get_analytics_token, args.analytics_username, analytics_login_url
    )
    check_analytics_resp_ok = partial(
        _check_resp_base, **{'token_getter': analytics_token_getter}
    )

    compass_token_getter = partial(
        _get_compass_token, args.compass_username, compass_login_url
    )
    check_compass_resp_ok = partial(
        _check_resp_base, **{'token_getter': compass_token_getter}
    )

    # Log in and set the auth header
    analytics_token_getter()
    compass_token_getter()

    # Get the documents to iterate over for streaming
    docs = _load_messages(args.input_file)

    # Stream until interrupted
    interrupted = False
    total = 0
    # To better simulate randomness, we can start from an arbitrary place in
    # the document list
    first = random.randint(1, len(docs))
    collected_docs = []
    while not interrupted:
        if first >= len(docs):
            first = 0
        batch_size = random.choice(BATCH_SIZES)
        interval = random.choice(INTERVALS)
        last = min(first + batch_size, len(docs))
        try:
            classify_messages(args,
                              docs[first:last],
                              collected_docs,
                              check_compass_resp_ok)

            total += last-first
            # The carriage return forces an overwrite over the same line.
            # The weird terminal character actually wipes the line, so we
            # don't leave old data on it.
            sys.stdout.write('\r' + ERASE_LINE)
            sys.stdout.write(('Classified %d messages against %s;'
                              ' collected %d messages;'
                              ' sleeping %d') %
                             (total, args.compass_pid,
                              len(collected_docs), interval))
            sys.stdout.flush()
        except Exception as e:
            _log_error('%r' % e)
            interrupted = True

        if len(collected_docs) >= REBUILD_THRESHOLD:
            try:
                update_analytics_project(
                    args, collected_docs, check_analytics_resp_ok
                )
                # Clearing collected_docs should always happen here, since at
                # this point we have already POSTed the documents to the
                # project.
                collected_docs = []
            except (AnalyticsNotReady,
                    requests.exceptions.ConnectionError) as e:
                # Analytics isn't responding correctly, for whatever reason.
                # This isn't showstopping; we can wait for it to come back up.
                _log_error('%r' % e)
            except Exception as e:
                _log_error('%r' % e)
                interrupted = True

        first = last
        time.sleep(interval)


def classify_messages(args, docs, collected_docs, check_resp):
    """
    Given:
    - `args` containing Compass information in compass_pid and compass_url
      fields
    - A set of `docs` (a list of dicts containing a 'text' field) to classify
    - A `check_resp` function to check the output of requests made to the
      Compass project (returning True if OK and False otherwise)

    Classifies `docs` against the Compass project, and modifies collected_docs
    _in place_ to include texts that are classified with any of the elements of
    TOPICS.
    """
    # Define API URL of interest
    url = '{}projects/{}/p/classify/'.format(
        args.compass_url, args.compass_pid
    )

    resp = requests.post(url, headers=COMPASS_HEADERS, data=json.dumps(docs))
    if check_resp(resp):
        # This behemoth collects any messages whose classifications
        # include any of the passed-in topics
        collected_docs.extend([
            r['text'] for r in resp.json() if any(
                [t['name'] in TOPICS for t in r['topics']]
            )
        ])

        # Trim excess documents; retain only the most recent ones
        if len(collected_docs) > REBUILD_THRESHOLD:
            collected_docs = collected_docs[
                (len(collected_docs) - REBUILD_THRESHOLD):
            ]


def update_analytics_project(args, collected_docs, check_resp):
    """
    Given:
    - `args` containing Analytics information in analytics_url, analytics_pid
      and analytics_username fields
    - A list of `collected_docs` (strings) to upload to the Analytics project
    - A `check_resp` function to check the output of requests made to the
      Analytics project (returning True if OK and False otherwise)

    Purges old documents from the Analytics project, uploads the contents of
    `collected_docs` to the project and triggers a rebuild on it.

    Raises an AnalyticsNotReady exception if an intermediate step fails.
    """
    # Define API URLs of interest
    upload_url = '{}v5/projects/{}/upload/'.format(
        args.analytics_url, args.analytics_pid
    )
    build_url = '{}v5/projects/{}/build/'.format(
        args.analytics_url, args.analytics_pid
    )
    documents_url = '{}v4/projects/{}/{}/docs/'.format(
        args.analytics_url, args.analytics_username, args.analytics_pid
    )

    # Purge older documents from the Analytics project
    resp = requests.get(
        documents_url,
        headers=ANALYTICS_HEADERS,
        params={'doc_fields': json.dumps(['date', '_id']),
                'limit': 25000}
    )
    if not check_resp:
        # For this and subsequent calls to the Analytics project: we
        # need these steps to succeed, so if an intermediate step
        # fails, skip the rest and wait for the next iteration (when
        # the project hopefully comes back up)
        raise AnalyticsNotReady(
            'Could not retrieve documents from Analytics project'
        )

    # Ensure that the window size is maintained on the project
    cutoff = max(
        0,
        len(resp.json()['result']) + len(collected_docs) - WINDOW_SIZE
    )
    delete_doc_ids = [d['_id'] for d in resp.json()['result']][:cutoff]

    if delete_doc_ids:
        n_docs_deleted = 0
        while n_docs_deleted < len(delete_doc_ids):
            resp = requests.delete(
                documents_url,
                headers=ANALYTICS_HEADERS,
                params={'ids': json.dumps(
                    delete_doc_ids[n_docs_deleted : n_docs_deleted+50]
                )}
            )
            if not check_resp(resp):
                raise AnalyticsNotReady(
                    'Could not delete documents from Analytics project'
                )
            n_docs_deleted += 50

        _log('Purged %d messages from %s' % (len(delete_doc_ids),
                                             args.analytics_pid))
            
    # POST collected docs to the Analytics project
    texts = [{'text': d} for d in collected_docs]
    resp = requests.post(
        upload_url,
        headers=ANALYTICS_HEADERS,
        data=json.dumps({'docs': _with_date_metadata(texts)})
    )
    if not check_resp(resp):
        raise AnalyticsNotReady(
            'Could not upload documents to Analytics project'
        )

    _log('POSTed %d messages to %s' % (len(collected_docs),
                                       args.analytics_pid))

    # Trigger a rebuild
    resp = requests.post(
        build_url,
        headers=ANALYTICS_HEADERS,
        data=json.dumps({})
    )
    if check_resp(resp):
        _log('Rebuilt project %s successfully' % args.analytics_pid)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument(
        'input_file', type=str,
        help='input file, in .jsons, .jsons.gz, or .csv format'
    )
    parser.add_argument(
        '-cu', '--compass_url', type=str, nargs='?',
        default='https://compass-staging.dev.int.luminoso.com/api/',
        help='Compass API url, with trailing slash (default: compass-staging)'
    )
    parser.add_argument(
        '-au', '--analytics_url', type=str, nargs='?',
        default='http://master-staging.int.luminoso.com/api/',
        help='Analytics API url, with trailing slash (default: master-staging)'
    )
    parser.add_argument(
        '-cn', '--compass_username', type=str, default='semi@deciduo.us',
        help=('Compass username with permissions to post messages' +
              ' (default: semi@deciduo.us)')
    )
    parser.add_argument(
        '-an', '--analytics_username', type=str, default='lumi-test',
        help=('Analytics username with permissions to post messages' +
              ' (default: lumi-test)')
    )
    parser.add_argument(
        '-cp', '--compass_pid', type=str, default='gbbj642t',
        help='8-character project ID for Compass project (default: gbbj642t)'
    )
    parser.add_argument(
        '-ap', '--analytics_pid', type=str, default='pr6svkfn',
        help='8-character project ID for Analytics project (default: pr6svkfn)'
    )
    parser.add_argument(
        '-i', '--intervals', type=str,
        help='a range of integers, used to select a random number of seconds'
             ' to sleep between requests'
    )
    parser.add_argument(
        '-b', '--batches', type=str,
        help='a range of integers, used to select a random number of messages'
             ' to post per request'
    )
    parser.add_argument(
        '-r', '--rebuild_threshold', type=int, default=REBUILD_THRESHOLD,
        help='how many messages to collect before attempting to rebuild the'
             ' Analytics project (default: %d, minimum: 50)' % REBUILD_THRESHOLD
    )
    parser.add_argument(
        '-w', '--window_size', type=int, default=WINDOW_SIZE,
        help='how many messages to retain on the Analytics project (default:'
             ' %d, minimum: 50, maximum: 25000)' % WINDOW_SIZE
    )
    parser.add_argument(
        '-t', '--topics', type=str, nargs='+', default=TOPICS,
        help='a space-separated list of topics that messages sent to Analytics'
             ' project must be classified with (default: %s)' % ' '.join(TOPICS)
    )

    main(parser.parse_args())
