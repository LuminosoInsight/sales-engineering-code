"""
Script that performs the following in an infinite loop:
- POSTs messages to a Compass project for classification, at random intervals,
  in batches of random size
- Collects messages that are classified with a certain topic
  (default: Unclassified, Other)
- Purges messages on the Compass project that are older than a day
- When enough messages are collected:
  - Purges old messages on the Analytics project
  - POSTs collected messages to the project
  - Trigger a rebuild of the project

Requires an input file as its first argument.

Optionally, the details (API URL, project ID, user name) of both of the Compass
and Analytics projects can be provided.

If either the Compass user or Analytics user's password is not stored in the
COMPASS_PASSWORD or ANALYTICS_PASSWORD environment variable, respectively, the
user will be prompted for it.

The input file may be in jsons, jsons.gz, or csv format, as long as it has one
document per line with the text of the message in a "text" field.

The random selection of both batch and interval sizes may be influenced or
eliminated altogether by passing an integer range to the "-b" or "-i" switches.
If, for example, you want to post 1 message with every request, use "-b 1-2".
If you want to space requests evenly at 5-second intervals, use "-i 5-6".

The -r and -w switches control the behavior of documents being posted to the
Analytics project. The -r switch specifies the amount of documents that should
be collected before attempting to modify the Analytics project. The -w switch
specifies how many documents to retain on the Analytics project in total.

The -t switch specifies a list of topics that determine whether a document is
passed to the Analytics project; if any of the classifiers return a topic on
this list, the document is sent. It requires a list of strings, separated by
spaces. If the topic has a space in it, it can be enclosed in quotation marks.
For example, '-t "topic 1" topic2' is valid.

Because the script must be runnable on any machine, it uses only packages
available in Python 2.6 or 2.7.
"""
import codecs
from datetime import datetime, timedelta
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

# Compass, by way of Arrow, expects this date format to filter messages
DATE_FMT = '%Y-%m-%d %H:%M:%S'
# Analytics, on the other hand, expects this date format
ANALYTICS_DATE_FMT = '%Y-%m-%dT%H:%M:%SZ'

# Terrible hack to circumvent certs errors
VERIFY = True

# Classifications to keep an eye out for
TOPICS = ('UNCLASSIFIED', 'Other')

RECALCULATE_THRESHOLD = 100
WINDOW_SIZE = 500

# See http://wiki.bash-hackers.org/scripting/terminalcodes for a more indepth
# explanation of these terminal codes
CURSOR_UP_ONE = '\x1b[1A'
ERASE_LINE = '\x1b[2K'


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
    except ValueError as e:
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


def _validate_recalc_threshold(threshold):
    if threshold < 50:
        raise RuntimeError('Minimum value for recalculation threshold is 50')


def _validate_window_size(window_size):
    if window_size < 50:
        raise RuntimeError('Minimum window size is 50')
    elif window_size > 25000:
        raise RuntimeError('Maximum window size is 25000')


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


def _with_timestamps(docs):
    """
    Add or modify the timestamp field with the current time (UTC) in each dict
    in the `docs` list with the current time (UTC) and return the list.
    """
    for d in docs:
        d['timestamp'] = datetime.utcnow().isoformat()
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
    resp = requests.post(url, data=user_data, verify=VERIFY)
    if not resp.ok:
        raise RuntimeError('Cannot log in as "%s": %s' % (username, resp.text))

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
        msg = '%s - %s' % (resp.status_code, resp.reason)
        _log_error('ERROR: %s' % msg)
    return False


def _validate_args(args):
    if args.intervals:
        global INTERVALS
        INTERVALS = _parse_range(args.intervals)

    if args.batches:
        global BATCH_SIZES
        BATCH_SIZES = _parse_range(args.batches)

    if args.recalc_threshold:
        global RECALCULATE_THRESHOLD
        _validate_recalc_threshold(args.recalc_threshold)
        RECALCULATE_THRESHOLD = args.recalc_threshold

    if args.window_size:
        global WINDOW_SIZE
        _validate_window_size(args.window_size)
        WINDOW_SIZE = args.window_size

    # Verification already handled by argparse
    global TOPICS
    TOPICS = args.topics


def main(args):
    _validate_args(args)

    # For convenience's sake, define partial functions to preload relevant
    # parameters into response checking, which can potentially re-log in as
    # necessary
    compass_login_url = args.compass_url + 'login/'
    analytics_login_url = args.analytics_url + 'v4/user/login/'

    get_analytics_token = partial(
        _get_analytics_token, args.analytics_username, analytics_login_url
    )
    check_analytics_resp_ok = partial(
        _check_resp_base, **{'token_getter': get_analytics_token}
    )

    get_compass_token = partial(
        _get_compass_token, args.compass_username, compass_login_url
    )
    check_compass_resp_ok = partial(
        _check_resp_base, **{'token_getter': get_compass_token}
    )

    # Log in and set the auth header
    get_analytics_token()
    get_compass_token()

    # Get the documents to iterate over for streaming
    docs = _load_messages(args.input_file)

    # Define API URLs
    compass_classify_url = '{}projects/{}/p/messages/'.format(
        args.compass_url, args.compass_pid
    )
    compass_purge_url = '{}projects/{}/p/purge/'.format(
        args.compass_url, args.compass_pid
    )

    analytics_upload_url = '{}v5/projects/{}/upload/'.format(
        args.analytics_url, args.analytics_pid
    )
    analytics_build_url = '{}v5/projects/{}/build/'.format(
        args.analytics_url, args.analytics_pid
    )
    analytics_documents_url = '{}v4/projects/{}/{}/docs/'.format(
        args.analytics_url, args.analytics_username, args.analytics_pid
    )

    # Stream until interrupted
    interrupted = False
    total = 0
    # To better simulate randomness, we can start from an arbitrary place in
    # the document list
    first = random.randint(1, len(docs))
    unclassified_docs = []
    while not interrupted:
        if first >= len(docs):
            first = 0
        batch_size = random.choice(BATCH_SIZES)
        interval = random.choice(INTERVALS)
        last = min(first + batch_size, len(docs))
        try:
            resp = requests.post(
                compass_classify_url, headers=COMPASS_HEADERS,
                data=json.dumps(_with_timestamps(docs[first:last])),
                verify=VERIFY
            )
            if check_compass_resp_ok(resp):
                total += len(resp.json())

                # This behemoth collects any texts which are classified as
                # 'Unclassified' or 'Other'. Currently, it filters if at least
                # one topic flags as such.
                unclassified_docs.extend([
                    r['text'] for r in resp.json() if any(
                        [t['name'] in TOPICS for t in r['topics']]
                    )
                ])

                # Trim excess documents; retain only the most recent ones
                if len(unclassified_docs) > RECALCULATE_THRESHOLD:
                    unclassified_docs = unclassified_docs[
                        len(unclassified_docs) - RECALCULATE_THRESHOLD:
                    ]

                # The carriage return forces an overwrite over the same line.
                # The weird terminal character actually wipes the line, so we
                # don't leave old data on it.
                sys.stdout.write('\r' + ERASE_LINE)
                sys.stdout.write(('POSTed %d messages to %s;'
                                  ' collected %d unclassified;'
                                  ' sleeping %d') %
                                 (total, args.compass_pid,
                                  len(unclassified_docs), interval))
                sys.stdout.flush()
        except Exception as e:
            _log_error('%r' % e)
            interrupted = True

        # Delete messages more than a day old
        to_date = (datetime.utcnow() - timedelta(days=1)).strftime(DATE_FMT)
        try:
            requests.delete(
                compass_purge_url + '?to_date=%s' % to_date,
                headers=COMPASS_HEADERS,
                verify=VERIFY
            )
        except Exception as e:
            _log_error(
                'Could not purge messages posted before %s: %r' %
                (to_date, e)
            )

        # This duplicates a bit of logic, but it saves the additional
        # indentation
        if len(unclassified_docs) < RECALCULATE_THRESHOLD:
            first = last
            time.sleep(interval)
            continue

        try:
            # Purge older documents from the Analytics project
            delete_doc_ids = []
            resp = requests.get(
                analytics_documents_url,
                headers=ANALYTICS_HEADERS,
                params={'doc_fields': json.dumps(['date', '_id']),
                        'limit': 25000},
                verify=VERIFY
            )
            if check_analytics_resp_ok(resp):
                # Ensure that the window size is maintained on the project
                cutoff = max(
                    0,
                    len(resp.json()['result']) + len(unclassified_docs) - WINDOW_SIZE
                )
                delete_doc_ids.extend(
                    [d['_id'] for d in resp.json()['result']][:cutoff]
                )
            else:
                # For this and subsequent calls to the Analytics project: we
                # need these steps to succeed, so if an intermediate step
                # fails, skip the rest and wait for the next iteration (when
                # the project hopefully comes back up)
                continue

            if delete_doc_ids:
                resp = requests.delete(
                    analytics_documents_url,
                    headers=ANALYTICS_HEADERS,
                    params={'ids': json.dumps(delete_doc_ids)},
                    verify=VERIFY
                )
                if check_analytics_resp_ok(resp):
                    _log('Purged %d messages' % len(delete_doc_ids))
                else:
                    continue

            # POST unclassified docs to the Analytics project
            texts = [{'text': d} for d in unclassified_docs]
            resp = requests.post(
                analytics_upload_url,
                headers=ANALYTICS_HEADERS,
                data=json.dumps({'docs': _with_date_metadata(texts)}),
                verify=VERIFY
            )
            if check_analytics_resp_ok(resp):
                _log('POSTed %d messages to %s' % (len(unclassified_docs),
                                                   args.analytics_pid))
            else:
                continue

            # Trigger a recalculate
            resp = requests.post(
                analytics_build_url,
                headers=ANALYTICS_HEADERS,
                data=json.dumps({}),
                verify=VERIFY
            )
            if check_analytics_resp_ok(resp):
                _log('Recalculate successful')

            # Clearing unclassified_docs should always happen regardless of the
            # result of the recalculate call, since we have already POSTed the
            # documents to the project.
            unclassified_docs = []

        except (ConnectionError, requests.exceptions.ConnectionError) as e:
            # Analytics might be down. This isn't showstopping; we can wait for
            # it to come back up.
            _log_error('%r' % e)
        except Exception as e:
            _log_error('%r' % e)
            interrupted = True

        first = last
        time.sleep(interval)


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
        help='8-character project ID for project (default: gbbj642t)'
    )
    parser.add_argument(
        '-ap', '--analytics_pid', type=str, default='pr6svkfn',
        help='8-character project ID for project (default: pr6svkfn)'
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
        '-r', '--recalc_threshold', type=int, default=201,
        help='how many messages to collect before attempting to rebuild the'
             ' Analytics project (default 201, minimum 50)'
    )
    parser.add_argument(
        '-w', '--window_size', type=int, default=997,
        help='how many messages to retain on the Analytics project (default'
             ' 997, minimum 50, maximum 25000'
    )
    parser.add_argument(
        '-t', '--topics', type=str, nargs='+', default=TOPICS,
        help=''
    )

    main(parser.parse_args())