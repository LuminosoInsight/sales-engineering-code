"""
Script that posts messages to a Compass project, in an infinite loop, at random
intervals, and in batches of random size. It also purges messages older than one
day to prevent the UI from becoming unresponsive.

Requires an input file as its first argument, followed optionally by a Compass
API url (the default is prod-compass), the project ID and the user name (who
must have write access to the project). If the user's password is not stored in
the COMPASS_PASSWORD environment variable, the user will be prompted for it.

The input file may be in jsons, jsons.gz, or csv format, as long as it has one
document per line with the text of the message in a "text" field.

The random selection of both batch and interval sizes may be influenced or
eliminated altogether by passing an integer range to the "-b" or "-i" switches.
If, for example, you want to post 1 message with every request, use "-b 1-2".
If you want to space requests evenly at 5-second intervals, use "-i 5-6".

Because the script must be runnable on any machine, it uses only packages
available in Python 2.6 or 2.7.
"""
import codecs
from datetime import datetime, timedelta
from getpass import getpass
import gzip
import json
import os
import random
import requests
import sys
import time


# Place to store (and refresh) the auth header for POST requests
HEADERS = None

# Default number of messages to post per request (1-15, inclusive)
BATCH_SIZES = [i for i in range(1,16)]
# Default number of seconds to wait between requests (0-10, inclusive)
INTERVALS = [i for i in range(11)]

DATE_FMT = '%Y-%m-%dT%H:%M:%S'


def _get_password():
    return os.environ.get('COMPASS_PASSWORD') or getpass()
    

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


def _login(url, username, password):
    """
    Return an Authorization header given the 'url', 'username', and `password`.
    """
    user_data = {'username': username, 'password': password}
    resp = requests.post(url, data=user_data)
    if not resp.ok:
        raise RuntimeError('Cannot log in as "%s": %s' % (username, resp.text))

    return {'Authorization': 'Token %s' % resp.json()['token'],
            'Content-Type': 'application/json'}


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
    Add or modify the timestamp field in each dict in the `docs` list with the
    current time (UTC) and return the list.
    """
    for d in docs:
        d['timestamp'] = datetime.utcnow().isoformat()
    return docs

    
def main(args):
    if args.intervals:
        global INTERVALS
        INTERVALS = _parse_range(args.intervals)
        
    if args.batches:
        global BATCH_SIZES
        BATCH_SIZES = _parse_range(args.batches)
        
    # Log in and set the auth header
    global HEADERS
    login_url = args.url + 'login/'
    HEADERS = _login(login_url, args.username, _get_password())

    # Get the documents to iterate over for streaming
    docs = _load_messages(args.input_file)
    messages_url = args.url + 'projects/%s/p/messages/' % args.project_id
    purge_url = args.url + 'projects/%s/p/purge/' % args.project_id
    
    # Stream until interrupted
    interrupted = False
    total = 0
    first = 0
    while not interrupted:
        if first >= len(docs):
            first = 0
        batch_size = random.choice(BATCH_SIZES)
        interval = random.choice(INTERVALS)
        last = min(first + batch_size, len(docs))
        try:
            resp = requests.post(
                messages_url, headers=HEADERS,
                data=json.dumps(_with_timestamps(docs[first:last]))
            )
            if resp.ok:
                total += len(resp.json())
                sys.stdout.write('\rPOSTED %d messages to %s (sleeping %d)' %
                                 (total, args.project_id, interval))
                sys.stdout.flush()

            # Token may have expired: get a fresh one
            elif resp.status_code == 401:
                HEADERS = _login(login_url, args.username, _get_password())
                sys.stdout.write('\nGot new token\n')
            else:
                sys.stderr.write('ERROR: %s' % resp.reason)
        except Exception as e:
            sys.stderr.write('%r\n' % e)
            interrupted = True

        # Delete messages more than a day old
        to_date = (datetime.utcnow() - timedelta(days=1)).strftime(DATE_FMT)
        try:
            requests.delete(
                purge_url + '?to_date=%s' % to_date, headers=HEADERS
            )
        except Exception as e:
            sys.stderr.write(
                '\nCould not purge messages posted before %s: %r' % (to_date, e)
            )

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
        'url', type=str, nargs='?',
        default='https://compass.luminoso.com/api/',
        help='Compass API url, with trailing slash (default: production)'
    )
    parser.add_argument(
        '-u', '--username', type=str, default='evergreen',
        help='username with permissions to post messages (default: evergreen)'
    )
    parser.add_argument(
        '-p', '--project_id', type=str, default='mfjwdkt4',
        help='8-character project ID for project (default: mfjwdkt4)'
    )
    parser.add_argument(
        '-i', '--intervals', type=str, 
        help='a range of integers, used to select a random number of seconds '
             'to sleep between requests'
    )
    parser.add_argument(
        '-b', '--batches', type=str, 
        help='a range of integers, used to select a random number of messages '
             'to post per request'
    )

    main(parser.parse_args())
