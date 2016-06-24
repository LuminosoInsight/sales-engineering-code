import argparse
import csv
import json

from luminoso_api import LuminosoClient

import logging
handler = logging.StreamHandler()
logging.root.addHandler(handler)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

FIELDS = ['text']


def download_docs(project, batch_size=25000):
    """
    Download all of the documents from a given project, given a LuminosoClient
    object pointed at that project.
    """
    docs = []
    batch = []
    logger.info('Beginning download')
    while len(docs) == 0 or len(batch) == batch_size:
        batch = project.get(
            'docs', limit=batch_size, offset=len(docs), doc_fields=FIELDS
        )
        docs += batch
        logger.info('Downloaded %s docs' % len(docs))
    return docs


def write_csv(docs, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for doc in docs:
            writer.writerow(doc)


def write_jsons(docs, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for doc in docs:
            json.dump(doc, f)
            f.write('\n')


def main():
    parser = argparse.ArgumentParser(
        description='Download documents from a Luminoso Analytics project.'
    )
    parser.add_argument(
        'account_id',
        help="The ID of the account that owns the project, such as 'demo'"
    )
    parser.add_argument(
        'project_id', help="The ID of the project to analyze, such as '2jsnm'"
    )
    parser.add_argument(
        'username', help="A Luminoso username with access to the project"
    )
    parser.add_argument(
        '-a', '--api-url', default='https://analytics.luminoso.com/api/v4',
        help="The base URL for the Luminoso API (defaults to the production"
             " API, %(default)s)"
    )
    parser.add_argument(
        '-f', '--format', default='csv', choices=('csv', 'jsons'),
        help="Output file format (defaults to %(default)s)"
    )
    parser.add_argument(
        '-n', '--name', help="Name of output file (defaults to project name)"
    )
    args = parser.parse_args()

    url = '%s/projects/%s/%s' % (args.api_url, args.account_id, args.project_id)
    project = LuminosoClient.connect(url, username=args.username)
    project_name = project.get()['name']
    logger.info('Connected to project %s' % project_name)

    docs = download_docs(project)
    if args.name:
        filename = args.name
    else:
        filename = '%s.%s' % (project_name, args.format)

    if args.format == 'csv':
        write_csv(docs, filename)
    else:
        write_jsons(docs, filename)
    logger.info('Wrote docs to %s' % filename)