from luminoso_api import V5LuminosoClient as LuminosoClient

import argparse
import csv
import json
import sys


def parse_url(url):
    api_root = url.strip('/ ').split('/app')[0]
    proj_id = url.strip('/').split('/')[6]
    return api_root + '/api/v5/projects/' + proj_id


def read_csv_file(filename):
    with open(filename, 'r', encoding='utf-8-sig') as f:
        return [row['text'] for row in csv.DictReader(f)]


def read_text_file(filename):
    with open(filename, 'r', encoding='utf-8-sig') as f:
        return f.readlines()


def ignore_terms(texts, client):
    # Get the current state of management, so we can see whether there are
    # changes by the end
    current_management = client.get('terms/manage')
    selector = [{'texts': [t]} for t in texts]
    concepts = client.get(
        'concepts',
        concept_selector={'type': 'specified', 'concepts': selector}
    )['result']

    to_ignore = {}
    for concept in concepts:
        exact_term_ids = concept['exact_term_ids']
        if exact_term_ids:
            to_ignore[exact_term_ids[0]] = {'action': 'ignore'}
    if not to_ignore:
        raise ValueError('No concepts found')
    client.put('terms/manage', term_management=to_ignore)
    new_management = client.get('terms/manage')
    if new_management == current_management:
        raise ValueError('No changes detected')
    return new_management


def main():
    parser = argparse.ArgumentParser(
        description='Ignore terms from a project.'
    )
    parser.add_argument('project_url',
                        help="URL of the project to ignore terms in")

    group = parser.add_mutually_exclusive_group()
    group.add_argument('-i', '--ignore_term', default=None, action='append',
                        help=('Term to ignore in project; can be specified'
                              ' multiple times'))
    group.add_argument('-f', '--filename', default=None,
                        help=('Name of the file to read list of terms to'
                              ' ignore from.  If the filename ends with .csv,'
                              ' concepts will be read from the "text" column.'
                              '  Otherwise, each line of the file will be'
                              ' treated as a concept to ignore'))
    args = parser.parse_args()

    endpoint = parse_url(args.project_url)
    client = LuminosoClient.connect(endpoint,
                                    user_agent_suffix='se_code:ignore_terms')

    if args.ignore_term:
        terms_to_ignore = args.ignore_term
    elif args.filename:
        if args.filename.endswith('.csv'):
            terms_to_ignore = read_csv_file(args.filename)
        else:
            terms_to_ignore = read_text_file(args.filename)
    else:
        ignore_term = ''
        while ignore_term.strip() == '':
            ignore_term = input('No term specified, please input a term now: ')
        terms_to_ignore = [ignore_term]

    try:
        ignore_result = ignore_terms(terms_to_ignore, client)
    except ValueError as e:
        print(f'Error encountered: {e}.  Not rebuilding!')
        sys.exit()

    print(json.dumps(ignore_result, ensure_ascii=False, indent=2))
    client.post('build')
    client.wait_for_build()
    print('Project rebuilt, terms ignored')


if __name__ == '__main__':
    main()
