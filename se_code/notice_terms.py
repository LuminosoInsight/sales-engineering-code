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


def notice_terms(texts, client):
    if not texts:
        raise ValueError('No concepts found')
    language = client.get(fields=('language',))['language']
    language_tag = '|' + language
    to_notice = {}
    for text in texts:
        if not text.endswith(language_tag):
            text += language_tag
        to_notice[text] = {'action': 'notice'}
    client.put('terms/manage', term_management=to_notice)
    return client.get('terms/manage')


def main():
    parser = argparse.ArgumentParser(
        description=(
            'Notice concepts in a project.  "Noticing" a concept means taking'
            ' a text that the project ignored as unimportant, such as a'
            ' number or a preposition, and telling the project not to ignore'
            ' it.')
    )
    parser.add_argument('project_url',
                        help="URL of the project to notice concepts in")

    group = parser.add_mutually_exclusive_group()
    group.add_argument('-n', '--notice_text', default=None, action='append',
                       help=('Text to notice in project; can be specified'
                             ' multiple times'))
    group.add_argument('-f', '--filename', default=None,
                       help=('Name of the file to read list of texts to'
                             ' notice from.  If the filename ends with .csv,'
                             ' concepts will be read from the "text" column.'
                             '  Otherwise, each line of the file will be'
                             ' treated as a concept to notice'))
    args = parser.parse_args()

    endpoint = parse_url(args.project_url)
    client = LuminosoClient.connect(endpoint,
                                    user_agent_suffix='se_code:notice_terms')

    if args.notice_text:
        texts_to_notice = args.notice_text
    elif args.filename:
        if args.filename.endswith('.csv'):
            texts_to_notice = read_csv_file(args.filename)
        else:
            texts_to_notice = read_text_file(args.filename)
    else:
        notice_text = ''
        while notice_text.strip() == '':
            notice_text = input('No text specified, please input a text now: ')
        texts_to_notice = [notice_text]

    try:
        notice_result = notice_terms(texts_to_notice, client)
    except ValueError as e:
        print(f'Error encountered: {e}.  Not rebuilding!')
        sys.exit()

    print(json.dumps(notice_result, ensure_ascii=False, indent=2))
    client.post('build')
    client.wait_for_build()
    print('Project rebuilt, concepts no longer ignored')


if __name__ == '__main__':
    main()
