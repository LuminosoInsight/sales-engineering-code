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


def ignore_concepts(texts, current_concepts, client, overwrite=False):

    selector = [{'texts': [t]} for t in texts]
    concepts = client.get(
        'concepts',
        concept_selector={'type': 'specified', 'concepts': selector}
    )['result']

    ignore_concepts = []
    for concept in concepts:
        name = concept['name']
        if name:
            ignore_concepts.append({'concept': name})
    current_concepts["ignore"] = ignore_concepts
    client.put('concepts/manage', concept_management=current_concepts,
               overwrite=overwrite)
    new_management = client.get('concepts/manage')
    if new_management['next_build'] == new_management['current_build']:
        raise ValueError('No changes detected')
    return new_management


def notice_concepts(texts, current_concepts, client,  overwrite=False):
    if not texts:
        raise ValueError('No concepts found')

    notice_concepts = []
    for text in texts:
        notice_concepts.append({'concept': text.strip()})
    current_concepts["notice"] = notice_concepts
    client.put('concepts/manage', concept_management=current_concepts,
               overwrite=overwrite)
    return client.get('concepts/manage')


def collocate_concepts(texts, current_concepts, client,  overwrite=False):
    if not texts:
        current_concepts["collocate"] = []
        client.put('concepts/manage', concept_management=current_concepts,
                overwrite=overwrite)
    else:
        collocate_concepts = []
        for text in texts:
            collocate_concepts.append({'concept': text.strip()})
        current_concepts["collocate"] = collocate_concepts
        client.put('concepts/manage', concept_management=current_concepts,
                overwrite=overwrite)
    return client.get('concepts/manage')


def merge_concepts(texts, current_concepts, merge_text, client,  overwrite=False):
    if not texts:
        raise ValueError('No concepts found')

    merge_concepts = []
    for text in texts:
        merge_concepts.append({'concept': text.strip(), 'merge_with': merge_text.strip()})
    current_concepts["merge"] = merge_concepts
    client.put('concepts/manage', concept_management=current_concepts,
               overwrite=overwrite)
    return client.get('concepts/manage')


def main():
    parser = argparse.ArgumentParser(
        description='Manage concepts from a project.'
    )
    parser.add_argument('project_url',
                        help="URL of the project to manage concepts in")

    parser.add_argument('-o', '--overwrite', default=False,
                        action='store_true',
                        help='Overwrites all existing concept management'
                             'information with the concept management object'
                             ' being sent. Otherwise appends new concepts')
    parser.add_argument('-b', '--build', default=False,
                        action='store_true',
                        help=('Has the project build immediately after managing'
                              ' the concepts. Otherwise adds the concepts to'
                              ' be built later'))
    manage_type = parser.add_mutually_exclusive_group()
    manage_type.add_argument('-i', '--ignore_concept', default=False,
                             action='store_true',
                             help=('Action to apply to concepts. Will ignore'
                                   ' any concepts specified'))
    manage_type.add_argument('-n', '--notice_concept', default=False,
                             action='store_true',
                             help=('Action to apply to concepts. Will notice any'
                                   ' concepts specified. "Noticing" a concept'
                                   ' means taking a concept that the project'
                                   ' ignored as unimportant and telling the'
                                   ' project not to ignore it'))
    manage_type.add_argument('-c', '--collocate_concepts', default=False,
                             action='store_true',
                             help=('Action to apply to concepts. Will collocate'
                                   ' any concepts specified'))
    manage_type.add_argument('-m', '--merge_concept', default=None,
                             help=('Action to apply to concepts. Will merge any'
                                   ' concepts specified to the concept that is'
                                   ' entered'))
    group = parser.add_mutually_exclusive_group()
    # equivalent to merge_from_text, ignore_term, notice_text
    group.add_argument('-s', '--single_concept', default=None, action='append',
                       help=('Concept to manage in project; can be specified'
                             ' multiple times'))
    group.add_argument('-f', '--filename', default=None,
                       help=('Name of the file to read list of concepts to'
                             ' ignore from.  If the filename ends with .csv,'
                             ' concepts will be read from the "text" column.'
                             '  Otherwise, each line of the file will be'
                             ' treated as a concept to ignore'))
    args = parser.parse_args()

    endpoint = parse_url(args.project_url)
    client = LuminosoClient.connect(endpoint,
                                user_agent_suffix='se_code:manage_concepts')

    if args.single_concept:
        concepts_to_manage = args.single_concept
    elif args.filename:
        if args.filename.endswith('.csv'):
            concepts_to_manage = read_csv_file(args.filename)
        else:
            concepts_to_manage = read_text_file(args.filename)
    else:
        if not args.build:
            manage_concept = ''
            while manage_concept.strip() == '':
                manage_concept = input('No concept specified, please input a concept now: ')
            concepts_to_manage = [manage_concept]

    # get the current list of concepts. On overwrite we only want to replace
    # the individual ignore/notice not both.
    mc_results = client.get('concepts/manage')
    current_concepts = mc_results['next_build']

    if args.ignore_concept is True:
        try:
            manage_result = ignore_concepts(concepts_to_manage, current_concepts, client, args.overwrite)
        except ValueError as e:
            print(f'Error encountered: {e}.  Not rebuilding!')
            sys.exit()
    elif args.notice_concept is True:
        try:
            manage_result = notice_concepts(concepts_to_manage, current_concepts, client, args.overwrite)
        except ValueError as e:
            print(f'Error encountered: {e}.  Not rebuilding!')
            sys.exit()
    elif args.collocate_concepts is True:
        try:
            manage_result = collocate_concepts(concepts_to_manage, current_concepts, client, args.overwrite)
        except ValueError as e:
            print(f'Error encountered: {e}.  Not rebuilding!')
            sys.exit()
    elif args.merge_concept:
        try:
            manage_result = merge_concepts(concepts_to_manage, current_concepts, args.merge_concept, client, args.overwrite)
        except ValueError as e:
            print(f'Error encountered: {e}.  Not rebuilding!')
            sys.exit()
    else:
        try:
            manage_result = client.get('concepts/manage')
        except ValueError as e:
            print(f'Error encountered: {e}.  Not rebuilding!')
            sys.exit()

    if args.build is True:
        print(json.dumps(manage_result, ensure_ascii=False, indent=2))
        client.post('build')
        client.wait_for_build()
        print('Project rebuilt, concepts managed')
    else:
        print(json.dumps(manage_result, ensure_ascii=False, indent=2))
        print('Concepts added for next project rebuild')


if __name__ == '__main__':
    main()
