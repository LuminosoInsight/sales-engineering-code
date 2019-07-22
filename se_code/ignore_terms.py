from luminoso_api import V5LuminosoClient as LuminosoClient

import json, csv, argparse

def parse_url(url):
    api_root = url.strip('/ ').split('/app')[0]
    proj_id = url.strip('/ ').split('/')[-1]
    return api_root + '/api/v5/projects/' + proj_id

def ignore_single_term(text, client):
    concept = client.get('concepts', concept_selector={'type':'specified','concepts':[{'texts':[text]}]})
    ignore_term = concept['result'][0]['exact_term_ids'][0]
    client.put('terms/manage', term_management={ignore_term:{'action':'ignore'}})
    ignore = client.get('terms/manage')
    return ignore

def read_csv_file(filename):
    with open(filename) as f:
        reader = csv.DictReader(f)
        table = [row for row in reader]
    return([t['text'] for t in table])

def ignore_multiple_terms(texts, client):
    concepts = [{'texts': [t]} for t in texts]
    concept = client.get('concepts', concept_selector={'type':'specified','concepts':concepts})['result']
    ignore_terms = [c['exact_term_ids'][0] for c in concept if len(c['exact_term_ids']) > 0]
    terms = {t:{'action':'ignore'} for t in ignore_terms}
    client.put('terms/manage', term_management=terms)
    ignore = client.get('terms/manage')
    return ignore
        

def main():
    parser = argparse.ArgumentParser(
        description='Ignore terms from a project.'
    )
    parser.add_argument('project_url', help="URL of the project to ignore terms in")
    parser.add_argument('-t', '--token', default=None, help="Authentication token for your Daylight account")
    parser.add_argument('-i', '--ignore_term', default=None, help="Term to ignore in project")
    parser.add_argument('-f', '--filename', default=None, help="Name of the file to read list of terms to ignore from")
    args = parser.parse_args()
    
    endpoint = parse_url(args.project_url)
    if args.token:
        client = LuminosoClient.connect(endpoint, token=args.token)
    else:
        client = LuminosoClient.connect(endpoint)
        
    if args.ignore_term:
        ignore = ignore_single_term(args.ignore_term, client)
    elif args.filename:
        texts = read_csv_file(args.filename)
        ignore = ignore_multiple_terms(texts, client)
    else:
        ignore_term = ''
        while ignore_term.strip() == '':
            ignore_term = input('No term specified, please input a term now: ')
        ignore = ignore_single_term(ignore_term, client)
    
    print(ignore)
    client.post('build')
    client.wait_for_build()
    print('Project rebuilt, terms ignored')
    
if __name__ == '__main__':
    main()