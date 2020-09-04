from luminoso_api import V5LuminosoClient as LuminosoClient

import json, csv, argparse

def parse_url(url):
    api_root = url.strip('/ ').split('/app')[0]
    proj_id = url.strip('/ ').split('/')[-1]
    return api_root + '/api/v5/projects/' + proj_id

def merge_single_term(text, merge_text, client):
    concept = client.get('concepts', concept_selector={'type':'specified','concepts':[{'texts':[text]}]})
    merge_concept = client.get('concepts', concept_selector={'type':'specified','concepts':[{'texts':[merge_text]}]})
    term = concept['result'][0]['exact_term_ids'][0]
    client.put('terms/manage', term_management={term:{'new_term_id':merge_concept['result'][0]['exact_term_ids'][0]}})
    term_manage = client.get('terms/manage')
    return term_manage

def read_csv_file(filename):
    with open(filename) as f:
        reader = csv.DictReader(f)
        table = [row for row in reader]
    return([t['text'] for t in table])

def merge_multiple_terms(texts, merge_text, client):
    concepts = [{'texts': [t]} for t in texts]
    concept = client.get('concepts', concept_selector={'type':'specified','concepts':concepts})['result']
    ignore_terms = [c['exact_term_ids'][0] for c in concept if len(c['exact_term_ids']) > 0]
    merge_concept = client.get('concepts', concept_selector={'type':'specified','concepts':[{'texts':[merge_text]}]})
    terms = {t:{'new_term_id':merge_concept['result'][0]['exact_term_ids'][0]} for t in ignore_terms}
    client.put('terms/manage', term_management=terms)
    term_manage = client.get('terms/manage')
    return term_manage

def main():
    parser = argparse.ArgumentParser(
        description='Restem certain terms together in a given project'
    )
    parser.add_argument('project_url', help="URL of the project to ignore terms in")
    parser.add_argument('merge_into_text', help="Text of the Concept to merge the following concepts into")
    parser.add_argument('-m', '--merge_from_text', default=None, help="Text to roll into merge_text in project")
    parser.add_argument('-f', '--filename', default=None, help="Name of the file to read list of text to merge")
    args = parser.parse_args()
    
    endpoint = parse_url(args.project_url)
    client = LuminosoClient.connect(endpoint)

    if args.merge_into_text and args.merge_from_text:
        term_manage = merge_single_text(args.merge_from_text, args.merge_into_text, client)
    elif args.merge_into_text and args.filename:
        texts = read_csv_file(args.filename)
        term_manage = merge_multiple_terms(texts, args.merge_into_text, client)
    else:
        merge_from_text = ''
        merge_into_text = ''
        while merge_from_text.strip() == '':
            merge_from_text = input('What text would you like to restem: ')
        while merge_into_text.strip() == '':
            merge_into_text = input('What would you like to restem it to: ')
        term_manage = merge_single_term(merge_from_text, merge_into_text, client)
    
    print(term_manage)
    client.post('build')
    client.wait_for_build()
    print('Project rebuilt, terms merged')
    
if __name__ == '__main__':
    main()