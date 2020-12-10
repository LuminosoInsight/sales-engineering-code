import argparse
import csv
import json

from luminoso_api import V5LuminosoClient as LuminosoClient
from se_code.copy_shared_concepts import delete_shared_concepts


def main():
    parser = argparse.ArgumentParser(
        description='Upload a file containing shared concepts to a target Daylight project.'
    )
    parser.add_argument('filename', help="Full name of the file containing shared concept definitions, including the extension. Must be CSV or JSON.")
    parser.add_argument('project_url', help="Full URL of the project to load the concepts into.")
    parser.add_argument('-d', '--delete', default=False, action="store_true", help="Whether to delete all the existing shared concepts or not.")
    parser.add_argument('-e', '--encoding', default="utf-8", help="Encoding type of the files to read from")
    args = parser.parse_args()
    
    root_url = args.project_url.strip('/ ').split('/app')[0]
    project_id = args.project_url.strip('/').split('/')[6]
    client = LuminosoClient.connect(url=root_url + '/api/v5/projects/' + project_id)
    
    filename = args.filename
    true_data = []
    if '.csv' in filename:
        with open(filename, encoding=args.encoding) as f:
            reader = csv.DictReader(f)
            data = [row for row in reader]
        true_data = {}
        for d in data:
            if d['concept_list_name'] not in true_data.keys():
                true_data[d['concept_list_name']] = {
                    "name":d['concept_list_name'],
                    "concepts":[]
                }
            if 'texts' not in [k.lower() for k in list(d.keys())] and 'texts' not in [k.lower() for k in list(d.keys())]:
                print('ERROR: File must contain a "text" column.')
                return
            row = {}
            for k in d:
                if 'text' in k.lower():
                    row['texts'] = [t.strip() for t in d[k].split(',')]
                if 'name' in k.lower():
                    row['name'] = d[k]
                if 'color' in k.lower():
                    row['color'] = d[k]
            true_data[d['concept_list_name']]['concepts'].append(row)
        
        # reformat true_data for export
        true_data = [{'concept_list_name':cl[1]['name'],
                      'concepts':cl[1]['concepts']} for cl in true_data.items()]

    elif '.json' in filename:
        true_data = json.load(open(filename, encoding=args.encoding))
        for clist in true_data:
            for c in clist['concepts']:
                c['texts'] = c['texts'].split(",")
    else:
        print('ERROR: you must pass in a CSV or JSON file.')
        filename = input('Please enter a valid filename (include the file extension): ')

    statement = 'New Shared Concepts added to project'
    if args.delete:
        delete_shared_concepts(client)
        statement += ' and old Shared Concepts deleted'

    for cl in true_data:
        client.post('concept_lists/', name=cl['concept_list_name'], concepts=cl['concepts'])

    print(statement)
    print('Project URL: %s' % args.project_url)


if __name__ == '__main__':
    main()
