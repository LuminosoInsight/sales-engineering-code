from luminoso_api import V5LuminosoClient as LuminosoClient
import csv, json, argparse

def main():
    parser = argparse.ArgumentParser(
        description='Upload a file containing topics to a target Daylight project.'
    )
    parser.add_argument('filename', help="Full name of the file containing topic definitions, including the extension. Must be CSV or JSON.")
    parser.add_argument('project_url', help="Full URL of the project to load the topics into.")
    parser.add_argument('-d', '--delete', default=False, action="store_true", help="Whether to delete existing topics or not.")
    parser.add_argument('-t', '--token', default=None, help="If Daylight token has not been saved to your machine, enter it here.")
    args = parser.parse_args()
    
    root_url = '/'.join(args.project_url.split('/')[:-5])
    project_id = args.project_url.strip('/').split('/')[-1]
    if args.token:
        client = LuminosoClient.connect(url=root_url + '/api/v5/projects/' + project_id, token=args.token)
    else:
        client = LuminosoClient.connect(url=root_url + '/api/v5/projects/' + project_id)
    
    correct = False
    filename = args.filename
    true_data = []
    while not correct:
        if '.csv' in filename:
            with open(filename) as f:
                reader = csv.DictReader(f)
                data = [row for row in reader]
            for d in data:
                if 'text' not in [k.lower() for k in list(d.keys())] and 'texts' not in [k.lower() for k in list(d.keys())]:
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
                true_data.append(row)
            correct = True
        elif '.json' in filename:
            data = json.load(open(filename))
            for d in data:
                if 'text' or 'texts' not in [k.lower() for k in list(d.keys())]:
                    print('ERROR: File must contain a "text" field.')
                    return
                row = {}
                for k in d:
                    if 'text' in k.lower():
                        row['texts'] = [t.strip() for t in d[k].split(',')]
                    if 'name' in k.lower():
                        row['name'] = d[k]
                    if 'color' in k.lower():
                        row['color'] = d[k]
                true_data.append(row)
            correct = True
        else:
            print('ERROR: you must pass in a CSV or JSON file.')
            filename = input('Please enter a valid filename (include the file extension): ')
            
    statement = 'New Saved Concepts added to project'
    if args.delete:
        saved_concepts = client.get('concepts/saved')
        saved_concept_ids = [c['saved_concept_id'] for c in saved_concepts]
        client.delete('concepts/saved', saved_concept_ids=saved_concept_ids)
        statement += ' and old Saved Concepts deleted'
    client.post('concepts/saved', concepts=true_data)
    print(statement)
    print('Project URL: %s' % args.project_url)
    
if __name__ == '__main__':
    main()
