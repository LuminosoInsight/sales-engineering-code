import csv, argparse, time
from datetime import datetime
from luminoso_api import V5LuminosoClient as LuminosoClient

DATE_FORMAT = '%Y-%m-%d'

def read_table_to_docs(table):
    '''
    Given a table read from a file, format the contents into Luminoso
    compatible documents
    '''
    docs = []
    for t in table:
        doc = {}
        if t.get('Content') and t['Content'].strip().lower() not in ['', 'null']:
            doc['text'] = t['Content']
        else:
            continue
        if t.get('Title') and t['Title'].strip().lower() not in ['', 'null']:
            doc['title'] = t['Title']
        else:
            doc['title'] = ''
        metadata = []
        if t.get('Source') and t['Source'].strip().lower() not in ['', 'null']:
            metadata.append({'type': 'string', 
                             'name': 'Data Source',
                             'value': t['Source']})
        if t.get('Date') and t['Date'].strip().lower() not in ['', 'null']:
            try:
                dt = datetime.strptime(t['Date'], DATE_FORMAT)
                unix_date = time.mktime(dt.timetuple())
                metadata.append({'type': 'date',
                                 'name': 'Date',
                                 'value': unix_date})
            except ValueError as e:
                print('Warning: %s' % e)
        if t.get('Name') and t['Name'].strip().lower() not in ['', 'null']:
            metadata.append({'type': 'string',
                             'name': 'Author Name',
                             'value': t['Name']})
        if t.get('Rating') and t['Rating'].strip().lower() not in ['', 'null']:
            metadata.append({'type': 'score',
                             'name': 'Rating',
                             'value': float(t['Rating'])})
        doc['metadata'] = metadata
        docs.append(doc)
    print(len(docs))
    docs = [d for d in docs if d['text'].strip() != '']
    return docs


def convert_docs_to_csv(docs):
    '''
    Converts Luminoso-compatible documents to CSV-writable table
    '''
    write_table = []
    for d in docs:
        row = {}
        row['title'] = d['title']
        row['text'] = d['text']
        for m in d['metadata']:
            if m['type'] == 'date' and m['value']:
                row[m['type'] + '_' + m['name']] = datetime.fromtimestamp(
                        int(m['value'])
                    ).strftime(DATE_FORMAT)
            else:
                row[m['type'] + '_' + m['name']] = m['value']
        write_table.append(row)
    return write_table

def write_upload_to_csv(filename, docs, encoding='utf-8'):
    '''
    Writes the upload-ready documents to CSVs as a backup
    '''
    write_table = convert_docs_to_csv(docs)
    
    root = filename.split('.csv')[0]
    
    fields = ['title', 'text']
    for d in write_table:
        for k in d:
            if k not in fields:
                fields.append(k)
    with open(root + '_upload.csv', 'w', encoding=encoding, newline='') as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(write_table)
        
        
        
def upload_docs_to_projects(docs, filename,
                            account_id=None, token=None, 
                            api_root='https://analytics.luminoso.com/api/v5'):
    '''
    Uploads Luminoso-compatible documents to 3 separate projects and prints
    their resulting URLs for easy access.
    '''
    if not token:
        client = LuminosoClient.connect('%s/projects/' % api_root)
    else:
        client = LuminosoClient.connect('%s/projects/' % api_root, token=token)
    name = filename.split('.csv')[0].split('/')[-1]
    if account_id:
        proj_id = client.post(name=name, 
                              language='en', 
                              account_id=account_id)['project_id']
    else:
        proj_id = client.post(name=name, 
                              language='en')['project_id']
    client = client.client_for_path(proj_id)
    client.post('upload', docs=docs)
    client.post('build')
    client.wait_for_build()
    
    url_root = api_root.split('api')[0] + 'app/projects/'
    account = client.get()['account_id']
    print('Completed project: %s' % (url_root + account + '/' + client.get()['project_id']))

    
    
def main():
    parser = argparse.ArgumentParser(
        description='Automatically upload scraped results into relevant Daylight projects.'
    )
    parser.add_argument('filename', help="Name of the CSV file outputted by the scraper")
    parser.add_argument('-a', '--account_id', default=None, 
                        help="The ID of the account that will own the project, such as 'demo'")
    parser.add_argument('-t', '--token', default=None,
                        help="Authentication token for your Daylight user")
    parser.add_argument('-e', '--encoding', default='utf-8',
                        help="Encoding of the file to read from and write to.")
    parser.add_argument('-r', '--api_root', default='https://analytics.luminoso.com/api/v5',
                        help="API Root for the Daylight environment to upload projects to")
    parser.add_argument('-s', '--save', default=False, action='store_true',
                        help="Whether or not to save upload files as a backup")
    parser.add_argument('-u', '--upload', default=False, action='store_true',
                        help="Do not upload the projects")
    args = parser.parse_args()
    
    with open(args.filename, encoding=args.encoding) as f:
        reader = csv.DictReader(f)
        table = [row for row in reader]
    
    docs = read_table_to_docs(table)
    
    if args.save:
        write_upload_to_csv(args.filename, docs, encoding=args.encoding)
    
    if not args.upload:
        upload_docs_to_projects(docs, args.filename,
                            token=args.token, account_id=args.account_id,
                            api_root=args.api_root)
    
if __name__ == '__main__':
    main()