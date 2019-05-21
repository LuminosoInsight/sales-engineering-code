import csv, argparse, time
from datetime import datetime
from luminoso_api import V5LuminosoClient as LuminosoClient

#DATE_FORMAT = '%d-%b-%y'
DATE_FORMAT = '%b %d, %Y'
TARGET_DATE_FORMAT = '%m/%d/%y'

def read_table_to_docs(table, field='pros'):
    '''
    Given a table read from a file, format the contents into Luminoso
    compatible documents
    '''
    docs = []
    if field == 'pros': 
        field_score = 5
    else:
        field_score = 0
    for t in table:
        row = {}
        score = int(t['score_Star Rating'])
        position = t['employeeStatus'].split('-')[-1].strip()
        if 'anonymous' in position.lower():
            position = 'Employee'
        status = t['employeeStatus'].split('Employee')[0].strip()
        row['title'] = '%s %s gave %d Stars' % (status, position, score)
        #row['title'] = '%d stars - %s' % (score, t['employeeStatus'])
        row['text'] = t['%sText' % field]
        metadata = []
        try:
            if t['date_Date'] != 'null':
                metadata.append({'type': 'date', 
                                 'name': 'Date', 
                                 'value': time.mktime(
                                     datetime.strptime(
                                         t['date_Date'], DATE_FORMAT
                                     ).timetuple())})
        except ValueError as e:
            print('Warning: %s' % e)
        if t['Location'] != 'null':
            metadata.append({'type': 'string', 
                             'name': 'Location', 
                             'value': t['Location']})
        if t['string_Recommendation'] != 'null':
            metadata.append({'type': 'string', 
                             'name': 'Recommendation', 
                             'value': t['string_Recommendation']})
        if t['string_Outlook'] != 'null':
            metadata.append({'type': 'string', 
                             'name': 'Outlook', 
                             'value': t['string_Outlook']})
        if t['string_CEO Approval'] != 'null':
            metadata.append({'type': 'string', 
                             'name': 'CEO Approval', 
                             'value': t['string_CEO Approval'].split('of')[0].strip()})
        metadata.append({'type': 'string', 
                         'name': 'Field', 
                         'value': field})
        metadata.append({'type': 'string', 
                         'name': 'Employee Status', 
                         'value': t['employeeStatus'].split('-')[0].strip()})
        metadata.append({'type': 'string', 
                         'name': 'Employee Title', 
                         'value': t['employeeStatus'].split('-')[-1].strip()})
        metadata.append({'type': 'score', 
                         'name': 'Star Rating', 
                         'value': score})
        metadata.append({'type': 'score', 
                         'name': 'Quasi-NPS', 
                         'value': score + field_score})
        row['metadata'] = metadata
        docs.append(row)
    docs = [d for d in docs if d['text'].strip() != '' and d['text'] != 'null']
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
                    ).strftime(TARGET_DATE_FORMAT)
            else:
                row[m['type'] + '_' + m['name']] = m['value']
        write_table.append(row)
    return write_table



def write_all_uploads_to_csvs(filename, pro_docs, con_docs, docs, 
                              encoding='utf-8'):
    '''
    Writes the upload-ready documents to CSVs as a backup
    '''
    write_pros = convert_docs_to_csv(pro_docs)
    write_cons = convert_docs_to_csv(con_docs)
    write_total = convert_docs_to_csv(docs)
    
    root = filename.split('.csv')[0]
    
    fields = ['title', 'text']
    for d in write_total:
        for k in d:
            if k not in fields:
                fields.append(k)
    with open(root + '_pros.csv', 'w', encoding=encoding, newline='') as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(write_pros)
    with open(root + '_cons.csv', 'w', encoding=encoding, newline='') as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(write_cons)
    with open(root + '_combined.csv', 'w', encoding=encoding, newline='') as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(write_total)
        
        
        
def upload_docs_to_projects(pro_docs, con_docs, docs, filename,
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
        pro_proj_id = client.post(name=name + ' Pros', 
                                  language='en', 
                                  account_id=account_id)['project_id']
        con_proj_id = client.post(name=name + ' Cons', 
                                  language='en', 
                                  account_id=account_id)['project_id']
        total_proj_id = client.post(name=name + ' Combined', 
                                    language='en', 
                                    account_id=account_id)['project_id']
    else:
        pro_proj_id = client.post(name=name + ' Pros', 
                                  language='en')['project_id']
        con_proj_id = client.post(name=name + ' Cons', 
                                  language='en')['project_id']
        total_proj_id = client.post(name=name + ' Combined', 
                                    language='en')['project_id']
    pro_client = client.client_for_path(pro_proj_id)
    pro_client.post('upload', docs=pro_docs)
    con_client = client.client_for_path(con_proj_id)
    con_client.post('upload', docs=con_docs)
    total_client = client.client_for_path(total_proj_id)
    total_client.post('upload', docs=docs)
    pro_client.post('build')
    con_client.post('build')
    total_client.post('build')
    pro_client.wait_for_build()
    con_client.wait_for_build()
    total_client.wait_for_build()
    
    url_root = api_root.split('api')[0] + 'app/projects/'
    account = pro_client.get()['account_id']
    print('Pros project: %s' % (url_root + account + '/' + pro_client.get()['project_id']))
    print('Cons project: %s' % (url_root + account + '/' + con_client.get()['project_id']))
    print('Combined project: %s' % (url_root + account + '/' + total_client.get()['project_id']))

    
    
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
    
    pro_docs = read_table_to_docs(table, field='pros')
    con_docs = read_table_to_docs(table, field='cons')
    docs = [d for d in pro_docs]
    docs.extend(con_docs)
    
    if args.save:
        write_all_uploads_to_csvs(args.filename, pro_docs, con_docs, docs, 
                                  encoding=args.encoding)
    
    if not args.upload:
        upload_docs_to_projects(pro_docs, con_docs, docs, args.filename,
                            token=args.token, account_id=args.account_id,
                            api_root=args.api_root)
    
if __name__ == '__main__':
    main()