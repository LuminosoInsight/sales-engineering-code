from luminoso_api import V5LuminosoClient as LuminosoClient
import csv, json, datetime, time, argparse

def get_all_docs(client):
    docs = []
    while True:
        new_docs = client.get('docs', limit=25000, offset=len(docs))
        if new_docs['result']:
            docs.extend(new_docs['result'])
        else:
            return docs
        
def get_fields(docs):
    fields = []
    for doc in docs:
        for key in doc:
            if key in ['text','title', 'metadata']:
                fields.append(key)
    fields = list(set(fields))
    return fields

def format_subsets(docs, fields, date_format):
    docs = [{k:v for k,v in d.items() if k in fields} for d in docs]
    subsets = []
    field_names = ['text']
    if 'title' in fields:
        field_names.append('title')
    if 'metadata' in fields:
        for doc in docs:
            for metadata in doc['metadata']:
                if metadata['type'] == 'date':
                    try:
                        doc['%s_%s' % (metadata['type'], metadata['name'])] = datetime.datetime.fromtimestamp(
                                                                                int(metadata['value'])).strftime(date_format)
                    except ValueError:
                        doc['%s_%s' % (metadata['type'], metadata['name'])] = '%s' % metadata['value']
                else:
                    doc['%s_%s' % (metadata['type'], metadata['name'])] = metadata['value']
                subsets.append('%s_%s' % (metadata['type'], metadata['name']))
            del doc['metadata']
    field_names.extend(list(set(subsets)))
    return docs, field_names
    
def write_to_csv(filename, docs, field_names, encoding='utf-8'):
    with open(filename, 'w', encoding=encoding) as f:
        writer = csv.DictWriter(f, field_names)
        writer.writeheader()
        writer.writerows(docs)
    print('Wrote %d docs to %s' % (len(docs), filename))
        
def main():
    parser = argparse.ArgumentParser(
        description='Download documents from an Analytics project and write to CSV.'
    )
    parser.add_argument('project_url', help="The URL of the project to analyze")
    parser.add_argument('filename', help="Name of CSV file to write project documents to")
    parser.add_argument('-t', '--token', default=None, help="Daylight token")
    parser.add_argument('-e', '--encoding', default='utf-8', help="Encoding type of the file to write to")
    parser.add_argument('-d', '--date_format', default='%Y-%m-%d', help="Format of timestamp")
    args = parser.parse_args()
    
    api_url = args.project_url.split('/app')[0]
    project_id = args.project_url.strip('/ ').split('/')[-1]
    if args.token:
        client = LuminosoClient.connect(url='%s/api/v5/projects/%s' % (api_url, project_id), token=args.token)
    else:
        client = LuminosoClient.connect(url='%s/api/v5/projects/%s' % (api_url, project_id))
    
    docs = get_all_docs(client)
    fields = get_fields(docs)
    docs, field_names = format_subsets(docs, fields, args.date_format)
    write_to_csv(args.filename, docs, field_names, encoding=args.encoding)
    
    
if __name__ == '__main__':
    main()
