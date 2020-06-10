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
  
def search_all_doc_ids(client, concepts):
    docs = []
    while True:
        new_docs = client.get('docs', search={"texts": concepts}, fields=["doc_id","match_score"], limit=25000, offset=len(docs))
        if new_docs['result']:
            docs.extend({'doc_id':d['doc_id'],'match_score':d['match_score']} for d in new_docs['result'])
        else:
            return docs
      
def add_relations(client,docs):
    # get the list of saved concepts
    concept_selector = {"type": "saved"}
    saved_concepts = client.get('concepts', concept_selector=concept_selector)['result']
    
    for sc in saved_concepts:
        sc['match_scores'] = search_all_doc_ids(client,sc['texts'])
        sc['match_scores_by_id'] = {c['doc_id']:c['match_score'] for c in sc['match_scores']}
        sc['docs_ids'] = [c['doc_id'] for c in sc['match_scores']]

    # loop through each doc and list of saved concepts finding if the doc is in that search result
    # add metadata for it's yes/no relation to each saved concept
    # if it is not associated with any saved concept, mark it as an outlier
    for d in docs:
        in_none = True
        for sc in saved_concepts:
            if (d['doc_id'] in sc['docs_ids']):
                in_none = False
                v = 'yes'
                d['metadata'].append({'name':sc['name'],'type':'string','value':v})
                d['metadata'].append({'name':sc['name']+' match_score','type':'number','value':sc['match_scores_by_id'][d['doc_id']]})
            else:
                v = 'no'
                d['metadata'].append({'name':sc['name'],'type':'string','value':v})
                d['metadata'].append({'name':sc['name']+' match_score','type':'number','value':0})
        
        if in_none:
            d['metadata'].append({'name':'doc_outlier','type':'string','value':'yes'})
        else:
            d['metadata'].append({'name':'doc_outlier','type':'string','value':'no'})

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
    parser.add_argument('-c', '--concept_relations', default=False, help="Add columns for saved concept relations and outliers")
    args = parser.parse_args()
    
    api_url = args.project_url.split('/app')[0]
    project_id = args.project_url.strip('/ ').split('/')[-1]

    account_id = args.project_url.strip('/').split('/')[5]
    project_id = args.project_url.strip('/').split('/')[6]
    api_url = '/'.join(args.project_url.strip('/').split('/')[:3]).strip('/') + '/api/v5'
    proj_apiv5 = '{}/projects/{}'.format(api_url, project_id)

    if args.token:
        client = LuminosoClient.connect(url=proj_apiv5, token=args.token)
    else:
        client = LuminosoClient.connect(url=proj_apiv5)
    
    docs = get_all_docs(client)
    if args.concept_relations:
        add_relations(client,docs)
    fields = get_fields(docs)
    docs, field_names = format_subsets(docs, fields, args.date_format)
    write_to_csv(args.filename, docs, field_names, encoding=args.encoding)
    

    
if __name__ == '__main__':
    main()
