from luminoso_api import V5LuminosoClient as LuminosoClient
import argparse, json
try:
    import pycld2 as cld2
except:
    import cld2

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]
    
def delete_docs(client, ids):
    docs = []
    offset = 0
    bad_batch = batch(ids,600)
    for bad_ids in bad_batch:
        client.post('docs/delete',doc_ids=bad_ids)
        
def get_all_docs(client, batch_size=20000):
    docs = []
    offset = 0
    while True:
        newdocs = client.get('docs', offset=offset, limit=batch_size)
        if not newdocs['result']:
            return docs
        docs.extend(newdocs['result'])
        offset += batch_size
        
def remove_foreign_lang(client,lang_code,threshold=.4):
    
    docs = get_all_docs(client)
    
    bad_doc_ids = []
    
    for doc in docs:
        try:
            isReliable, textBytesFound, details = cld2.detect(doc['text'].encode('utf-8'))
        except ValueError:
            bad_doc_ids.append(doc['doc_id'])
            continue
        except cld2.error:
            continue
        except Error:
            continue
        if (details[0][1] != lang_code) or (details[0][1] == lang_code and (not isReliable or (details[0][2] - details[1][2] <= (threshold * 100)))):
                bad_doc_ids.append(doc['doc_id'])
    delete_docs(client,bad_doc_ids)
    client.post('build')
    client.wait_for_build()
    print('{} documents not identified as "{}" removed from project.'.format(len(bad_doc_ids),lang_code))
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('project_url', help="The URL of the project to analyze")
    parser.add_argument('lang_code', default='en', help="The 2 character language code to retain ex. en, fr")
    parser.add_argument('-t', '--token', default=None, help="Authentication token for Daylight")
    parser.add_argument('--threshold', default=.4, type=float, help="Minimum threshold for desired language (ex .95 for 95%%)")
    args = parser.parse_args()
    
    api_url = args.project_url.split('/app')[0]
    project_id = args.project_url.strip('/ ').split('/')[-1]
    if args.token:
        client = LuminosoClient.connect(url='{}/api/v5/projects/{}/'.format(api_url, project_id), token=args.token)
    else:
        client = LuminosoClient.connect(url='{}/api/v5/projects/{}/'.format(api_url, project_id))
    remove_foreign_lang(client,args.lang_code,args.threshold)
    
if __name__ == '__main__':
    main()