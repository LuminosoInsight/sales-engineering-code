from luminoso_api import LuminosoClient
import pycld2 as cld2
import argparse, json

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]
    
def delete_docs(client, ids):
    docs = []
    offset = 0
    bad_batch = batch(ids,600)
    for bad_ids in bad_batch:
        client.delete('docs',ids=bad_ids)
        
def get_all_docs(client, batch_size=20000):
    docs = []
    offset = 0
    while True:
        newdocs = client.get('docs', offset=offset, limit=batch_size)
        if not newdocs:
            return docs
        docs.extend(newdocs)
        offset += batch_size
        
def remove_foreign_lang(client,lang_code,threshold=0):
    
    docs = get_all_docs(client)
    
    bad_doc_ids = []
    
    for doc in docs:
        try:
            isReliable, textBytesFound, details = cld2.detect(doc['text'])
        except ValueError:
            bad_doc_ids.append(doc['_id'])
            continue
        if not details[0][1] == lang_code and isReliable or details[0][2] < threshold:
                bad_doc_ids.append(doc['_id'])
    delete_docs(client,bad_doc_ids)
    client.post('docs/recalculate', language=lang_code)
    print('{} documents not identified as "{}" removed from project.'.format(len(bad_doc_ids),lang_code))
    
def main(args):
    root_url = args.project_url.split('/app')[0] + '/api/v4'
    account_id = args.project_url.split('/')[-2]
    project_id = args.project_url.split('/')[-1]
    if not args.lang_code or args.lang_code not in [b for a,b in cld2.LANGUAGES]:
        args.lang_code = input('Enter the 2 letter language code to be retained: ')
        
    client = LuminosoClient.connect(url=root_url,username=args.username)
    client = client.change_path('/projects/{}/{}/'.format(account_id,project_id))
    remove_foreign_lang(client,args.lang_code,args.threshold)
    
if __name__ == '__main__':
    print("Remove Foreign Language Tool:"
          "\nRemoves documents from a project if they cannot be positively identified as the target language.")
    print("\nScript can be run with arguments or interactively. Run with -h to see help.\n")   
    print("\nBy default this script retains only documents in English ('en')")

    parser = argparse.ArgumentParser()
    parser.add_argument('project_url', help="The URL of the Daylight project to remove foreign language from")
    parser.add_argument('-l','--lang_code', help="The 2 character language code to retain ex. en, fr", default='en')
    parser.add_argument('-t','--threshold', help="Add a minimum threshold for the desired language (ex. 95 for 95%%)",
                        default=0,type=int)
    parser.add_argument('-username','--username', help="Luminoso username")
    args = parser.parse_args()
    main(args)