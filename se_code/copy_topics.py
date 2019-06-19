from luminoso_api import V5LuminosoClient as LuminosoClient

import json, argparse, getpass

def get_topics(client):
    topics = client.get('concepts/saved')
    return topics

def delete_topics(client):
    topics = client.get('concepts/saved')
    if len(topics)>0:
        ids_to_delete = []
        for t in topics:
            ids_to_delete.append(t['saved_concept_id'])
        client.delete('concepts/saved', saved_concept_ids=ids_to_delete)
        print('Old topics deleted')
        
def copy_topics(client, topics):
    concepts = []
    for i, t in enumerate(topics):
        client.post('concepts/saved', concepts=[{'texts': t['texts'],'name': t['name'],'color': t['color']}], position=i)
    print('Topics copied')

def main():
    parser = argparse.ArgumentParser(
        description='Export data to Tableau compatible CSV files.'
    )
    parser.add_argument('from_url', help="The URL of the project to copy all topics from")
    parser.add_argument('to_url', help="The URL of the project to copy all topics into")
    parser.add_argument('--keep', default=False, action='store_true', help="Use this flag to specify if you want to keep the existing topics")
    parser.add_argument('-t1', '--token1', default=None, help="Authentication token for first Daylight project")
    parser.add_argument('-t2', '--token2', default=None, help="Authentication token for second Daylight project")
    
    args = parser.parse_args()
    from_proj = args.from_url.strip('/ ').split('/')[-1]
    from_root = args.from_url.split('/app')[0]
    to_proj = args.to_url.strip('/ ').split('/')[-1]
    to_root = args.to_url.split('/app')[0]
    if args.token1:
        from_client = LuminosoClient.connect(url='%s/api/v5/projects/%s' % (from_root, from_proj), token=args.token1)
    else:
        from_client = LuminosoClient.connect(url='%s/api/v5/projects/%s' % (from_root, from_proj))
    if args.token2:
        to_client = LuminosoClient.connect(url='%s/api/v5/projects/%s' % (to_root, to_proj), token=args.token2)
    else:
        to_client = LuminosoClient.connect(url='%s/api/v5/projects/%s' % (to_root, to_proj))
        
    topics = from_client.get('concepts/saved')
    if not args.keep:
        delete_topics(to_client)
    copy_topics(to_client, topics)
    
if __name__ == '__main__':
    main()
