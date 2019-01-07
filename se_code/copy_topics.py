from luminoso_api import V5LuminosoClient as LuminosoClient

import json, argparse, getpass

def get_topics(client):
    topics = client.get('concepts/saved')
    return topics

def delete_topics(client):
    topics = client.get('concepts/saved')
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
    
    args = parser.parse_args()
    from_proj = args.from_url.strip('/ ').split('/')[-1]
    from_root = args.from_url.split('/app')[0]
    #from_root = '/'.join(args.from_url.strip('/ ').split('/')[:-4])
    to_proj = args.to_url.strip('/ ').split('/')[-1]
    to_root = args.to_url.split('/app')[0]
    #to_root = '/'.join(args.to_url.strip('/ ').split('/')[:-4])
    count = 0
    while count < 3:
        token = input('Token: ')
        try:
            from_client = LuminosoClient.connect(url='%s/api/v5/projects/%s' % (from_root, from_proj), token=token)
            to_client = LuminosoClient.connect(url='%s/api/v5/projects/%s' % (to_root, to_proj), token=token)
            break
        except:
            print('Incorrect credentials, please re-enter token')
            count += 1
            continue
    if count >= 3:
        print('Invalid credentials.')
        return
    topics = from_client.get('concepts/saved')
    if not args.keep:
        delete_topics(to_client)
    copy_topics(to_client, topics)
    
if __name__ == '__main__':
    main()