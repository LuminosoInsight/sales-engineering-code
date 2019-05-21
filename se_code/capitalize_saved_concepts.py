from luminoso_api import V5LuminosoClient
import argparse

def main():
    parser = argparse.ArgumentParser(
        description='Capitalize Saved Concepts in a project'
    )
    parser.add_argument('url', help="The URL of the project to Capitalize Saved Concepts for.")
    parser.add_argument('-t', '--token', default=None, help="Daylight Authentication token.")
    args = parser.parse_args()
    
    api_root = args.url.split('app/')[0].strip('/ ')
    project_id = args.url.strip('/ ').split('/')[-1]
    
    if args.token:
        client = V5LuminosoClient.connect('%s/api/v5/projects/%s' % (api_root, project_id), token=args.token)
    else:
        client = V5LuminosoClient.connect('%s/api/v5/projects/%s' % (api_root, project_id))
    saved_concepts = client.get('concepts/saved')
    concepts_to_update = [{'saved_concept_id': c['saved_concept_id'], 
                           'name': c['name'].capitalize()} for c in saved_concepts]
    client.put('concepts/saved', concepts=concepts_to_update)
    print("Saved Concepts Capitalized")
    
if __name__ == '__main__':
    main()