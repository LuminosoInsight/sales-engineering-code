from luminoso_api import V5LuminosoClient as LuminosoClient
import argparse

def get_all_docs(client):
    '''
    Get all docs
    '''
    docs = []
    while True:
        newdocs = client.get('docs', limit=25000, offset=len(docs))
        if newdocs['result']:
            docs.extend(newdocs['result'])
        else:
            return docs
        
def copy_projects_to_accounts(all_projects, client, to_account):
    for from_project in all_projects:
        # Connect to project
        from_client_project = client.client_for_path(from_project['project_id'])
        print('Connected to project: ' +  from_project['name'])
        language = from_project['language']

        # Copy topics & documents
        from_topics = from_client_project.get('concepts/saved')
        from_docs = get_all_docs(from_client_project)
        from_docs = [{'text': d['text'],
                  'title': d['title'],
                  'metadata': d['metadata']} for d in from_docs]
        # Create a new project
        client = client.client_for_path('/projects')
        to_project = client.post(name=from_project['name'],language=language,account_id=to_account)
        to_project_client = client.change_path(to_project['project_id'])
        to_project_client.post('upload', docs=from_docs)
        for i, t in enumerate(from_topics):
            to_project_client.post('concepts/saved', concepts=[{'name': t['name'],
                                                               'color': t['color'],
                                                               'texts': t['texts']}], position=i)
        to_project_client.post('build')    
        print('Copied project: ' +  from_project['name'])
        
        
def main():
    parser = argparse.ArgumentParser(
        description='Copy all projects from an account on one cloud into another account on another cloud.'
    )
    parser.add_argument('from_url', help="The URL of the account that owns the current projects")
    parser.add_argument('to_url', help="The URL of the account to copy all projects to")
    args = parser.parse_args()
    
    from_api_url = '/'.join(args.from_url.split('/')[:-4])
    to_api_url = '/'.join(args.to_url.split('/')[:-4])
    from_account = args.from_url.split('/')[-1]
    to_account = args.to_url.split('/')[-1]
    client = LuminosoClient.connect(url=from_api_url + '/api/v5/projects')
    all_projects = client.get()
    all_projects = [p for p in all_projects if p['account_id'] == from_account]
    print('There are {} projects to be copied'.format(len(all_projects)))
    
    copy_projects_to_accounts(all_projects, client, to_account)
    
    
if __name__ == '__main__':
    main()