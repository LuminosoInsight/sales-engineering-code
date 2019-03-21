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
        
def copy_projects_to_accounts(all_projects, from_client, to_client, to_account):
    for from_project in all_projects:
        # Connect to project
        from_client_project = from_client.client_for_path(from_project['project_id'])
        print('Connected to project: ' +  from_project['name'])
        language = from_project['language']

        # Copy topics & documents
        from_topics = from_client_project.get('concepts/saved')
        from_docs = get_all_docs(from_client_project)
        from_docs = [{'text': d['text'],
                  'title': d['title'],
                  'metadata': d['metadata']} for d in from_docs]
        # Create a new project
        client = to_client.client_for_path('/projects')
        to_project = client.post(name=from_project['name'],language=language,account_id=to_account)
        to_project_client = client.client_for_path(to_project['project_id'])
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
    parser.add_argument('-t1', '--from_token', default=None, help="Authentication token for the 'from' Daylight project")
    parser.add_argument('-t2', '--to_token', default=None, help="Authentication token for the 'to' Daylight project")
    args = parser.parse_args()
    
    from_api_url = args.from_url.split('/app')[0]
    to_api_url = args.to_url.split('/app')[0]
    from_account = args.from_url.strip('/ ').split('/')[-1]
    to_account = args.to_url.strip('/ ').split('/')[-1]
    if args.from_token:
        from_client = LuminosoClient.connect(url=from_api_url + '/api/v5/projects', token=args.from_token)
    else:
        from_client = LuminosoClient.connect(url=from_api_url + '/api/v5/projects')
    if args.to_token:
        to_client = LuminosoClient.connect(url=to_api_url + '/api/v5/projects', token=args.to_token)
    else:
        to_client = LuminosoClient.connect(url=to_api_url + '/api/v5/projects')
    all_projects = from_client.get()
    all_projects = [p for p in all_projects if p['account_id'] == from_account]
    print('There are {} projects to be copied'.format(len(all_projects)))
    
    copy_projects_to_accounts(all_projects, from_client, to_client, to_account)
    
    
if __name__ == '__main__':
    main()