import argparse

from luminoso_api import V5LuminosoClient as LuminosoClient
from se_code.copy_shared_concepts import copy_shared_concepts


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

        # get the docs and filter for new project
        from_docs = get_all_docs(from_client_project)
        from_docs = [{'text': d['text'],
                      'title': d['title'],
                      'metadata': d['metadata']} for d in from_docs]

        # Create a new project
        client = to_client.client_for_path('/projects')
        to_project = client.post(name=from_project['name'], language=language, 
                                 workspace_id=to_account)
        to_client_project = client.client_for_path(to_project['project_id'])
        to_client_project.post('upload', docs=from_docs)

        # Copy shared concept lists
        copy_shared_concepts(from_client_project, to_client_project)

        to_client_project.post('build')    
        print('Copied project: ' + from_project['name'])
        
        
def main():
    parser = argparse.ArgumentParser(
        description='Copy all projects from an account on one cloud into another account on another cloud.'
    )
    parser.add_argument('from_url', help="The URL of the account that owns the current projects")
    parser.add_argument('to_url', help="The URL of the account to copy all projects to")
    args = parser.parse_args()
    
    from_api_url = args.from_url.split('/app')[0]
    to_api_url = args.to_url.split('/app')[0]

    from_account = args.from_url.strip('/').split('/')[5]
    to_account = args.to_url.strip('/').split('/')[5]

    from_client = LuminosoClient.connect(url=from_api_url + '/api/v5/projects')
    to_client = LuminosoClient.connect(url=to_api_url + '/api/v5/projects')
    all_projects = from_client.get()
    all_projects = [p for p in all_projects if p['account_id'] == from_account]

    print('There are {} projects to be copied'.format(len(all_projects)))
    
    copy_projects_to_accounts(all_projects, from_client, to_client, to_account)
    
    
if __name__ == '__main__':
    main()