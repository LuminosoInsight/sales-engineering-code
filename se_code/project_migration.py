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


def copy_projects_to_workspace(all_projects, from_client, to_client,
                               to_workspace):
    for from_project in all_projects:
        project_name = from_project['name']

        # Connect to project
        from_client_project = from_client.client_for_path(
            from_project['project_id']
        )
        print('Connected to project: ' + project_name)
        # get the docs and filter for new project
        from_docs = get_all_docs(from_client_project)
        from_docs = [{'text': d['text'],
                      'title': d['title'],
                      'metadata': d['metadata']} for d in from_docs]

        # Create a new project
        to_project = to_client.post(
            name=project_name, description=from_project['description'],
            language=from_project['language'], workspace_id=to_workspace
        )
        to_client_project = to_client.client_for_path(to_project['project_id'])
        to_client_project.post('upload', docs=from_docs)

        # Copy shared concept lists
        copy_shared_concepts(from_client_project, to_client_project)

        to_client_project.post('build')
        to_client_project.wait_for_sentiment_build()
        print('Copied project: ' + project_name)
        
        
def main():
    parser = argparse.ArgumentParser(
        description=('Copy all projects from a workspace on one cloud into a'
                     ' workspace on another cloud.')
    )
    parser.add_argument(
        'from_url',
        help='The URL of a project in the workspace to copy from'
    )
    parser.add_argument(
        'to_url',
        help='The URL of a project in the workspace to copy to'
    )
    args = parser.parse_args()
    
    from_api_url = args.from_url.split('/app')[0]
    to_api_url = args.to_url.split('/app')[0]

    from_workspace = args.from_url.strip('/').split('/')[5]
    to_workspace = args.to_url.strip('/').split('/')[5]

    from_client = LuminosoClient.connect(
        url=from_api_url + '/api/v5/projects',
        user_agent_suffix='se_code:project_migration:from'
    )
    to_client = LuminosoClient.connect(
        url=to_api_url + '/api/v5/projects',
        user_agent_suffix='se_code:project_migration:to'
    )
    all_projects = from_client.get(
        fields=('project_id', 'name', 'description', 'language',
                'document_count'),
        workspace_id=from_workspace
    )

    print('There are {} projects to be copied'.format(len(all_projects)))
    
    copy_projects_to_workspace(all_projects, from_client, to_client,
                               to_workspace)
    
    
if __name__ == '__main__':
    main()