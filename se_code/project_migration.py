import argparse

from luminoso_api import V5LuminosoClient as LuminosoClient
from se_code.copy_shared_concepts import copy_shared_concepts


BATCH_SIZE = 1000


def copy_projects_to_workspace(all_projects, from_client, to_client,
                               to_workspace, batch_size=BATCH_SIZE):
    for from_project in all_projects:
        project_name = from_project['name']

        # Connect to project
        from_client_project = from_client.client_for_path(
            from_project['project_id']
        )
        print('Connected to project: ' + project_name)

        # Create a new project
        to_project = to_client.post(
            name=project_name, description=from_project['description'],
            language=from_project['language'], workspace_id=to_workspace
        )
        to_client_project = to_client.client_for_path(to_project['project_id'])

        # get the docs
        offset = 0
        while offset < from_project['document_count']:
            docs = from_client_project.get(
                'docs', limit=batch_size, offset=offset,
                fields=('text', 'title', 'metadata')
            )['result']
            to_client_project.post('upload', docs=docs)
            offset += batch_size

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