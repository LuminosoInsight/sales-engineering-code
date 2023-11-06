'''
rebuild_all_projects.py

This will rebuild all the projects in a given workspace. It can also
only rebuild projects that have failed a sentiment build.
'''
import argparse

from luminoso_api import V5LuminosoClient as LuminosoClient
from luminoso_api import LuminosoError


def main():
    parser = argparse.ArgumentParser(
        description='Rebuild all the projects in a specified workspace.'
    )
    parser.add_argument(
        'workspace_id',
        help='The workspace to rebuild all projects in.')
    parser.add_argument(
        '-u', '--host_url',
        required=False,
        default='https://daylight.luminoso.com',
        help='Luminoso API endpoint (e.g., https://daylight.luminoso.com)',
    )
    parser.add_argument(
        '-t', '--test',
        default=False,
        required=False,
        action='store_true',
        help="Just test, don't start the rebuilds")
    parser.add_argument(
        '-s', '--sentiment',
        default=False,
        required=False,
        action='store_true',
        help='Only rebuild projects that have not finished sentiment build')
    parser.add_argument(
        '-w', '--wait_for_build',
        default=False,
        required=False,
        action='store_true',
        help='Only wait for build if this flag is set')
    args = parser.parse_args()

    workspace_id = args.workspace_id
    only_if_sentiment_stalled = args.sentiment

    print('workspace_id: {}'.format(workspace_id))
    print('test: {}'.format(args.test))

    api_url = args.host_url+'/api/v5/'
    client = LuminosoClient.connect(url=api_url, user_agent_suffix='se_code:rebuild_all_projects')

    projects = client.get('/workspaces/'+workspace_id)['projects']
    if not projects:
        print('no projects in workspace_id: {}'.format(workspace_id))
        return

    for p in projects:
        print('considering {}:{}'.format(p['project_id'], p['name']))
        pclient = client.client_for_path('/projects/{}/'.format(p['project_id']))
        pinfo = pclient.get('/', fields=['last_build_info'])['last_build_info']

        is_sentiment_built = pinfo.get('sentiment', {}).get('success')
        if only_if_sentiment_stalled and is_sentiment_built:
            print('  sentiment okay, skipping build')
            continue
        
        if 'start_time' not in pinfo:
            print(f"Skipping empty project :{p['project_id']}")

        elif pinfo['stop_time'] is None:

            if args.test:
                print('  project already building, would wait for completion')
                continue
            print('  project already building, skipping build start')
            print('  waiting for completion...')
            try:
                if args.wait_for_build:
                    pclient.wait_for_sentiment_build()
            except LuminosoError as e:
                print('  Error:', str(e))

        else:
            if args.test:
                print('  would start rebuild')
                continue
            try:
                pclient.post('/build/')
                print('  rebuild started, waiting for completion...')
                if args.wait_for_build:
                    pclient.wait_for_sentiment_build()
            except LuminosoError as e:
                print('  Error:', str(e))


if __name__ == '__main__':
    main()