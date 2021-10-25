import argparse

from luminoso_api import V5LuminosoClient as LuminosoClient

'''
rebuild_all_projects.py 

This will rebuild all the projects in a given workspace. It can also
only rebuild projects that have failed a sentiment build.
'''


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
        help="Only rebuild projects that have not finished sentiment build")
    args = parser.parse_args()

    workspace_id = args.workspace_id
    only_if_sentiment_stalled = args.sentiment

    print("workspace_id: {}".format(workspace_id))
    print("test: {}".format(args.test))

    api_url = args.host_url+"/api/v5/"
    client = LuminosoClient.connect(url=api_url, user_agent_suffix='se_code:rebuild_all_projects')

    projects = client.get("/workspaces/"+workspace_id)['projects']
    if not projects:
        print("no projects in workspace_id: {}".format(workspace_id))
        return

    for p in projects:
        if args.test:
            print("test: rebuild project not started: {}:{}".format(p['project_id'], p['name']))
        else:

            print("considering {}:{}".format(p['project_id'], p['name']))
            pclient = client.client_for_path('/projects/{}/'.format(p['project_id']))
            pinfo = pclient.get("/")
            if ('sentiment' in pinfo['last_build_info']) and ('success' in pinfo['last_build_info']['sentiment']):
                is_sentiment_built = pinfo['last_build_info']['sentiment']['success']
            else:
                is_sentiment_built = False

            try:
                if only_if_sentiment_stalled:
                    if not is_sentiment_built:
                        pclient.post('/build/')
                        print("  rebuild started, waiting for completion...")
                    else:
                        print("  sentiment okay, skipping build")
                elif pinfo['last_build_info']['stop_time'] is not None:
                    pclient.post('/build/')
                    print("  rebuild started, waiting for completion...")
                else:
                    print("  project all ready building, skipping build start")
                    print("  waiting for completion...")

                pclient.wait_for_sentiment_build()
            except Exception as e:
                print(e)


if __name__ == '__main__':
    main()