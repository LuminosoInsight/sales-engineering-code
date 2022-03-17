from luminoso_api import V5LuminosoClient as LuminosoClient

import argparse
import json


def load_filter(args):
    try:
        if args.filter:
            return json.loads(args.filter)
        elif args.filter_json_file:
            with open(args.filter_json_file, "r") as read_file:
                return json.load(read_file)
        else:
            print("No filter given, must be a string or file")
            return None
    except Exception as e:
        print("ERROR parsing filter: {}".format(e))
        return None

def main():
    parser = argparse.ArgumentParser(
        description='Copy project with filter.'
    )
    parser.add_argument('project_url',
                        help="URL of the project to copy from")
    parser.add_argument('-w', '--workspace_id', default=None,
                        help=('Workspace id to use if creating new project'))
    parser.add_argument('-m', '--metadata', default=False,
                        action='store_true',
                        help=('Just show metadata for project_url useful for examining projects before copy'))
    parser.add_argument('-b', '--wait_for_build', default=False,
                        action='store_true',
                        help=('After the copy, wait for the build to complete.'))

    parser.add_argument('-c', '--concepts', default=None,
                        help=('A string of concepts to copy over. Example ["lettuce","cucumber"]'))
    parser.add_argument('-t', '--match_type', default="exact",
                        help=('Type of concept match to use. Values: exact, conceptual, both'))

    filter_group = parser.add_mutually_exclusive_group()
    filter_group.add_argument('-f', '--filter', default=None, 
                              help=('JSON string filter to use in selecting documents to copy. Example: [{"name": "Bank Name", "values": ["Your Bank"]},{"name": "Version", "values": ["2.2.108"]}]'))
    filter_group.add_argument('-j', '--filter_json_file', default=None,
                              help=('JSON file filter to use in selecting documents to copy'))

    to_group = parser.add_mutually_exclusive_group()
    to_group.add_argument('-n', '--project_name', default=None,
                          help=('Name of new project to create'))
    to_group.add_argument('-u', '--to_url', default=None,
                          help=('URL of existing project to copy into'))
    args = parser.parse_args()

    project_id = args.project_url.strip('/').split('/')[6]
    api_url = '/'.join(args.project_url.strip('/').split('/')[:3]) + '/api/v5'
    proj_apiv5 = '{}/projects/{}'.format(api_url, project_id)

    project_client = LuminosoClient.connect(proj_apiv5, user_agent_suffix='se_code:copy_project_with_filter')

    if args.concepts:
        try:
            search_concepts = {"texts": json.loads(args.concepts)}
        except Exception as e:
            print('ERROR parsing concepts. Should be in ["conceptA","conceptB"] format. Error : {}'.format(e))
            return None

    if args.metadata:
        # just show the metadata
        md_data = project_client.get("metadata/")
        print("Project metadata\n{}".format(json.dumps(md_data['result'], indent=2)))
    elif args.project_name:
        # create the new project

        # load the filter
        jfilter = load_filter(args)
        if not jfilter:
            return

        try:
            if args.workspace_id:
                if args.concepts:
                    to_project_info = project_client.post("copy/", workspace_id=args.workspace_id,
                                        filter=jfilter, search=search_concepts,
                                        match_type=args.match_type,
                                        name=args.project_name)
                else:
                    to_project_info = project_client.post("copy/", workspace_id=args.workspace_id,
                                        filter=jfilter, name=args.project_name)
            else:
                if args.concepts:
                    to_project_info = project_client.post("copy/", filter=jfilter,
                                        search=search_concepts,
                                        match_type=args.match_type,
                                        name=args.project_name)
                else:
                    to_project_info = project_client.post("copy/", filter=jfilter,
                                        name=args.project_name)
        except Exception as e:
            print("ERROR copying data: {}".format(e))
            return

        print('Project copy to new project complete')

        if (args.wait_for_build):
            print('Waiting for build')
            project_to_client = LuminosoClient.connect(
                api_url+"/projects/{}/".format(to_project_info['project_id']),
                user_agent_suffix='se_code:copy_project_with_filter')
            project_to_client.wait_for_build(wait_for_sentiment=True)
            print('Build complete.')

    elif args.to_url:
        # copy to existing project

        # get the to project id from the to_project_url
        to_project_id = args.to_url.strip('/').split('/')[6]

        # make sure the to_project isn't currently building.
        # msg user and wait for build to complete if it is
        project_to_client = LuminosoClient.connect(
            api_url+"/projects/{}/".format(to_project_id),
            user_agent_suffix='se_code:copy_project_with_filter')
        project_to_info = project_to_client.get("/")
        if (('success' not in project_to_info['last_build_info']) or
           (not project_to_info['last_build_info']['success'])):
            print('The to_project is building. Waiting for build to complete before copying.')
            project_to_client.wait_for_build(wait_for_sentiment=True)

        # load the filter
        jfilter = load_filter(args)
        if not jfilter:
            return

        try:
            if args.concepts:
                project_client.post('copy/', destination=to_project_id,
                                    filter=jfilter,
                                    search=search_concepts,
                                    match_type=args.match_type)
            else:
                project_client.post('copy/', destination=to_project_id,
                                    filter=jfilter)
        except Exception as e:
            print("ERROR copying data: {}".format(e))
            return

        # kick off the build - api auto builds now and may go away in the copy call.
        try:
            # rebuild with sentiment
            project_client.post('/build/')
        except LuminosoClientError as e:
            details = e.args[0]
            # fail PROJECT_LOCKED silently as the api is going to change and
            # no longer build automatically. When it does, the build will kick
            # off normally here.
            if details['error'] != 'PROJECT_LOCKED':
                raise

        print('Project copy to existing project complete - rebuild started')

        if (args.wait_for_build):
            print('Waiting for build to complete.')
            project_to_client.wait_for_build(wait_for_sentiment=True)
            print('Build complete.')

    else:
        print('ERROR must have either --project_name or --to_url parameter. Received neither')


if __name__ == '__main__':
    main()
