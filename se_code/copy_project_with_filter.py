from luminoso_api import V5LuminosoClient as LuminosoClient

import argparse
import json


def load_filter(args):
    if args.filter:
        return json.loads(args.filter)
    elif args.filter_json_file:
        with open(args.filter_json_file, "r") as read_file:
            return json.load(read_file)
    else:
        print("No filter given, must be a string or file")
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
                        help=('just show metadata for project_url useful for examining projects before copy'))

    filter_group = parser.add_mutually_exclusive_group()
    filter_group.add_argument('-f', '--filter', default=None, 
                              help=('JSON string filter to use in selecting documents to copy. Example: [{"name": "Bank Name", "values": ["Your Bank"]},{"name": "Version", "values": ["2.2.108"]}]"]'))
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

    project_client = LuminosoClient.connect(proj_apiv5)

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

        if args.workspace_id:
            project_client.post("copy/", workspace_id=args.workspace_id,
                                filter=jfilter, name=args.project_name)
        else:
            project_client.post("copy/", filter=jfilter, 
                                name=args.project_name)

        print('Project copy to new project complete')
    elif args.to_url:
        # copy to existing project

        # get the to project id from the to_project_url
        to_project_id = args.to_url.strip('/').split('/')[6]

        # load the filter
        jfilter = load_filter(args)
        if not jfilter:
            return

        project_client.post("copy/", destination=to_project_id,
                            filter=jfilter)
        print('Project copy to existing project complete - rebuild started')

    else:
        print("ERROR must have either --project_name or --to_url parameter. Received neither")


if __name__ == '__main__':
    main()
