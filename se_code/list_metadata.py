import argparse
import json

from luminoso_api import V5LuminosoClient as LuminosoClient


def main():
    parser = argparse.ArgumentParser(
        description='Show the metadata in a specific project'
    )
    parser.add_argument('project_url', help="The complete URL of the Daylight project")
    parser.add_argument('-j', '--json', action='store_true', required=False, help="Output a json string")
    parser.add_argument('-t', '--type_map', action='store_true', required=False, help="Ouput a typemap in json")
    args = parser.parse_args()

    root_url = args.project_url.strip('/ ').split('/app')[0]
    api_url = root_url + '/api/v5'

    project_id = args.project_url.strip('/').split('/')[6]

    client = LuminosoClient.connect(url='%s/projects/%s' % (api_url.strip('/'), project_id))

    result = client.get('/metadata')['result']

    if args.types_only:
        metatypes = {md['name']:md['type'] for md in result}
        print(json.dumps(metatypes))
    elif args.json:
        print(json.dumps(result))
    else:
        print("name,type,unique_values")
        for md in result:
            if 'values' in md:
                unique_values = len(md['values'])
            else:
                unique_values = 0
            print("{},{},{}".format(md['name'], md['type'], unique_values))


if __name__ == '__main__':
    main()
