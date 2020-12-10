import argparse

from luminoso_api import V5LuminosoClient as LuminosoClient


def main():
    parser = argparse.ArgumentParser(
        description='Show the metadata in a specific project'
    )
    parser.add_argument('project_url', help="The complete URL of the Daylight project")
    args = parser.parse_args()

    root_url = args.project_url.strip('/ ').split('/app')[0]
    api_url = root_url + '/api/v5'

    project_id = args.project_url.strip('/').split('/')[6]

    client = LuminosoClient.connect(url='%s/projects/%s' % (api_url.strip('/'), project_id))

    result = client.get('/metadata')['result']
    for md in result:
        if 'values' in md:
            md['unique_values'] = len(md['values'])
        else:
            md['unique_values'] = 0

    metadata = [{'name': md['name'], 'type': md['type'], 'unique_values': md['unique_values']} for md in result]

    print("name,type,unique_values")
    for md in metadata:
        print("{},{},{}".format(md['name'], md['type'], md['unique_values']))


if __name__ == '__main__':
    main()
