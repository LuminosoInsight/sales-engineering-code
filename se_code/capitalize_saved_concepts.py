import argparse

from luminoso_api import V5LuminosoClient


def main():
    parser = argparse.ArgumentParser(
        description='Capitalize shared conecept list concepts in a project'
    )
    parser.add_argument('url', help="The URL of the project to capitalize shared concept list concepts for.")
    args = parser.parse_args()

    root_url = args.url.strip('/ ').split('/app')[0]
    api_url = root_url + '/api/v5'
    project_id = args.url.strip('/').split('/')[6]

    client = V5LuminosoClient.connect('%s/projects/%s' % (api_url, project_id))

    # get the list of shared concepts
    concept_lists_raw = client.get("concept_lists/")
    for cl in concept_lists_raw:
        # capitalize all the concept names
        for c in cl['concepts']:
            c['name'] = c['name'].capitalize()

        # write the list back to the saved list
        client.put('concept_lists/{}/concepts/'.format(cl['concept_list_id']), concepts=cl['concepts'])

    print("Shared Concepts Capitalized")


if __name__ == '__main__':
    main()