from luminoso_api import V5LuminosoClient as LuminosoClient
from luminoso_api import LuminosoError

from se_code.bi_tool_export import parse_url

import argparse
import time


def delete_shared_concepts(client):
    scl = client.get('concept_lists/')
    ids_to_delete = [cl['concept_list_id'] for cl in scl]

    if ids_to_delete:
        for id in ids_to_delete:
            client.delete('concept_lists/{}/'.format(id))
        print('{} shared concept lists deleted'.format(len(ids_to_delete)))
    else:
        print('No concept lists to delete')


def copy_shared_concepts(from_client, to_client, overwrite=False):
    concept_lists_from = from_client.get("concept_lists/")
    concept_list_to = to_client.get("concept_lists/")

    # create a dict mapping the list names to id
    to_ids = {cl['name']:cl['concept_list_id'] for cl in concept_list_to}

    add_count = 0
    skip_count = 0
    overwrite_count = 0
    for cl in concept_lists_from:
        if (cl['name'] in to_ids.keys()) and (overwrite):
            # delete the to list
            to_client.delete("concept_lists/{}/".format(to_ids[cl['name']]))
            overwrite_count += 1

            # add count will get incremented in the next step fixing the accounting
            add_count -= 1

        if (not cl['name'] in to_ids.keys()) or (overwrite):           
            # clear all the ids out of the list of concepts
            for c in cl['concepts']:
                c.pop('shared_concept_id', None)

            retry = True
            retry_count = 0
            while retry and retry_count < 3:
                retry_count += 1
                retry = False
                try:
                    to_client.post('concept_lists/', name=cl['name'], concepts=cl['concepts'])
                except LuminosoError as e:
                    eobj = e.args[0]
                    if eobj['error'] == 'TOO_MANY_REQUESTS':
                        time.sleep(1)
                        print("  API retry")
                        retry = True
                    else:
                        print(f"  Error: {e}")
            add_count += 1
        else:
            skip_count += 1

    print('Added {} concept lists, skipped {}, overwrote {}'.format(add_count, skip_count, overwrite_count))


def main():
    parser = argparse.ArgumentParser(
        description='Copy shared concept lists from one project to another.'
    )
    parser.add_argument('from_url', help="The URL of the project to copy all topics from")
    parser.add_argument('to_url', help="The URL of the project to copy all topics into")
    parser.add_argument('--delete', default=False, action='store_true', help="Use this flag to delete all concept listgs in destination project")
    parser.add_argument('--overwrite', default=False, action='store_true', help="Use this flag to overwrite shared concept lists in destination project")

    args = parser.parse_args()

    # parse the from url
    froot_url, fapi_url, faccount_id, fproject_id = parse_url(args.from_url)
    # parse the to url
    troot_url, tapi_url, taccount_id, tproject_id = parse_url(args.to_url)

    from_client = LuminosoClient.connect(url='%s/projects/%s' % (fapi_url, fproject_id), user_agent_suffix='se_code:copy_shared_concepts:from')
    to_client = LuminosoClient.connect(url='%s/projects/%s' % (tapi_url, tproject_id), user_agent_suffix='se_code:copy_shared_concepts:to')
        
    if args.delete:
        delete_shared_concepts(to_client)
    copy_shared_concepts(from_client, to_client, args.overwrite)


if __name__ == '__main__':
    main()