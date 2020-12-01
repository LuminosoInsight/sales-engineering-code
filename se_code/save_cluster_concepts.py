from luminoso_api import V5LuminosoClient as LuminosoClient

import argparse


def save_cluster_concepts(
    client, num_clusters, num_cluster_concepts, shared_list_name=None
):
    concept_selector = {
        'type': 'suggested',
        'num_clusters': num_clusters,
        'num_cluster_concepts': num_cluster_concepts,
    }
    suggested_concepts = client.get('concepts', concept_selector=concept_selector)[
        'result'
    ]

    save_list = [{'name': c['name'], 'texts': c['texts']} for c in suggested_concepts]

    if shared_list_name:
        try:
            print('{}'.format(save_list))
            result = client.post(
                '/concept_lists', name=shared_list_name, concepts=save_list
            )
            print('list_id: {}'.format(result['concept_list_id']))
        except Exception as e:
            print('Error: Does the shared concept list already exist?')
            print('Error: {}'.format(e))
    else:
        saved_concepts = client.get('concepts/saved')

        if len(saved_concepts) > 0:
            # out with the old
            saved_concept_ids = [sc['saved_concept_id'] for sc in saved_concepts]

            # use the older (soon deprecated) saved concepts endpoint
            client.delete('concepts/saved', saved_concept_ids=saved_concept_ids)

        client.post('concepts/saved', concepts=save_list)


def main():
    parser = argparse.ArgumentParser(
        description='Export data to Tableau compatible CSV files.'
    )
    parser.add_argument(
        'project_url', help='The URL of the project to save the clusters in'
    )
    parser.add_argument(
        '-c', '--num_clusters', required=True, help='The number of clusters'
    )
    parser.add_argument(
        '-n',
        '--num_cluster_concepts',
        required=True,
        help='The number of concepts per cluster',
    )
    parser.add_argument(
        '-l', '--list_name', default=None, help='The name of this shared concept list'
    )

    args = parser.parse_args()
    proj_id = args.project_url.strip('/').split('/')[6]
    proj_root = '/'.join(args.project_url.strip('/').split('/')[:3]).strip('/')

    client = LuminosoClient.connect(url='%s/api/v5/projects/%s' % (proj_root, proj_id))

    save_cluster_concepts(
        client, int(args.num_clusters), int(args.num_cluster_concepts), args.list_name
    )


if __name__ == '__main__':
    main()
