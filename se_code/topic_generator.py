from luminoso_api import LuminosoClient
import argparse


# Copy the 7 colors from the Luminoso Analytics UI

COLORS = [
    '#00aaff',
    '#15cf20',
    '#ffbb2a',
    '#f07600',
    '#ff0032',
    '#ff379c',
    '#7d27bc'
]


def run(account_id, project_id, username, num_clusters, num_cluster_terms,
        api_url='https://analytics.luminoso.com/api/v4', create=False):
    client = LuminosoClient.connect(
        '%s/projects/%s/%s/' % (api_url, account_id, project_id),
        username=username
    )
    selected_clusters = client.get(
        'terms/clusters', num_clusters=num_clusters,
        num_cluster_terms=num_cluster_terms
    )
    print("\nSelected clusters:")
    for cluster in selected_clusters:
        print(', '.join([term['text'] for term in cluster]))

    if create:
        existing_topics = client.get('topics/')
        for topic in existing_topics:
            if topic['name'].endswith('(auto)'):
                print("Deleting existing topic: %s" % topic['name'])
                topic_id = topic['_id']
                client.delete('topics/id/%s/' % topic_id)

        pos = 0
        for i, cluster in enumerate(selected_clusters):
            if i < 7:
                color = COLORS[i]
            else:
                color = '#808080'
            for term in cluster:
                print("Creating topic: %s" % term['text'])
                client.post(
                    'topics/',
                    text=term['text'],
                    name='%s (auto)' % term['text'],
                    color=color,
                    position=pos
                )
                pos += 1
    return selected_clusters


def main():
    parser = argparse.ArgumentParser(
        description='Automatically find representative topics for a Luminoso project.'
    )
    parser.add_argument('account_id', help="The ID of the account that owns the project, such as 'demo'")
    parser.add_argument('project_id', help="The ID of the project to analyze, such as '2jsnm'")
    parser.add_argument('username', help="A Luminoso username with access to the project")
    parser.add_argument('-n', '--num-colors', type=int, default=7, help="Number of topic colors to generate (max 7)")
    parser.add_argument('-t', '--topics-per-color', type=int, default=4, help="Number of topics of each color to generate")
    parser.add_argument('-a', '--api-url', default='https://analytics.luminoso.com/api/v4', help="The base URL for the Luminoso API (defaults to the production API, https://analytics.luminoso.com/api/v4)")
    parser.add_argument('-c', '--create', action='store_true', help="Actually create the topics, marking them as (auto) and deleting previous auto-topics")
    args = parser.parse_args()
    run(args.account_id, args.project_id, args.username, args.num_colors,
        args.topics_per_color, args.api_url, args.create)


if __name__ == '__main__':
    main()
