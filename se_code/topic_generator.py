from luminoso_api import LuminosoClient
from se_code.reptree import RepTree
from pack64 import unpack64
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


def fetch_terms_from_project(client, count=500):
    """
    Fetch a desired number of terms from a given project.

    Returns [{term:, text:, vector:, relevance:}, ...]
    """

    raw_terms = client.get('/terms/', limit=count)
    return [dict(term=rt['term'],
                 text=rt['text'],
                 vector=unpack64(rt['vector']),
                 relevance=rt['score'])
            for rt in raw_terms if rt['vector'] is not None]


def find_topics(client, num_topics):
    """
    Get terms from a project, arrange them into a RepTree, and extract
    a flat list of topics from them.

    A text representation of the RepTree will also be displayed.
    """
    terms = fetch_terms_from_project(client, count=1000)
    tree = RepTree.from_term_list(terms)
    print("Topic tree:")
    print(tree.show_tree(min_score=21, max_depth=10))
    topic_list = tree.flat_topic_list(count=num_topics)
    return topic_list


def run(account_id, project_id, username, num_groups, topics_per_group,
        api_url='https://analytics.luminoso.com/api/v4', create=False):
    client = LuminosoClient.connect(
        '%s/projects/%s/%s/' % (api_url, account_id, project_id),
        username=username
    )
    selected_topic_groups = find_topics(client, num_groups)
    print("\nSelected topic groups:")
    for group in selected_topic_groups:
        print('-', group.describe(topics_per_group))

    if create:
        existing_topics = client.get('topics/')
        for topic in existing_topics:
            if topic['name'].endswith('(auto)'):
                print("Deleting existing topic: %s" % topic['name'])
                topic_id = topic['_id']
                client.delete('topics/id/%s/' % topic_id)

        pos = 0
        for i, group in enumerate(selected_topic_groups):
            topic_texts = group.term_texts(topics_per_group)
            if i < 7:
                color = COLORS[i]
            else:
                color = '#808080'
            for topic_text in topic_texts:
                print("Creating topic: %s" % topic_text)
                client.post(
                    'topics/',
                    text=topic_text,
                    name='%s (auto)' % topic_text,
                    color=color,
                    position=pos
                )
                pos += 1
    return selected_topic_groups


def main():
    parser = argparse.ArgumentParser(
        description='Automatically find representative topics for a Luminoso project.'
    )
    parser.add_argument('account_id', help="The ID of the account that owns the project, such as 'demo'")
    parser.add_argument('project_id', help="The ID of the project to analyze, such as '2jsnm'")
    parser.add_argument('username', help="A Luminoso username with access to the project")
    parser.add_argument('-n', '--num-colors', type=int, default=5, help="Number of topic colors to generate (max 7)")
    parser.add_argument('-t', '--topics-per-color', type=int, default=3, help="Number of topics of each color to generate")
    parser.add_argument('-a', '--api-url', default='https://analytics.luminoso.com/api/v4', help="The base URL for the Luminoso API (defaults to the production API, https://analytics.luminoso.com/api/v4)")
    parser.add_argument('-c', '--create', action='store_true', help="Actually create the topics, marking them as (auto) and deleting previous auto-topics")
    args = parser.parse_args()
    run(args.account_id, args.project_id, args.username, args.num_colors,
        args.topics_per_color, args.api_url, args.create)


if __name__ == '__main__':
    main()
