from luminoso_api import LuminosoClient
from pack64 import unpack64

def get_all_topics(client):
    topics = client.get('topics')
    topic_list = []
    for i in len(topics):
        topic_ids.append((topics[i]['_id'],topics[i]['text']))
    return topic_list

def uncolor_all_topics(client, topic_list):
    for i in len(topic_list):
        client.put('topics/id/{}/'.format(topic_list[i][0]), text=topic_list[i][1], name=topic_list[i][1])
        
def delete_all_topics(client, topic_list):
    for i in len(topic_list):
        client.delete('topics/id/{}/'.format(topic_list[i][0]))
        
def add_plutchik(client):
    joy = ['ecstasy', 'joy', 'serene', 'happy']
    trust = ['admire', 'trust', 'accept']
    terror = ['fear', 'apprehension', 'terror', 'scare']
    surprise = ['amaze', 'surprise', 'distract', 'shock']
    sad = ['grief', 'sad', 'pensive', 'miserable']
    disgust = ['loathe', 'disgust', 'bored', 'gross']
    anger = ['rage', 'angry', 'annoy', 'wrath']
    anticipate = ['vigilant', 'anticipate', 'interest', 'await']
    plutchik = [joy, trust, terror, surprise, sad, disgust, anger, anticipate]
    colors = ['#f9ef2a', '#b2ed28', '#42a010', '#24e2d9', '#2d47bc', '#af48f9', '#ce2e23', '#e88f29']
    for i in range (0, 8):
        for j in range(0, len(plutchik[i])):
            client.post('topics', text=plutchik[i][j], color=colors[i])
            
def main(args):
    client = LuminosoClient.connect(args.api_url, args.username)
    client = client.change_path('/')
    client = client.change_path('/projects/{}/{}'.format(args.acct_id, args.proj_id))
    topic_list = get_all_topics(client)
    if args.uncolor:
        uncolor_all_topics(client, topic_list)
    if args.delete:
        delete_all_topics(client, topic_list)
    add_plutchik(client)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Create a classification model based on an existing project using subsets as labels.'
    )
    parser.add_argument(
        'acct_id',
        help="The ID of the account that owns the testing project"
        )
    parser.add_argument(
        'proj_id',
        help="The ID of the project to add Plutchik's wheel to"
        )
    parser.add_argument(
        '-u', '--username',
        help='Username (email) of Luminoso account'
        )
    parser.add_argument(
        '-a', '--api_url',
        help='URL of Luminoso API endpoint (https://eu-analytics.luminoso.com/api/v4)'
        )
    parser.add_argument(
        '-u', '--uncolor', default=False, action='store_true',
        help='If we want to remove the color from existing topics'
        )
    parser.add_argument(
        '-d', '--delete', default=False, action='store_true',
        help="If we want to delete the existing topics"
        )
    args = parser.parse_args()
    main(args)