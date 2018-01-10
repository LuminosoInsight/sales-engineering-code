from luminoso_api import LuminosoClient
from pack64 import unpack64

import argparse

def get_all_topics(client):
    topics = client.get('topics')
    topic_list = []
    for i in range(len(topics)):
        topic_list.append((topics[i]['_id'],topics[i]['text'],topics[i]['name']))
    return topic_list

def uncolor_all_topics(client, topic_list):
    for i in range(len(topic_list)):
        client.put('topics/id/{}/'.format(topic_list[i][0]), text=topic_list[i][1], name=topic_list[i][2])
        
def delete_all_topics(client, topic_list):
    for i in range(len(topic_list)):
        client.delete('topics/id/{}/'.format(topic_list[i][0]))
        
def add_plutchik(client):
    joy = ['ecstasy', 'joy', 'serene', 'happy']
    trust = ['admire', 'trust', 'accept', 'faith']
    terror = ['fear', 'apprehension', 'terror', 'scare']
    surprise = ['amaze', 'surprise', 'distract', 'shock']
    sad = ['grief', 'sad', 'pensive', 'miserable']
    disgust = ['loathe', 'disgust', 'bored', 'gross']
    anger = ['rage', 'angry', 'annoy', 'wrath']
    anticipate = ['vigilant', 'anticipate', 'interest', 'await']
    plutchik = [joy, trust, terror, surprise, sad, disgust, anger, anticipate]
    emotions = ['joy', 'trust', 'terror', 'surprise', 'sad', 'disgust', 'anger', 'anticipate']
    colors = ['#f9ef2a', '#b2ed28', '#42a010', '#24e2d9', '#2d47bc', '#af48f9', '#ce2e23', '#e88f29']
    for i in range (0, 8):
        topic_to_add = ''
        topic_negative = ''
        #for j in range(0, len(plutchik[i])):
        #    if j == 0:
        #        topic_to_add += plutchik[i][j]
        #    else:
        #        topic_to_add += (' ' + plutchik[i][j])
        #for j in range(0, len(plutchik[(i + 4) % 8])):
        #    if j == 0:
        #        topic_negative += plutchik[(i + 4) % 8][j]
        #    else:
        #        topic_negative += (' ' + plutchik[(i + 4) % 8][j])
        negative_dict = {'text': emotions[(i + 4) % 8]}
        client.post('topics', text=emotions[i], color=colors[i], name=emotions[i], negative=negative_dict)
        #negative_dict = {'text': topic_negative}
        #client.post('topics', text=topic_to_add, color=colors[i], name=emotions[i], negative=negative_dict)
        
def copy_project(client, acct_id, name):
    current_name = client.get()['name']
    new_proj = client.post('copy', destination=current_name + name)
    proj_id = new_proj['project_id']
    client = client.change_path('/')
    client = client.change_path('/projects/{}/{}'.format(acct_id, proj_id))
    return client
            
def main(args):
    client = LuminosoClient.connect(args.api_url, args.username)
    client = client.change_path('/')
    client = client.change_path('/projects/{}/{}'.format(args.acct_id, args.proj_id))
    if not args.no_copy:
        client = copy_project(client, args.acct_id, args.copy_name)
    
    topic_list = get_all_topics(client)
    if args.uncolor:
        uncolor_all_topics(client, topic_list)
    if not args.dont_delete:
        delete_all_topics(client, topic_list)
    if not args.no_plutchik:
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
        '-c', '--uncolor', default=False, action='store_true',
        help='If we want to just remove the color from existing topics'
        )
    parser.add_argument(
        '-d', '--dont_delete', default=False, action='store_true',
        help="If we don\'t want to delete the existing topics"
        )
    parser.add_argument(
        '-n', '--no_plutchik', default=False, action='store_true',
        help="If you don\'t want to add the Plutchik wheel emotions"
        )
    parser.add_argument(
        '-s', '--no_copy', default=False, action='store_true',
        help="If you want to modify the original project instead of creating a copy"
        )
    parser.add_argument(
        '-b', '--copy_name', default='_plutchik',
        help="The name of the project copy"
    )
    args = parser.parse_args()
    main(args)