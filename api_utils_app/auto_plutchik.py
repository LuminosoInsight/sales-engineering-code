from luminoso_api import LuminosoClient
from pack64 import unpack64

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
        negative_dict = {'text': emotions[(i + 4) % 8]}
        client.post('topics', text=emotions[i], color=colors[i], name=emotions[i], negative=negative_dict)
        
def copy_project(client, acct_id, name):
    new_proj = client.post('copy', destination=name)
    proj_id = new_proj['project_id']
    client = client.change_path('/')
    client = client.change_path('/projects/{}/{}'.format(acct_id, proj_id))
    return client
