from luminoso_api import V5LuminosoClient as LuminosoClient
from pack64 import unpack64

def get_all_topics(client):
    topics = client.get('concepts/saved')
    topic_list = []
    for i in range(len(topics)):
        topic_list.append((topics[i]['saved_concept_id'], ','.join(topics[i]['texts']), topics[i]['name']))
    return topic_list

def uncolor_all_topics(client, topic_list):
    concepts = [{'saved_concept_id': t['saved_concept_id'],
                 'name': t['name'],
                 'texts': t['texts'],
                 'color': '#aaaaaa'} for t in topic_list]
    client.put('concepts/saved/', concepts=concepts)
        
def delete_all_topics(client, topic_list):
    topic_ids = [t['saved_concept_id'] for t in topic_list]
    client.delete('concepts/saved', saved_concept_ids=topic_ids)
        
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
    concepts = [{'texts': plutchik[i], 'name': emotions[i], 'color': colors[i]} for i in range(0, 8)]
    client.post('concepts/saved', concepts=concepts)
        
def copy_project(client, acct_id, name):
    new_proj = client.post('copy', name=name, account=acct_id)
    proj_id = new_proj['project_id']
    client = client.client_for_path('/')
    client = client.client_for_path('/projects/{}'.format(proj_id))
    return client
