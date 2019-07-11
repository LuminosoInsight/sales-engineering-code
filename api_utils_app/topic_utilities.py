from luminoso_api import V5LuminosoClient as LuminosoClient
import re

def del_topics(cli):
    """ Delete all topics in a project """
    topics = cli.get('concepts/saved')
    topic_ids = [t['saved_concept_id'] for t in topics]
    cli.delete('concepts/saved', saved_concept_ids=topic_ids)

def __post_topic(cli, topic):
    """ Post a topic to a project """
    if 'exact_term_ids' in topic:
        del topic['exact_term_ids']
    if 'vector' in topic:
        del topic['vector']
    del topic['saved_concept_id']
    cli.post('/concepts/saved/', concepts=[topic])

def copy_topics(cli, from_proj, to_proj):
    """ Copy all topics from a project to another project """
    src_proj = cli.client_for_path(from_proj)
    dest_proj = cli.client_for_path(to_proj)
    src_topics = src_proj.get('/concepts/saved') #get topics from source project
    for topic in reversed(src_topics): 
        __post_topic(dest_proj, topic)

def parse_url(url):
    api_url = url.partition('.com')[0] + '.com/api/v5/'
    proj = url.split('app/projects/')[1].strip('/ ').split('/')[-1]
    return (api_url, proj)
