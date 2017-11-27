from luminoso_api import LuminosoClient
import re

def del_topics(cli, acct_id, proj_id):
    """ Delete all topics in a project """
    cli = cli.change_path(acct_id+'/'+proj_id)
    topics = cli.get('/topics')
    topic_ids = [t['_id'] for t in topics]
    for tid in topic_ids:
        cli.delete('/topics/id/' + tid)

def __post_topic(cli, topic):
    """ Post a topic to a project """
    del topic['vector']
    del topic['_id']
    cli.post('/topics', **topic)

def copy_topics(cli, from_acct, from_proj, to_acct, to_proj):
    """ Copy all topics from a project to another project """
    src_proj = cli.change_path(from_acct + '/' + from_proj)
    dest_proj = cli.change_path(to_acct + '/' + to_proj)        
    src_topics = src_proj.get('/topics') #get topics from source project
    for topic in reversed(src_topics): 
        __post_topic(dest_proj, topic)

def parse_url(url):
    if '?account=' in url: #old url format
        acct = re.search("\?account=(.*)&", url).group(1)
        proj = re.search("&projectId=(.*)", url).group(1)
    else:
        acct,proj = url.split('app/#/projects/')[1].split('/')
    return (acct, proj)