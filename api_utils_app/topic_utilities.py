from luminoso_api import LuminosoClient

def del_topics(cli, acct_id, proj_id):
    """ Delete all topics in a project """
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