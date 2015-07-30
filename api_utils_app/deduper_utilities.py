import random
import sys
sys.path.insert(0, '../se_code/')
from deduper import Deduper

def __get_token(cli):
    cli2 = cli.change_path('/')
    return cli2.get('/user/tokens/')[0]['token']

def __retain_shortest(docs):
    return sorted(docs, key = lambda d: len(d['text']))[0]

def __retain_longest(docs):
    return sorted(docs, key = lambda d: len(d['text']))[-1]

def dedupe(acct, proj, cli, reconcile_func=None, copy=False):
    if copy:
        new_proj = cli.post('copy')    
    initial_count = len(cli.get('docs/ids'))
    if reconcile_func == 'shortest':
        reconcile_func = __retain_shortest
    elif reconcile_func == 'longest':
        reconcile_func = __retain_longest
    else:
        reconcile_func = None
    token = __get_token(cli)
    if initial_count > 40000:
        deduper = Deduper(acct=acct, proj=proj, token=token,
                    split_amt=40000, reconcile_func=reconcile_func)
    else:
        deduper = Deduper(acct=acct, proj=proj, token=token,
                    reconcile_func=reconcile_func)
    deduper.dedupe()
    final_count = len(cli.get('docs/ids'))
    num_deleted = str(initial_count - final_count)
    if copy:
        pid = new_proj['project_id']
        url = 'https://dashboard.luminoso.com/v4/explore.html?account='+acct+'&projectId='+pid
        return {'num':num_deleted, 'url':url}
    return {'num':num_deleted, 'url':''}
