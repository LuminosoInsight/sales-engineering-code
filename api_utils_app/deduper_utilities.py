import sys
import time
sys.path.insert(0, '../se_code/')
from se_code.deduper import Deduper

def __get_token(cli):
    cli2 = cli.change_path('/')
    return cli2.get('/user/tokens/')[0]['token']

def __retain_shortest(docs):
    return sorted(docs, key = lambda d: len(d['text']))[0]

def __retain_longest(docs):
    return sorted(docs, key = lambda d: len(d['text']))[-1]

def dedupe(acct, proj, cli, recalc=True, reconcile_func=None, copy=False):
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
    num_deleted = deduper.dedupe()
    url = 'https://analytics.luminoso.com/explore.html?account='+acct+'&projectId='+proj
    while recalc and cli.get('jobs'):
        time.sleep(1)
    return {'num':num_deleted, 'url':url}
