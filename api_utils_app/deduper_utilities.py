import sys
import time
from se_code.deduper import Deduper,__retain_shortest,__retain_longest
from luminoso_api import V5LuminosoClient as LuminosoClient

#def __get_token(cli):
#    cli2 = cli.change_path('/')
#    return cli2.get('/user/tokens/')[0]['token']

def dedupe(cli, recalc=True, reconcile_func=None, copy=False):
    if copy:
        copy_info = cli.post('copy')
        cli = cli.client_for_path("/projects/"+copy_info['project_id'])

        print("Waiting for copy to complete.")
        cli.wait_for_build()
        print("Copy/Build done.")

    initial_count = cli.get('/')['document_count']

    if reconcile_func == 'shortest':
        reconcile_func = __retain_shortest
    elif reconcile_func == 'longest':
        reconcile_func = __retain_longest
    else:
        reconcile_func = None
        
    if initial_count > 40000:
        deduper = Deduper(cli, token=token,
                    split_amt=40000, reconcile_func=reconcile_func)
    else:
        deduper = Deduper(cli, reconcile_func=reconcile_func)
    num_deleted = deduper.dedupe()

    p_info = cli.get('/')
    url = 'https://analytics.luminoso.com/app/projects/{}/{}/highlights'.format(p_info['account_id'],p_info['project_id'])
   
    if recalc:
        cli.wait_for_build()
    return {'num':num_deleted, 'url':url}
