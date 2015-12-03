from luminoso_api import LuminosoClient
from networkx import Graph, connected_components

def get_terms(cli):
    return [(t['text'], t['term']) for t in cli.get('terms', limit=750)]

def get_term_to_text_mapping(cli):
    terms = cli.get('terms', limit=50000)
    return {t['term']:t['text'] for t in terms}

def ignore_terms(cli, terms):
    for term in terms:
        cli.put('/terms/ignorelist/', term=term)
    cli.wait_for(cli.post('docs/recalculate'))
    return __list_ignored(cli)

def __list_ignored(cli):
    return cli.get('/terms/ignorelist/')

def __list_merged(cli):
    term_names = get_term_to_text_mapping(cli)
    restems = list(cli.get('/terms/restem/').items())
    restem_groups = connected_components(Graph(restems))
    return {"Group "+str(i+1):r for i,r in enumerate(restem_groups)}

def merge_terms(cli, terms):
    """ restems all the terms to the first term """
    for term in terms[1:]:
        cli.put('/terms/restem/', from_term=term, to_term=terms[0])
    cli.wait_for(cli.post('docs/recalculate'))
    return __list_merged(cli)