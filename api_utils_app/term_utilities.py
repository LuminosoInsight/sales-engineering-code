from luminoso_api import V5LuminosoClient as LuminosoClient
from networkx import Graph, connected_components

def get_terms(cli):
    return [(t['name'], t['exact_term_ids'][0]) for t in cli.get('concepts', concept_selector={'type': 'top', 'limit':750})['result']]

def get_term_to_text_mapping(cli):
    concepts = cli.get('concepts', concept_selector={'type': 'top', 'limit': 50000})['result']
    return {t['exact_term_ids'][0]:t['name'] for t in concepts}

def ignore_terms(cli, terms):
    concept = [{'texts': [term]} for term in terms]
    concepts = cli.get('concepts', concept_selector={'type': 'specified', 'concepts': concept})['result']
    actions = {c['exact_term_ids'][0]: {'action': 'ignore'} for c in concepts}
    cli.put('terms/manage', actions)
    cli.post('build')
    cli.wait_for_build()
    return manage(cli)

def manage(cli):
    return cli.get('/terms/manage/')

def merge_terms(cli, terms):
    """ restems all the terms to the first term """
    concept = [{'texts': [term]} for term in terms]
    concepts = cli.get('concepts', concept_selector={'type': 'specified', 'concepts': concept})['result']
    actions = {c['exact_term_ids'][0]: {'new_term_id': concepts[0]['exact_term_ids'][0]} for c in concepts[1:]}
    cli.post('build')
    cli.wait_for_jobs()
    return manage(cli)