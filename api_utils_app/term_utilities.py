from luminoso_api import LuminosoClient

def search_terms(query, cli):
    results = cli.get('/terms/search/', text=query, limit=100)['search_results']
    return [(r[0]['text'],r[0]['term']) for r in results]

def ignore_terms(cli, terms):
    for term in terms:
        cli.put('/terms/ignorelist/', term=term)

def list_ignored_terms(cli, terms):
    return cli.get('/terms/ignorelist/')

def merge_terms(cli, terms):
    """ restems all the terms to the first term """
    for term in terms[1:]:
        cli.put('/terms/restem/', from_term=term, to_term=terms[0])
    return