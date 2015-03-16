from luminoso_api import LuminosoClient
import csv

"""
This script is essentially a wrapper for the GET /docs/correlations/ endpoint,
and outputs a human-readable CSV with topic-document correlations.
"""

ACCOUNT_ID = 'a11a111a'
PROJECT_ID = 'a1a1a'
USERNAME = 'user@luminoso.com'
PROJECT_PATH = 'projects/%s/%s' % (ACCOUNT_ID, PROJECT_ID)

project = LuminosoClient.connect(PROJECT_PATH, username=USERNAME)
DOC_FIELDS = ['title', 'text', '_id']

def download_docs(project, doc_fields=DOC_FIELDS):
    """
    Given a project, download all the documents from it. There is NO SANITY
    CHECKING for project size, so this should only be used on relatively small
    (<100K docs) projects.
    """
    i = 0
    docs = []
    while True:
        batch = project.get('docs', limit=25000, offset=i, 
                            doc_fields=doc_fields)
        if batch == []:
            break
        docs += batch
        i += len(batch)
    return docs


def doc_to_row(doc, corrs, topics):
    """
    Given a document and a topic-document correlation dictionary, make a pretty
    CSV row with all the appropriate information.
    """
    doc_corrs = [corrs[doc['_id']][topic['_id']] for topic in topics]
    if 'title' not in doc:
        doc['title'] = ''
    return [doc['title'], doc['text']] + doc_corrs


docs = download_docs(project)
topics = project.get('topics')
corrs = project.get('docs/correlations')
header = [['title', 'text'] + [topic['name'] for topic in topics]]
rows = header + [doc_to_row(doc, corrs, topics) for doc in docs]

with open(project.get()['name'] + '_corrs.csv', 'w') as f:
    csv.writer(f).writerows(rows)
