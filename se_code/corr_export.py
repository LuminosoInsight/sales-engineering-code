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


def doc_to_row(doc, corrs, topics):
    """
    Given a document and a topic-document correlation dictionary, make a pretty
    CSV row with all the appropriate information.
    """
    doc_corr_dict = corrs[doc['_id']]
    doc_corrs = [doc_corr_dict[topic['_id'] for topic in topics]
    return [doc.get('title'), doc['text']] + doc_corrs


topics = project.get('topics')
corrs = project.get('docs/correlations')
header = [['title', 'text'] + [topic['name'] for topic in topics]]

with open(project.get()['name'] + '_corrs.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    i = 0
    docs = []
    while True:
        batch = project.get('docs', limit=25000, offset=i, 
                            doc_fields=DOC_FIELDS)
        if batch == []:
            break
        i += len(batch)
        for doc in batch:
            writer.writerow(doc_to_row(doc, corrs, topics))
