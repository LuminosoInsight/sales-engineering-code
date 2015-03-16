from luminoso_api import LuminosoClient
from collections import defaultdict
from string import ascii_letters
import random
import re

"""
Take a Luminoso project built on Twitter data and run it through a
deduplication step, then identify long collocations and remove documents
containing them as spam. Requires multiple builds and dataset downloads/
uploads, so takes a LONG time.
"""

AT_RE = re.compile(r'(\A|\s)@(\w+)', re.I)


def upload(client, docs, project_name, batch_len=100000, max_ngram_length=15,
           sample_thresh=1):
    """
    Required parameters:
        client - a LuminosoClient object pointed at the appropriate account
        docs - a list of Luminoso documents
        project_name - the name of the project to be created

    Optional parameters:
        batch_len - the number of documents in each upload batch. Defaults to
            100000.
        max_ngram_length - the maximum length of collocations to be found.
            Defaults to 15.
        sample

    Returns:
        project - a LuminosoClient object pointed at the newly created project

    Upload a list of documents to a new project. If the project name already
    exists, it is deleted and a new one is created in its place. Uploads are
    batched, and collocations up to length max_ngram_length are found.
    """
    
    try:
        client.delete(client.get(name=project_name)[0]['project_id'])
    except IndexError:
        pass
    project_id = client.post(name=project_name)['project_id']
    project = client.change_path(project_id)

    batch_num = 0
    while batch_num * batch_len < len(docs):
        batch = docs[batch_len * batch_num: batch_len * (batch_num + 1)]
        if sample_thresh < 1:
            batch = [doc for doc in batch if random.random() < sample_thresh]
        project.upload('docs', batch)
        batch_num += 1
        print('Batch %d uploaded' % batch_num)
    project.post('docs/recalculate', max_ngram_length=max_ngram_length)
    return project


def dedupe(docs):
    """
    Required parameters:
        docs - a list of Luminoso documents

    Returns:
        texts - a defaultdict that counts the number of times a specific
            piece of text appears
        users - a defaultdict that counts the number of times a specific
            title, representing a Twitter user, appears
    """
    texts = defaultdict(int)
    users = defaultdict(int)
    for i, doc in enumerate(docs):
        doc['text'] = re.sub(AT_RE, '_____', doc['text'])
        doc['term_text'] = ' '.join([triple[0] for triple in doc['terms']])
        docs[i] = doc
        texts[doc['term_text']] += 1
        users[doc['title']] += 1
    docs = [doc for doc in docs if texts[doc['term_text']] <= 1 and 
                                   users[doc['title']] < 10]
    return docs


def get_collocation_texts(project, min_ngram_length=6, all_spam=True):
    """
    Required parameters:
        project - A LuminosoClient object pointed at the appropriate project

    Optional parameters:
        min_ngram_length - the minimum length of collocations to return.
            Defaults to 6.
        all_spam - Designate all long collocations as spam, otherwise prompt
            the user to identify which collocations are and are not spam.
            Defaults to False.

    Returns:
        spam_strings - a list of strings that indicate spam

    Goes through top 10000 terms in the given project, finds all collocations
    of greater than min_ngram_length, and optionally prompts the user to
    determine whether or not it's spam.
    """
    terms = project.get('terms', limit=10000)
    question = '-------------------------\n\n%s\nIs this spam? '
    collocations = [term for term in terms 
                    if len(term['term'].split()) >= min_ngram_length]
    if all_spam:
        spam_terms = collocations
    else:
        spam_terms = [term for term in collocations 
                if input(question % term['text']) == 'y']

    return spam_terms


def get_docs(project, doc_fields=['title', 'text', 'date', 'source', 'terms']):
    """
    Required parameters:
        project - A LuminosoClient object pointed at the appropriate project
    
    Optional parameters:
        doc_fields - The fields of each document to include. Defaults to 
            ['title', 'text', 'date', 'source']

    Returns:
        docs - a list of Luminoso documents

    A wrapper for the GET /docs/ endpoint of the Luminoso API.
    """
    i = 0
    docs = []
    new_docs = None

    while new_docs != []:
        print(i, 'docs downloaded')
        new_docs = project.get('docs', limit=25000, offset=i,
                               doc_fields=doc_fields)
        docs += new_docs
        i += len(new_docs)
    return docs


def is_spam(doc, spam_terms):
    return any([spam_term['term'] == triple[0] 
                for triple in doc['terms'] 
                for spam_term in spam_terms])


if __name__ == '__main__':
    url = 'http://api.staging.lumi/v4/projects/'
    username = 'user@luminoso.com'
    account_id = 'a11a111a'
    project_id = 'a1a1a'
    client = LuminosoClient.connect(url + account_id, username=username)
    project = client.change_path(project_id)
    project_name = project.get()['name'] + ' Cleaned'
    docs = dedupe(get_docs(project))
    for i in range(2):
        spam_terms = get_collocation_texts(project, min_ngram_length=6)
        print('----------------\n\nSpam Round %s:\n' % str(i+1))
        for term in spam_terms:
            print(term['text'])
        docs = [doc for doc in docs if not is_spam(doc, spam_terms)]
        project = upload(client, docs, project_name)
        print('\nProject calculating')
        project.wait_for(1)
        docs = get_docs(project)
