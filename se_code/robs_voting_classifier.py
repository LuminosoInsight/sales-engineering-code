from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer

import numpy as np
from pack64 import pack64, unpack64
import json
import wordfreq
from luminoso_api import LuminosoClient

PAGE_SIZE = 10000


def sklearn_text(termlist, lang='en'):
    """
    Convert a list of Luminoso terms, possibly multi-word terms, into text that
    the tokenizer we get from `make_term_vectorizer` below will tokenize into
    those terms.

    Yes, the tokenizer will basically be undoing what this function does, but it
    means we also get the benefit of sklearn's TF-IDF.
    """
    langtag = '|' + lang
    fixed_terms = [
        term.replace(langtag, '').replace(' ', '_')
        for term, _tag, _span in termlist
        if '\N{PILCROW SIGN}' not in term
    ]
    return ' '.join(fixed_terms)


def make_term_vectorizer():
    """
    Return a sklearn vectorizer whose tokenizer only splits on whitespace.
    This is for text that we have already tokenized, in the Luminoso way, and
    then stuck together with whitespace.
    """
    return TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words=None, token_pattern=r'\S+')


def make_simple_vectorizer():
    """
    Return a sklearn vectorizer that does sklearn's usual thing with arbitrary
    English text.
    """
    return TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')


def binary_rating_labeler(doc):
    """
    An example that produces labels from Amazon book reviews.

    The classes it produces are 'pos' or 'neg' depending on whether the
    document has a 'rating' of more or less than 3. It returns None for a rating
    of exactly 3, saying to skip that document.

    You'll want to change this to something else for basically any other project.
    """
    rating = doc['predict']['rating']
    if rating == 3:
        return None
    return ('pos' if rating > 3 else 'neg')


def get_labeled_luminoso_docs(project_client, label_func, stepsize=1, max_docs=100000):
    """
    Produces training data with labels by iterating through the documents in
    a project. The documents it returns are Luminoso documents, except that
    they have an additional 'label' field, indicating the label each document
    should train the classifier with.

    These labels are provided by the provided `label_func`. An example of a
    `label_func` is `binary_rating_labeler` above.

    If the `label_func` returns None, it means to skip that document.
    """
    offset = 0
    all_docs = []
    while True:
        docs = project_client.get('docs', limit=PAGE_SIZE, offset=offset)
        for i, doc in enumerate(docs):
            if i % stepsize != 0:
                continue
            label = label_func(doc)
            if label is None:
                continue
            doc['label'] = label
            all_docs.append(doc)
        offset += PAGE_SIZE
        print("Downloaded %d documents" % len(all_docs))
        if len(docs) < PAGE_SIZE:
            break
        if len(all_docs) >= max_docs:
            break
    return all_docs


def test_reviews_from_file(filename, max_docs=1000):
    """
    Test data consists of dictionaries with 'text' and 'label' values. It doesn't
    need other fields. This means it can come from outside of a Luminoso project
    if necessary.

    For reference, here's the data source Rob used when testing on a particular
    dataset of Amazon reviews.
    """
    all_docs = []
    with open(filename) as infile:
        n_docs = 0
        for i, line in enumerate(infile):
            doc = json.loads(line.rstrip())

            # Specific to this data set: there are two classes. The class is
            # 'pos' if the rating is greater than 3, and 'neg' if less than 3.
            # Reviews with a rating of exactly 3 are skipped.
            label = binary_rating_labeler(doc)
            if label is None:
                continue

            all_docs.append({
                'text': doc['text'],
                'label': label
            })
            if len(all_docs) >= max_docs:
                break
    return all_docs


def train_classifier(docs):
    """
    Train a classifier.

    Input: a list of training documents, Luminoso documents with the 'label' item added.
    Output: a pair of (classifiers, vectorizers).

    The returned items represent three different sklearn classifiers and their
    corresponding vectorizers. These should be passed on to the `test_classifier`
    function.
    """
    labels = [doc['label'] for doc in docs]
    assert len(docs) > 0

    term_vectorizer = make_term_vectorizer()
    simple_vectorizer = make_simple_vectorizer()
    vectorizers = {'simple': simple_vectorizer, 'term': term_vectorizer}

    classifiers = {
        style: LinearSVC(
            C=1.0, class_weight=None, dual=False, fit_intercept=True,
            intercept_scaling=1, loss='squared_hinge', max_iter=1000, multi_class='ovr',
            penalty='l1', random_state=None, tol=0.0001, verbose=0
        )
        for style in ('simple', 'term', 'vector')
    }

    simple_vecs = vectorizers['simple'].fit_transform([doc['text'] for doc in docs])
    term_vecs = vectorizers['term'].fit_transform([sklearn_text(doc['terms']) for doc in docs])
    luminoso_vecs = [unpack64(doc['vector']) for doc in docs]
    classifiers['simple'].fit(simple_vecs, labels)
    classifiers['term'].fit(term_vecs, labels)
    classifiers['vector'].fit(luminoso_vecs, labels)
    return (classifiers, vectorizers)


def test_classifier(client, docs, classifiers, vectorizers):
    """
    Inputs:

    * `client`: a LuminosoClient pointing to the root of a project, which will
      be used to vectorize the test documents.
    * `docs`: test documents, which must have at least a `text` item.
    * `classifiers` and `vectorizers` are the result of running
      `train_classifier` on training data.

    Returns a list of classes assigned to the documents in order, and the
    decision matrix, whose dimensions are (n_docs, n_classes).
    """
    docs_out = client.upload('docs/vectors', [{'text': doc['text']} for doc in docs])
    simple_vecs = vectorizers['simple'].transform([doc['text'] for doc in docs_out])
    term_vecs = vectorizers['term'].transform([sklearn_text(doc['terms']) for doc in docs_out])
    luminoso_vecs = [unpack64(doc['vector']) for doc in docs_out]

    decision_mat = (
        classifiers['simple'].decision_function(simple_vecs)
        + classifiers['term'].decision_function(term_vecs)
        + classifiers['vector'].decision_function(luminoso_vecs)
    )

    # Annoyingly, sklearn returns a different shape of data when there are
    # 2 classes, representing just the predictions for class 1. Fix that.
    if len(decision_mat.shape) == 1:
        decision_mat = np.vstack([-decision_mat, decision_mat]).T

    # Find the index of the maximum of each row, representing the class
    # to classify each example as
    best_class_indices = np.argmax(decision_mat, axis=1)
    class_labels = classifiers['simple'].classes_
    results = [class_labels[idx] for idx in best_class_indices]
    return results, decision_mat


def run():
    project = '/projects/zoo/rjcfz'
    client = LuminosoClient.connect().change_path(project)

    # Use a stepsize of 100 here because the reviews are ordered by book, and
    # not every training example should be for "The Prophet" by Kahlil Gibran
    test_docs = test_reviews_from_file('example_data/books-test.luminoso.jsons', max_docs=1000)
    train_docs = get_labeled_luminoso_docs(client, binary_rating_labeler, stepsize=10, max_docs=10000)

    print('Training')
    t_classifiers, t_vectorizers = train_classifier(train_docs)
    print('Testing')
    test_labels, decision_ = test_classifier(client, test_docs, t_classifiers, t_vectorizers)

    n_correct = sum([
        test_label == doc['label']
        for doc, test_label in zip(test_docs, test_labels)
    ])
    print(n_correct / len(test_docs))


if __name__ == '__main__':
    run()
