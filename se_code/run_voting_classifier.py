import argparse
import csv
import numpy as np
import json
from sklearn.model_selection import train_test_split

from luminoso_api import LuminosoClient
from voting_classifier.util import train_classifier, classify_documents
from voting_classifier.serialization import serialize, deserialize, validate


def extract_labels(labels):
    '''
    Given a list of subset names, extract label value.
    Subset names must separate prefix and value with a colon.
    '''

    extracted_labels = []
    for label in labels:

        if ':' in label:
            extracted_labels.append(label.partition(':')[2].strip())
        else:
            raise ValueError("Subset prefix must contain colon between prefix and value.")

    return extracted_labels


def get_all_docs(client, subset_field, batch_size=20000):
    '''
    Pull all docs from project, using a particular subset as the LABEL
    '''
    if ':' in subset_field:
        subset_field = subset_field.lower()
    else:
        subset_field = '{}:'.format(subset_field.lower())

    docs = []
    offset = 0
    while True:
        newdocs = client.get('docs', offset=offset, limit=batch_size)

        if not newdocs:

            docs, labels = zip(*[(doc, s)
                                 for doc in docs
                                 for s in doc['subsets']
                                 if s.lower().startswith(subset_field)])

            labels = extract_labels(labels)

            return docs, labels

        docs.extend(newdocs)
        offset += batch_size


def split_train_test(docs, labels, split=0.3):
    '''
    Split documents & labels into a single dict for both testing and training
    '''

    labels_without_enough_samples = [label for label in set(labels)
                                     if labels.count(label) == 1]

    if labels_without_enough_samples:
        indexes = [i
                   for i, label in enumerate(labels)
                   if label not in labels_without_enough_samples]

        labels = [labels[i] for i in indexes]
        docs = [docs[i] for i in indexes]
    return train_test_split(np.array(docs), np.array(labels), test_size=split, random_state=32, stratify=np.array(labels))


def get_test_docs_from_file(filename, label_func=None):
    '''
    Test data consists of dictionaries with 'text' and 'label' values.
    It doesn't need other fields.
    This means it can come from outside of a Luminoso project if necessary.

    label_func specifies a transformation function applied to inbound labels
    '''

    all_docs = []
    all_labels = []
    with open(filename) as infile:

        for line in infile:
            doc = json.loads(line.rstrip())

            if label_func:
                label = label_func(doc['label'])
            else:
                label = doc['label'].strip()

            if label is None:
                continue

            all_docs.append({
                'text': doc['text']})

            all_labels.append(label)

    return all_docs, all_labels


def classify_test_documents(train_client, test_docs, test_labels, classifiers,
                            vectorizers, save_results=False):
    '''
    Inputs:

    * `train_client`: a LuminosoClient pointing to the root of a project,
    which will be used to vectorize the test documents.
    * `test_docs`: test documents, which must have at least a `text` item.
    * `test_labels`: test labels, which must be in same order as test_docs.
    * `classifiers` and `vectorizers` are the result of running
      `train_classifier` on training data.
    * 'save_results' flag to indicate whether results should be
    saved to a CSV for exploration/visualization

    Returns a list of classes assigned to the documents in order, and the
    decision matrix, whose dimensions are (n_docs, n_classes).
    '''

    processed_test_docs = []
    for i in range(0, len(test_docs), 5000):
        processed_test_docs.extend(train_client.upload('docs/vectors', test_docs[i:i+5000]))

    classification = classify_documents(processed_test_docs, classifiers, vectorizers)

    if save_results:
        results_dict = [dict({'text': z[0]['text'], 'truth': z[1]},
                             **dict(zip(list(classifiers['simple'].classes_), z[2])))
                        for z in zip(processed_test_docs, test_labels, classification)]

        with open('results.csv', 'w', encoding='utf-8') as file:
            writer = csv.DictWriter(file, ['text', 'truth'] +
                                    list(classifiers['simple'].classes_))
            writer.writeheader()
            writer.writerows(results_dict)

    return classification


def return_label(new_text, classifiers, vectorizers, train_client):
    '''
    Return label function for operating in a live demo
    Returns best class and "confidence score"
    '''

    test_doc = train_client.upload('docs/vectors', [{'text': new_text}])[0]
    classification = classify_documents([test_doc], classifiers, vectorizers)

    best_class = np.argmax(classification, axis=1)[0]
    return classifiers['simple'].classes_[best_class], classification[0][best_class]


def score_results(test_labels, classifiers, classification):
    '''
    Return the overall accuracy of the classifier on the test set
    '''

    if not any([label in classifiers['simple'].classes_ 
                for label in set(test_labels)]):
        raise ValueError("Test labels do not match training labels")

    best_class = np.argmax(classification, axis=1)
    gold = np.array([list(classifiers['simple'].classes_).index(label)
                    for label in test_labels]
                    )
    # Above fails if test labels don't match training labels

    accuracy = (gold == best_class).sum() / len(gold)
    return accuracy


def main(args):
    '''
    Collect required arguments if not supplied.

    For demo purposes, a single project can be used for both training/testing,
    for POC purposes, projects should be split into training & test.
    '''

    if not args.account_id:
        args.account_id = input('Enter the account id: ')
    if not args.training_project_id:
        args.training_project_id = input('Enter the id of the training project: ')
    if not args.testing_data:
        args.testing_data = input('Enter the id of the testing project: ')
    if not args.subset_field:
        args.subset_field = input('Subset field prefix holding the label("Category label"): ')

    client = LuminosoClient.connect(url=args.api_url, username=args.username)

    train_client = client.change_path('/projects/{}/{}'.format(args.account_id, args.training_project_id))

    print('Loading Testing & Training documents...')
    if args.csv_file:
        test_docs, test_labels = get_test_docs_from_file(args.testing_data)
        train_docs, train_labels = get_all_docs(train_client, args.subset_field)
    elif args.testing_data == args.training_project_id:
        docs, labels = get_all_docs(train_client, args.subset_field)
        train_docs, test_docs, train_labels, test_labels = split_train_test(docs, labels)
    else:
        test_client = client.change_path('/projects/{}/{}'.format(args.account_id, args.testing_data))
        train_docs, train_labels = get_all_docs(train_client, args.subset_field)
        test_docs, test_labels = get_all_docs(test_client, args.subset_field)

    # Allows for live demo-ing in Python notebook
    if args.live:
        print('Training classifier...')
        classifiers, vectorizers = train_classifier(
            train_client, train_docs, train_labels
            )
        print('Classifier trained. Enter example text below or "exit" to exit.\n\n')
        while True:
            new_text = input('Enter text to be classified: ')
            if new_text == 'exit':
                break
            else:
                print('The predicted value is: "{0}".\n The model is {1:.2%} confident.\n'.format(
                    *return_label(new_text, classifiers, vectorizers, train_client))
                      )
    else:
        if args.pickle_path:
            try:
                classifiers, vectorizers, _, _ = deserialize(args.pickle_path)
            except FileNotFoundError:
                print('No classifier found in {}.'.format(args.pickle_path))
                print('Training classifier...')
                classifiers, vectorizers = train_classifier(train_docs, train_labels)
                serialize(classifiers, vectorizers, args.pickle_path)
            except PermissionError:
                print('No access to {}, cannot save/load classifier.'.format(args.pickle_path))
        else:
            print('Training classifier...')
            classifiers, vectorizers = train_classifier(train_docs, train_labels)

        print('Testing classifier...')
        classification = classify_test_documents(
            train_client, test_docs, test_labels, classifiers,
            vectorizers, args.save_results
            )
        print('Test Accuracy:{:.2%}'.format(
            score_results(test_labels, classifiers, classification))
            )

if __name__ == '__main__':
    '''
    BENCHMARK PROJECTS
    USAA: (a53y655v 54hdb 9b2fw -f "Label: ") Accuracy:69.98%
    Pandora: (h82y756m vnfzx vnfzx -f "Category Tag: ") Accuracy:81.11%
    Fidelity: (a53y655v sv5pn sv5pn -f "CED: ") Accuracy:83.08%
    Fidelity: (a53y655v sv5pn sv5pn -f "COSMO_SEMANTIC_TAG: ") Accuracy:80.20%
    SuperCell: (a53y655v 6bsv2 6bsv2 -f "Type: ") Accuracy:83.30%
    Subaru: (a53y655v fpdxb fpdxb -f "Site: ") 
    '''

    parser = argparse.ArgumentParser(
        description='Create a classification model based on an existing project using subsets as labels.'
    )
    parser.add_argument(
        'account_id',
        help="The ID of the account that owns the project, such as 'demo'"
        )
    parser.add_argument(
        'training_project_id',
        help="The ID of the project that contains the training data"
        )
    parser.add_argument(
        'testing_data',
        help="The ID of the project, or name of the CSV containing testing data"
        )
    parser.add_argument(
        '-u', '--username',
        help='Username (email) of Luminoso account'
        )
    parser.add_argument(
        '-a', '--api_url',
        help='URL of Luminoso API endpoint (https://eu-analytics.luminoso.com/api/v4)'
        )
    parser.add_argument(
        '-c', '--csv_file', default=False, action='store_true',
        help="CSV file with testing data: (text,label) columns"
        )
    parser.add_argument(
        '-f', '--subset_field',
        help='A prefix on the subset names that will be used for classification.'
        'These subset names should begin with the prefix, '
        'followed by a colon, such as "Label: positive".'
        )
    parser.add_argument(
        '-l', '--live', default=False, action='store_true',
        help="Run the classifier in live mode (classifying entries via terminal/notebook)",
        )
    parser.add_argument(
        '-s', '--save_results', default=False, action='store_true',
        help="Save the results of the test set to a CSV file named results.csv"
        )
    parser.add_argument(
        '-p', '--pickle_path',
        help="Specify a path to save the classifier to or load a classifier from"
        "If a classifier is found, it will be loaded, if not one will be created"
        )
    args = parser.parse_args()
    main(args)
