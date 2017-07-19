import argparse
import csv
import numpy as np
import sys
import json
import ntpath
from sklearn.model_selection import train_test_split

from luminoso_api import LuminosoClient
from voting_classifier.util import train_classifier, classify_documents
from voting_classifier.serialization import serialize, deserialize, validate


def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

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
                if 'label' in doc:
                    label = doc['label'].strip()
                else:
                    label = ''

            if label is None:
                continue

            all_docs.append({
                'text': doc['text']})

            all_labels.append(label)

    return all_docs, all_labels

# ADDED FLAG
def classify_test_documents(train_client, test_docs, test_labels, classifiers,
                            vectorizers, filename, flag, save_results=False):
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
    test_docs = train_client.upload('docs/vectors', test_docs)
    classification = classify_documents(test_docs, classifiers, vectorizers)
    
    if save_results:
        print('Saving results to ' + filename +'.csv file...')
        if flag == 3:
            results_dict = [dict({'text': z[0]['text'],
                                  'label': list(classifiers['simple'].classes_)[np.argmax(z[2])],
                                  'max_score': np.max(z[2])},
                                 **dict(zip(list(classifiers['simple'].classes_), z[2])))
                            for z in zip(test_docs, test_labels, classification)]
        else:    
            results_dict = [dict({'text': z[0]['text'],
                                  'truth': z[1],
                                  'prediction': list(classifiers['simple'].classes_)[np.argmax(z[2])],
                                  'correct': z[1]==list(classifiers['simple'].classes_)[np.argmax(z[2])],
                                  'max_score': np.max(z[2])},
                                 **dict(zip(list(classifiers['simple'].classes_), z[2])))
                            for z in zip(test_docs, test_labels, classification)]

        with open(filename + '.csv', 'w', encoding='utf-8') as file:
            if flag == 3:
                writer = csv.DictWriter(file, ['text', 'label', 'max_score'] + 
                                        list(classifiers['simple'].classes_))
            else:
                writer = csv.DictWriter(file, ['text', 'truth', 'prediction', 'correct', 'max_score'] +
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


    client = LuminosoClient.connect(url=args.api_url, username=args.username)

    train_client = client.change_path('/projects/{}/{}'.format(args.account_id, args.training_project_id))

    print('Loading Testing & Training documents...')
    # ADDED FLAG
    if args.csv_file:
        test_docs, test_labels = get_test_docs_from_file(args.testing_data)
        train_docs, train_labels = get_all_docs(train_client, args.subset_field)
        flag = 1
    elif args.testing_data == args.training_project_id:
        docs, labels = get_all_docs(train_client, args.subset_field)
        train_docs, test_docs, train_labels, test_labels = split_train_test(docs,labels,args.split)
        flag = 2
    # ADDED FOURTH OPTION
    elif args.no_test:
        test_client = client.change_path('/projects/{}/{}'.format(args.account_id, args.testing_data))
        train_docs, train_labels = get_all_docs(train_client, args.subset_field)
        #test_docs = test_client.get('docs')
        test_docs = []
        offset = 0
        batch_size = 20000
        newdocs = test_client.get('docs', offset=offset, limit=batch_size)
        test_docs.extend(newdocs)
        offset += batch_size
        while newdocs:
            newdocs = test_client.get('docs', offset=offset, limit=batch_size)
            #if not newdocs:
            #    break
            test_docs.extend(newdocs)
            offset += batch_size
            
        test_labels = []
        for i in range(0, len(test_docs)):
            test_labels.append('Other')
        flag = 3
        print(len(test_docs))
    else:
        test_client = client.change_path('/projects/{}/{}'.format(args.account_id, args.testing_data))
        train_docs, train_labels = get_all_docs(train_client, args.subset_field)
        test_docs, test_labels = get_all_docs(test_client, args.subset_field)
        flag = 4

    
    if args.pickle_path:
        try:
            classifiers, vectorizers, _, _ = deserialize(args.pickle_path)
            print('Loaded classifier from {}.'.format(args.pickle_path))
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

    if args.live:
        print('Live Demo Mode:\nEnter example text below or "exit" to exit.\n\n')
        while True:
            new_text = input('Enter text to be classified: ')
            if new_text == 'exit':
                break
            else:
                print('The predicted value is: "{0}",\n with a confidence score of {1:.2}.\n'.format(
                    *return_label(new_text, classifiers, vectorizers, train_client))
                      )
    else:
    # ADDED FLAG SEPARATION
        if flag == 1:
            filename = path_leaf(args.testing_data)
        elif flag == 2:
            filename = train_client.get()['name']
        else:
            filename = test_client.get()['name']
        print('Testing classifier...')
        classification = classify_test_documents(
            train_client, test_docs, test_labels, classifiers,
            vectorizers, filename, flag, args.save_results
        )
        if flag != 3:
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
        'subset_field',
        help='A prefix on the subset names that will be used for classification.'
        'These subset names should begin with the prefix, '
        'followed by a colon, such as "Label: positive".'
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
        '-l', '--live', default=False, action='store_true',
        help="Run the classifier in live mode (classifying entries via terminal/notebook)",
        )
    parser.add_argument(
        '-s', '--save_results', default=False, action='store_true',
        help="Save the results of the test set to a CSV file named after the file or project name"
        )
    parser.add_argument(
        '-p', '--pickle_path',
        help="Specify a path to save the classifier to or load a classifier from"
        "If a classifier is found, it will be loaded, if not one will be created"
        )
    parser.add_argument(
        '-z', '--split',
        help="Fraction of documents to hold for testing set. (.3 = 30%%) "
        "For when training/testing documents are in the same project.",
        default=.3
        )
    # ADDED THIS
    parser.add_argument(
        '-n', '--no_test', default=False, action='store_true',
        help="Simply run the classifier, do not compare to validation set (Because there is no validation label data."
        )
    args = parser.parse_args()
    main(args)
