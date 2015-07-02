
# coding: utf-8

# In[58]:

from luminoso_api import LuminosoClient
import argparse
import csv


def collect_doc_dicts(client, batch_size=25000):
    docs = []
    offset = 0
    while True:
        found = client.get('docs', limit=batch_size, offset=offset, doc_fields=['_id', 'text', 'terms', 'fragments'])
        if len(found) == 0:
            break
        for doc in found:
            docs.append(doc)
        offset += batch_size
        print("Got %d documents" % offset)
    return docs


def collect_topic_mapping(client):
    topic_dicts = client.get('topics')
    for topic_dict in topic_dicts:
        topic_id = topic_dict['_id']
        topic_search = client.get('docs/search', topic=topic_id, doc_fields=['_id'], limit=1)
        topic_dict['exact_terms'] = set(topic_search['exact_terms'])
        topic_dict['related_terms'] = set(topic_search['related_terms'])

    doc_dicts = collect_doc_dicts(client)
    for doc_dict in doc_dicts:
        tokens = doc_dict['terms'] + doc_dict['fragments']
        terms = set([token[0] for token in tokens])
        doc_dict['exact_topics'] = set()
        doc_dict['related_topics'] = set()
        for topic_dict in topic_dicts:
            topic_id = topic_dict['_id']
            if terms & topic_dict['exact_terms']:
                doc_dict['exact_topics'].add(topic_id)
            elif terms & topic_dict['related_terms']:
                doc_dict['related_topics'].add(topic_id)

    return topic_dicts, doc_dicts


def write_csv(topic_dicts, doc_dicts, out_filename):
    with open(out_filename, 'w', encoding='utf-8') as out:
        writer = csv.writer(out, dialect='excel')
        topic_ids = [t['_id'] for t in topic_dicts]
        topic_names = [t['name'] for t in topic_dicts]
        padding = len(topic_ids) - 1
        header1 = ['', 'Exact matches'] + [''] * padding + ['Conceptual matches'] + [''] * padding + ['']
        header2 = ['id'] + topic_names + topic_names + ['text']
        writer.writerow(header1)
        writer.writerow(header2)

        for doc_dict in doc_dicts:
            exact_matches = [''] * len(topic_ids)
            conceptual_matches = [''] * len(topic_ids)
            for i, tid in enumerate(topic_ids):
                if tid in doc_dict['exact_topics']:
                    exact_matches[i] = '1'
            for i, tid in enumerate(topic_ids):
                if tid in doc_dict['related_topics']:
                    conceptual_matches[i] = '1'
            text = doc_dict['text'].replace('\n', ' ')
            doc_row = [doc_dict['_id']] + exact_matches + conceptual_matches + [text]
            writer.writerow(doc_row)


def run(project_id, username, out_filename):
    client = LuminosoClient.connect(
        'https://api.luminoso.com/v4/projects/%s' % project_id,
        username=username
    )

    topic_dicts, doc_dicts = collect_topic_mapping(client)
    write_csv(topic_dicts, doc_dicts, out_filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('project_id', help="The ID of the project to analyze, such as 'demo/2jsnm'")
    parser.add_argument('username', help="A Luminoso username with access to the project")
    parser.add_argument('output', help="The filename to write CSV output to")

    args = parser.parse_args()
    run(args.project_id, args.username, args.output)

