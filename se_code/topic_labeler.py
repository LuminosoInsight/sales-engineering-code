from luminoso_api import LuminosoClient
import argparse
import csv


def get_doc_dicts(client, batch_size=25000):
    """
    Use a LuminosoClient connection to get selected fields from
    all of a project's documents.
    """
    docs = []
    offset = 0
    while True:
        found = client.get('docs', limit=batch_size, offset=offset, doc_fields=['_id', 'text', 'terms', 'fragments'])
        if len(found) == 0:
            break
        docs.extend(found)
        offset += batch_size
        print("Got %d documents" % len(docs))
    return docs


def get_topic_dicts(client):
    """
    Get information about the topics defined in a project.
    """
    topic_dicts = client.get('topics')
    for topic_dict in topic_dicts:
        topic_id = topic_dict['_id']

        # Find out the exact and related terms that match this topic, by
        # running a search that asks for 0 documents matching the topic, and
        # extracting the data about which terms were searched for.
        topic_search = client.get('docs/search', topic=topic_id, limit=0)
        topic_dict['exact_terms'] = set(topic_search['exact_terms'])
        topic_dict['related_terms'] = set(topic_search['related_terms'])
    return topic_dicts


def collect_topic_mapping(client):
    """
    Retrieve documents and topics from the project, and then collect
    information about which documents match which topics.

    Returns the list of topics and the list of documents, where the
    documents will have extra 'exact_topics' and 'related_topics'
    fields indicating the topics they matched.
    """
    doc_dicts = get_doc_dicts(client)
    topic_dicts = get_topic_dicts(client)
    for doc_dict in doc_dicts:
        tokens = doc_dict['terms'] + doc_dict['fragments']
        terms = set([token[0] for token in tokens])
        doc_dict['exact_topics'] = set()
        doc_dict['related_topics'] = set()
        for topic_dict in topic_dicts:
            topic_id = topic_dict['_id']
            if terms & topic_dict['exact_terms']:
                # This document has an exact match for this topic
                doc_dict['exact_topics'].add(topic_id)
            elif terms & topic_dict['related_terms']:
                # This document has a related (not exact) match for this topic
                doc_dict['related_topics'].add(topic_id)

    return topic_dicts, doc_dicts


def write_csv(topic_dicts, doc_dicts, out_filename):
    """
    Given the `topic_dicts` and `doc_dicts` returned by the
    `collect_topic_mapping` function, write them into a CSV file so they
    can be manipulated as a spreadsheet.

    In this CSV, the columns represent topics, and the rows represent
    documents. Each document is labeled with its document ID at the start of
    the row, and its full text at the end. Each topic appears twice, once
    for exact matches and once for related matches.

    When a document matches a topic, the appropriate cell for that
    document and that topic (and the kind of match) contains a '1'. Otherwise,
    the cell is blank.
    """
    with open(out_filename, 'w', encoding='utf-8') as out:
        writer = csv.writer(out, dialect='excel')

        # Make two rows of headers. The first row shows the grouping of
        # exact matches vs. conceptual matches. The second row labels the
        # topics with their names.
        topic_ids = [t['_id'] for t in topic_dicts]
        topic_names = [t['name'] for t in topic_dicts]
        padding = len(topic_ids) - 1
        padding = (len(topic_ids) - 1) * ['']
        header1 = ['', 'Exact matches'] + padding + ['Conceptual matches'] + padding + ['']
        header2 = ['id'] + topic_names + topic_names + ['text']
        writer.writerow(header1)
        writer.writerow(header2)

        for doc_dict in doc_dicts:
            exact_matches = ['1' if tid in doc_dict['exact_topics'] else '' for tid in topic_ids]
            conceptual_matches = ['1' if tid in doc_dict['related_topics'] else '' for tid in topic_ids]
            text = doc_dict['text'].replace('\n', ' ')
            doc_row = [doc_dict['_id']] + exact_matches + conceptual_matches + [text]
            writer.writerow(doc_row)


def run(account_id, project_id, username, out_filename):
    """
    Get topics and documents from the project with the given `account_id`
    and `project_id`, using a LuminosoClient that logs in as `username`.
    Write the results in CSV form to `out_filename`.
    """
    client = LuminosoClient.connect(
        'https://api.luminoso.com/v4/projects/%s/%s' % (account_id, project_id),
        username=username
    )

    topic_dicts, doc_dicts = collect_topic_mapping(client)
    write_csv(topic_dicts, doc_dicts, out_filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('account_id', help="The ID of the account that owns the project, such as 'demo'")
    parser.add_argument('project_id', help="The ID of the project to analyze, such as '2jsnm'")
    parser.add_argument('username', help="A Luminoso username with access to the project")
    parser.add_argument('output', help="The filename to write CSV output to")

    args = parser.parse_args()
    run(args.account_id, args.project_id, args.username, args.output)

