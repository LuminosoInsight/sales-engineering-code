from luminoso_api import V5LuminosoClient as LuminosoClient
import argparse
import csv


def get_all_docs(client, batch_size=25000):
    docs = []
    while True:
        new_docs = client.get('docs', limit=batch_size, offset=len(docs))['result']
        if new_docs:
            docs.extend(new_docs)
        else:
            return docs

        
def get_topic_dicts(client):
    """
    Get information about the saved concepts defined in a project
    """
    topic_dicts = client.get('concepts/saved', include_science=True)
    for topic_dict in topic_dicts:
        topic_id = topic_dict['saved_concept_id']
        topic_search = client.get('docs', 
                                  search={'saved_concept_id': topic_dict['saved_concept_id']})['search']
        topic_dict['exact_terms'] = set(topic_search['exact_term_ids'])
        topic_dict['related_terms'] = set(topic_search['related_term_ids'])
    return topic_dicts


def collect_topic_mapping(client):
    """
    Retrieve documents and topics from the project, and then collect
    information about which documents match which topics.

    Returns the list of topics and the list of documents, where the
    documents will have extra 'exact_topics' and 'related_topics'
    fields indicating the topics they matched.
    """
    doc_dicts = get_all_docs(client)
    topic_dicts = get_topic_dicts(client)
    for doc_dict in doc_dicts:
        tokens = doc_dict['terms'] + doc_dict['fragments']
        #print(tokens)
        terms = set([token['term_id'] for token in tokens])
        doc_dict['exact_topics'] = set()
        doc_dict['related_topics'] = set()
        for topic_dict in topic_dicts:
            topic_id = topic_dict['saved_concept_id']

            # Look for intersections between this document's terms and those
            # that define the topic.
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
        topic_ids = [t['saved_concept_id'] for t in topic_dicts]
        topic_names = [t['name'] for t in topic_dicts]
        padding = (len(topic_ids) - 1) * ['']
        header1 = (['', 'Exact matches'] + padding +
                   ['Conceptual matches'] + padding + [''])
        header2 = ['id'] + topic_names + topic_names + ['text']
        writer.writerow(header1)
        writer.writerow(header2)

        for doc_dict in doc_dicts:
            exact_matches = ['1' if tid in doc_dict['exact_topics']
                             else '' for tid in topic_ids]
            conceptual_matches = ['1' if tid in doc_dict['related_topics']
                                  else '' for tid in topic_ids]
            text = doc_dict['text'].replace('\n', ' ')
            doc_row = ([doc_dict['doc_id']] + exact_matches +
                       conceptual_matches + [text])
            writer.writerow(doc_row)


def run(project_url, out_filename, token):
    """
    Get topics and documents from the project with the given `account_id`
    and `project_id`, using a LuminosoClient that logs in as `username`.
    Write the results in CSV form to `out_filename`.
    """
    api_root = project_url.strip('/ ').split('/app')[0]
    project_id = project_url.strip('/ ').split('/')[-1]
    if not token:
        client = LuminosoClient.connect('%s/api/v5/projects/%s' % (api_root, project_id))
    else:
        client = LuminosoClient.connect('%s/api/v5/projects/%s' % (api_root, project_id),
                                        token=token)

    topic_dicts, doc_dicts = collect_topic_mapping(client)
    write_csv(topic_dicts, doc_dicts, out_filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('project_url', help="The URL of the project")
    parser.add_argument('output', help="The filename to write CSV output to")
    parser.add_argument('-t', '--token', default=None, help="Authentication token for Daylight")

    args = parser.parse_args()
    run(args.project_url, args.output, args.token)

