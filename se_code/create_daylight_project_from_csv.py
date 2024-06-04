'''
Create and/or upload to a Luminoso Daylight project from a CSV file. Useful if the file is too
large for UI or building without the UI for things like search_enhancement
'''
from luminoso_api import V5LuminosoClient as LuminosoClient

import argparse
import csv
import os
import time

# Python's csv module limits cell sizes to 131072 characters (i.e., 2^17, or
# 128k).  Some CSVs have extremely long text, so we set max cell size to four
# times that; bear in mind that that's already longer than the 500,000
# characters that the API allows in a document.
csv.field_size_limit(2 ** 19)

# Number of documents to load at once
BLOCK_SIZE = 1000

DATE_FORMATS = [
    '%Y-%m-%dT%H:%M:%S.%fZ',
    '%Y-%m-%dT%H:%M:%SZ',
    '%Y-%m-%d',
    '%m/%d/%Y',
    '%m/%d/%y',
    '%m/%d/%y %H:%M'
]

FIELD_TYPES = ['date', 'number', 'score', 'string']


def create_project(client, project_name, workspace_id):
    # create the project
    print('Creating project named: ' + project_name)
    project_info = client.post(name=project_name, language='en',
                               workspace_id=workspace_id)
    print('New project info = ' + str(project_info))

    return client.client_for_path(project_info['project_id'])


def upload_documents(client_prj, input_file, offset=0, keyword_expansion_terms=None, max_len=0, skip_build=False, skip_sentiment_build=False):
    # convert max_len into None if it's 0, to allow indexing later
    if not max_len:
        max_len = None

    # Extract the documents from the CSV file and upload them in batches.
    with open(input_file, 'r', encoding='utf-8-sig') as file:
        for i, docs in parse_csv_file(file, max_len):

            done = False
            tries = 0
            while (not done):
                try:
                    # i is where the file pointer is after the read (at the end of docs)
                    if i-len(docs) >= offset:
                        print('Uploading at {}, {} new documents'.format((i-len(docs)), len(docs)))
                        client_prj.post('upload', docs=docs)
                        done = True
                    elif i > offset:
                        # need to do a partial write
                        print('Uploading at {}, {} new documents'.format(offset, (i-offset)))
                        client_prj.post('upload', docs=docs[offset-(i-len(docs)):])
                        done = True
                    else:
                        done = True
                except ConnectionError:
                    tries = tries + 1
                    if tries > 5:
                        print('Upload failed - connection error aborting')
                        raise
    print('Done uploading.')

    options = {}
    if skip_sentiment_build:
        options['sentiment_configuration'] = {'type': 'none'}

    if keyword_expansion_terms:
        keyword_expansion_filter = []
        for entry in keyword_expansion_terms.split('|'):
            field_and_val = entry.split('=')
            print('fv {}'.format(field_and_val))
            field_values = field_and_val[1].split(',')
            keyword_expansion_filter.append({'name': field_and_val[0],
                                             'values': field_values})

        keyword_expansion_dict = {'limit': 20,
                                  'filter': keyword_expansion_filter}
        print('keyword filter = {}'.format(keyword_expansion_dict))
        options['keyword_expansion'] = keyword_expansion_dict

    if not skip_build:
        client_prj.post('build', **options)
        print('Build started')

    return client_prj


def parse_csv_file(file, max_text_length):
    """
    Given a file and a length at which to truncate document text, yield batches
    of documents suitable for uploading.
    """
    csv.field_size_limit(100000000)

    # Note that we don't use a DictReader here because there may be
    # multiple columns with the same header
    reader = csv.reader(file)

    # Parse the first row of the CSV as a header row, storing the results in
    # a list whose elements are:
    # * for text/title: the string "text" or "title"
    # * for metadata fields: the pair (data_type, field_name)
    # * for unparseable headers: None
    columns = []
    for col_num, cell in enumerate(next(reader), start=1):
        data_type, _, data_name = cell.partition('_')
        data_type = data_type.strip().lower()
        if data_type in ('text', 'title') and not data_name:
            columns.append(data_type)
        elif data_type in FIELD_TYPES:
            columns.append((data_type, data_name))
        else:
            print('Uninterpretable header "{}" in column'
                  ' {}'.format(cell, col_num))
            columns.append(None)

    # If there is not exactly one "text" column, raise an error
    text_count = columns.count('text')
    if text_count != 1:
        raise RuntimeError('Must have exactly one text column,'
                           ' found {}'.format(text_count))

    docs = []
    i = None
    for i, row in enumerate(reader, start=1):
        new_doc = {'metadata': []}
        for header, cell_value in zip(columns, row):
            # For each cell in the row: if the header was unparseable, skip it;
            # if the header is text/title, add that to the document; otherwise,
            # parse it as metadata
            if header is None:
                continue
            elif header == 'text':
                new_doc['text'] = cell_value[:max_text_length]
            elif header == 'title':
                new_doc['title'] = cell_value[:64]
            else:
                # Blank cells indicate no metadata value
                cell_value = cell_value.strip()
                if not cell_value:
                    continue
                try:
                    metadata_field = parse_metadata_field(header, cell_value)
                    if "|" in metadata_field['value']:
                        # need to expand out all the field values
                        value_list = metadata_field['value'].split("|")
                        for v in value_list:
                            mfcopy = metadata_field.copy()
                            mfcopy['value'] = v
                            new_doc['metadata'].append(mfcopy)
                    else:
                        new_doc['metadata'].append(metadata_field)
                except ValueError as e:
                    print(
                        'Metadata error in document {}: {}'.format(i, str(e))
                    )
        docs.append(new_doc)

        if len(docs) >= BLOCK_SIZE:
            yield i, docs
            docs = []

    if i is None:
        raise RuntimeError('No documents found')

    if docs:
        yield i, docs


def parse_metadata_field(type_and_name, cell_value):
    """
    Given a (type, name) pair and a value, return a metadata dict with type,
    name, and the parsed value.  Raises a ValueError if "value" cannot be
    parsed as the given type.
    """
    data_type, field_name = type_and_name
    value = None
    if data_type == 'date':
        if cell_value.isnumeric():
            value = int(cell_value)
        else:
            for df in DATE_FORMATS:
                try:
                    value = int(time.mktime(time.strptime(cell_value, df)))
                except ValueError:
                    continue
                break
    elif data_type in ('number', 'score'):
        try:
            value = float(cell_value.strip())
        except ValueError:
            pass
    elif data_type == 'string':
        value = cell_value
    if value is None:
        raise ValueError(
            'Cannot parse "{}" value "{}" as {}'.format(
                field_name, cell_value, data_type
            )
        )
    return {'type': data_type, 'name': field_name, 'value': value}


def split_url(project_url):
    workspace_id = project_url.strip('/').split('/')[5]
    project_id = project_url.strip('/').split('/')[6]
    api_url = '/'.join(project_url.strip('/').split('/')[:3]).strip('/') + '/api/v5'
    proj_api = '{}/projects/{}'.format(api_url, project_id)

    return (workspace_id, project_id, api_url, proj_api)


def main():
    parser = argparse.ArgumentParser(
        description='Create (or upload documents to) a Luminoso project using a CSV file.'
    )
    parser.add_argument('input_file', help='CSV file with project data')
    parser.add_argument(
        '-n', '--project_name', default=None, required=False,
        help='New project name'
    )
    parser.add_argument(
        '-w', '--workspace_id', default='', required=False,
        help='Luminoso account ID'
    )
    parser.add_argument(
        '-u', '--api_url', default='https://daylight.luminoso.com/api/v5/',
        help='The host url. Default=https://daylight.luminoso.com/api/v5/'
    )
    parser.add_argument(
        '-k', '--keyword_expansion_terms', default=None, required=False,
        help='field list of metadata field=data,data to expand. '
             'search_doc_type=primary,primary2|search_doc_type2=secondary '
    )
    parser.add_argument(
        '-m', '--max_text_length', default=0, type=int, required=False,
        help='The maximum length to limit text fields'
    )
    parser.add_argument(
        '-p', '--project_url', default=None, required=False,
        help='If this is provided upload to this project instead of creating a new project'
    )
    parser.add_argument(
        '-o', '--offset', default=0, required=False,
        help='Start the upload at the document offset specified instead of the beginning of the input file'
    )
    parser.add_argument(
        '-s', '--skip_build', action='store_true', default=False,
        help='Skip the project build after upload'
    )
    parser.add_argument(
        '-ss', '--skip_sentiment_build', action='store_true', default=False,
        help='Allows the build to skip the sentiment build'
    )
    parser.add_argument(
        '-b', '--wait_for_build_complete', action='store_true', default=False,
        help='Wait for the build to complete'
    )
    args = parser.parse_args()

    input_file = args.input_file
    workspace_id = args.workspace_id
    max_len = args.max_text_length

    # if no project name is given, use the input file
    if not args.project_name:
        project_name = args.input_file.split(os.sep)[-1]
    else:
        project_name = args.project_name

    api_url = args.api_url

    # get the default account id if none given
    if len(workspace_id) == 0:
        # connect to v5 api
        client = LuminosoClient.connect(
            url=api_url,
            user_agent_suffix='se_code:create_daylight_project_from_csv'
        )
        workspace_id = client.get('/profile')['default_workspace']
        client = client.client_for_path('/projects/')
    else:
        # connect to v5 api
        client = LuminosoClient.connect(
            url=api_url + '/projects/',
            user_agent_suffix='se_code:create_daylight_project_from_csv'
        )

    try:
        if args.project_url is None:
            # create a new project if no url was provided
            client_prj = create_project(client, project_name, workspace_id)
            prj_info = client_prj.get("/")
            # show the Daylight UI url
            host_url = api_url.split('api')[0]
            print("project created: {}app/projects/{}/{}/".format(host_url,
                                                                 prj_info['workspace_id'],
                                                                 prj_info['project_id']))
        else:
            # create a client from the url provided
            workspace_id, project_id, api_url, proj_api = split_url(args.project_url)

            # connect to v5 api again - in case the project's host is different than the default
            client = LuminosoClient.connect(
                url=api_url + '/projects/',
                user_agent_suffix='se_code:create_daylight_project_from_csv'
            )
            client_prj = client.client_for_path(project_id)

        upload_documents(client_prj, input_file,
                         offset=int(args.offset),
                         keyword_expansion_terms=args.keyword_expansion_terms,
                         max_len=max_len,
                         skip_build=args.skip_build,
                         skip_sentiment_build=args.skip_sentiment_build)
    except RuntimeError as e:
        parser.exit(1, 'Error creating project: {}'.format(str(e)))
        return  # unreachable; lets the IDE knows client_prj has been defined

    if (args.wait_for_build_complete) and (not args.skip_build):
        print('waiting for build to complete...')
        client_prj.wait_for_build()


if __name__ == '__main__':
    main()
