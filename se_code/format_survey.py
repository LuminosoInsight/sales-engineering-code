import argparse
import csv
import time

from luminoso_api import LuminosoClient


def file_to_dict(file_name, encoding="utf-8"):

    # the header names are combinations of the first two rows.

    # only show unknown types once
    unknown_types = set()

    with open(file_name, encoding=encoding) as f:
        column_types, column_names, *rows = list(csv.reader(f))

    column_types = [ct.strip().lower() for ct in column_types]
    column_names = [cn.strip() for cn in column_names]

    # convert all the reality field names to lumi
    title_count = column_types.count('title')
    table = []
    for row in rows:
        new_row = {}
        titles = []
        for col_type, col_name, value in zip(column_types, column_names, row):
            if col_type == 'title':
                titles.append(value)
                if title_count > 1:
                    new_row['string_' + col_name] = value
            else:
                andkeys = col_type.split('&')
                for ak in andkeys:
                    ak = ak.strip()
                    if ak in ['score', 'date', 'number', 'text', 'string']:
                        new_row[ak + "_" + col_name] = value
                    elif ak not in unknown_types:
                        print(f'Warning: Unknown type [{ak}]')
                        unknown_types.add(ak)
        if titles:
            new_row['title'] = '_'.join(titles)
        table.append(new_row)
    return table


# Number of documents to load at once
BLOCK_SIZE = 1000

DATE_FORMATS = [
    '%Y-%m-%dT%H:%M:%SZ',
    '%Y-%m-%d',
    '%m/%d/%Y',
    '%m/%d/%y',
    '%m/%d/%y %H:%M'
]

FIELD_TYPES = ['date', 'number', 'score', 'string']


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
            f'Cannot parse "{field_name}" value "{cell_value}" as {data_type}'
            )
    return {'type': data_type, 'name': field_name, 'value': value}


def format_for_lumi(table):

    # a list whose elements are:
    # * for text/title: the string "text" or "title"
    # * for metadata fields: the pair (data_type, field_name)
    # * for unparseable headers: None
    columns = []
    for cell, val in table[0].items():
        data_type, _, data_name = cell.partition('_')
        data_type = data_type.strip().lower()
        if data_type in ('text', 'title') and not data_name:
            columns.append(data_type)
        elif data_type in FIELD_TYPES:
            columns.append((data_type, data_name))
        else:
            print(f'Uninterpretable header "{cell}": value'
                  f' {val}')
            columns.append(None)

    # If there is not exactly one "text" column, raise an error
    text_count = columns.count('text')
    if text_count != 1:
        raise RuntimeError('Must have exactly one text column,'
                           f' found {text_count}')

    docs = []
    i = None
    for i, row in enumerate(table, start=1):
        new_doc = {'metadata': []}
        for header, cell_value in zip(columns, row):
            # For each cell in the row: if the header was unparseable, skip it;
            # if the header is text/title, add that to the document; otherwise,
            # parse it as metadata
            if header is None:
                continue
            elif header == 'text':
                new_doc['text'] = row[cell_value]
            elif header == 'title':
                new_doc['title'] = row[cell_value]
            else:
                # Blank cells indicate no metadata value
                cell_value = cell_value.strip()
                if not cell_value:
                    continue
                try:
                    metadata_field = parse_metadata_field(header, row[cell_value])
                    new_doc['metadata'].append(metadata_field)
                except ValueError as e:
                    print(
                        f'Metadata error in document {i}: {str(e)}'
                    )
        docs.append(new_doc)

        if len(docs) >= BLOCK_SIZE:
            yield i, docs
            docs = []

    if i is None:
        raise RuntimeError('No documents found')

    if docs:
        yield i, docs


def create_project(client, project_name, workspace_id, docs):

    # create the project
    print('Creating project named: ' + project_name)
    project_info = client.post(name=project_name, language='en',
                               workspace_id=workspace_id)
    print('New project info = ' + str(project_info))

    client_prj = client.client_for_path(project_info['project_id'])

    for i, docs2 in format_for_lumi(docs):
        print(f'Uploading at {i}, {len(docs2)} new documents')
        client_prj.post('upload', docs=docs2)

    print('Done uploading. Starting build')

    client_prj.post('build')
    print('Build started')
    return client_prj


def dict_to_file(table, file_name, encoding="utf-8"):
    fields = []
    for key in table[0]:
        fields.append(key)
    with open(file_name, 'w', newline='', encoding=encoding) as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(table)


def main():
    parser = argparse.ArgumentParser(
        description='Changing a CSV with multiple text columns into a Luminoso ready CSV with one text column and a metadata field describing each text field.'
    )

    parser.add_argument(
        'input_file',
        help="Name of the file that we want to modify."
        )

    parser.add_argument(
        '-p', '--project', default=False,
        action='store_true',
        help="Create a project instead of a file")

    parser.add_argument(
        'output_name',
        help="Name of the output file or project. Default = file"
        )
    parser.add_argument(
        '--encoding',
        default='utf-8',
        help="Encoding type of the files to read from"
    )
    parser.add_argument(
        '-u', '--api_url', default='https://daylight.luminoso.com/api/v5/',
        help='The host url. Default=https://daylight.luminoso.com/api/v5/'
    )
    parser.add_argument(
        '-w', '--workspace_id', default='', required=False,
        help='Luminoso account ID'
    )

    args = parser.parse_args()

    table = file_to_dict(args.input_file, encoding=args.encoding)

    write_table = []
    text_fields = [field for field in table[0] if 'text_' in field.lower() or 'text' == field.lower().strip()]
    for read_row in table:
        for key in read_row:
            if key.lower().startswith('text_') or key.lower().strip().startswith('text'):
                write_row = {k: v for k, v in read_row.items() if k not in text_fields}
                write_row.update({'text': read_row[key]})
                if key.lower().startswith('text_'):
                    # case sensitive text_
                    write_row.update({'string_' + "Question": key[5:]})
                else:
                    write_row.update({'string_' + args.column_dest: 'text'})
                write_table.append(write_row)

    if not args.project:
        dict_to_file(write_table, args.output_name, encoding=args.encoding)
    else:
        project_name = args.output_name
        api_url = args.api_url

        # get the default workspace id if none given
        workspace_id = args.workspace_id
        if len(workspace_id) == 0:
            # connect to v5 api
            client = LuminosoClient.connect(
                url=api_url,
                user_agent_suffix='realith_check:format_realitycheck_fields.py'
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
            create_project(
                client, project_name, workspace_id, write_table
            )
        except RuntimeError as e:
            parser.exit(1, f'Error creating project: {str(e)}')


if __name__ == '__main__':
    main()
