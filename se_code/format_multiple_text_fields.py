import csv
import argparse


def file_to_dict(file_name, encoding="utf-8"):
    table = []
    with open(file_name, encoding=encoding) as f:
        reader = csv.reader(f)

        # found some csv files have same name columns
        # map and change those to unique names so can still use as dict
        raw_header = next(reader)
        header_map = {}
        header = []
        for name in raw_header:
            if name in header:
                idx = 2
                new_name = "{}_{}".format(name, idx)
                while new_name in header:
                    idx += 1
                    new_name = "{}_{}".format(name, idx)
                header_map[new_name] = name

                name = new_name
            header.append(name)

        for row in reader:
            table.append(dict(zip(header, row)))
    return table, header_map


def file_to_list(file_name, encoding="utf-8"):
    table = []
    with open(file_name, encoding=encoding) as f:
        reader = csv.reader(f)
        for row in reader:
            table.append(row)
    return table


def dict_to_file(table, file_name, encoding="utf-8", header_map={}):
    fields = []
    for key in table[0]:
        if key in header_map:
            fields.append(header_map[key])
        else:
            fields.append(key)

    with open(file_name, 'w', newline='', encoding=encoding) as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        for row in table:
            writer.writerow(row.values())


def list_to_file(table, file_name, encoding="utf-8"):
    with open(file_name, 'w', newline='', encoding=encoding) as f:
        writer = csv.writer(f)
        writer.writerows(table)


def char_position(letters):
    index = 0
    for i, char in enumerate(letters):
        index += ((ord(char.lower()) - 97) + (i * 26))
    return index


def main():
    parser = argparse.ArgumentParser(
        description='Changing a CSV with multiple text columns into a CSV with one text column and a metadata field describing the name of the original column.'
    )
    parser.add_argument(
        'input_file',
        help="Name of the file that we want to modify."
        )
    parser.add_argument(
        'output_file',
        help="Name of the output file (after modifications are complete)."
        )
    parser.add_argument(
        'column_dest',
        help="Name of the new column to use as metadata to determine which original column the text came from."
        )
    parser.add_argument(
        '--encoding',
        default='utf-8',
        help="Encoding type of the files to read from"
    )
    args = parser.parse_args()

    write_table = []
    table, header_map = file_to_dict(args.input_file, encoding=args.encoding)
    text_fields = [field for field in table[0] if 'text_' in field.lower() or 'text' == field.lower().strip()]
    for read_row in table:
        for key in read_row:
            if key.lower().startswith('text_') or key.lower().strip().startswith('text'):
                write_row = {k: v for k, v in read_row.items() if k not in text_fields}
                write_row.update({'Text': read_row[key]})
                if key.lower().startswith('text_'):
                    # case sensitive text_
                    splitat = key[0:5]
                    write_row.update({'string_' + args.column_dest: key.split(splitat)[1]})
                else:
                    write_row.update({'string_' + args.column_dest: 'Text'})
                if len(write_row['Text'].strip()) > 0:
                    write_table.append(write_row)
    dict_to_file(write_table, args.output_file, encoding=args.encoding, header_map=header_map)


if __name__ == '__main__':
    main()
