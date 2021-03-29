import csv
import argparse

def file_to_dict(file_name, encoding="utf-8"):
    table = []
    with open(file_name, encoding=encoding) as f:
        reader = csv.DictReader(f)
        for row in reader:
            table.append(row)
    return table

def file_to_list(file_name, encoding="utf-8"):
    table = []
    with open(file_name, encoding=encoding) as f:
        reader = csv.reader(f)
        for row in reader:
            table.append(row)
    return table
    
def dict_to_file(table, file_name, encoding="utf-8"):
    fields = []
    for key in table[0]:
        fields.append(key)
    with open(file_name, 'w', newline='', encoding=encoding) as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(table)
        
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
    table = file_to_dict(args.input_file, encoding=args.encoding)
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
                write_table.append(write_row)
    dict_to_file(write_table, args.output_file, encoding=args.encoding)


if __name__ == '__main__':
    main()
