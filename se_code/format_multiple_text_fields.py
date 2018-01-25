import csv
import argparse

def file_to_dict(file_name):
    table = []
    with open(file_name) as f:
        reader = csv.DictReader(f)
        for row in reader:
            table.append(row)
    return table

def file_to_list(file_name):
    table = []
    with open(file_name) as f:
        reader = csv.reader(f)
        for row in reader:
            table.append(row)
    return table
    
def dict_to_file(table, file_name):
    fields = []
    for key in table[0]:
        fields.append(key)
    with open(file_name, 'w') as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(table)
        
def list_to_file(table, file_name):
    with open(file_name, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(table)
        
def char_position(letters):
    index = 0
    for i, char in enumerate(letters):
        index += ((ord(char.lower()) - 97) + (i * 26))
    return index

def main():
    parser = argparse.ArgumentParser(
        description='Format a CSV given an input file'
    )
    parser.add_argument(
        'input_file',
        help="Name of the file that we want to modify"
        )
    parser.add_argument(
        'output_file',
        help="Name of the file to write the modification to"
        )
    parser.add_argument(
        'column_dest',
        help="Name of column for combined data to be written to"
        )
    parser.add_argument(
        '-b', '--blanks', default=False, action='store_true',
        help="Retain blank text to accurately represent share of voice"
        )
    args = parser.parse_args()
    
    write_table = []
    table = file_to_dict(args.input_file)
    text_fields = [field for field in table[0] if 'text_' in field.lower() or 'text' == field.lower()]
    for read_row in table:
        for key in read_row:
            if 'text_' in key.lower() or 'text' == key.lower():
                write_row = {k: v for k, v in read_row.items() if k not in text_fields}
                write_row.update({'Text': read_row[key]})
                if 'text_' in key.lower():
                    write_row.update({'string_' + args.column_dest: key.split('ext_')[1]})
                else:
                    write_row.update({'string_' + args.column_dest: 'Text'})
                if read_row[key] != '' or args.blanks:
                    write_table.append(write_row)
    dict_to_file(write_table, args.output_file)
    
if __name__ == '__main__':
    main()
