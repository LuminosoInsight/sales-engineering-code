import csv
import argparse

def file_to_dict(file_name):
    table = []
    with open(file_name) as f:
        reader = csv.DictReader(f)
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
    args = parser.parse_args()
    
    write_table = []
    table = file_to_dict(args.input_file)
    text_fields = [field for field in table[0] if 'text_' in field.lower()]
    for read_row in table:
        for key in read_row:
            if 'text_' in key.lower():
                write_row = {k: v for k, v in read_row.items() if k not in text_fields}
                write_row.update({'Text': read_row[key]})
                write_row.update({args.column_dest: key.split('ext_')[1]})
                if read_row[key] != '':
                    write_table.append(write_row)
    dict_to_file(write_table, args.output_file)
    
if __name__ == '__main__':
    main()