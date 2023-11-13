import argparse
import csv
import json

from argparse import RawTextHelpFormatter

from luminoso_api import V5LuminosoClient as LuminosoClient
from se_code.copy_shared_concepts import delete_shared_concepts


def main():
    parser = argparse.ArgumentParser(
        description=('Upload a file containing shared concepts '
                     'to a target Daylight project.'
                     '\n\nExample csv:'
                     '\n concept_list_name,texts, name, color'
                     '\n lista,"tablet,ipad",tablet,#808080'
                     '\n lista,device,device,#800080'
                     '\n lista,speakers, speakers,#800000'
                     '\n listb,"usb,usb-c,micro-usb","cable","#008080"'
                     '\n listb,"battery","battery","#000080"'
                     '\n listb,"switch","switch","#808000"'
                     '\n\nExample JSON:'
                     '\n [{"concept_list_name":"lista"'
                     '\n   "concepts": ['
                     '\n    {"texts":"tablet,ipad","name":"tablet","color":"#808080"},'
                     '\n    {"texts":"device", "name":"device","color":"#800080"},'
                     '\n    {"texts":"speakers", "name":"speakers","color":"#800000"}]},'
                     '\n  {"concept_list_name":"listb",'
                     '\n  "concepts": ['
                     '\n    {"texts":"usb,usb-c,micro-usb","name":"cable","color":"#008080"},'
                     '\n    {"texts":"battery", "name":"battery","color":"#000080"},'
                     '\n    {"texts":"switch", "name":"switch","color":"#808000"}]}'
                     '\n ]'), 
        formatter_class=RawTextHelpFormatter
    )
    parser.add_argument("project_url", help="The URL of the project to use.")
    parser.add_argument(
        'filename',
        help=('Full name of the file containing shared concept definitions,'
              ' including the extension. Must be CSV or JSON.'
             )
        )
    parser.add_argument('-d', '--delete', default=False, action="store_true", 
        help="Whether to delete all the existing shared concepts or not.")
    parser.add_argument('-g', '--download',
                        action="store_true",
                        default=False,
                        help="Download the shared concept lists from a project to a json file.")
    parser.add_argument('-e', '--encoding', default="utf-8", 
        help="Encoding type of the files to read from")
    args = parser.parse_args()

    root_url = args.project_url.strip('/ ').split('/app')[0]
    project_id = args.project_url.strip('/').split('/')[6]
    client = LuminosoClient.connect(url=root_url + '/api/v5/projects/' + project_id)

    filename = args.filename
    true_data = []
    if args.download:
        scl = client.get('concept_lists/')

        # remove all 'concept_list_id' and 'shared_concept_id' so it can be re-uploaded
        for cl in scl:
            cl.pop('concept_list_id', None)
            for c in cl['concepts']:
                c.pop('shared_concept_id', None)

        with open(filename, 'w') as f:
            json.dump(scl, f, indent=2)
        statement = f'Shared concept lists downloaded to {filename}'

    elif '.csv' in filename:
        with open(filename, encoding=args.encoding) as f:
            reader = csv.DictReader(f)
            true_data = {}
            for d in reader:
                if d['concept_list_name'] not in true_data.keys():
                    true_data[d['concept_list_name']] = {
                        "name": d['concept_list_name'],
                        "concepts": []
                    }
                if 'texts' not in [k.lower() for k in list(d.keys())]:
                    print('ERROR: File must contain a "text" column.')
                    return
                row = {}
                for k, v in d.items():
                    if 'text' in k.lower():
                        row['texts'] = [t.strip() for t in v.split(',')]
                    if 'name' in k.lower():
                        row['name'] = v
                    if 'color' in k.lower():
                        row['color'] = v
                true_data[d['concept_list_name']]['concepts'].append(row)
            
        # reformat true_data for export
        true_data = [{'concept_list_name':cl[1]['name'],
                      'concepts':cl[1]['concepts']} for cl in true_data.items()]

    elif '.json' in filename:
        true_data = json.load(open(filename, encoding=args.encoding))
        for clist in true_data:
            for c in clist['concepts']:
                c['texts'] = c['texts'].split(",")
    else:
        print('ERROR: you must pass in a CSV or JSON file.')
        return

    if not args.download:
        statement = 'New Shared Concepts added to project'
        if args.delete:
            delete_shared_concepts(client)
            statement += ' and old Shared Concepts deleted'

        for cl in true_data:
            client.post('concept_lists/', name=cl['concept_list_name'], concepts=cl['concepts'])

    print(statement)
    print('Project URL: %s' % args.project_url)


if __name__ == '__main__':
    main()
