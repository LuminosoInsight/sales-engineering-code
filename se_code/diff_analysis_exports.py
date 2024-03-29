import csv, json, argparse
from luminoso_api import V5LuminosoClient as LuminosoClient
FIELDS = ['Relevance Ranking', 'Concept', 'Exact Matches', 'Exact Match Percent of All Documents', 'Conceptual Matches', 'Conceptual Match Percent of All Documents', 'Total Matches', 'Total Match Percent of All Documents']

def format_write_table(match_counts):
    total_count = match_counts['total_count']
    write_table = [{'Relevance Ranking': (i + 1),
                    'Concept': c['name'],
                    'Exact Matches': c['exact_match_count'],
                    'Exact Match Percent of All Documents': '%.2f%%' % (c['exact_match_count'] / total_count * 100),
                    'Conceptual Matches': c['match_count'] - c['exact_match_count'],
                    'Conceptual Match Percent of All Documents': '%.2f%%' % (((c['match_count'] - c['exact_match_count']) / total_count) * 100),
                    'Total Matches': c['match_count'],
                    'Total Match Percent of All Documents': '%.2f%%' % (c['match_count'] / total_count * 100)} for i, c in enumerate(match_counts['match_counts'])]
    return write_table

def write_to_csv(write_table, filename, fields, encoding="utf-8"):
    with open(filename, 'w', newline='', encoding=encoding) as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(write_table)
    print('Wrote %d lines of data to %s.' % (len(write_table), filename))

def main():
    parser = argparse.ArgumentParser(
        description='Downloads data to mimic the old v4 terms export format.'
    )
    parser.add_argument('project_url_1', help="The full URL of the first project to compare with diff analysis.")
    parser.add_argument('project_url_2', help="The full URL of the second project to compare with diff analysis.")
    parser.add_argument('-l', '--limit', type=int, default=500, help="number of concepts to download from projects.")
    parser.add_argument('-f1', '--filename_1', default='baseline_term_counts_export.csv', help="Name of file to write first project's output to.")
    parser.add_argument('-f2', '--filename_2', default='new_data_term_counts_export.csv', help="Name of file to write second project's output to.")
    parser.add_argument('-e', '--encoding', default='utf-8', help="Encoding type of file to write to")
    args = parser.parse_args()
    
    if args.filename_1.split('.')[-1] != 'csv' or args.filename_2.split('.')[-1] != 'csv':
        print("Files must be of CSV format and filenames must include '.csv' extension")
        return
    
    root_1 = args.project_url_1.split('/app')[0]
    project_1_id = args.project_url_1.strip('/').split('/')[-1]
    root_2 = args.project_url_2.split('/app')[0]
    project_2_id = args.project_url_2.strip('/').split('/')[-1]
    
    client_1 = LuminosoClient.connect(url='%s/api/v5/projects/%s' % (root_1, project_1_id), user_agent_suffix='se_code:diff_analysis:client1')
    client_2 = LuminosoClient.connect(url='%s/api/v5/projects/%s' % (root_2, project_2_id), user_agent_suffix='se_code:copy_shared_concepts:from')
    
    match_counts_1 = client_1.get('concepts/match_counts', concept_selector={'type': 'top', 'limit': int(args.limit)})
    match_counts_2 = client_2.get('concepts/match_counts', concept_selector={'type': 'top', 'limit': int(args.limit)})
    
    write_table_1 = format_write_table(match_counts_1)
    write_table_2 = format_write_table(match_counts_2)

    write_to_csv(write_table_1, args.filename_1, FIELDS, encoding=args.encoding)
    write_to_csv(write_table_2, args.filename_2, FIELDS, encoding=args.encoding)

if __name__ == '__main__':
    main()
