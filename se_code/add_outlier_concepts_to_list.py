import argparse
import json

from luminoso_api import V5LuminosoClient as LuminosoClient

from se_code.doc_downloader import get_all_docs, add_relations


def format_docs_for_upload(docs):
    new_docs = []
    for d in docs:
        new_docs.append({
            'text': d['text'],
            'title': d['title'],
            'metadata': d['metadata']
        })
    return new_docs


def write_documents(client, docs):
    offset = 0
    while offset < len(docs):
        end = offset+1000
        client.post('upload', docs=docs[offset:end])
        offset = end


def create_and_build(root_client, old_project_info, docs):
    new_proj_info = root_client.post("/projects", name=old_project_info['name']+" (plus relations)", language=old_project_info['language'], workspace_id=old_project_info['workspace_id'])
    new_proj_client = root_client.client_for_path("/projects/{}".format(new_proj_info['project_id']))
    write_documents(new_proj_client, docs)
    new_proj_client.post("/build")
    new_proj_client.wait_for_build()


def main():
    parser = argparse.ArgumentParser(
        description='Calculate the outlier concepts on a concept list and add those concepts to the list.'
    )
    parser.add_argument('project_url', help="The URL of the project to analyze")
    parser.add_argument('-c', '--concept_lists', default=None,
                        help='Comma separated concept list names to create outlier lists for. Default=ALL'
                             ' Samp: "list one,list two"')
    parser.add_argument('-m', '--match_type', default="both",
                        help='Type of match: both, exact, conceptual. default=both')
    parser.add_argument('-n', '--num_outlier_concepts', 
                        default=10,
                        help='The number of outlier concepts to create. Default=10')
    args = parser.parse_args()

    project_id = args.project_url.strip('/').split('/')[6]
    api_url = '/'.join(args.project_url.strip('/').split('/')[:3]).strip('/') + '/api/v5'
    proj_apiv5 = '{}/projects/{}'.format(api_url, project_id)

    client = LuminosoClient.connect(url=proj_apiv5)

    # get project info for calculating coverage
    proj_info = client.get("/")

    # get the list of shared concepts
    all_concept_lists = client.get('concept_lists/')
    all_concept_list_names = [cl['name'] for cl in all_concept_lists]

    concept_lists = all_concept_lists
    if args.concept_lists is not None:
        argclnames = args.concept_lists.split(',')
        concept_lists = [cl for cl in all_concept_lists if cl['name'] in argclnames]

        for cname in argclnames:
            if cname not in all_concept_list_names:
                print("Concept list: '{}' not in saved concept lists.".format(cname))
                print("Available list names:\n  {}".format("\n  ".join(all_concept_list_names)))
                print("Failed")
                exit()

    # create the concept selector
    for cl in concept_lists:
        outlier_cs = {"type": "concept_list", "concept_list_id": cl['concept_list_id']}

        setup_outlier_results = client.post("concepts/outliers/", 
                                            concept_selector=outlier_cs, 
                                            match_type=args.match_type)

        coverage_pct = 100-((setup_outlier_results['filter_count'] / proj_info['document_count']) * 100)

        # get the list of outlier concepts
        outlier_filter = [{"special": "outliers"}]
        unique_cs = {"type": "unique_to_filter", "limit": args.num_outlier_concepts}
        outlier_concepts = client.get("concepts", concept_selector=unique_cs, filter=outlier_filter)

        if (len(outlier_concepts['result']) > 0):
            # build the new shared concept list
            save_list = [{'name': c['name'], 'texts': c['texts'], 'color': c['color']} for c in cl['concepts']]
            for c in outlier_concepts['result']:
                save_list.append({'name': c['name'], 'texts': c['texts']})
            new_name = cl['name']+' with Top {} outliers ({:.2f}%) coverage'.format(len(outlier_concepts['result']),coverage_pct)

            result = client.post("/concept_lists", name=new_name, concepts=save_list)
            print("new list created: "+new_name)
        else:
            top_cs = {"type": "top", "limit": args.num_outlier_concepts}
            top_concepts = client.get("concepts", concept_selector=top_cs, filter=outlier_filter)
            top_concepts = [c['name'] for c in top_concepts['result']]

            print("No outlier concepts found in list: {}. Coverage [{:.2f}%] ".format(cl['name'], coverage_pct))
            print("No concept list created.")
            print("Some prevalent concepts in those outlier documents: [{}]".format(", ".join(top_concepts)))
    print("Done")


if __name__ == '__main__':
    main()
