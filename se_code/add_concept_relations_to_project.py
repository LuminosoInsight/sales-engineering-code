from luminoso_api import V5LuminosoClient as LuminosoClient
import csv, json, datetime, time, argparse

from doc_downloader import get_all_docs, search_all_doc_ids, add_relations

def format_docs_for_upload(docs):
    new_docs = []
    for d in docs:
        new_docs.append({
            'text': d['text'],
            'title': d['title'],
            'metadata': d['metadata']
        })
    return new_docs

def write_documents(client,docs):
    offset = 0
    while offset<len(docs):
        end = min(len(docs),offset+1000)
        result = client.post('upload', docs=docs[offset:end])
        offset = end

def create_and_build(root_client, old_project_info, docs):
    new_proj_info = root_client.post("/projects", name=old_project_info['name']+" (plus relations)", language=old_project_info['language'], workspace_id=old_project_info['workspace_id'])
    new_proj_client = root_client.client_for_path("/projects/{}".format(new_proj_info['project_id']))
    write_documents(new_proj_client, docs)
    new_proj_client.post("/build")
    new_proj_client.wait_for_build()

def main():
    parser = argparse.ArgumentParser(
        description='Download all docs from a luminoso project, calculate which saved concepts each doc is aligned to and create a new project with them'
    )
    parser.add_argument('project_url', help="The URL of the project to analyze")
    args = parser.parse_args()
    
    api_url = args.project_url.split('/app')[0]
    project_id = args.project_url.strip('/ ').split('/')[-1]

    account_id = args.project_url.strip('/').split('/')[5]
    project_id = args.project_url.strip('/').split('/')[6]
    api_url = '/'.join(args.project_url.strip('/').split('/')[:3]).strip('/') + '/api/v5'
    proj_apiv5 = '{}/projects/{}'.format(api_url, project_id)

    client = LuminosoClient.connect(url=proj_apiv5)
    root_client = client.client_for_path("/")

    print("Reading documents")
    docs = get_all_docs(client)
    print("Done reading: {} documents".format(len(docs)))
    add_relations(client,docs)
    print("Uploading documents")
    new_docs = format_docs_for_upload(docs)
    project_info = client.get("/")

    print("Building project")
    create_and_build(root_client,project_info,new_docs)
    print("Done")
    
if __name__ == '__main__':
    main()
