import argparse

from luminoso_api import V5LuminosoClient as LuminosoClient

from se_code.doc_downloader import get_all_docs, add_relations
from se_code.copy_shared_concepts import copy_shared_concepts
from se_code.copy_shared_views import copy_shared_views


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
        if offset % 1000 == 0:
            print("Uploaded {} total documents".format(offset))


def create_and_build(client, old_project_info, docs):
    root_client = client.client_for_path("/")

    new_proj_info = root_client.post("/projects", name=old_project_info['name']+" (plus relations)", language=old_project_info['language'], workspace_id=old_project_info['workspace_id'])
    new_proj_client = root_client.client_for_path("/projects/{}".format(new_proj_info['project_id']))
    write_documents(new_proj_client, docs)

    print("Copying shared views and concept lists")
    copy_shared_concepts(client, new_proj_client)
    copy_shared_views(client, new_proj_client)

    print("Building project")
    new_proj_client.post("/build")
    new_proj_client.wait_for_build()


def main():
    parser = argparse.ArgumentParser(
        description='Download all docs from a luminoso project, calculate which shared concepts each doc is aligned to and create a new project with them'
    )
    parser.add_argument('project_url', help="The URL of the project to analyze")

    parser.add_argument(
        "-mt",
        "--match_type",
        default="both",
        help="For concept relations use exact, conceptual or both when searching",
    )

    args = parser.parse_args()

    project_id = args.project_url.strip('/').split('/')[6]
    api_url = '/'.join(args.project_url.strip('/').split('/')[:3]).strip('/') + '/api/v5'
    proj_apiv5 = '{}/projects/{}'.format(api_url, project_id)

    client = LuminosoClient.connect(url=proj_apiv5)

    print("Reading documents")
    docs = get_all_docs(client)
    print("Done reading: {} documents".format(len(docs)))
    print("Calculating concept relations")
    add_relations(client, docs, True, match_type=args.match_type)
    print("Uploading documents")
    new_docs = format_docs_for_upload(docs)
    project_info = client.get("/")

    create_and_build(client, project_info, new_docs)
    print("Done")


if __name__ == '__main__':
    main()
