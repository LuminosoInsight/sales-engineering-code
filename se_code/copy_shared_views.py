from luminoso_api import V5LuminosoClient as LuminosoClient
import argparse
import time

def parse_url(url):
    api_root = url.strip('/ ').split('/app')[0]
    proj_id = url.strip('/').split('/')[6]
    return api_root + '/api/v5/projects/' + proj_id

def copy_shared_views(from_client, to_client):
    # grabs all the views from a project
    shared_views = from_client.get("shared_views")
    
    # this gets the ids for the shared views that use lists and the list ids
    views = []
    mult = set()
    for list in shared_views:
        views.append(list)
        try:
            info = shared_views[list]['concept_list_id'] 
        except:
            continue
        else:
            if info not in mult:
                mult.add(info)
    
    # gets concept lists relevant to the views
    concept_list = from_client.get("/concept_lists/")
    concept_list[:] = [concept for concept in concept_list if concept['concept_list_id'] in mult]  
    
    # creates the necessary concept lists in the to project
    concept_id_dict = {}
    i = 0
    while i < len(concept_list):
        concepts = []
        for concept in concept_list[i]['concepts']:
            concept.pop('shared_concept_id')
            concepts.append(concept)
        list_name = concept_list[i]['name']
        new_concept = to_client.post("/concept_lists/",name=list_name,concepts=concepts)
        concept_id_dict[concept_list[i]['concept_list_id']] = new_concept['concept_list_id']
        i += 1
        
    # copies all views into the to project
    for k,sv in shared_views.items():
        try:
            sv['concept_list_id'] = concept_id_dict[sv['concept_list_id']]
            time.sleep(0.05)
            to_client.post("/shared_views",shared_view=sv)
        except:
            time.sleep(0.05)
            to_client.post("/shared_views",shared_view=sv)
            
    print('Copy of shared views complete')

    

def main():
    parser = argparse.ArgumentParser(
        description='Copy shared views and their associated concept lists from one project to another.'
    )
    parser.add_argument('from_url', help="The URL of the project to copy all topics from")
    parser.add_argument('to_url', help="The URL of the project to copy all topics into")
    
    args = parser.parse_args()
    
    from_api_url = parse_url(args.from_url)
    to_api_url = parse_url(args.to_url)
    
    from_client = LuminosoClient.connect(from_api_url, user_agent_suffix='se_code:copy_shared_views:from')
    to_client = LuminosoClient.connect(to_api_url, user_agent_suffix='se_code:copy_shared_views:to')
    
    copy_shared_views(from_client, to_client)
    
if __name__ == '__main__':
    main()