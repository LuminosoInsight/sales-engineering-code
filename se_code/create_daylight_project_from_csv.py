from luminoso_api import V5LuminosoClient as LuminosoClient

import argparse
import datetime, getpass, time, json, os, csv
import numpy, pack64

'''
Create a Luminoso Daylight project from a CSV file. Usefull if the file is too 
large for UI or building without the UI for things like search_enhancement

'''

def write_documents(client,docs):
    offset = 0
    while offset<len(docs):
        end = min(len(docs),offset+1000)
        result = client.post('upload', docs=docs[offset:end])
        offset = end

def create_project(client, input_file, project_name, account_id, keyword_expansion_terms=None, max_len=0, skip_sentiment_build=False):
    
    block_size = 5000

    # create the project
    print("Creating project named: "+project_name)
    project_info = client.post(name=project_name,language="en",account_id=account_id)
    print("New project info = "+str(project_info))

    client_prj = client.client_for_path(project_info['project_id'])

    # load the csv
    lumi_data_types = ["string","number","score","date"]
    reader = csv.DictReader(open(input_file, 'r', encoding='utf-8-sig'))
    docs = []
    for i, row in enumerate(reader):
        # print(str(row))
        new_doc = {}
        new_doc['metadata']=[]
        for k in row.keys():
            ksplit = k.split("_")
            if k.strip().lower() == "text":
                #if max_len>0:
                #    new_doc['text'] = row[k][0:max_len]
                #else:
                new_doc['text'] = row[k]
            if k.strip().lower() == "title":
                new_doc['title'] = row[k]
            if (len(ksplit)>1):
                field_name = "_".join(ksplit[1:])
                if ksplit[0].lower().strip() == 'date':
                    try:
                        if len(row[k])>0:
                            if (row[k].isnumeric()):
                                edate = int(row[k])
                            else:
                                date_formats = ["%Y-%m-%dT%H:%M:%SZ","%Y-%m-%d","%m/%d/%Y","%m/%d/%y","%m/%d/%y %H:%M"]
                                edate = None
                                for df in date_formats:
                                    try:
                                        edate = int(time.mktime(time.strptime(row[k], df)))
                                    except:
                                        pass
                                if edate == None:
                                    print("Error in date format: {}".format(row[k]))
                                    print("{} is numeric {}".format(row[k],row[k].isnumeric()))
                                else:
                                    new_doc['metadata'].append({"type":ksplit[0],"name":field_name,"value":edate})
                    except:
                        print("date error: {}".format(row[k]))
                elif ksplit[0].lower().strip() in 'number':
                    try:
                        new_doc['metadata'].append({"type":ksplit[0],"name":field_name,"value":float(row[k].strip()) if row[k].strip() else 0})                
                    except:
                        print("number error")
                elif ksplit[0].lower().strip() in 'score':
                    try:
                        new_doc['metadata'].append({"type":ksplit[0],"name":field_name,"value":float(row[k].strip()) if row[k].strip() else 0})
                    except:
                        print("score error")
                elif ksplit[0] in lumi_data_types:
                    new_doc['metadata'].append({"type":ksplit[0],"name":field_name,"value":row[k]})
        docs.append(new_doc)

        if len(docs)>=block_size:
            print("Uploading {} documents".format(len(docs)))
            write_documents(client_prj, docs)

            docs = []
    
    if len(docs)>0:
        print("Uploading {} documents".format(len(docs)))
        write_documents(client_prj, docs)
    
    sentiment_configuration = {"type":"full"}
    if (skip_sentiment_build):
        sentiment_configuration = {"type":"none"}

    print("Done uploading. Starting build")
    if keyword_expansion_terms:

        keyword_expansion_filter = []
        for entry in keyword_expansion_terms.split("|"):
            field_and_val = entry.split("=")
            print("fv {}".format(field_and_val))
            field_values = field_and_val[1].split(",")
            keyword_expansion_filter.append({"name":field_and_val[0],
                                            "values":field_values})

        keyword_expansion_dict = {"limit":20,
                                "filter":keyword_expansion_filter}
        print("keyword filter = {}".format(keyword_expansion_dict))

        client_prj.post("build",
                keyword_expansion=keyword_expansion_dict, 
                sentiment_configuration=sentiment_configuration)
    else:
        client_prj.post('build', sentiment_configuration=sentiment_configuration)
    
    print("Build started")

    return client_prj

def main():
    parser = argparse.ArgumentParser(
        description='Create a Luminoso project using a CSV file.'
    )
    parser.add_argument('input_file', help="CSV file with project data")
    parser.add_argument('-n', '--project_name', default="", required=True, help="New project name")
    parser.add_argument('-a', '--account_id', default="", required=False, help="Luminoso account ID")
    parser.add_argument('-u', '--api_url', default='https://daylight.luminoso.com/api/v5/projects/', help='The host url. Default=https://daylight.luminoso.com/api/v5/projects/')
    parser.add_argument('-k', '--keyword_expansion_terms', default=None, required=False, help="field list of metadata field=data,data to expand. search_doc_type=primary,primary2|search_doc_type2=secondary")
    parser.add_argument('-m', '--max_text_length', default="0", required=False, help="The maximum length to limit text fields")
    parser.add_argument('-s', '--skip_sentiment_build', action="store_true", default=False, help="Allows the build to skip the sentiment build")
    parser.add_argument('-w', '--wait_for_build_complete', action="store_true", default=False, help="Wait for the build to complete")
    args = parser.parse_args()
    
    project_name = args.project_name
    input_file = args.input_file
    account_id = args.account_id
    max_len = int(args.max_text_length)

    api_url = args.api_url

    # get the default account id if none given
    if len(account_id)==0:
        # use the v5 url to build the v4 url
        urllist = api_url.strip('/').split('/')
        api_urlv4 = "/".join(urllist[0:3])+"/api/v4"

        # connect to luminoso and get the client info
        clientv4 =  LuminosoClient.connect(url=api_urlv4)
        profile = clientv4.get("/user/profile/")
        if not profile['error']:
            account_id = profile["result"]["default_account"]
        else:
            print("error retrieving account_id: {}".format(profile['error']))

    # connect to v5 api
    client = LuminosoClient.connect(url=api_url, user_agent_suffix='se_code:create_daylight_project_from_csv')

    client_prj = create_project(client, input_file, project_name, account_id, keyword_expansion_terms=args.keyword_expansion_terms, max_len=max_len, skip_sentiment_build=args.skip_sentiment_build)

    if (args.wait_for_build_complete):
        print("waiting for build to complete...")
        client_prj.wait_for_build()

if __name__ == '__main__':
    main()