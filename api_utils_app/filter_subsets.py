client = LuminosoClient.connect('/')

client = client.change_path('/projects/d53m338v/prv28wdf')

subsets = client.get('subsets/stats')

subsets

subsets_to_remove = []
for subset in subsets:
    if subset['count'] < 150:
        subsets_to_remove.append(subset['subset'])

docs = []
while True:
    new_docs = client.get('docs', limit=25000, offset=len(docs))
    if new_docs:
        docs.extend(new_docs)
    else:
        break

for doc in docs:
    subset = []
    for doc_subset in doc['subsets']:
        if doc_subset not in subsets_to_remove:
            subset.append(doc_subset)
    doc['subsets'] = subset
        

client = LuminosoClient.connect()
proj_id = client.post(name='AT Kearney Glacier NP Reviews')['project_id']
client = client.change_path(proj_id)
batch = 0
batch_size = 10000
total_size = len(docs)
while batch < total_size:
    end = min(total_size, batch + batch_size)
    client.upload('docs', docs=docs[batch:end])
    batch += batch_size
client.post('docs/recalculate')