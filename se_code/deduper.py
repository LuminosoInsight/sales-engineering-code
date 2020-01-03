import argparse
from itertools import chain
from sklearn.feature_extraction.text import TfidfVectorizer
from luminoso_api import V5LuminosoClient as LuminosoClient
from networkx import Graph, connected_components

def __retain_shortest(docs):
    return sorted(docs, key = lambda d: len(d['text']))[0]

def __retain_longest(docs):
    return sorted(docs, key = lambda d: len(d['text']))[-1]

class Deduper(object):
    """ This class is initialized by pointing it at a project. It then downloads
        all of the documents, vectorizes them based on unprocessed surface forms,
        finds the pairwise cosine similarity, groups documents in to sets of highly
        similar documents, and then deletes all but one from the project.
        
        The choice of which one to delete (or to create and retain a new 'hybrid'
        document) is left to the reconcile function, which can also be specified
        upon initialization. Since the complexity is n^2 (more precisely n choose 2),
        the split_amt parameter allows the deduplication process to be batched
        for large projects. Note that this batching may cause a few dupes to be
        missed, and is only effective for projects with dates that have the property
        of dupes being likely to occur close to each other in time. In practice,
        the data sets that have dupes have been small enough to do in one batch, but
        when that changes the batching algorithm could be improved or the entire thing
        could be overhauled to use LSH which should reduce the complexity to n*log(n).
        
        If you wish to use a custom reconcile function, you should pass a function
        that takes a list of document dictionaries, and returns a single document
        dictionary to be retained in the project. """

    def __init__(self, client, split_amt=10**10, reconcile_func=None, thresh=0.95):
        """ split_amt is the number of docs per batch  """
        self.thresh = thresh
        self.reconcile_func = reconcile_func
        self.split_amt = split_amt
        self.cli = client
        
    def intervals(self, maximum, interval):
        """ Creates intervals of size 'interval' up to maximum """
        return [(i, min(i + interval, maximum)) for i in range(0, maximum, interval)]

    def chunks(self, l, n):
        """ Partitions a list in to a list of lists """
        return [l[i:i + n] for i in range(0, len(l), n)]

    def all_docs(self):
        """ Fetches all documents from a project """
        limit = 25000
        offset = 0
        docs = []
        while True:
            batch = self.cli.get('docs', limit=limit, offset=offset)['result']
            docs.extend(batch)
            if len(batch) < limit:
                return docs
            offset += limit

    def get_similar(self, similarity_matrix):
        """ Return all pairs with similarity > self.thresh.
            This is batched to save memory """
        similar = []
        for lower, upper in self.intervals(similarity_matrix.shape[0], 5000):
            sim = similarity_matrix[lower:upper,] > self.thresh
            sim = sim.nonzero()
            similar.extend([(x[0] + lower, x[1]) for x in zip(*sim)])
            print("processed rows " + str(lower) + " to row " + str(upper))
        return [s for s in similar if s[0] != s[1]]
            
    def get_dupes(self, dupe_list):
        """ Turn these dupe pairs in to a graph and return the connected
            components (full sets of dupes). Note: in some cases two docs
            will be considered dupes even with similarity a bit less than
            self.thresh due to transitivity """
        return connected_components(Graph(dupe_list))

    def reconcile_dupes(self, dupes):
        """ Dedupe reconciliation process. Currently retains the first dupe.
        In many cases, this function should be modified depending on the 
        nature of the dupes, e.g. to retain the one with the shortest text,
        or create a new document according to some rules. """
        if self.reconcile_func:
            return self.reconcile_func(dupes)
        return dupes[0]

    def dedupe(self):
        """ Main method that should be called to dedupe the project """
        
        print("Fetching documents from project")
        documents = self.all_docs()
        n_docs = len(documents)

        if self.split_amt < n_docs:
            try:
                documents = sorted(documents, key=lambda d: d['date'])
            except KeyError:
                print("""WARNING: The deduplication process is being batched,
                but your data does not contain a date field. Thus it is likely
                that many duplicates are being missed across batches.""")

        batches = self.chunks(documents, self.split_amt)

        if n_docs > 50000 and self.split_amt > n_docs:
            print("""WARNING: This script may fail due to memory constraints.
            If it does, then re-run with split_amt parameter set at 40000 or lower.""")
    
        for i,batch in enumerate(batches):
            print("Starting batch " + str(i+1) + " of " + str(len(batches)))

            # want to vectorize based on surface forms for duplicate detection
            tfidf = TfidfVectorizer().fit_transform([d['text'] for d in batch])
            print("Finished vectorizing documents")

            pairwise_similarity = tfidf * tfidf.T
            print("Finished constructing similarity matrix")

            print("Starting to identify duplicates and near-duplicates")
            dupe_sets = self.get_dupes(self.get_similar(pairwise_similarity))
            print("Finished identifying duplicates")

            # get duplicates and near-duplicates and reconcile them.
            dupe_ids = [batch[i]['doc_id'] for i in chain(*dupe_sets)]

            dupes_to_retain = []
            for dupe_set in dupe_sets:
                docs = [batch[i] for i in dupe_set]
                dupes_to_retain.append(self.reconcile_dupes(docs))

            # send delete request to delete all dupes. Partitioned due to URI limitations.
            for d in self.chunks(dupe_ids, 100):
                self.cli.post('/docs/delete', doc_ids=d)
            print("Finished deleting duplicates from project \n")

            # need to remove the doc_id field otherwise it will get deleted
            # upon recalc (if it's the same as one of the deleted docs,
            # which it often is depending on the reconcile function).
            for d in dupes_to_retain:
                if 'doc_id' in d:
                    del d['doc_id']

            if (len(dupes_to_retain)>0):
                print("d0={}".format(dupes_to_retain[0]))
                # upload the dupes we have chosen to keep
                self.cli.post('docs', docs=dupes_to_retain)

            #calculate number deleted
            num_deleted = len(dupe_ids) - len(dupes_to_retain)
        
        print("Deduping finished. Project is now recalculating.")
        self.cli.post('build')

        return num_deleted

def main():
    parser = argparse.ArgumentParser(
        description='Dedupe documents from a project'
    )
    parser.add_argument('project_url', help="The complete URL of the Analytics project")
    parser.add_argument('-t', '--token', default=None, help="Authentication Token for Daylight")
    parser.add_argument('-copy', '--copy', action='store_true', help="Use a copy of the project")
    parser.add_argument('-func', '--func', default='None', help="Reconcile function to use [shortest,longest,None]")
    args = parser.parse_args()
    
    project_url = args.project_url.strip('/')
    api_url = project_url.split('/app')[0].strip() + '/api/v5'
    project_id = project_url.split('/')[6].strip()
    workspace_id = project_url.split('/')[5].strip()
    
    print("opening client: {}  - {}/{}".format(api_url,project_id,workspace_id))
    if args.token:
        client = LuminosoClient.connect(url='%s/projects/%s' % (api_url.strip('/'), project_id), token=args.token)
    else:
        client = LuminosoClient.connect(url='%s/projects/%s' % (api_url.strip('/'), project_id))
    
    
    print("prjinfo={}".format(client.get('/')))
    #print('Getting Docs...')
    #docs = get_all_docs(client)

    print('Dedupe starting...')
    if args.copy:
        copy_info = client.post('copy')
        client = client.client_for_path("/projects/"+copy_info['project_id'])

        print("Waiting for copy to complete.")
        client.wait_for_build()
        print("Done.")

    initial_count = client.get('/')['document_count']

    if args.func == 'shortest':
        reconcile_func = __retain_shortest
    elif args.func == 'longest':
        reconcile_func = __retain_longest
    else:
        reconcile_func = None

    if initial_count > 40000:
        deduper = Deduper(client,split_amt=40000, reconcile_func=reconcile_func)
    else:
        deduper = Deduper(client, reconcile_func=reconcile_func)
    num_deleted = deduper.dedupe()

    p_info = client.get('/')
    url = 'https://analytics.luminoso.com/app/projects/{}/{}/highlights'.format(p_info['account_id'],p_info['project_id'])
    
    client.wait_for_build()
    print("num deleted:{}, url:{}".format(num_deleted,url))

    return {'num':num_deleted, 'url':url}

if __name__ == '__main__':
    main()