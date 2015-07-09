from itertools import chain
from sklearn.feature_extraction.text import TfidfVectorizer
from luminoso_api import LuminosoClient
import networkx 
from networkx.algorithms.components.connected import connected_components

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

    def __init__(self, acct, proj, token, split_amt=10**10, reconcile_func=None, thresh=0.95):
        """ split_amt is the number of docs per batch  """
        self.thresh = thresh
        self.reconcile_func = reconcile_func
        self.split_amt = split_amt
        self.cli = LuminosoClient.connect('/projects/'+acct+'/'+proj, token=token)

    def intervals(self, maximum, interval):
        """ Creates intervals of size 'interval' up to maximum """
        return [(i, min(i + interval, maximum)) for i in range(0, maximum, interval)]

    def chunks(self, l, n):
        """ Partitions a list in to a list of lists """
        n = max(1, n)
        return [l[i:i + n] for i in range(0, len(l), n)]

    def all_docs(self):
        """ Fetches all documents from a project """
        limit = 25000
        offset = 0
        docs = []
        while True:
            batch = self.cli.get('docs', limit=limit, offset=offset)
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

    # get_similar outputs a list of dupe pairs. The following graph methods turn
    # these dupe pairs in to a graph and return the connected components (full
    # sets of dupes). Note: in some cases two docs will be considered dupes even
    # with similarity a bit less than self.thresh due to transitivity
    def to_graph(self, l):
        G = networkx.Graph()
        for part in l:
            G.add_nodes_from(part) # each sublist is a bunch of nodes
            G.add_edges_from(self.to_edges(part)) # it also implies a number of edges
        return G

    def to_edges(self, l):
        it = iter(l)
        last = next(it)
        for current in it:
            yield last, current
            last = current
            
    def get_dupes(self, dupe_list):
        return connected_components(self.to_graph(dupe_list))

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
            # Deduped is a list of docs to keep
            dupe_ids = [batch[i]['_id'] for i in chain(*dupe_sets)]

            dupes_to_retain = [] #will contain the duplicates that should be retained
            for dupe_set in dupe_sets:
                docs = [batch[i] for i in dupe_set]
                dupes_to_retain.append(self.reconcile_dupes(docs))

            # send delete request to delete all dupes. Partitioned due to URI limitations.
            for d in self.chunks(dupe_ids, 100):
                self.cli.delete('/docs', ids=d)
            print("Finished deleting duplicates from project \n")

            # need to remove the _id field otherwise it will get deleted
            # upon recalc (if it's the same as one of the deleted docs,
            # which it often is depending on the reconcile function).
            for d in dupes_to_retain:
                if '_id' in d:
                    del d['_id']

            # upload the dupes we have chosen to keep
            self.cli.upload('docs', dupes_to_retain)
        
        print("Deduping finished. Project is now recalculating.")
        self.cli.post('docs/recalculate')