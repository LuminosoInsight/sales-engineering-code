"""
Convenience class of wrapper objects around Luminoso V5 API Clients.
"""

import numpy as np
import pandas as pd

from luminoso_api import V5LuminosoClient
from pack64 import unpack64


def make_uploadable_doc(doc):
    """
    Clean out all keys from the given document except those acceptable in a
    document to upload to a project.
    """
    return dict(
        text=doc.get("text", ""),
        title=doc.get("title", ""),
        metadata=doc.get("metadata", []),
    )


class LuminosoClientHolder:
    """
    Convenience objects wrapping a Luminoso V5 API client.
    """

    def __init__(self, client, parent=None):
        """
        Make a wrapper object around the given client.
        """
        self.client = client  # a luminoso api client
        self.parent = parent  # a luminoso client wrapper (or None)

    @classmethod
    def from_root_url(cls, url="http://localhost:8000/api/v5", **kwargs):
        """
        Create and save an initialized LuminosoClient for interacting with a
        Daylight stack.  (Depends on the existence of an access token
        in ~/.luminoso/tokens.json, unless you specify a token with the
        argument token=<your-token-here>.)
        """
        client = V5LuminosoClient.connect(url=url, **kwargs)
        return cls(client=client, parent=None)

    @property
    def root(self):
        """
        The client wrapper object for the whole stack.  This will be the
        parent for child (per-project) wrapper objects, and self for root
        (all-project) objects.
        """
        return self.parent or self

    def is_root(self):
        """
        Is this a wrapper for a space of projects or a single project?
        """
        return self.root == self

    @property
    def project_id(self):
        """
        For per-project wrapper objects, return the project id of the
        associated project.
        """
        if self.is_root():
            raise ValueError("Can only get the id from a per-project wrapper.")
        info = self.client.get("")
        result = info["project_id"]
        return result

    @property
    def project_name(self):
        """
        For per-project wrapper objects, return the name of the associated
        project.
        """
        if self.is_root():
            raise ValueError("Can only get the name from a per-project wrapper.")
        info = self.client.get("")
        result = info["name"]
        return result

    def get_project_info(self, project_name=None, project_id=None):
        """
        Returns the descriptor(s) (if any) for the project.  Search may be by
        id (exact matches only) or name (for which partial matches are allowed
        and all matching projects will be returned).  Returns a list of all
        matches found.
        """
        info_list = self.root.client.get("/projects/")
        if project_id is not None:
            info_list1 = [
                info for info in info_list if info["project_id"] == project_id
            ]
            if len(info_list1) > 1:
                raise ValueError("Multiple projects with id {}.".format(project_id))
        elif project_name is not None:
            info_list1 = [
                info for info in info_list if info["name"].find(project_name) != -1
            ]
        else:
            raise ValueError("Must specify one of project_id or project_name")
        return info_list1

    def get_project(self, project_id):
        """
        Returns a client wrapper for interacting with a single project.
        """
        project_client = self.root.client.client_for_path(
            "/projects/{}".format(project_id)
        )
        result = LuminosoClientHolder(project_client, parent=self.root)
        return result

    def new_project_from_docs(self, project_name, lang="en", docs=[], **kwargs):
        """
        Creates a project with the given name from the given list of docs.
        Returns a wrapper object for the project created.
        """
        print("Starting build of {}.".format(project_name))
        project_id = self.root.client.post(
            "/projects/", name=project_name, language=lang, **kwargs
        )["project_id"]
        project = self.get_project(project_id)
        buffer = []
        for doc in docs:
            doc = make_uploadable_doc(doc)
            buffer.append(doc)
            if len(buffer) >= 1000:
                project.client.post("upload", docs=buffer)
                buffer = []
        if len(buffer) > 0:
            project.client.post("upload", docs=buffer)
            buffer = []
        project.client.post("build")
        print("Waiting for build of {} ({}).".format(project_name, project_id))
        project.client.wait_for_build()
        print("Done building {}.".format(project_id))
        return project

    def delete_project(self, project_id):
        """
        Delete the project with the given id.  Deleting the project associated
        with a child wrapper object is kosher, but further operations on that
        project won't be (of course).
        """
        path = "/projects/{}/".format(project_id)
        self.root.client.delete(path)

    def get_docs(self, **kwargs):
        """
        Generates documents from the project (client).
        """
        if self.is_root():
            raise ValueError("Can only get docs from a per-project wrapper.")
        project_id = self.project_id
        offset = 0
        limit = 1000
        while True:
            doc_dict = self.client.post("/docs/", offset=offset, limit=limit, **kwargs)
            n_new_docs = len(doc_dict["result"])
            offset += n_new_docs
            if n_new_docs < 1:
                return
            if offset % 5000 == 0:
                print("Fetched {} documents from {}.".format(offset, project_id))
            for doc in doc_dict["result"]:
                yield doc

    def get_term_ids(self):
        """
        Returns a set of all term ids present in documents of the project.
        """
        if self.is_root():
            raise ValueError("Can only get terms from a per-project wrapper.")
        term_ids = set()
        for doc in self.get_docs():
            for term in doc["terms"]:
                term_ids.add(term["term_id"])
        return term_ids

    def get_term_vectors(self, term_ids=None, **kwargs):
        """
        Returns a dataframe mapping the specified term ids (by default all
        terms from the project) to their vectors.
        """
        if self.is_root():
            raise ValueError("Can only get terms from a per-project wrapper.")
        project_id = self.project_id
        if term_ids is None:
            term_ids = sorted(self.get_term_ids(**kwargs))
        valid_term_ids = [tid for tid in term_ids if tid.count("|") == 1]
        found_term_ids = []
        embeddings = []
        offset = 0
        limit = 1000
        while True:
            next_offset = np.min([offset + limit, len(valid_term_ids)])
            if next_offset <= offset:
                break
            new_terms = self.client.post(
                "terms", term_ids=valid_term_ids[offset:next_offset]
            )
            offset = next_offset
            new_ids = [t["term_id"] for t in new_terms if t.get("vector") is not None]
            if len(new_ids) > 0:
                new_vectors = np.vstack(
                    [
                        unpack64(t["vector"])
                        for t in new_terms
                        if t.get("vector") is not None
                    ]
                )
                found_term_ids.extend(new_ids)
                embeddings.append(new_vectors)
            if offset % 5000 == 0:
                print("Fetched {} vectors from {}.".format(offset, project_id))
        if len(embeddings) > 0:
            vectors = np.vstack(embeddings)
        else:
            vectors = np.empty((0, 150), dtype=np.float32)
        result = pd.DataFrame(index=found_term_ids, data=vectors)
        return result

    def get_concept_term_ids(self, **kwargs):
        """
        Returns a list of the term ids of concepts of the project.  By default
        these will be the top concepts, but the concepts to return can be
        specified by the keyword argument concept_selector.
        """
        if self.is_root():
            raise ValueError("Can only get concepts from a per-project wrapper.")
        concept_data = self.client.get("/concepts", **kwargs)
        concepts = concept_data["result"]
        if len(concepts) > 0 and "relevance" in concepts[0]:
            concepts.sort(key=lambda concept: concept["relevance"])
        term_ids = [c["exact_term_ids"][0] for c in concepts]
        return term_ids

    def get_concept_associations(self, **kwargs):
        """
        Returns a dict mapping pairs of concepts (represented as term id's)
        to their associations, and a list of those concepts.  The concepts to
        associate may be specified by giving a value to the keyword argument
        concept_selector, and defaults to the top concepts of the project.
        """
        concepts = self.client.get("/concepts/concept_associations", **kwargs)
        if len(concepts) > 0 and "relevance" in concepts[0]:
            concepts.sort(key=lambda concept: concept["relevance"])
        term_ids = [c["exact_term_ids"][0] for c in concepts]
        associations = [c["associations"] for c in concepts]
        association_map = {}
        for term_id, assoc_list in zip(term_ids, associations):
            other_terms = [a["exact_term_ids"][0] for a in assoc_list]
            assoc_scores = [a["association_score"] for a in assoc_list]
            for other_term_id, score in zip(other_terms, assoc_scores):
                association_map[(term_id, other_term_id)] = score
        return association_map, term_ids
