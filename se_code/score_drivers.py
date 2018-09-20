from luminoso_api.v5_client import LuminosoClient
from pack64 import unpack64

import numpy as np
import pandas as pd
from statsmodels.stats.weightstats import DescrStatsW


NTERMS = 2000


def download_docs(client):
    """
    Get the documents from a project via the API.
    """
    # FIXME: replace with the appropriate v5 invocation to get all documents,
    # even if there are more than 25000
    return client.get('docs', limit=25000)['result']


def get_term_data(client, n_terms=2000):
    """
    Returns two dataframes, indexed by the same list of term_ids, for the top
    `n_terms` domain terms in the project.

    The first output, `term_info_frame`, contains descriptive information about
    each term -- its preferred label and its relevance score.

    The second output, `term_vec_frame`, contains the term vectors.
    """
    terms = client.get(
        'concepts', concept_selector={'type': 'top', 'limit': NTERMS},
        include_vectors=True
    )['result']

    term_ids = [term['exact_term_ids'][0] for term in terms]
    term_info_frame = pd.DataFrame(
        {
            'label': [term['texts'][0] for term in terms],
            'relevance': [term['relevance'] for term in terms],
        },
        index=term_ids
    )
    term_vec_frame = pd.DataFrame(
        [unpack64(term['vector']) for term in terms],
        index=term_ids
    )
    return term_info_frame, term_vec_frame


def get_sparse_doc_frame(docs, term_info):
    """
    Get a dataframe whose rows are documents and whose columns are domain
    terms. It contains 1.0 if the document contains the given term, and
    0.0 otherwise.
    """
    term_ids = term_info.index
    sparse_vecs = []
    for doc in docs:
        doc_terms = set(term['term_id'] for term in doc['terms'])
        sparse_vec = [(term in doc_terms) for term in term_ids]
        sparse_vecs.append(sparse_vec)

    sparse_doc_frame = pd.DataFrame(
        np.vstack(sparse_vecs).astype('f'),
        columns=term_ids
    )
    return sparse_doc_frame


def get_fuzzy_doc_frame(docs, term_info, term_vecs):
    """
    The 'fuzzy document frame' has the same form as the 'sparse document
    frame', but it takes conceptual matches into account. We start with the
    sparse document frame and add an additional value to each entry
    (0 < value < 1) when the document contains a conceptual match for the term.

    Overall, the total weight for each term will roughly double, as it will
    add up to the original count of each term, plus that count being redistributed
    among conceptual matches.
    """

    assoc_values = term_vecs.dot(term_vecs.T)

    # Put conceptual matches on a non-linear scale:
    # An exact match (correlation = 1) gets a relative weight of 1.
    # A match with a correlation of 0.5 or less gets a weight of 0.
    # A match with a correlation of 0.75 gets a weight of 1/8.
    # The weight grows with the cube of the correlation score.
    closeness = ((assoc_values - .5) ** 3).clip(lower=0)

    # Scale the 'closeness' to add up for 1 for each term being distributed.
    closeness /= closeness.sum(axis=0)

    # Drop terms with no conceptual matches.
    interpolated = closeness.T.dropna()

    # Smudge the sparse doc matrix, which originally contains just exact matches,
    # so that some of the weight is on conceptual matches.
    sparse_docs = get_sparse_doc_frame(docs, term_info)
    semi_dense_docs = (sparse_docs + sparse_docs.dot(interpolated.T))
    return semi_dense_docs


def get_doc_metadata(docs, instance_field='hotel_id', predicted_field='rating'):
    """
    Get a dataframe whose rows are documents and whose columns are:

    - 'score': the score we are evaluating/predicting on the documents, such as
      an NPS score or a star rating.

    - 'instance': the thing being compared. If we are comparing hotel reviews, for
      example, each distinct hotel ID is an instance.
    """
    doc_metadata = []
    for doc in docs:
        score = None
        for md_item in doc['metadata']:
            if md_item['name'] == predicted_field:
                score = md_item['value']
            elif md_item['name'] == instance_field:
                instance = md_item['value']

        assert score is not None, doc['metadata']
        doc_metadata.append({
            'score': score,
            'instance': instance
        })
    return pd.DataFrame(doc_metadata)


def weighted_t_test(values1, weights1, values2, weights2):
    """
    Student's t-test determines whether the mean of one batch of data is
    larger or smaller than the mean of another batch of data.

    We want to be able to run an unpaired t-test where some of the samples
    are more important than others -- having more impact on the mean and
    variance. In particular, we want exact matches to be more important than
    conceptual matches.

    statsmodels provides basically everything we need to do this, but it
    doesn't have a complete implementation of the unpaired t-test that takes
    weights.

    What it has, though, is a class called DescrStatsW that computes
    descriptive statistics, such as mean and variance, over weighted data.
    The t-test is defined in terms of these statistics, so once we have those,
    we can plug them into the formula for the t-test.

    For details on the formula, see:
    https://en.wikipedia.org/wiki/Student%27s_t-test#Independent_two-sample_t-test
    """
    # We also need to know the raw number of data points, without weights.
    # (Adding up the weights doesn't seem to do things right -- you could get a sum
    # less than 1, and statistics don't work with N < 1.)
    n1 = len(weights1)
    n2 = len(weights2)

    # Get the descriptive statistics for the two samples to compare
    stats1 = DescrStatsW(values1, weights=weights1, ddof=0)
    stats2 = DescrStatsW(values2, weights=weights2, ddof=0)

    # Calculate the pooled standard deviation.
    s_p = (((n1 - 1) * stats1.var + (n2 - 1) * stats2.var) / (n1 + n2 - 2)) ** 0.5

    # Calculate the difference in (weighted) means.
    mean_diff = stats1.mean - stats2.mean

    # The t_value -- a signed value representing the confidence that one mean
    # is higher or lower than the other -- is calculated from the mean difference,
    # the pooled variance, and the sample size.
    t_value = mean_diff / (s_p * (1 / n1 + 1 / n2) ** 0.5)

    # Return the mean difference and the t_value.
    return (mean_diff, t_value)


def weighted_score_drivers(scores, fuzzy_docs, term_info):
    """
    Create a DataFrame of score drivers for this set of documents.

    The inputs are:

    - scores: a Series indicating the score of each document
    - fuzzy_docs: a sparse-ish DataFrame indicating whether each document has an
      exact or conceptual match for each term
    - term_info: a DataFrame that you get from `get_term_info()`, containing the
      label and relevance score for each term

    The result is a table containing:

    - The term labels
    - An 'impact' value for each term: how much higher or lower in score are
      documents that match this term, compared to the average?
    - 'count': the number of exact matches
    - 'weight': the effective number of matches, including conceptual
    - 'relevance': the term's global relevance in this project
    - 't_value': a positive or negative number that is larger in magnitude when
      we are more certain that this is a score driver
    - 'importance': a combination of many of these factors used for ranking the
      results
    """
    records = []
    for term in fuzzy_docs.columns:
        # Separate the documents that contain a match for the term
        term_present_selector = fuzzy_docs[term] > 0.

        # The ones that match the term should be weighted, so that exact matches
        # cont for more than conceptual matches
        term_present_weights = fuzzy_docs[term_present_selector][term]

        # If a term has weight of 1 or more in a document, it must have started
        # as an exact match. Keep track of this because we want the number of
        # exact matches as a column in the table.
        term_count = (fuzzy_docs[term] >= 1).sum()

        # We can calculate a score driver value as long as we have at least 2
        # positive examples.
        if term_present_selector.sum() > 1:
            term_present_scores = scores[term_present_selector]
            term_present_weights = fuzzy_docs[term_present_selector][term]
            constant_weights = np.ones(len(scores))

            # Run a t-test comparing (weighted) documents that contain the term
            # to (unweighted) documents in general
            mean_diff, t_value = weighted_t_test(
                term_present_scores, term_present_weights,
                scores, constant_weights
            )

            # Calculate the row of the table for this term
            term_record = term_info.loc[term]
            records.append({
                'term': term,
                'label': term_record['label'],
                'impact': mean_diff,
                'count': term_count,
                'weight': term_present_weights.sum(),
                'relevance': term_record['relevance'],
                't_value': t_value,
            })

    dataframe = pd.DataFrame(records).set_index('term')

    # Calculate the 'importance' of a score driver, helping rank the terms in a
    # way that emphasizes useful score drivers that we have information about.
    #
    # The 'importance' is the geometric mean of the score impact, its confidence
    # (t) value, the term's overall relevance in the project, and the total
    # weight of the term in this subset of the data.
    #
    # The impact and t-value can both be negative, but they always have the same
    # sign, causing their contribution to the importance value to be positive.
    #
    # This value is on an arbitrary scale, so we rescale it to a maximum of 1.
    importance = (
        dataframe['impact'] * dataframe['t_value'] * dataframe['relevance'] * dataframe['weight']
    ) ** (1 / 4)
    dataframe['importance'] = importance / importance.max()
    return dataframe[['label', 'impact', 'count', 'weight', 'relevance', 't_value', 'importance']]


def tab_separated(frame):
    return frame.to_csv(sep='\t', float_format='%.3f')


def run():
    """
    Run a demo of score drivers. The documents are hotel reviews, the scores
    are the review scores on a 0-100 scale, and the instances are the different
    hotels being reviewed.
    
    For each hotel, we show its top score drivers, then output a TSV file of all
    of the score drivers.
    """
    root_client = LuminosoClient.connect()
    client = root_client.client_for_path('/projects/prq5w34f')

    print('Getting terms')
    term_info_frame, term_vec_frame = get_term_data(client)
    print('Getting documents')
    docs = download_docs(client)
    doc_metadata = get_doc_metadata(docs, instance_field='hotel_id', predicted_field='rating')
    print('Finding conceptually related terms')
    fuzzy_doc_frame = get_fuzzy_doc_frame(docs, term_info_frame, term_vec_frame)

    print('Calculating global score drivers')
    # Get the score drivers that aren't specific to an instance. We're adding a
    # column called 'instance' to indicate which instance the drivers are for,
    # so for these we'll set it to '__all__'.
    global_drivers = weighted_score_drivers(doc_metadata['score'], fuzzy_doc_frame, term_info_frame)
    global_drivers['instance'] = '__all__'

    # Show the top 200 score drivers so we have something to look at while this runs
    top_global_drivers = global_drivers.sort_values('importance')[-200:]

    print(top_global_drivers)

    # Find out what instances exist in the data, so we can iterate over them
    instances = sorted(set(doc_metadata['instance']))

    print('Calculating instance score drivers')
    all_instance_drivers = [global_drivers]
    for instance in instances:
        # Filter the documents and scores to just this instance
        selector = doc_metadata['instance'] == instance
        instance_frame = fuzzy_doc_frame[selector]
        instance_scores = doc_metadata[selector]['score']

        # Calculate the score drivers and add the 'instance' column
        instance_drivers = weighted_score_drivers(instance_scores, instance_frame, term_info_frame)
        instance_drivers['instance'] = instance
        all_instance_drivers.append(instance_drivers)

        # Show the top 50 instance score drivers
        top_instance_drivers = instance_drivers.sort_values('importance')[-50:]
        print(top_instance_drivers)
    
    # Write a TSV file of the score driver results for every term and every instance
    result = pd.concat(all_instance_drivers, axis=0)
    result_csv = tab_separated(result)
    
    print("Writing score-drivers.tsv")
    with open('score-drivers.tsv', 'w', encoding='utf-8') as out:
        print(result_csv, file=out)


if __name__ == '__main__':
    run()
