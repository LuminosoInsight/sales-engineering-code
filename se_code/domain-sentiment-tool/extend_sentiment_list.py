import json

import click
import numpy as np
from lumi_science.sentiment import SentimentScorer
from lumi_science.text_readers import get_reader
from luminoso_api import V5LuminosoClient as LuminosoClient
from sentiment_topics import SentimentTopics


def create_domain_sentiment_list(sentiment_topics, output_filename, n_results):
    """
    Create and save a list of at most n domain sentiment terms. The terms are sorted by
    total_doc_count to surface the terms that may have the most influence on the accuracy of the
    classification.

    Additionally, identify potential conflicts of the domain sentiment list and the general
    sentiment list.
        - check if any term on the domain sentiment list was assigned a positive score even
        though we identified it as a negative term (and vice versa)
        - check if any term on the general sentiment list has a positive score even though we
        identified it as a negative term (and vice versa).
        - check which scores for on the general sentiment list changed the most when assigned a
        new score
    """
    pos_terms = sentiment_topics.get_domain_sentiment_terms('pos')
    neg_terms = sentiment_topics.get_domain_sentiment_terms('neg')
    pos_set = set(term['term'] for term in pos_terms)
    neg_set = set(term['term'] for term in neg_terms)

    sent_terms = pos_terms + neg_terms
    client = sentiment_topics.client
    language = client.get()['language']
    sentiment_scorer = SentimentScorer(language=language)

    # ----- Create and save a list of domain sentiment terms -----

    # Assign a score to each term
    sent_terms = assign_sentiment_scores_n_sim(sent_terms, client, sentiment_scorer)

    # Divide into novel terms and known terms
    novel_terms, known_terms = get_novel_known_terms(sent_terms, sentiment_scorer)

    # Sort novel terms by their document count for prioritizing
    sorted_sent_terms = sort_terms_by_doc_count(novel_terms)[:n_results]

    save_terms(output_filename, sorted_sent_terms, language)

    # ----- Identify conflicts with the general sentiment list -----

    # Cross-reference new and known sent scores with pos and neg terms
    conflicts_novel_terms = check_against_pos_neg_terms(sorted_sent_terms, pos_set, neg_set)
    conflicts_known_terms = check_against_pos_neg_terms(known_terms, pos_set, neg_set)

    # Cross-reference known sentiment scores and new sentiment scores
    most_changed_known = get_most_changed(known_terms)


def assign_sentiment_scores_n_sim(sent_terms, client, sentiment_scorer):
    """
    Assign a new sentiment score to a term using the scores of its neighbors. First, query the
    project for the most similar terms to that term, then take the top 5 most similar terms
    that are present on the general sentiment list. The new score is an average of the scores of
    these 5 most similar sentiment terms, weighted by their similarity to the term in question.

    If no sentiment terms are present among the top 200 most similar terms, assign a classifier
    score using a linear function (5 * new-sentiment-score, which is the similarity of a term to
    a sentiment axis).
    """
    for term in sent_terms:
        sim_terms = client.get('concepts', concept_selector={'type': 'related',
                                                             'search_concept': {'texts': [term['name']]},
                                                             'limit': 200})['result']
        similar_terms = []
        for result in sim_terms:
            sentiment_score = sentiment_scorer.term_sentiment(result['exact_term_ids'][0])
            if sentiment_score != 0:
                similar_terms.append((sentiment_score, result['match_score'])
        similar_terms = similar_terms[:5]

        if similar_terms:
            sent_scores, weights = zip(*similar_terms)
            score = np.average(sent_scores, weights=weights)
            term['classifier-score'] = round(score, 2)
        else:
            term['classifier-score'] = term['new-sentiment-score'] * 5
            if term['sentiment-label'] == 'neg':
                term['classifier-score'] *= -1
    return sent_terms


def get_novel_known_terms(sent_terms, sentiment_scorer):
    """
    Separate sent_terms into the terms that have scores assigned to them by the general sentiment
    list (known_sentiment_terms) and those that don't (novel terms).
    """
    novel_sentiment_terms = []
    known_sentiment_terms = []
    for term in sent_terms:
        sent_score = sentiment_scorer.term_sentiment(term['term'])
        term['orig-sent-score'] = sent_score
        if sent_score == 0:
            novel_sentiment_terms.append(term)
        else:
            known_sentiment_terms.append(term)
    return novel_sentiment_terms, known_sentiment_terms


def sort_terms_by_doc_count(sent_terms):
    """
    Sort terms by total_doc_count, which helps prioritize the documents with the biggest impact
    on the sentiment classification task.
    """
    return sorted(sent_terms, key=lambda x: x['total_doc_count'], reverse=True)


def check_against_pos_neg_terms(sent_terms, pos_set, neg_set):
    """
    Identify the terms which:
        - have a positive score assigned to them even though we identified them as negative terms
        - have a negative score assigned to them even though we identified them as positive terms
    """
    conflicts = []
    for term in sent_terms:
        if term['classifier-score'] > 0 and term['term'] in neg_set:
            conflicts.append(term)
        if term['classifier-score'] < 0 and term['term'] in pos_set:
            conflicts.append(term)
    return conflicts


def get_most_changed(sent_terms):
    """
    Identify the terms on the general sentiment list which scores changed the most when assigned
    a new score.
    """
    for term in sent_terms:
        if term['orig-sent-score'] != 0:
            change = abs(term['orig-sent-score'] - term['classifier-score'])
            term['change'] = change
        else:
            term['change'] = 0
    return sorted(sent_terms, key=lambda x: x['change'], reverse=True)[:50]


def save_terms(output_filename, sent_terms, language):
    reader = get_reader(language)
    with open(output_filename, 'w') as output_file:
        for term in sent_terms:
            text = reader.delangtag_skipping_untagged(term['term'])
            output_file.write('{},{}\n'.format(text, term['classifier-score']))


@click.command()
@click.argument('project_id')
@click.argument('output-file-name')
@click.option('--api', default='https://analytics.luminoso.com/api/v5/projects')
@click.option('--n-terms', default=1000)
@click.option('--n-results', default=100)
def main(project_id, output_file_name, api, n_terms, n_results):
    client = LuminosoClient.connect('{}/{}'.format(api, project_id))
    sentiment_topics = SentimentTopics(client, n_terms, n_results * 2)
    create_domain_sentiment_list(sentiment_topics, output_file_name, n_results)

if __name__ == '__main__':
    main()
