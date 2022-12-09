import argparse

from luminoso_api import V5LuminosoClient as LuminosoClient

def parse_url(url):
    api_root = url.strip('/ ').split('/app')[0]
    proj_id = url.strip('/').split('/')[6]
    return api_root + '/api/v5/projects/' + proj_id

def get_concepts(client):
    cs = {"type": "sentiment_suggested", "limit": 500}
    sentiment_concepts = client.get("/concepts/sentiment", concept_selector=cs)['match_counts']
    return sentiment_concepts

def get_sentiment_value(c):
    negative = c['sentiment_share']['negative']
    neutral = c['sentiment_share']['neutral']
    positive = c['sentiment_share']['positive']
    negative = -negative + min(negative,(neutral*0.5))
    positive = positive - min(positive,(neutral*0.5))
    return negative + positive

def code_for_net_sentiment(sentiment_concepts):
    net_sentiment_colors = ["#ff0000","#ff7f7f","#f0f0f0","#90ee90","#008000"]

    for c in sentiment_concepts:
        sv = get_sentiment_value(c)
        sentiment_color_index = round((sv+1) * (4/2))
        c['net_sentiment_color'] = net_sentiment_colors[sentiment_color_index]

def calc_net_sentiment(c):
    return c['sentiment_share']['positive'] - c['sentiment_share']['negative']

def calc_sentiment_polarization(c):
    return min(c['sentiment_share']['negative'], c['sentiment_share']['positive'])*2

def code_for_sentiment_polarization(sentiment_concepts):
    # two lists, separated by low and high, the red/neg blue/neu or green/pos
    high_polar_colors = ["#ff0000", "#8000ff", "#0000ff"]
    low_polar_colors = ["#ff8000", "#808080", "#00cc22"]
    polar_colors = [low_polar_colors, high_polar_colors]

    for c in sentiment_concepts:
        ns = calc_net_sentiment(c)
        if ns<-0.25:
            ns_index = 0
        elif ns<=0.25:
            ns_index = 1
        else:
            ns_index = 2

        p = calc_sentiment_polarization(c)
        if p<0.60:
            p_index = 0   # low polarization
        else:
            p_index = 1   # high polarization
        c['polar_sentiment_color'] = polar_colors[p_index][ns_index]

def get_color_concept_list(sentiment_concepts, color_field):
    cl = []
    for c in sentiment_concepts:
        cl.append({
            'name': c['name'],
            'texts': c['texts'],
            'color': c[color_field]
        })
    return cl

def main():
    parser = argparse.ArgumentParser(
        description='Creating two shared concept lists based on the sentiment calculated for each concept and the color associated with that sentiment.'
    )
    parser.add_argument(
        'project_url',
        help="The URL of the source project to create the shared concept lists in and get the concepts from"
        )
    args = parser.parse_args()

    endpoint = parse_url(args.project_url)
    client = LuminosoClient.connect(endpoint,
                                user_agent_suffix='se_code:sentiment_colorization')

    concepts = get_concepts(client)

    code_for_net_sentiment(concepts)
    code_for_sentiment_polarization(concepts)

    concept_list = get_color_concept_list(concepts,  'net_sentiment_color')
    client.post("/concept_lists",
                name="Net Sentiment Colors",
                concepts=concept_list,
                overwrite=True
                )

    concept_list = get_color_concept_list(concepts,  'polar_sentiment_color')
    client.post("/concept_lists",
                name="Sentiment Polarization Colors",
                concepts=concept_list,
                overwrite=True
                )

print("Done. Two new shared concept lists have been created.")

if __name__ == "__main__":
    main()
