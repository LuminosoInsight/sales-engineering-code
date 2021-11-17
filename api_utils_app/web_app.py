import datetime
import redis

from flask import Flask, jsonify, render_template, request, session, url_for

from luminoso_api import LuminosoClient
from deduper_utilities import dedupe
from reddit_utilities import get_reddit_api, get_posts_from_past, get_posts_by_name, get_docs_from_comments, write_to_csv
from term_utilities import get_terms, ignore_terms, merge_terms
from se_code.copy_shared_concepts import copy_shared_concepts, delete_shared_concepts
from se_code.create_train_test_split import create_train_test
from se_code.bi_tool_export import pull_lumi_data, create_doc_table, create_doc_term_table, create_doc_subset_table, create_themes_table, create_skt_table, create_drivers_table, write_table_to_csv, create_terms_table, create_sentiment_table, create_sdot_table, get_date_field_by_name, create_drivers_with_subsets_table, parse_url
from subset_utilities import search_subsets, calc_metadata_vectors

# Implement this for login checking for each route http://flask.pocoo.org/snippets/8/

app = Flask(__name__)
app.secret_key = 'secret_key_that_we_need_to_have_to_use_sessions'
red = redis.StrictRedis()


def connect_to_client(url):
    root_url, api_url, acct, proj = parse_url(url)

    client = LuminosoClient.connect_with_username_and_password(url=api_url,
                                                               username=session['username'],
                                                               password=session['password'])
    client = client.client_for_path('projects/{}'.format(proj))
    return client


@app.route('/')
def home():
    return render_template('login.html')


@app.route('/', methods=['GET', 'POST'])
def login():
    session['username'] = request.form['username']
    session['password'] = request.form['password']
    session['apps_to_show'] = [
        ('Concept List', ('Copy Concept Lists', url_for('copy_shared_concepts_page')), 
                  ('Delete Concept Lists', url_for('delete_shared_concept_page'))),
        ('Term', ('Merge Terms', url_for('term_merge_page')),
                 ('Ignore Terms', url_for('term_ignore_page'))),
        ('Subsets', ('Conceptual Subset Search', url_for('subset_search'))),
        ('Cleaning', ('Deduper', url_for('deduper_page')),
                     ('Create Train Test Split', url_for('create_train_test_page'))),
                     # ('Boilerplate Cleaner',url_for('boilerplate_page'))),
        ('Exports', ('BI Tool Export', url_for('bi_tool_export_page'))),
        ('Connectors', ('Reddit by Time', url_for('reddit_by_time_page')),
                       ('Reddit by Name', url_for('reddit_by_name_page')))]
    try:
        LuminosoClient.connect_with_username_and_password('/projects', username=session['username'],
                                                                       password=session['password'])

        return render_template('welcome.html', urls=session['apps_to_show'])
    except Exception as e:
        print(e)
        error = 'Invalid_credentials'
        return render_template('login.html', error=error)


@app.route('/index')
def index():
    return render_template('index.html', urls=session['apps_to_show'])


@app.route('/reddit_by_time_page', methods=['GET'])
def reddit_by_time_page():
    SEARCH_TYPES = ['top', 'controversial', 'new']
    SEARCH_PERIODS = ['week', 'hour', 'day', 'month', 'year', 'all']
    return render_template('reddit_by_timeframe.html', urls=session['apps_to_show'],
                           types=SEARCH_TYPES,
                           periods=SEARCH_PERIODS)


@app.route('/reddit_by_time', methods=['POST'])
def reddit_by_time():
    fields = ['text', 'title', 'date_Post Date', 'string_Author Name', 'string_Comment Type', 'string_Thread', 'string_Reddit Post', ]
    subreddit_name = request.form['subreddit'].strip()
    start_date = request.form.get('start_date')
    start_time = request.form.get('start_time')
    start_datetime = datetime.datetime.strptime(' '.join([start_date, start_time]), '%Y-%m-%d %H:%M')
    sort_type = request.form['type']
    time_period = request.form['period']
    reddit = get_reddit_api()
    posts = get_posts_from_past(
        reddit, subreddit_name, start_datetime, sort_type, time_period
        )
    docs = get_docs_from_comments(posts, reddit)
    write_to_csv('%s docs.csv' % subreddit_name, docs, fields)
    SEARCH_TYPES = ['top', 'controversial', 'new']
    SEARCH_PERIODS = ['week', 'hour', 'day', 'month', 'year', 'all']
    return render_template('reddit_by_timeframe.html', 
                           urls=session['apps_to_show'],
                           types=SEARCH_TYPES,
                           periods=SEARCH_PERIODS)


@app.route('/reddit_by_name_page', methods=['GET'])
def reddit_by_name_page():
    return render_template('reddit_by_name.html', 
                           urls=session['apps_to_show'])


@app.route('/reddit_by_name', methods=['POST'])
def reddit_by_name():
    fields = ['text', 'title', 'date_Post Date', 'string_Author Name', 'string_Comment Type', 'string_Thread', 'string_Reddit Post', ]
    subreddit_name = request.form['subreddit'].strip()
    post_names = request.form['post_list'].strip()
    post_names = post_names.split(';')
    reddit = get_reddit_api()
    posts = get_posts_by_name(reddit, subreddit_name, post_names)
    docs = get_docs_from_comments(posts, reddit)
    write_to_csv('%s docs.csv' % subreddit_name, docs, fields)
    return render_template('reddit_by_name.html', 
                           urls=session['apps_to_show'])


@app.route('/bi_tool_export_page', methods=['GET'])
def bi_tool_export_page():
    return render_template('bi_tool_export.html', urls=session['apps_to_show'])


@app.route('/bi_tool_export', methods=['POST'])
def bi_tool_export():
    url = request.form['url'].strip()
    root_url, api_url, acct, proj = parse_url(url)
    ui_project_url = root_url + '/app/projects/' + acct + '/' + proj

    foldername = request.form['folder_name'].strip()
    concept_count = request.form['term_count'].strip()
    if concept_count == '':
        concept_count = 100
    else:
        concept_count = int(concept_count)
    skt_limit = request.form['skt_limit'].strip()
    if skt_limit == '':
        skt_limit = 20
    else:
        skt_limit = int(skt_limit)
        
    term_table = (request.form.get('terms') == 'on')
    doc_term = (request.form.get('doc_term') == 'on')
    doc_subset = (request.form.get('doc_subset') == 'on')
    themes_on = (request.form.get('themes') == 'on')
    skt_on = (request.form.get('skt') == 'on')
    drivers_on = (request.form.get('drivers') == 'on')
    driver_subsets = (request.form.get('driver_subsets') == 'on')
    driver_subset_fields = request.form['driver_subset_fields'].strip()

    sentiment = (request.form.get('sentiment') == 'on')
    topic_drive = (request.form.get('topic_drive') == 'on')
    
    sdot_on = (request.form.get('sdot') == 'on')
    sdot_end = ''
    if 'sdot_end' in request.form:
        sdot_end = request.form['sdot_end'].strip()
    sdot_iterations = 7
    if 'sdot_iterations' in request.form:
        sdot_iterations = request.form['sdot_iterations'].strip()
    sdot_range_type = ''
    if 'sdot_range_type' in request.form:
        sdot_range_type = request.form['sdot_range'].strip()
    sdot_date_field_name = ''
    if 'sdot_date_field_name' in request.form:
        sdot_date_field_name = request.form['sdot_date_field_name'].strip()
    
    score_drivers, docs, scl_match_counts, concepts, metadata, driver_fields, skt, themes = pull_lumi_data(proj, api_url, skt_limit=int(skt_limit), concept_count=int(concept_count))
    client = score_drivers.client

    doc_table, xref_table, metadata_map = create_doc_table(docs, metadata, themes, sentiment=sentiment)
    write_table_to_csv(doc_table, foldername+'doc_table.csv', calc_keys=True)
    write_table_to_csv(xref_table, foldername+'xref_table.csv')
    
    if sentiment:
        sentiment_table = create_sentiment_table(client, scl_match_counts)
        write_table_to_csv(sentiment_table, foldername+'sentiment.csv', calc_keys=True)

    if term_table:
        terms_table = create_terms_table(concepts, scl_match_counts)
        write_table_to_csv(terms_table, foldername+'terms_table.csv', calc_keys=True)
    if doc_term:
        doc_term_table = create_doc_term_table(docs, concepts, scl_match_counts)
        write_table_to_csv(doc_term_table, foldername+'doc_term_table.csv')
    
    # if doc_topic:
    #    doc_topic_table = create_doc_topic_table(docs, saved_concepts)
    #    write_table_to_csv(doc_topic_table, foldername+'doc_topic_table.csv')
        
    if doc_subset:
        doc_subset_table = create_doc_subset_table(docs, metadata_map)
        write_table_to_csv(doc_subset_table, foldername+'doc_subset_table.csv')
    
    if themes_on:
        themes_table = create_themes_table(client, themes)
        write_table_to_csv(themes_table, foldername+'themes_table.csv')

    if skt_on:
        skt_table = create_skt_table(client, skt)
        write_table_to_csv(skt_table, foldername+'skt_table.csv')
    
    if drivers_on:
        driver_table = create_drivers_table(client, driver_fields, topic_drive)
        write_table_to_csv(driver_table, foldername+'drivers_table.csv')

    if driver_subsets:
        driver_table = create_drivers_with_subsets_table(client, driver_fields, topic_drive, subset_fields=driver_subset_fields)
        write_table_to_csv(driver_table, 'subset_drivers_table.csv')
     
    if sdot_on:
        print("SDOT {},{},{},{}".format(sdot_end, sdot_iterations, sdot_range_type, sdot_date_field_name))

        if len(sdot_date_field_name)==0:
            date_field_info = score_drivers.get_first_date_field()
            if date_field_info == None:
                print("ERROR no date field in project")
                return
        else:
            date_field_info = get_date_field_by_name(sdot_date_field_name)
            if not date_field_info:
                print("ERROR: no date field name: {}".format(sdot_date_field_name))
                return

        sdot_table = create_sdot_table(client, driver_fields, date_field_info, sdot_end, int(sdot_iterations), sdot_range_type, topic_drive, root_url='', docs=docs)
        write_table_to_csv(sdot_table, foldername+'sdot_table.csv')
    
    #if trends:
    #    trends_table, trendingterms_table = create_trends_table(terms, topics, docs)
    #    write_table_to_csv(trends_table, foldername, 'trends_table.csv')
    #    write_table_to_csv(trendingterms_table, foldername, 'trendingterms_table.csv')
    
    return render_template('bi_tool_export.html', urls=session['apps_to_show'])


@app.route('/subset_search', methods=['GET','POST'])
def subset_search():
    
    global client, metadata_with_vects, field
    
    if request.method == 'POST':
        if 'url' in request.form:
            url = request.form['url'].strip()
            field = request.form['field'].strip()
            client = connect_to_client(url)

            project = client.get()['name']
            metadata_with_vects = client.get('/metadata')['result']

            if field:
                metadata_with_vects = calc_metadata_vectors(client, metadata_with_vects, field)
            else:
                field = None
                metadata_with_vects = calc_metadata_vectors(client, metadata_with_vects)
        else:
            project = client.get()['name']
            question = request.form['text']
            include_docs = False
            if request.form.get('include_docs'):
                include_docs = True

            result = search_subsets(client, [question], metadata_with_vects, field=field, sample_docs=include_docs)

            return render_template('subset_search.html',
                                   urls=session['apps_to_show'],
                                   query_info=question,
                                   results=result[question],
                                   project=project)
    else:
        project = ''
        
    return render_template('subset_search.html', urls=session['apps_to_show'], project=project)


@app.route('/concept_list_utils')
def concept_list_utils():
    return render_template('concept_list_utils.html', urls=session['apps_to_show'])


@app.route('/concept_list_utils/copy', methods=['POST'])
def concept_list_utils_copy():
    # NOTE: Should add a checkbox for if the existing concept lists should be deleted first
    url = request.form['url'].strip()
    to_delete = (request.form.get('delete') == 'on')
    # parse the from url
    froot_url, fapi_url, faccount_id, fproject_id = parse_url(url)

    # api_url, from_proj = parse_url(url)
    # client = connect_to_client(url)
    # client = client.client_for_path('/')
    # client = client.client_for_path('projects')
    dests = [url.strip() for url in request.form['dest_urls'].split(',')]

    from_client = LuminosoClient.connect(url='%s/projects/%s' % (fapi_url, fproject_id))

    for dest_proj in dests:
        # parse the to url
        troot_url, tapi_url, taccount_id, tproject_id = parse_url(dest_proj)
        to_client = LuminosoClient.connect(url='%s/projects/%s' % (tapi_url, tproject_id))

        # api_url, to_proj = parse_url(dest_proj)
        if to_delete:
            delete_shared_concepts(to_client)
        copy_shared_concepts(
            from_client,
            to_client
        )
    
    # NOTE: ADD A FLASH CONFIRMATION MESSAGE HERE
    return render_template('copy_shared_concepts.html', urls=session['apps_to_show'])


@app.route('/concept_list_delete', methods=['POST'])
def concept_list_delete():
    dests = [url.strip() for url in request.form['urls'].split(',')]

    for dest_proj in dests:
        client = connect_to_client(dest_proj)
        delete_shared_concepts(client)
    
    # NOTE: ADD A FLASH CONFIRMATION MESSAGE HERE
    return render_template('delete_shared_concept_lists.html', urls=session['apps_to_show'])


@app.route('/term_utils')
def term_utils():
    return render_template('term_utils.html', urls=session['apps_to_show'])


@app.route('/term_merge_page')
def term_merge_page():
    return render_template('term_merge.html', urls=session['apps_to_show'])


@app.route('/term_ignore')
def term_ignore_page():
    return render_template('term_ignore.html', urls=session['apps_to_show'])


@app.route('/copy_shared_concepts_page')
def copy_shared_concepts_page():
    return render_template('copy_shared_concepts.html', urls=session['apps_to_show'])


@app.route('/delete_shared_concept_page')
def delete_shared_concept_page():
    return render_template('delete_shared_concept_lists.html', urls=session['apps_to_show'])


@app.route('/term_utils/search', methods=['GET', 'POST'])
def term_utils_search():
    url = request.args.get('url', 0, type=str).strip()
    cli = connect_to_client(url)
    return jsonify(get_terms(cli))


@app.route('/term_utils/merge')
def term_utils_merge():
    url = request.args.get('url', 0, type=str)
    terms = eval(request.args.get('terms', 0, type=str))
    cli = connect_to_client(url)
    return jsonify(merge_terms(cli, terms))


@app.route('/term_utils/ignore')
def term_utils_ignore():
    url = request.args.get('url', 0, type=str).strip()
    terms = eval(request.args.get('terms', 0, type=str))
    cli = connect_to_client(url)
    return jsonify(ignore_terms(cli, terms))


@app.route('/deduper_page')
def deduper_page():
    return render_template('dedupe.html', urls=session['apps_to_show'])


@app.route('/dedupe')
def dedupe_util():
    url = request.args.get('url', 0, type=str)
    api_url, proj = parse_url(url)
    copy = (request.args.get('copy') == 'true')
    recalc = (request.args.get('recalc') == 'true')
    reconcile = request.args.get('reconcile')
    cli = LuminosoClient.connect('{}/projects/{}'.format(api_url, proj))

    return jsonify(dedupe(cli, recalc=recalc, reconcile_func=reconcile, copy=copy))


@app.route('/create_train_test_page', methods=['GET'])
def create_train_test_page():
    return render_template('create_train_test.html', urls=session['apps_to_show'])


@app.route('/create_train_test_util', methods=['POST'])
def create_train_test_util():
    read_file = request.form['read_file'].strip()
    train_write_file = request.form['output_train'].strip()
    request.args.get('output_train', 0, type=str)
    test_write_file = request.form['output_test'].strip()
    request.args.get('output_test', 0, type=str)
    encoding = request.form['encoding'].strip()
    split = float(request.form['split_size'].strip())
    text_index = request.form['text_index'].strip()
    label_index = request.form['label_index'].strip()
    create_train_test(read_file, train_write_file, test_write_file, split, encoding, text_index, label_index)
    
    return render_template('create_train_test.html', urls=session['apps_to_show'])


###
# BEGIN Boilerplate code, some of which will be moved to separate file
# note: may need to use the session['username'] to uniquely identify
#       the pubsub messages for each user, if red is shared for all sessions.
###
def event_stream():
    pubsub = red.pubsub()
    pubsub.subscribe('boilerplate')
    for message in pubsub.listen():
        if message['type'] == 'subscribe':
            pass
        else:
            yield str('data: %s\n\n' % message['data'])

#@app.route('/boilerplate_page')
#def boilerplate_page():
#    return render_template('boilerplate.html', urls=session['apps_to_show'])

#@app.route('/boilerplate_stream')
#def boilerplate_stream():
#    return Response(event_stream(), content_type='text/event-stream')

#@app.route('/boilerplate_run')
#def bp_run():
#    thresh = request.args.get('thresh', 6, type=int)
#    window_size = request.args.get('window_size', 7,  type=int)
#    use_gaps = request.args.get('use_gaps', "on", type=str)
#    sample_docs = request.args.get('sample_docs', 10, type=int)
#    url = request.args.get('url', type=str)
#    api_url, acct, proj = parse_url(url)
#    bp = BPDetector()
#    bp.threshold = thresh
#    bp.window_size = window_size
#    bp.use_gaps = use_gaps == "on"
#    output_fp, name, acct = bp.run(api_url=api_url, acct=acct, proj=proj,
#            user=session['username'],
#            passwd=session['password'],
#            sample_docs=sample_docs,
#            redis=red,
#            train=True,
#            tokens_to_scan=1000000,
#            verbose=True)
#    return jsonify({'output': output_fp, 'name': name, 'acct': acct})

#@app.route('/boilerplate_new_proj')
#def bp_create_proj():
#    filepath = request.args.get('docs_path', type=str)
#    name = request.args.get('name', type=str)
#    acct = request.args.get('acct', type=str)
#    recalc = (request.args.get('recalc') == 'true')
#    acct, proj = boilerplate_create_proj(filepath, name, acct, recalc,
#                            username = session['username'],
#                            password = session['password'])
#    url = 'https://analytics.luminoso.com/explore.html?account='+acct+'&projectId='+proj
#    return jsonify({'proj_url': url})



###
# END Boilerplate code
###

if __name__ == '__main__':
    app.run()
