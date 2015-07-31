@app.route('/')
def index():
    return render_template('qual_simple.html')

@app.route('/_step1')
def step1():
    """From the given token, get name and id of surveys"""
    token = request.args.get('token', 0, type=str)
    info = get_name_id(token) #returns a dictionary
    return jsonify(**info)

@app.route('/_step2')
def step2():
    """Given sid and token, return the question info"""
    sid = request.args.get('sid', 0, type=str)
    token = request.args.get('token', 0, type=str)
    return jsonify(get_question_descriptions(sid, token))

@app.route('/_step3')
def step3():
    """Given sid, token, text question ID, subset question IDs,
       a luminoso account and token, create a proj in Analytics"""
    sid = request.args.get('sid', 0, type=str)
    token = request.args.get('token', 0, type=str)
    text_qs = eval(request.args.get('text_qs', 0, type=str))
    subset_qs = eval(request.args.get('subset_qs', 0, type=str))
    title = request.args.get('title', 0, type=str)
    proj_url = build_analytics_project(sid, token, text_qs, subset_qs,
                                       lumi_account, lumi_token, title+" (Imported from Qualtrics)")
    return jsonify({"url":proj_url})

if __name__ == '__main__':
    app.run(debug=True)