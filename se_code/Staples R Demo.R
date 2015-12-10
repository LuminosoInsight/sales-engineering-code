library(httr)
library(jsonlite)

# Get this in the main page under the dropdown in upper-right
# account dropdown -> "User Settings" -> "API Tokens"
TOKEN = 'UyCWnoF7z5fVk5TYVMcIKjU2RAMBfUJs'

# Simple function to make get and post requests to the API
api_call = function(type, path, params = list()) {
  if (type == "get") {
    resp = GET(paste(c('https://api.luminoso.com/v4',path), collapse=''),
               add_headers(Authorization = paste('Token', TOKEN)),
               query = if (length(params)) params else '')
    return(content(resp))
  }
  else if (type == "post") {
    resp = POST(paste(c('https://api.luminoso.com/v4',path), collapse=''),
                add_headers("Authorization" = paste('Token', TOKEN)),
                body = params)
    return(content(resp))
  }
  else if (type == "upload") {
    resp = POST(paste(c('https://api.luminoso.com/v4',path), collapse=''),
                add_headers("Authorization" = paste('Token', TOKEN),
                            "Content-Type" = "application/json"),
                body = params)
    return(content(resp))
  }
  else {
    stop("You must specify 'get' or 'post' or 'upload' as the request type")
  }
}

# This function is used to unpack vectors returned by the API
unpack64 = function(string) {
  chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_'
  chars_to_indices = structure(lapply((1:64), function(i) i),
                               names=sapply((1:64), function(i) substr(chars, i, i)))
  SIGN_BIT = 2**17
  ROUND_MARGIN = SIGN_BIT / (SIGN_BIT-0.5)
  increment = 2**(chars_to_indices[substr(string,1,1)][[1]] - 40)
  str_tail = substr(string, 2, nchar(string))
  numbers = unlist(lapply(strsplit(str_tail, list())[[1]],
                          function(x) chars_to_indices[x][[1]]-1), use.names = FALSE)
  highplace = numbers[seq(1,length(numbers), 3)]
  midplace = numbers[seq(2,length(numbers), 3)]
  lowplace = numbers[seq(3,length(numbers), 3)]
  values = 4096 * highplace + 64 * midplace + lowplace
  signs = values >= SIGN_BIT
  values = values - 2 * signs * SIGN_BIT
  return(round((values * increment)/2, 6))
}

#####################
# Basic quickstart
#####################

# get a list of projects in your account
projs = api_call('get', '/projects/v43y563b/')

# the number of projects in the account
length(projs$result)

# get project_id for the fourth project
proj_id = projs$result[[4]]['project_id']

# get the topics from the above project
topics = api_call('get', paste(c('/projects/v43y563b/', proj_id, '/topics', collapse='')))
names(topics) 

# view the info on the first and second topics returned
topic1 = topics$result[[1]]
topic2 = topics$result[[2]]

# unpack the vector representation of the above topic
vec1 = unpack64(topic1$vector)
vec2 = unpack64(topic2$vector)

# get the association score (dot product) between topic1 and topic2
vec1 %*% vec2

# create a new topic, including parameters in the POST body
api_call('post',
         paste(c('/projects/v43y563b/', proj_id, '/topics', collapse='')),
         params=list(text="test", name="test topic"))

# perform a document search on some text, "shipping" in this example
shipping_docs = api_call('get',
                       paste(c('/projects/v43y563b/',proj_id,'/docs/search/'), collapse=''),
                       params=list(text='shipping'))

# print out the text of the first document returned
shipping_docs$result$search_results[[1]][[1]]$document$text

#####################
# Create a new project and upload documents
#####################

# create the project
new_proj = api_call('post',
                    '/projects/temp/',
                    params=list(name='Staples Demo Project'))

new_proj_path = new_proj$result$path

# convert to JSON for uploading to the new project
sample_docs_json =
  '[
    {"text":"this is the text of the first document.",
     "subsets":["sample_subset1", "sample_subset2"],
     "title":"document1 title"},
    {"text":"a man a plan a canal panama",
     "subsets":["palindromes"],
     "title":"Palindrome"},
    {"text":"I sing the body electric, The armies of those I love engirth me and I engirth them, They will not let me off till I go with them, respond to them, And discorrupt them, and charge them full with the charge of the soul.",
     "subsets":["poems", "whitman"],
     "title":"I sing the body electric"},
    {"text":"In Xanadu did Kubla Khan A stately pleasure-dome decree: Where Alph, the sacred river, ran Through caverns measureless to man Down to a sunless sea. So twice five miles of fertile ground With walls and towers were girdled round; And there were gardens bright with sinuous rills, Where blossomed many an incense-bearing tree; And here were forests ancient as the hills, Enfolding sunny spots of greenery.",
     "subsets":["poems", "coleridge"],
     "title":"kubla khan"}
  ]'

# this just cleans up the string above, since R messes it up otherwise
# in reality you'll likely be loading from a csv to a data frame, and
# then using a json library to convert that data frame to JSON 
sample_docs_json = toJSON(fromJSON(sample_docs_json))

# so that we have enough conent, let's re-upload those same docs 300 times
upload_msgs = replicate(300, api_call('upload',
                        paste(c(new_proj_path, 'docs/'), collapse=''),
                        params=sample_docs_json))

# now we must tell the project to calculate to make it usable
api_call('post',
         paste(c(new_proj_path, 'docs/recalculate/'), collapse=''),
         params=list(language='en', max_ngram_length=7))


#####################
# Get top 5 terms and create topics for them,
# then get topic-topic association scores
#####################

# get the top 5 terms
terms = api_call('get',
                 paste(c('/projects/v43y563b/', proj_id, '/terms', collapse='')),
                 params=list(limit=5))

# get the surface form text for each term
term_text = sapply(terms$result, function(x) x$text) 

# for each term, create a new topic defined by the term text
for (text in term_text) {
  api_call('post',
           paste(c('/projects/v43y563b/', proj_id, '/topics', collapse='')),
           params=list(text=text, name=text))
}

# get the topic-topic association scores from the project in JSON format
top_top_assoc = api_call('get',
                         paste(c('/projects/v43y563b/', proj_id, '/topics/correlation/', collapse='')),
                         params=list(format='json'))

# download a CSV file of the topic-topic association scores
download.file("https://api.luminoso.com/v4/projects/v43y563b/3x7cp/topics/correlation/",
              '/Users/tobrien/Downloads/Staples_topic_associations.csv',
              'curl',
              extra=paste('-H', '"Authorization:', 'Token', TOKEN, '"'))
