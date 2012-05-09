import re

import word_category_counter
from collections import Counter
from mpqa import mpqa
from parsing import tokenizer

FEATURES = ['basic_lengths', 'bigrams', 'initialisms',
            'gen_dep', 'LIWC', 'liwc_dep', 'opin_dep',
            'pos_dep','repeated_punct', 'unigrams']

stemmer = None
def get_ngrams(feature_vector, tokens, prefix=None, n=1, add_null_tokens=False, binary_output=False, stem=False, use_lowercase=True):
    global stemmer
    if prefix == None:
        prefix = __get_measure(n)
        if stem:
            prefix='stem'+prefix
    if use_lowercase:
        tokens=[token.lower() for token in tokens]
    if stem:
        if stemmer == None:
            import nltk.stem
            stemmer = nltk.stem.PorterStemmer()
        tokens = [stemmer.stem(word) for word in tokens]

    if n==1:
        n_grams = tokens
    else:
        n_grams = list()
        if add_null_tokens:
            unigrams = ['-nil-' for i in range(n-1)]+tokens+['-nil-' for i in range(n-1)]
        else:
            unigrams=tokens
        for i in range(len(unigrams)-n+1):
            n_grams.append(' '.join(unigrams[i:i+n]))
    word_counts = Counter(n_grams)
    total_words = len(n_grams)
    for word, count in word_counts.items():
        if binary_output:
            feature_vector[prefix+':'+word]=True
        else:
            feature_vector[prefix+':'+word]=count/float(total_words)

def get_initialisms(feature_vector, tokens, min_length=None, max_length=2, use_lowercase=False, finalism=False):
    if min_length is None:
        min_length=max_length
    for length in range(min_length,max_length+1):
        measure = __get_measure(length)
        if finalism:
            ngram = ' '.join(tokens[-length:])
        else:
            ngram = ' '.join(tokens[:length])
        if use_lowercase:
            ngram = ngram.lower()
        if finalism:
            feature_vector['finalism:'+measure+'_'+ngram]=True
        else:
            feature_vector['initialism:'+measure+'_'+ngram]=True

def __get_measure(length):
    if length == 1:
        return 'unigram'
    elif length == 2:
        return 'bigram'
    elif length == 3:
        return 'trigram'
    return str(length)+'_gram'



def get_basic_lengths(feature_vector, text, sentences, words):
    feature_vector['length:num_characters']=len(text)
    if len(words) != 0: feature_vector['length:ave_word']=len(''.join(words))/float(len(words))
    if len(sentences) != 0: feature_vector['length:ave_sentence']=sum([len(sent) for sent in sentences])/float(len(sentences))
    feature_vector['length:num_sentence']=len(sentences)
    feature_vector['length:num_words']=len(words)

repeated_punct_re = re.compile('[!?][!?]+')
sentence_ending_punct_re = re.compile('[.?!]+')
def get_repeated_punct(feature_vector, text):
    #TODO: room for improvement here
    total_punct_count = float(len(sentence_ending_punct_re.findall(text)))
    for punct in repeated_punct_re.findall(text):
        if '!' in punct and '?' in punct:
            if 'repeated_punct:?!' not in feature_vector:
                feature_vector['repeated_punct:?!']=0
            feature_vector['repeated_punct:?!']+=1
        elif '!' in punct:
            if 'repeated_punct:!!' not in feature_vector:
                feature_vector['repeated_punct:!!']=0
            feature_vector['repeated_punct:!!']+=1
        elif '?' in punct:
            if 'repeated_punct:??' not in feature_vector:
                feature_vector['repeated_punct:??']=0
            feature_vector['repeated_punct:??']+=1
    for feature in ['repeated_punct:??', 'repeated_punct:?!', 'repeated_punct:!!']:
        if feature in feature_vector:
            feature_vector[feature]/= total_punct_count

def get_quoted_terms(feature_vector, text, max_token_count=3):
    """regexes weren't cutting it
    you're invited to improve/modify
    can't handle quotes within quotes

    """
    max_token_count = max_token_count+max_token_count-1+1#adding spaces & a quote
    tokens = tokenizer.tokenize(text, leave_whitespace=True)
    quote_start = -1
    quote_token = None
    for index in range(len(tokens)):
        token = tokens[index]
        if token=='"' or token =='\'':
            if quote_start == -1:
                quote_start = index
                quote_token=token
            else:
                if token==quote_token:
                    if index-quote_start < max_token_count:
                        term = ''.join(tokens[quote_start:index+1])
                        feature_vector['quoted_term:'+term]=True
                    quote_start=-1

def get_LIWC(feature_vector, text, normalize_to_percents=True):
    """@summary: Requires word_category_counter.set_default_dictionary(LIWC_dictionary_filename)"""
    scores = word_category_counter.score_text(text)
    if normalize_to_percents:
        scores = word_category_counter.normalize(scores)
    for category, score in scores.items():
        feature_vector['LIWC:'+category]=score

def get_dependency_features(feature_vector, dependency_list, generalization=None):
    """@param generalization: the generalization to apply, if any.
        Appropriate values are: None, 'pos', 'opinion', 'liwc', 'neg_dist_opinion'

    """

    for i in range(len(dependency_list)):
        dependency_list[i]['governor_index'] = i


    negations = dict([(dep['governor_index'],dep) for dep in dependency_list if dep['relation']=='neg'])

    #Could stand to be reworked
    for dep in dependency_list:
        relation = dep['relation'].lower()
        governor = dep['governor'].lower()
        dependent = dep['dependent'].lower()

        if generalization==None:
            feature_vector['dependency:'+relation+'('+governor+', '+dependent+')']=True

        elif generalization=='pos':
            feature_vector['dep_'+generalization+'_generalized:'+relation+'('+mpqa.convert_pos(dep['governor_pos'])+', '+dependent+')']=True

        elif generalization=='opinion':
            gov_polarity = mpqa.lookup(governor, dep['governor_pos'])
            if gov_polarity != None:
                feature_vector['dep_'+generalization+'_generalized:'+relation+'('+gov_polarity['polarity']+', '+dependent+')']=True
            dep_polarity = mpqa.lookup(dependent, dep['dependent_pos'])
            if dep_polarity != None:
                feature_vector['dep_'+generalization+'_generalized:'+relation+'('+governor+', '+dep_polarity['polarity']+')']=True

        elif generalization=='neg_dist_opinion':
            for element in ['dependent','governor']:
                word = dep[element].lower()
                polarity_dict = mpqa.lookup(word, dep[element+'_pos'])
                if polarity_dict is not None:
                    polarity = polarity_dict['polarity']
                    element_index = dep[element+'_index']
                    if element_index in negations and relation!='neg':
                        if polarity=='negative':
                            polarity = 'positive'
                        elif polarity=='positive':
                            polarity = 'negative'
                        else: continue #TODO: This forces only flipped polarity deps to be included... is this desired?
                    if element == 'dependent':
                        feature_vector['dep_'+generalization+'_generalized:'+relation+'('+governor+', '+polarity+')']=True
                    else:
                        feature_vector['dep_'+generalization+'_generalized:'+relation+'('+polarity+', '+dependent+')']=True

        elif generalization=='liwc':
            gov_categories = word_category_counter.score_word(governor).keys()
            gov_categories.append(governor)
            dep_categories = word_category_counter.score_word(dependent).keys()
            dep_categories.append(dependent)
            for gov_category in gov_categories:
                for dep_category in dep_categories:
                    if gov_category==governor and dep_category==dependent: continue #avoids the no generalization case
                    feature_vector['dep_'+generalization+'_generalized:'+relation+'('+gov_category+', '+dep_category+')']=True

def get_features_by_type(feature_vector, features=None, text_obj=None, dependency_list=None):
    features = FEATURES if features == None else features

    for feature in features:
        if feature.startswith('bas'):
            get_basic_lengths(feature_vector, text_obj.text, text_obj.sentences, text_obj.tokens)
        elif feature.startswith('bi'):
            get_ngrams(feature_vector, text_obj.tokens, n=2)
        elif feature.startswith('init'):
            get_initialisms(feature_vector, text_obj.tokens)
        elif feature.startswith('gen'):
            if dependency_list is not None:
                get_dependency_features(feature_vector, dependency_list)
        elif feature == 'LIWC':
            get_LIWC(feature_vector, text_obj.text)
        elif feature == 'liwc_dep':
            if dependency_list is not None:
                get_dependency_features(feature_vector, dependency_list, 'liwc')
        elif feature.startswith('opin'):
            if dependency_list is not None:
                get_dependency_features(feature_vector, dependency_list, generalization='opinion')
        elif feature.startswith('pos_dep'):
            if dependency_list is not None:
                get_dependency_features(feature_vector, dependency_list, generalization='pos')
        elif feature.startswith('repeated_punct'):
            get_repeated_punct(feature_vector, text_obj.text)
        elif feature.startswith('uni'):
            get_ngrams(feature_vector, text_obj.tokens)
