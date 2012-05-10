from collections import defaultdict, Counter

from jsondataset import  JSONDataset

from nlp.text_obj import TextObj
from pprint import pprint
from corelex import lookup

def match(mention, dependencies, stem_pos=False, corelex=False):
    mention = mention.split()
    results = list()
    events = list()
    dependent_event = ['dependent', 'governor_pos']
    governor_event = ['dependent_pos', 'governor']
    for dependency in dependencies:
        if dependency['dependent']==mention[-1]:
            if (dependency['governor_pos'].startswith('JJ')
                or dependency['governor_pos'].startswith('VB')):
                results.append(dependency)
                if stem_pos:
                    dependency['governor_pos'] = dependency['governor_pos'][:2]
                if corelex:
                    dependent = lookup(dependency['dependent'])
                    if dependent != '':
                        dependency['dependent'] = dependent
                event = "{} {} _".format(*map(lambda key: dependency[key], dependent_event))
                events.append(event)
            continue
        if dependency['governor']==mention[-1]:
            if dependency['dependent'] not in mention:
                if (dependency['dependent_pos'].startswith('JJ')
                    or dependency['dependent_pos'].startswith('VB')):
                    results.append(dependency)
                    if stem_pos:
                        dependency['dependent_pos'] = dependency['dependent_pos'][:2]
                    if corelex:
                        governor = lookup(dependency['governor'])
                        if governor != '':
                            dependency['governor'] = governor
                    event = "_ {} {}".format(*map(lambda key: dependency[key], governor_event))
                    events.append(event)
    return results, events

def get_events(feature_vector, qr_post, post_type='quote', stem_pos=True, corelex=True):
    post = {'dep': qr_post['{}_dep'.format(post_type)],
            'coref': qr_post['{}_coref'.format(post_type)]
            }
    sentences_dependencies = post['dep']
    for coreference in post['coref']:
        #startIndex = coreference['startIndex']
        mentionSpan = coreference['mentionSpan']
        sentNum = coreference['sentNum']
        #t = mentionSpan, sentNum, startIndex
        r, events = match(mention=mentionSpan,
            dependencies=sentences_dependencies[sentNum - 1],
            stem_pos=stem_pos,
            corelex=corelex)
        for event in events:
            feature_vector[event] = True

class GenerateEvents:


    def __init__(self, dataset):
        self.dataset = dataset

    def generate(self):
        for qr_post in self.dataset.posts():
            quote = {'text': TextObj(qr_post['quote_text']),
                     'dep' : qr_post['quote_dep'],
                     'pos' : qr_post['quote_pos'],
                     'coref': qr_post['quote_coref']}
            response = {'text': TextObj(qr_post['response_text']),
                        'dep': qr_post['response_dep'],
                        'pos': qr_post['response_pos'],
                        'coref': qr_post['response_coref']}
            yield quote, response



    def chains(self):
        c = Counter()
        for quote, response in self.generate():
            sentences_dependencies = quote['dep']
            for coreference in quote['coref']:
                startIndex = coreference['startIndex']
                mentionSpan = coreference['mentionSpan']
                sentNum = coreference['sentNum']
                t = mentionSpan, sentNum, startIndex
                r, events = self.match(mention=mentionSpan, dependencies=sentences_dependencies[sentNum - 1])
                if len(r) > 0:
                    #print mentionSpan, sentNum, startIndex
                    c.update(events)
                    print events, r
                    print coreference
                    #for event in events:
                        #pprint(event, indent=4)

            break
        print c.most_common(n=20)

    def count(self):
        max_length, max_chain = 0, []
        counter = Counter()
        for quote, response in self.generate():
            chains = defaultdict(list)
            for type in [quote, response]:
                for coref in type['coref']:
                    chains[coref['corefClusterID']].append(coref)

                for id, chain in chains.iteritems():
                    #chain = map(lambda x: x['mentionSpan'], chain)
                    counter.update({len(chain) : 1})
                    length = len(chain)
                if length > max_length:
                    max_length, max_chain = length, chain
            print quote['text'].tokens
            y, x = chains.items()[0]
            print map(lambda i: (i['mentionSpan'], i['startIndex'], i['sentNum']), x)
            print quote['text'].sentences


        print max_length, max_chain
        total = float(sum(counter.values()))
        for key, value in counter.most_common()[:10]:
            print "%d : %.2f" % (key, value/total)

if __name__ == '__main__':
    generate_events = GenerateEvents(dataset=JSONDataset('instances'))
    generate_events.chains()