from collections import defaultdict

from jsondataset import  JSONDataset

from nlp.text_obj import TextObj


class GenerateEvents:
    def __init__(self, dataset):
        self.dataset = dataset

    def generate(self):
        max_length, max_chain = 0, []
        all_chains = defaultdict(int)
        for qr_post in self.dataset.posts():
            quote = {'text': TextObj(qr_post['quote_text']),
                     'dep' : qr_post['quote_dep'],
                     'pos' : qr_post['quote_pos'],
                     'coref': qr_post['quote_coref']}
            response = {'text': TextObj(qr_post['response_text']),
                        'dep': qr_post['response_dep'],
                        'pos': qr_post['response_pos'],
                        'coref': qr_post['response_coref']}

            chains = defaultdict(list)
            for coref in quote['coref']:
                chains[coref['corefClusterID']].append(coref)

            for id, chain in chains.iteritems():
                chain = map(lambda x: x['mentionSpan'], chain)
                length = len(chain)
            if length > max_length:
                max_length, max_chain = length, chain
            continue

            tokens = zip(quote['text'].tokens, xrange(len(quote['text'].tokens)))

            for coref in quote['coref']:
                print quote['text'].tokens
                print coref['startIndex'], coref['mentionSpan']
                print filter(lambda x: x[0]==coref['mentionSpan'].split(' ')[0], tokens)

            break
        print max_length, max_chain
if __name__ == '__main__':
    generate_events = GenerateEvents(dataset=JSONDataset('instances'))
    generate_events.generate()
