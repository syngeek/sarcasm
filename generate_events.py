from collections import defaultdict, Counter

from jsondataset import  JSONDataset

from nlp.text_obj import TextObj


class GenerateEvents:
    def __init__(self, dataset):
        self.dataset = dataset

    def generate(self):
        max_length, max_chain = 0, []
        all_chains = defaultdict(int)
        items = Counter()
        counter = Counter()
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
            for type in [quote, response]:
                for coref in type['coref']:
                    chains[coref['corefClusterID']].append(coref)

                for id, chain in chains.iteritems():
                    chain = map(lambda x: x['mentionSpan'], chain)

                    counter.update({len(chain) : 1})

                    length = len(chain)
                if length > max_length:
                    max_length, max_chain = length, chain


        print max_length, max_chain
        total = float(sum(counter.values()))
        for key, value in counter.most_common()[:10]:
            print "%d : %.2f" % (key, value/total)
if __name__ == '__main__':
    generate_events = GenerateEvents(dataset=JSONDataset('instances'))
    generate_events.generate()
