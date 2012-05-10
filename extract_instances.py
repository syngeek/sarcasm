import json
from collections import defaultdict

from nlp import stanford_nlp
from nlp.text_obj import TextObj
from misc import utils

from progress_reporter.progress_reporter import ProgressReporter

from grab_data.discussion import Dataset, results_root_dir

class InstanceExtractor:
    def __init__(self, classification_feature='sarcasm'):
        self.classification_feature = classification_feature
        self.task1 = {'agreement':['disagree','agree'],
                      'attack':['attacking','supportive'],
                      'fact-feeling':['feelings','fact'],
                      'nicenasty':['nasty','nice'],
                      'sarcasm':['sarcastic','not sarcastic']}
        self.results_dir = results_root_dir
        for key in self.task1.keys():
            self.task1[key+'_unsure' ]=['unsure','sure']

    def main(self):
        self.extract()

    def extract(self):
        feature_vectors = dict()
        dataset = Dataset('fourforums',annotation_list=['qr_dependencies'])
        current_index = 0
        for discussion in dataset.get_discussions(annotation_label='mechanical_turk'):

            if 'qr_meta' not in discussion.annotations['mechanical_turk']: continue

            #dependencies = defaultdict(lambda:defaultdict(list))
            #if 'qr_dependencies' in discussion.annotations:
            #    for dep in discussion.annotations['qr_dependencies']:
            #        dependencies[dep['qr_key']][dep['source']].append(dep)
            discussion_items = discussion.annotations['mechanical_turk']['qr_resample'].items()
            for (key, entry) in discussion_items:
                if entry['resampled']==True:

                    #some conditions
                    if discussion.annotations['mechanical_turk']['qr_meta'][key]['quote_post_id']==None:
                        continue
                    if discussion.annotations['mechanical_turk']['qr_meta'][key]['task1 num annot']==None:
                        continue

                    attributes = self.get_post_attributes(discussion=discussion, key_id=key, dependencies=None)
                    if attributes is None: continue
                    json.dump(attributes, open('instances/%s.json' % current_index, 'w'))
                    current_index += 1

    def get_post_attributes(self, discussion, key_id, dependencies):

        meta_entry=discussion.annotations['mechanical_turk']['qr_meta'][key_id]
        average_entry=discussion.annotations['mechanical_turk']['qr_averages'][key_id]

        label = self.get_label(average_entry)
        if label == None: return None

        feature_vector = dict()
        feature_vector[self.classification_feature] = label
        feature_vector['key'] = "%r" % key_id
        for elem in ['quote', 'response']:
            rawtext = utils.ascii_only(meta_entry[elem])
            #text = TextObj(rawtext.decode('utf-8', 'replace'))
            try:
                pos, tree, dep, coref = stanford_nlp.get_parses(rawtext, coreferences=True)
                mapping = {'text': rawtext,
                           'pos': pos,
                           #'tree': tree, <- sets aren't serializable and i cant see any need for them
                           'dep': dep,
                           'coref': coref}
                for key, value in mapping.iteritems():
                    feature_vector["%s_%s" % (elem, key)] = value
            except Exception, e:
                print e, key_id
                return None
        return feature_vector

    def get_label(self, average_entry, threshold=0.5):
        if average_entry[self.classification_feature]>=threshold:
            return self.task1[self.classification_feature][0]
        elif average_entry[self.classification_feature]==0:
            return self.task1[self.classification_feature][1]
        else:
            return None


if( __name__ == '__main__'):
    instance_extractor = InstanceExtractor()
    instance_extractor.main()

