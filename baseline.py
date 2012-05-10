import math
import cPickle
import os
import re
from collections import defaultdict

from nlp import feature_extractor
from nlp.text_obj import TextObj
from misc import utils
from file_formatting import arff_writer, csv_wrapper
from machine_learning import weka_interface
from progress_reporter.progress_reporter import ProgressReporter

from grab_data.discussion import Dataset, results_root_dir

from feature_extractor import get_features_by_type

from jsondataset import JSONDataset

from generate_events import get_events

FEATURES = ['basic_lengths', 'initialisms', 
            'gen_dep', 'LIWC', 'liwc_dep', 'opin_dep', 
            'pos_dep','repeated_punct', 'unigrams', 'meta',
            'annotations']

#FEATURES = ['unigrams']

class ClassificationBaseline:
    def __init__(self, classification_feature='sarcasm'):
        self.classification_feature = classification_feature
        self.task1 = {'agreement':['disagree','agree'], 'attack':['attacking','supportive'], 'fact-feeling':['feelings','fact'], 'nicenasty':['nasty','nice'], 'sarcasm':['sarcastic','not sarcastic']}
        self.results_dir = results_root_dir
        for key in self.task1.keys():
            self.task1[key+'_unsure' ]=['unsure','sure']
            
    def main(self, features=None):
        with utils.random_guard(95064):
            feature_vectors = self.get_feature_vectors(features)
            print 'Derived features for', len(feature_vectors),'instances'
            feature_vectors = utils.balance(feature_vectors, self.classification_feature)
            print 'writing arff file with', len(feature_vectors),'instances'

            #filename = self.results_dir+self.classification_feature+'/baseline/arffs/'
            filename = 'baseline/arffs'
            detailed_features, instance_keys = arff_writer.write(filename, feature_vectors, classification_feature=self.classification_feature, write_many=True, minimum_instance_counts_for_features=2)
            self.write_arff_instance_data(filename, instance_keys, detailed_features)
            #self.run_experiments(filename)

    def write_arff_instance_data(self, filename, instance_keys, detailed_features):
        instance_data = list()
        for arff_index in range(len(instance_keys)):
            row = dict()
            row['key']=instance_keys[arff_index]
            row['arff_index']=arff_index
            instance_data.append(row)
        csv_wrapper.write_csv(filename+'instances.csv', instance_data, ['key'], get_keys_from_first_row=True)
        pkl_file = open(filename+'detailed_features.pkl','w')
        cPickle.dump(detailed_features, pkl_file,cPickle.HIGHEST_PROTOCOL)
        
    def get_feature_vectors(self, features=None):
        feature_vectors = dict()
        dataset = JSONDataset('instances')
        for index, qr_post in enumerate(dataset.posts()):
            #print index, qr_post['sarcasm']
            #feature_vectors[index] = {'index': index, 'sarcasm': qr_post['sarcasm']}
            feature_vector = self.extract_features(qr_post=qr_post, features=features)
            if feature_vector == None:
               continue
            feature_vectors[index] = feature_vector
        return feature_vectors


    def extract_features(self, qr_post, features=None):
        features = FEATURES if features is None else features

        feature_vector = dict()
        feature_vector[self.classification_feature] = qr_post[self.classification_feature]
        
        for elem in ['quote', 'response']:
            text = qr_post["%s_text" % elem]
            text = TextObj(text.decode('utf-8', 'replace'))

            elem_features = dict()
            #dependencies = dict()
            get_features_by_type(elem_features, features, text)#, dependencies[key_id][elem])
            get_events(elem_features, qr_post=qr_post, post_type=elem)

            for key, value in elem_features.items():
                feature_vector[elem+'_'+key]=value

        return feature_vector
    
    def get_local_features(self, text):
        feature_vector = dict()
        text_obj = TextObj(text)
        feature_extractor.get_ngrams(feature_vector, text_obj.tokens)
        feature_extractor.get_ngrams(feature_vector, text_obj.tokens, n=2)
        feature_extractor.get_initialisms(feature_vector, text_obj.tokens)
        feature_extractor.get_basic_lengths(feature_vector, text_obj.text, text_obj.sentences, text_obj.tokens)
        feature_extractor.get_repeated_punct(feature_vector, text_obj.text)
        feature_extractor.get_LIWC(feature_vector, text_obj.text)
        
        return feature_vector

    def get_dependency_features(self, dependency_list):
        feature_vector = dict()
        feature_extractor.get_dependency_features(feature_vector, dependency_list)          
        feature_extractor.get_dependency_features(feature_vector, dependency_list, generalization='pos')
        feature_extractor.get_dependency_features(feature_vector, dependency_list, generalization='opinion')
#         feature_extractor.get_dependency_features(feature_vector, dependency_list, generalization='neg_dist_opinion')
        feature_extractor.get_dependency_features(feature_vector, dependency_list, generalization='liwc')
        return feature_vector


if( __name__ == '__main__'):
    classification_baseline = ClassificationBaseline(classification_feature='sarcasm')
    classification_baseline.main()
    
