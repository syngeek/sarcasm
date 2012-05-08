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
"""
FEATURES = ['basic_lengths', 'initialisms', 
            'gen_dep', 'LIWC', 'liwc_dep', 'opin_dep', 
            'pos_dep','repeated_punct', 'unigrams', 'meta',
            'annotations']
"""
FEATURES = ['annotations']
class AgreementBaseline:
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

            filename = self.results_dir+self.classification_feature+'/baseline/arffs/'
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
        dataset = Dataset('fourforums',annotation_list=['qr_dependencies'])

        for discussion in dataset.get_discussions(annotation_label='mechanical_turk'):

            if 'qr_meta' not in discussion.annotations['mechanical_turk']: continue
                    
            dependencies = defaultdict(lambda:defaultdict(list))
            if 'qr_dependencies' in discussion.annotations:
                for dep in discussion.annotations['qr_dependencies']:
                    dependencies[dep['qr_key']][dep['source']].append(dep)
            
            for key, entry in discussion.annotations['mechanical_turk']['qr_resample'].items():
                if entry['resampled']==True:
                    
                    #some conditions
                    if discussion.annotations['mechanical_turk']['qr_meta'][key]['quote_post_id']==None:
                        continue
                    if discussion.annotations['mechanical_turk']['qr_meta'][key]['task1 num annot']==None:
                        continue
                    
                    feature_vector = self.extract_features(discussion, key, dependencies, features)
                    if feature_vector is None: continue
                    feature_vectors[key]=feature_vector
        return feature_vectors
    
    def extract_features(self, discussion, key_id, dependencies, features=None):
        features = FEATURES if features is None else features
        meta_entry=discussion.annotations['mechanical_turk']['qr_meta'][key_id]
        average_entry=discussion.annotations['mechanical_turk']['qr_averages'][key_id]
        label = self.get_label(average_entry)
        if label == None: return None
        
        feature_vector = dict()
        feature_vector[self.classification_feature]=label
        
        for elem in ['quote', 'response']:
            text = meta_entry[elem]
            text = TextObj(text.decode('utf-8', 'replace'))
            elem_features = dict()
            feature_extractor.get_features_by_type(elem_features, features, text, dependencies[key_id][elem])
            
            for key, value in elem_features.items():
                feature_vector[elem+'_'+key]=value
                
        if 'meta' in features: feature_vector.update(self.get_meta_features(discussion, meta_entry));
        if 'annotations' in features: feature_vector.update(self.get_annot_features(discussion, average_entry));
        feature_vector.update(self.get_advanced_features(discussion, meta_entry));
        return feature_vector
    
        
    def get_advanced_features(self,discussion, meta_entry):
        feature_vector = dict()
        #feature_vector['advanced_contextual:cosine similarity']=tf_idf.get_cosine_similarity_from_text(quote, response, idf)
        return feature_vector
    
    def get_meta_features(self, discussion, meta_entry):
        feature_vector = dict()
        
        quote = meta_entry['quote']
        response = meta_entry['response']
        quoted_post = discussion.posts[meta_entry['quote_post_id']]
        response_post = discussion.posts[meta_entry['response_post_id']]
        
        feature_vector['meta:response_longer']=(len(response)> len(quote))
        feature_vector['meta:time_between_posts']=response_post.timestamp - quoted_post.timestamp
        if feature_vector['meta:time_between_posts'] > 0:
            feature_vector['meta:log_time_between_posts']=math.log10(float(feature_vector['meta:time_between_posts']))
        elif feature_vector['meta:time_between_posts'] < 0:
            print 'uhhh... B comes before A???', feature_vector['meta:time_between_posts']
        
        feature_vector['meta:same_author']=(quoted_post.author==response_post.author)
        if len(quote.strip())<len(quoted_post.delete_ranges('quotes').strip()):
            feature_vector['meta:percent_quoted']=len(quote)/float(len(quoted_post.delete_ranges('quotes')))
        feature_vector['meta:number_of_other_quotes']=len(response_post.get_ranges('quotes', sort=False))-1
        feature_vector['meta:response_to_response']=(quoted_post.parent_id in discussion.posts and discussion.posts[quoted_post.parent_id].author == response_post.author)
        feature_vector['meta:mention_of_quote_author']=(quoted_post.author.lower() in response_post.delete_ranges('quotes').lower())
        
        feature_vector['author:response_'+response_post.author]=True
        feature_vector['author:quote_'+quoted_post.author]=True
        feature_vector['author:pair_'+str(sorted([response_post.author,quoted_post.author]))]=True
        
        return feature_vector
    
    def get_annot_features(self, discussion, average_entry):
        feature_vector = dict()
        for key, value in average_entry.items():
            if value == None: continue
            if key not in self.task1: continue
            if key == self.classification_feature: continue
            feature_vector['annot:'+key]=value
        return feature_vector
     
    def get_label(self, average_entry, threshold=0.5):
        if average_entry[self.classification_feature]>=threshold:
            return self.task1[self.classification_feature][1]
        elif average_entry[self.classification_feature]==0:
            return self.task1[self.classification_feature][0]
        else:
            return None

    
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
    
    def run_experiments(self, arff_folder):
        runs = 10
        all_arffs = [os.path.basename(filename)[:-5] for filename in os.listdir(arff_folder) if filename.endswith('.arff')]
        
        quote_local = set([arff for arff in all_arffs if re.match('quote.*', arff)])
        response_local = set([arff for arff in all_arffs if re.match('response.*', arff)])
        meta = set(['meta'])
        author = set(['author'])
        bow = set([arff for arff in all_arffs if re.match('.*(unigram|bigram|LIWC|initialism)', arff)])
        uni = set([arff for arff in all_arffs if re.match('.*(unigram|initialism)', arff)])

        experiments = dict()
        experiments['dep_opinion']=set([arff for arff in all_arffs if re.match('.*opinion.*', arff)])
        experiments['uni']=uni
        experiments['meta']=meta
        experiments['author']=author
        experiments['bow']=bow
        experiments['quote local']=quote_local
        experiments['response local']=response_local
        experiments['both local']=response_local.union(quote_local)
        experiments['meta+local']=response_local.union(quote_local).union(meta)
        experiments['all']=set()
        for name, arffs in experiments.items():
            if name is not 'all':
                experiments['all'].update(arffs)
            arffs.add(self.classification_feature)
        classifiers = ['weka.classifiers.bayes.NaiveBayes']
        
        progress = ProgressReporter(total_number=len(classifiers)*len(experiments)*runs)
        results = defaultdict(dict) #run->classifier->featureset->results
        for classifier_name in classifiers:
            for featureset, arffs in experiments.items():
                print 'running: classifier: '+classifier_name+', featureset: '+featureset
                experiment_accuracy = 0.0
                with utils.random_guard(95064):
                    for run in range(runs):
                        run_results = weka_interface.cross_validate(arff_folder, arffs, classifier_name=classifier_name, classification_feature=self.classification_feature, n=10)
                        
                        #Pickle can't handle lambdas, so using this for now
                        if classifier_name not in results[run]:
                            results[run][classifier_name]=dict()
                            
                        results[run][classifier_name][featureset] = run_results
                        right = sum([1 for entry in run_results.values() if entry['right?']])
                        run_accuracy = right/float(len(run_results))
                        experiment_accuracy+=run_accuracy
                        progress.report()
                experiment_accuracy = experiment_accuracy/float(runs)
                print 'Accuracy:',experiment_accuracy
        pkl_file = open(arff_folder+'../weka_results.pkl','w')
        cPickle.dump(results, pkl_file, cPickle.HIGHEST_PROTOCOL)
if( __name__ == '__main__'):
    agreement_baseline = AgreementBaseline()
    agreement_baseline.main()
    
