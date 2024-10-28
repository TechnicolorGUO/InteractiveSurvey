#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   text_classifier.py
@Time    :   2022/02/23 16:48:55
@Author  :   Shuaiqi 
@Version :   1.0
@Contact :   shuaiqizju@gmail.com
@Desc    :   None
'''

import pdb
import torch
from nltk.tokenize import word_tokenize,sent_tokenize


MODEL_PATH_DICT={}
#MODEL_PATH_DICT['asg_pegasus']={}
#MODEL_PATH_DICT['asg_pegasus']['background']="/home/disk1/data/shuaiqi/survey_gen/models/pegasus/survey_mds_background_3000_300/checkpoint-10170"
MODEL_PATH_DICT['bert_base_3sent_classify']={}
MODEL_PATH_DICT['bert_base_3sent_classify']['CSAbstruct']="/home/disk1/data/shuaiqi/survey_gen/models/classify/bert_base_3sent/CSAbstruct/checkpoint-1418"


BERT_SEP_TOKEN = " [SEP] "
ROBERTA_SEP_TOKEN = " </s></s> "
SEGMENT_SYMBOL = ' story_separator_special_tag '
TEST_BATCH_SIZE=16

class Text_Classifier(object):

    def __init__(
            self,
            classifier_dict, 
            train_set_name,classifier_name       
    ):
        import torch
        self.classifier_dict = classifier_dict
        self.torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        #self.input_text = input_text
        #self.section_name = section_name
        #self.summarizer_name = summarizer_name
        #self.output_len = output_len
        #self.input_truncate_len = input_len
        if classifier_name == "bert_base_3sent_classify":
            #pdb.set_trace()

            from transformers import BertTokenizer, BertForSequenceClassification
            model_name = self.classifier_dict['bert_base_3sent_classify'][train_set_name]#['background']
            #model_name = "/home/disk1/data/shuaiqi/survey_gen/models/pegasus/survey_mds_background_3000_300/checkpoint-10170"

            self.tokenizer = BertTokenizer.from_pretrained(model_name, problem_type="multi_label_classification",num_labels=5)
            self.model = BertForSequenceClassification.from_pretrained(model_name,num_labels=5).to(self.torch_device)


    '''
    # load summarizer model
    def load_model(self, section_name,summarizer_name):

        if summarizer_name == "asg_pegasus":
            from transformers import PegasusForConditionalGeneration, PegasusTokenizer
            model_name = self.summarizer_dict['asg_pegasus'][section_name]#['background']
            #model_name = "/home/disk1/data/shuaiqi/survey_gen/models/pegasus/survey_mds_background_3000_300/checkpoint-10170"

            self.tokenizer = PegasusTokenizer.from_pretrained(model_name)
            self.model = PegasusForConditionalGeneration.from_pretrained(model_name).to(self.torch_device)

        #return model,tokenizer
        return self.model,self.tokenizer
    '''
    
    def preprocess_input_text(self, input_str):
        import re
        input_str = str(input_str).strip().lower()
        input_str = input_str.replace('\\n',' ').replace('\n',' ').replace('\r',' ').replace('——',' ').replace('——',' ').replace('__',' ').replace('__',' ').replace('........','.').replace('....','.').replace('....','.').replace('..','.').replace('..','.').replace('..','.').replace('. . . . . . . . ','. ').replace('. . . . ','. ').replace('. . . . ','. ').replace('. . ','. ').replace('. . ','. ')
        input_str = input_str.encode('unicode_escape').decode('ascii')
        input_str = re.sub(r'\\u[0-9a-z]{4}', ' ', input_str)
        input_str = input_str.replace('\\ud835',' ').replace('    ',' ').replace('  ',' ').replace('  ',' ')
        return input_str
        
    def preprocess_input_abs_list(self, input_abs_str_list):
        processed_abs_list=[]
        processed_3sents_list=[]
        for abs_index, input_abs_str in enumerate(input_abs_str_list):
            input_abs_str = self.preprocess_input_text(input_abs_str)
            input_sent_list = sent_tokenize(input_abs_str)
            input_sent_list_temp=[]
            for input_sent in input_sent_list:
                if len(input_sent)>3:
                    input_sent_list_temp.append(input_sent)
            input_sent_list = input_sent_list_temp        
            combined_sents_str=""
            processed_3sents_list=[]
            for sent_index, input_sent in enumerate(input_sent_list):
                combined_sents_str = input_sent
                #if sent_index==len(input_sent_list)-1 and sent_index-1>=0:
                if sent_index-1>=0:
                    combined_sents_str = combined_sents_str + BERT_SEP_TOKEN + input_sent_list[sent_index-1]
                else:
                    combined_sents_str = combined_sents_str + BERT_SEP_TOKEN
                #if sent_index == 0 and sent_index<=len(input_sent_list)-2:
                if sent_index<=len(input_sent_list)-2:
                    combined_sents_str = combined_sents_str + BERT_SEP_TOKEN + input_sent_list[sent_index+1]
                else:
                    combined_sents_str = combined_sents_str + BERT_SEP_TOKEN

                #elif sent_index<=len(input_sent_list)-2:
                #    combined_sents_str = input_sent + BERT_SEP_TOKEN + input_sent_list[sent_index-1] + BERT_SEP_TOKEN + input_sent_list[sent_index+1]
                processed_3sents_list.append(combined_sents_str)
            processed_abs_list.append(processed_3sents_list)

        return processed_abs_list

    def classify_sent(self, input_abs_str_list ,batch_size=16 , input_max_length = 2048):
        output_text = ""
        #batch_size = TEST_BATCH_SIZE
        
        processed_abs_list = self.preprocess_input_abs_list(input_abs_str_list)
        pred_abs_id_list=[]
        pred_abs_results_list=[]
        ref_abs_list=[]
        for abs_id,processed_input_sent_list in enumerate(processed_abs_list):
            #test_batch = [processed_input_text]
            test_batches = [processed_input_sent_list[i:i+batch_size] for i in range(0,len(processed_input_sent_list),batch_size)]

            #torch_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            #pdb.set_trace()
            pred_batch_results_list=[]
            
            for batch_id,test_batch in enumerate(test_batches):
            
                inputs = self.tokenizer(test_batch,truncation=True, padding=True, max_length=input_max_length, return_tensors='pt').to(self.torch_device)

                #if summarizer_name == "pegasus":
                #predictions = self.model.generate(**inputs,max_length=output_len+100,min_length=output_len,num_beams=5,length_penalty=2.0,no_repeat_ngram_size=3)
                #predictions = self.tokenizer.batch_decode(predictions)

                outputs = self.model(**inputs)
                batch_preds = outputs['logits'].argmax(-1)+1
                if batch_id==0:
                    preds = batch_preds#np.argmax(outputs['logits'], axis=1)
                else:
                    preds = torch.cat([preds,batch_preds])
                
                pred_batch_results_list.append(preds)
                

            pred_abs_id_list.append(abs_id)
            pred_abs_results_list.append(pred_batch_results_list)
            ref_abs_list.append(test_batches)
            #pdb.set_trace()

        background_sent_list = []
        method_sent_list = []
        assert(len(pred_abs_id_list)==len(pred_abs_results_list))
        assert(len(ref_abs_list)==len(pred_abs_results_list))
        id_to_label_list = ['background', 'objective', 'method', 'result', 'other']

        #pdb.set_trace()
        for abs_index, pred_abs_results in enumerate(pred_abs_results_list):
            pred_abs_id = pred_abs_id_list[abs_index]
            ref_abs = ref_abs_list[abs_index]
            for batch_index,pred_batch_results in enumerate(pred_abs_results):
                ref_abs_batch = ref_abs[batch_index]
                for sent_index,pred_sent_result in enumerate(pred_batch_results):
                    #pdb.set_trace()
                    ref_abs_sent = ref_abs_batch[sent_index].split(BERT_SEP_TOKEN)[0]
                    if id_to_label_list[pred_sent_result-1] == 'background':
                        background_sent_list.append(ref_abs_sent)
                    elif id_to_label_list[pred_sent_result-1] == 'method':
                        method_sent_list.append(ref_abs_sent)

        #pdb.set_trace()

        return background_sent_list,method_sent_list
