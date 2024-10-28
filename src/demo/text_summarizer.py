#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   text_summarizer.py
@Time    :   2022/02/23 16:48:55
@Author  :   Shuaiqi 
@Version :   1.0
@Contact :   shuaiqizju@gmail.com
@Desc    :   None
'''

import pdb
import torch

MODEL_PATH_DICT={}
MODEL_PATH_DICT['asg_pegasus']={}
MODEL_PATH_DICT['asg_pegasus']['background']="/home/disk1/data/shuaiqi/survey_gen/models/pegasus/survey_mds_background_3000_300/checkpoint-10170"


class Text_Summarizer(object):

    def __init__(
            self,
            summarizer_dict, 
            section_name,summarizer_name       
    ):
        import torch
        self.summarizer_dict = summarizer_dict
        self.torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        #self.input_text = input_text
        #self.section_name = section_name
        #self.summarizer_name = summarizer_name
        #self.output_len = output_len
        #self.input_truncate_len = input_len
        if summarizer_name == "asg_pegasus":
            #pdb.set_trace()

            from transformers import PegasusForConditionalGeneration, PegasusTokenizer
            model_name = self.summarizer_dict['asg_pegasus'][section_name]#['background']
            #model_name = "/home/disk1/data/shuaiqi/survey_gen/models/pegasus/survey_mds_background_3000_300/checkpoint-10170"

            self.tokenizer = PegasusTokenizer.from_pretrained(model_name)
            self.model = PegasusForConditionalGeneration.from_pretrained(model_name).to(self.torch_device)

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
        

    def summarize(self, input_text, output_len=300, input_max_length = 2048):
        output_text = ""
        processed_input_text = self.preprocess_input_text(input_text)
        test_batch = [processed_input_text]

        #torch_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        inputs = self.tokenizer(test_batch,truncation=True, padding=True, max_length=input_max_length, return_tensors='pt').to(self.torch_device)
        #if summarizer_name == "pegasus":
        predictions = self.model.generate(**inputs,max_length=output_len+100,min_length=output_len,num_beams=5,length_penalty=2.0,no_repeat_ngram_size=3)
        predictions = self.tokenizer.batch_decode(predictions)

        
        output_text = predictions[0].strip()
        output_sent_list = output_text.split('. ')
        for output_sent_id in range(len(output_sent_list)):
            output_sent_list[output_sent_id]=output_sent_list[output_sent_id].capitalize()

        if output_text[-1]!='.':
            output_sent_list = output_sent_list[:-1]
        output_text='. '.join(output_sent_list)+'.'

        print(predictions)
        print(output_text)

        #pdb.set_trace()

        return output_text
