#!/usr/bin/env python
# encoding: utf-8
'''
@author: shuaiqi
@file: ref_paper_desp.py
@time: 2020/6/15 16:13
@desc:
'''
import pandas as pd
import nltk
from summa.summarizer import summarize
import spacy
nlp = spacy.load('en_core_web_sm')
import pyinflect
#from pattern.en import conjugate,lemma
#nltk.download("wordnet")
#nltk.download("averaged_perceptron_tagger")
#nltk.download("punkt")
#nltk.download("maxent_treebank_pos_tagger")
from nltk.corpus import wordnet
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
import pdb

class ref_desp(object):
    def __init__(
            self,
            input_DF
    ):
        self.input_DF = input_DF

    def get_wordnet_pos(self, treebank_tag):
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return None
    def lemmatize_sentence(self, sentence):
        res = ''
        lemmatizer = WordNetLemmatizer()
        words_list = word_tokenize(sentence)
        i = 0
        for word, pos in pos_tag(words_list):
            wordnet_pos = self.get_wordnet_pos(pos) or wordnet.NOUN
            if (i < 4 and wordnet_pos == 'v' ):
                word = lemmatizer.lemmatize(word, pos=wordnet_pos) #change to base form of verb
                token = nlp(word)
                word2 = token[0]._.inflect('VBZ') #base form of verb -> third person singular verb
                '''
                try:
                    word = conjugate(word,person=3)# ,tense ='PRESENT',person=3,number = "singular"
                except  RuntimeError:
                    print()
                '''
                if word2 != None:
                    res = res + word2 + ' '
                else:
                    res = res + word + ' '

            else:
                res = res + str(word) + ' '
            i = i + 1
        return res

    def sentence_selection(self, ref_abs):
        keywords = ['we ', 'this paper', 'this survey','our ']
        # ,'in this paper','proposes','propose','presents','prove','proves','study','studies','focuses','focus','show','shows','introduce','introduces'
        matched_sentence = ''
        matched_keyword = ''
        item = ref_abs
        item = item.lower()
        sentences = nltk.tokenize.sent_tokenize(item)
        for sentence in sentences:

            match_label = -1
            for keyword in keywords:
                match_label = sentence.find(keyword)
                if (match_label != -1):
                    matched_sentence = sentence
                    matched_keyword = keyword
                    break
            if (match_label != -1): break

        return matched_sentence, matched_keyword

    def ref_desp_generator(self):

        train_tsv = self.input_DF
        #data_without_NaN = train_tsv.dropna(axis=0)
        #train_data = data_without_NaN[data_without_NaN['ref_context'] != '[]']
        data_without_NaN = train_tsv.fillna('')
        train_data = data_without_NaN

        # extract from abstract
        ref_abs = list(data_without_NaN['abstract'])
        ref_entry = list(data_without_NaN['ref_entry'])
        ref_title = list(data_without_NaN['ref_title'])
        ref_description_list = []
        abs_summary_list = []

        for abs in ref_abs:
            summary = ''
            sentence_type = ''
            matched_sent, matched_keyword = self.sentence_selection(abs)
            if matched_sent != '':
                summary = matched_sent
                sentence_type = matched_keyword
            else:
                summary = summarize(abs, words=20)
                sentence_type = 'textrank'
            abs_summary_list.append([summary, sentence_type])

        # convert the sentence
        for i in range(len(abs_summary_list)):
            ref_description = ''

            # get the name of the first author
            # names = ref_entry[i].split(str=".")[0]
            split_entry = ref_entry[i].split()
            if len(split_entry) < 2:
                name = ' '
            else:
                name = ref_entry[i].split()[1]
                name = name + ' et al. ' + '[' + str(i + 1) + '] '

            # delete content before the subject_word ['we ','this paper','this survey','our ']
            text = abs_summary_list[i][0]
            subject_word = abs_summary_list[i][1]
            #if subject_word == 'in this paper':
            #    ref_description = text.replace('this paper', name)
            if subject_word != 'textrank':
                match_label = text.find(subject_word)
                SVO_text = text[match_label:]
                if SVO_text == 'this paper.':
                    ref_description = text.replace(subject_word, name)
                else:
                    # convert verb
                    SVO_text = self.lemmatize_sentence(SVO_text)

                    ref_description = SVO_text.replace(subject_word, name)
            else:
                SVO_text = text
                ref_description = SVO_text[:-1]+ ' ['+str(i)+'].'
                #print()

            ref_description = ref_description.replace('  ', ' ')
            ref_description = ref_description.replace(' ,', ',')
            ref_description = ref_description.replace(' .', '.')
            ref_description = ref_description.replace('( ', '(')
            ref_description = ref_description.replace(' )', ')')


            ref_description_list.append([ref_title[i],ref_description])
        #pdb.set_trace()
        return ref_description_list

if __name__ == "__main__":

    input_folder_dir =  'D:\\dataset\\autosurvey\\2830555.tsv'#'D:\\dataset\\autosurvey\\2830555.tsv'
    train_tsv = pd.read_csv(input_folder_dir, sep='\t', header=0, index_col=0)
    DF_shape = train_tsv.shape
    mask_list = [1 for i in range(DF_shape[0])] # use 0  no use 1

    # add a new column 'mask'
    train_tsv['mask'] = mask_list
    # select rows ['mask'] == 1
    input_DF = train_tsv[train_tsv['mask'] == 1]
    #feed the new df
    ref_desp_1 = ref_desp(input_DF)
    ref_description_list = ref_desp_1.ref_desp_generator()

    for item in ref_description_list:
        print(item)
