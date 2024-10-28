#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 15:28:09 2020

@author: yangruosong
"""
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from summa.summarizer import summarize
from . import views

import pandas as pd
import numpy as np

from .text_summarizer import Text_Summarizer
from .text_classifier import Text_Classifier
import torch
import pdb

# line 66, 88 change the title for the category descroption
no2word = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}

Survey_dict = views.Survey_dict
Survey_Topic_dict = views.Survey_Topic_dict

'''

MODEL_PATH_DICT={}
MODEL_PATH_DICT['asg_pegasus']={}
MODEL_PATH_DICT['asg_pegasus']['background']="/home/disk1/data/shuaiqi/survey_gen/models/pegasus/survey_mds_background_3000_300/checkpoint-10170"
MODEL_PATH_DICT['asg_pegasus']['method']="/home/disk1/data/shuaiqi/survey_gen/models/pegasus/survey_mds_method_3000_300/checkpoint-9980"

MODEL_PATH_DICT['bert_base_3sent_classify']={}
MODEL_PATH_DICT['bert_base_3sent_classify']['CSAbstruct']="/home/disk1/data/shuaiqi/survey_gen/models/classify/bert_base_3sent/CSAbstruct/checkpoint-1418"

'''

def readReference(df):
    refs = []
    train_tsv = df

    abstract = train_tsv["abstract"]
    intro = train_tsv["intro"]
    refs = [(a, i) for a, i in zip(abstract, intro)]
    refs = [refs[i] for i in range(len(refs))]
    # print(len(refs))
    return refs


def getReferenceCategories(fileID, mask):
    cate_topics = []
    train_tsv = pd.read_csv(fileID + ".tsv", sep='\t', header=0, index_col=0)
    ls = train_tsv["label"]
    ds = train_tsv["topic_word"]
    cate_topics = [(d, l) for d, l in zip(ds, ls)]
    cate_topics = [cate_topics[i] for i in range(len(cate_topics)) if mask[i] == 1]
    topics = []
    labels = {}
    for topic, label in cate_topics:
        topic = eval(topic)
        if label not in labels:
            labels[label] = " ".join(topic)
    topics = sorted([(key, labels[key]) for key in labels])
    topics = [topic[1] for topic in topics]
    return topics


# pay attention to the index of category description
def getReferenceTypes(fileID, mask):
    cate_topics = []
    train_tsv = pd.read_csv(fileID + ".tsv", sep='\t', header=0, index_col=0)
    ls = train_tsv["label"]
    ds = train_tsv["topic_word"]
    cs = train_tsv["cate_desc"]
    cate_topics = [(d, c, l) for d, c, l in zip(ds, cs, ls)]
    # replace the category description index with 1 in train_tsv.iloc[i][10]
    cate_topics = [cate_topics[i] for i in range(len(cate_topics)) if mask[i] == 1]
    topics = []
    labels = {}
    for topic, des, label in cate_topics:
        topic = eval(topic)
        if label not in labels:
            labels[label] = [" ".join(topic), des]
    topics = sorted([(key, labels[key]) for key in labels])
    tops = [topic[1][0] for topic in topics]
    descriptions = [topic[1][1] for topic in topics]
    return tops, descriptions


# pay attention to the index of category description
def getReferenceDes(df):
    cate_topics = []
    train_tsv = df
    # replace the category description index with the forth element of 1 in train_tsv.iloc[i][10]
    ls = train_tsv["label"]
    ds = train_tsv["topic_word"]

    # cs = train_tsv["cate_desc"]
    cs = train_tsv["label"] ## convert

    rs = train_tsv["ref_entry"]
    rds = train_tsv["description"]
    cate_topics = [(r, d, rd, c, l) for r, d, rd, c, l in zip(rs, ds, rds, cs, ls)]
    cate_topics = [cate_topics[i] for i in range(len(cate_topics))]
    topics = []
    labels = {}
    for ref, topic, ref_des, des, label in cate_topics:
        # topic = eval(topic)
        if label not in labels:
            labels[label] = [" ".join(topic), des, [(ref, ref_des)]]
        else:
            labels[label][2].append((ref, ref_des))
    topics = sorted([(key, labels[key]) for key in labels])
    tops = [topic[1][0] for topic in topics]
    descriptions = [topic[1][1] for topic in topics]
    refs = []
    idx = 1
    ref_deses = []
    for topic in topics:
        each_topic = []
        for i, t in enumerate(topic[1][2]):
            ref, ref_des = t
            refs.append("[" + str(idx) + "]. " + ref)
            each_topic.append(topic[1][2][i][1].replace("[NO]", "[" + str(idx) + "]"))
            idx += 1
        ref_deses.append(each_topic)
    return refs, ref_deses


def cleanText(text):
    texts = text.strip().split("\n")
    sents = []
    for t in texts:
        sents += sent_tokenize(t)
    return sents


def sentText(refs):
    newRefs = []
    aiRefs = []
    abslist = []
    introlist = []
    for ref in refs:
        newAbs = [word_tokenize(sent) for sent in cleanText(ref[0])]
        newIntro = [word_tokenize(sent) for sent in cleanText(ref[1])]
        newRefs.append([newAbs, newIntro])
        aiRefs.append(newAbs + newIntro)
        abslist.append(newAbs)
        introlist.append(newIntro)

    return newRefs, aiRefs, abslist, introlist


def testTopicSentence(topic, sent):
    flag = 0
    for word in topic:
        if word in sent:
            flag += 1
    if flag == len(topic):
        return True
    else:
        return False


def testContainTopic(topic, abstract):
    for sent in abstract:
        if testTopicSentence(topic, sent):
            return True

    return False


def extractTopicRef(topic, abslist, introlist):
    newabs = []
    newintro = []
    for al, il in zip(abslist, introlist):
        if testContainTopic(topic, al):
            newabs.append(al)
            newintro.append(il)
    return newabs, newintro


def extractTopicSent(abslist, introlist):
    newabs = []
    newintro = []
    for al, il in zip(abslist, introlist):
        newal = []
        for a in al:
            if "we" not in a and "paper" not in a and "We" not in a:
                newal.append(a)
            else:
                if len(newal) > 0:
                    newabs.append(newal)
                    newintro.append(il)
                    newal = []
                break
    return newabs, newintro


def combineSentence(sents):
    if len(sents) == 1:
        if len(sents[0]) > 1:
            text = " ".join(sents[0])
        else:
            text = " ".join(sents[0][0])
    else:
        sentences = []
        for sent in sents:
            s = " ".join(sent)
            sentences.append(s)
        # sentences = [" ".join(sent) for sent in sents]
        text = " ".join(sentences)
    return text


'''----------------------------sentence similarity-----------------------------------'''


def JaccardSim(sent1, sent2):
    sent1 = set(sent1)
    sent2 = set(sent2)
    sim = float(len(sent1 & sent2)) / len(sent1 | sent2)
    return sim


def testSim(sents):
    similarity = []
    for i in range(len(sents) - 1):
        for j in range(i + 1, len(sents), 1):
            sim = JaccardSim(sents[i], sents[j])
            similarity.append((i, j, sim))
    return similarity


def cleanSummary(summ):
    summary = [word_tokenize(sent) for sent in sent_tokenize(summ)]
    similarity = testSim(summary)
    clean_ids = []
    for i, j, sim in similarity:
        if sim > 0.3:
            clean_ids.append(j)
    clean_ids = set(clean_ids)
    newSum = []
    for i in range(len(summary)):
        if i not in clean_ids:
            newSum.append(summary[i])

    if len(newSum) == 1:
        text = " ".join(newSum[0])
    else:
        sent = [" ".join(s) for s in newSum]
        text = " ".join(sent)
    return text

def cleanComma(text):
    word = word_tokenize(text)
    comma = [",", ".", ")"]
    new_text = []
    for i in range(len(word) - 1):
        if word[i] in comma and word[i + 1] in comma:
            continue
        elif word[i] in comma and word[i + 1] not in comma:
            new_text.append(word[i])
        else:
            new_text.append(word[i])
    # print(word)
    new_text.append(word[-1])
    clean_text = ""
    for word in new_text:
        if len(clean_text) == 0:
            clean_text += word
        else:
            if word[0].isalpha() or word[0].isdigit():
                clean_text += " " + word
            else:
                if word == ")" or word == "(":
                    clean_text += " " + word
                else:
                    clean_text += word

    if clean_text[-1] != ".":
        clean_text += "."
    return clean_text

'''--------------------------------------abstract generation--------------------------------------------'''


def absGen(fileID, df_selected, category_label):  # Abstract Section generation



    topic = Survey_Topic_dict[fileID]

    print(topic)

    summ = None
    # refs = readReference()
    refs = readReference(df_selected)
    newRefs, aiRefs, abslist, introlist = sentText(refs)

    newabs, newintro = extractTopicRef(topic, abslist, introlist)
    if len(newabs) == 0 or len(newintro) == 0:
        newabs = abslist
        newintro = introlist

    newabs, newintro = extractTopicSent(newabs, newintro)

    # print(len(newabs))
    sents = []
    for abstext in newabs:
        sents += abstext
    # sents = topicStartSentence(topic, sents)
    # print(sents)
    text = combineSentence(sents)
    summ = summarize(text, words=100)
    print(summ)
    clean_sum = cleanSummary(summ)
    clean_sum = cleanComma(clean_sum)
    # categories = getReferenceCategories(fileID)  # return the categories of exisitng works
    categories = category_label
    template = "In this survey, we conduct a comprehensive overview of " + Survey_dict[
        fileID].lower() + "." + " We classify existing methods into " + no2word[
                   len(categories)] + " categories: "
    cate_des = ""
    for i in range(len(categories)):
        if i != len(categories) - 1:
            cate_des += categories[i]
            cate_des +=', '
        else:
            cate_des += ' and '
            cate_des += categories[i]


    template += cate_des + "."
    # abstract = clean_sum + " " + template
    return clean_sum, template


'''------------------------------------introduction generation---------------------------------------------'''


def cleanIntroText(text):
    texts = text.strip().split("\n")
    sents = []
    for t in texts:
        sents.append(sent_tokenize(t))
    return sents


def sentIntroText(refs):
    newRefs = []
    aiRefs = []
    abslist = []
    introlist = []
    for ref in refs:
        newAbs = [word_tokenize(sent) for sent in cleanText(ref[0])]
        newIntro = []
        for para in cleanIntroText(ref[1]):
            cur_para = [word_tokenize(sent) for sent in para]
            newIntro.append(cur_para)
        newRefs.append([newAbs, newIntro])
        aiRefs.append(newAbs + newIntro)
        abslist.append(newAbs)
        introlist.append(newIntro)

    return newRefs, aiRefs, abslist, introlist


def extractTopicIntro(introlist):
    newintro = []
    for il in introlist:  # il refer to each introduction
        newil = []
        for i in il:  # i refer to each paragraph
            newit = []
            for sent in i:  # sent refer to each sentence
                if "we" not in sent and "We" not in sent and "this" not in sent and "This" not in sent and "paper" not in sent and "papers" not in sent:  # and "paper" not in sent
                    newit.append(sent)
                else:
                    if len(newit) > 0:
                        newil.append(newit)
                        newit = []
                    break
        if len(newil) > 0:
            newintro.append(newil)
    return newintro


def getIntroPara(newintro, NO):
    intros = []
    for intro in newintro:
        if len(intro) < 2 or len(intro) <= NO:
            continue
            # intros.append(extractTopicIntro(intro))
        else:
            intros.append(intro[NO])

    return intros


def combinePara(intros):
    texts = []
    for intro in intros:
        text = combineSentence(intro)
        texts.append(text)
    introparas = " ".join(texts)
    return introparas


def extractSimilarity(similarity, sent_len):

    weight = [0.] * sent_len
    for i, j, s in similarity:
        weight[i] += s
        weight[j] += s

    idx = sent_len - np.argsort(np.array(weight))
    sent_id = [i for i in range(len(idx)) if idx[i] < 20]
    return sent_id


def selectIntroSentences(intros):
    sentences = []
    for intro in intros:
        sentences += intro
    similarity = testSim(sentences)
    similarity = sorted(similarity, key=lambda x: x[2], reverse=True)
    sent_id = extractSimilarity(similarity, len(sentences))
    sentences = [sentences[i] for i in sent_id]
    return sentences

'''
def introGen(fileID, df_selected, category_label, category_description):  # Introduction Section generation

    topic = Survey_Topic_dict[fileID]
    One = False
    introNO = 0
    refs = readReference(df_selected)
    newRefs, aiRefs, abslist, introlist = sentIntroText(refs)
    newabs, newintro = extractTopicRef(topic, abslist, introlist)


    if  len(newintro) == 0:

        newintro = introlist



    newintro = extractTopicIntro(newintro)

    if len(newintro) == 0:
        newintro = introlist



    if One:
        intros = getIntroPara(newintro, introNO)

    else:
        intros = []
        for intro in newintro:
            its = []
            for i in intro:
                its += i
            intros.append(its)
    sentences = selectIntroSentences(intros)
    if len(sentences) == 0:
        sentences = intros

    introPara = combineSentence(sentences)

    if One:
        summ = summarize(introPara, words=50)
    else:
        summ = summarize(introPara, words=400)

    if summ is None or len(summ) == 0:
        summ = introPara

    sent = sent_tokenize(summ)

    summ = " ".join([s.strip() for s in sent])
    summ = summ.replace("\n", "")
    summ = summ.replace("< NO >", "")
    summ = summ.replace("< NO>", "")
    summ = summ.replace("<NO >", "")

    summ = cleanComma(summ)
    categories = category_label
    des = category_description
    # types, des = getReferenceTypes(fileID, mask)

    template = "In this paper, we reviewed existing works and classify them into " + no2word[
        len(categories)] + " types namely: "

    keywords_des = ""
    for i in range(len(categories)):
        if i != len(categories) - 1:
            keywords_des += categories[i]
            keywords_des += ', '
        else:
            keywords_des += ' and '
            keywords_des += categories[i]

    keywords_des += "."
    template += keywords_des
    types_des = " ".join(des)
    template += " " + types_des
    introduction = summ + "\n" + template
    conjunction = " In the next section, we will introduce existing works in each types with details."
    introduction += conjunction
    # introduction = introduction.replace("< NO >", "")
    return introduction
'''

def clean_wzy(text):
    text = text.replace("\n", "")
    text = text.replace("< NO >", "")
    text = text.replace("< NO>", "")
    text = text.replace("<NO >", "")
    # text = ' '.join(extractTopicIntro(text))
    # text = cleanComma(text)
    return text


# def introGen(fileID, df_selected, category_label, category_description):
#
#
#     # refs = readReference(df_selected)
#     # newRefs, aiRefs, abslist, introlist = sentIntroText(refs)
#     # newabs, newintro = extractTopicRef(topic, abslist, introlist)
#     #
#     # if len(newintro) == 0:
#     #     newintro = introlist
#     #
#     # newintro = extractTopicIntro(newintro)
#     # if len(newintro) == 0:
#     #     newintro = introlist
#
#     ## ==== Background begin ====
#     abs_list = '\n'.join([' '.join(i.split(' ')[:50]) for i in df_selected.abstract])
#     intro_list = ' '.join([i.split('\n')[0] for i in df_selected.intro])
#     text = abs_list + ' ' + intro_list
#     background = summarize(text, words=150)
#     background = clean_wzy(background)
#     ## ==== Background end ====
#
#     import nltk
#     from nltk.tokenize import WordPunctTokenizer
#     sen_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
#     whole_text = [i for i in df_selected.abstract] + [i for i in df_selected.intro]
#     whole_text = sen_tokenizer.tokenize(' '.join(whole_text))
#
#     ## ==== Problem_def begin ====
#     topic = Survey_Topic_dict[fileID]
#     topic_sents = []
#     for t in topic:
#         for sent in whole_text:
#             if t in sent.split():
#                 topic_sents.append(sent)
#     topic_sents = set(topic_sents)
#     topic_introduction = ' '.join([i for i in topic_sents])
#     topic_intro = summarize(topic_introduction, words=100)
#     topic_intro = clean_wzy(topic_intro)
#     ## ==== Problem_def end ====
#
#     ## ==== Challenges begin ====
#     key_words = ['challenge', 'challenges', 'challenging', 'difficult', 'difficulties']
#     challenges = ' '.join([i for i in whole_text if (key_words[0] in i.split() or key_words[1] in i.split() or
#                                                              key_words[2] in i.split() or key_words[3] in i.split() or
#                                                              key_words[4] in i.split())])
#     # challenges = ' '.join(extractTopicIntro(challenges))
#     challenges = summarize(challenges, words=100)
#     challenges = clean_wzy(challenges)
#     begin = 'There are several challenges of ' + Survey_dict[fileID] + '.'
#     challenges = begin + challenges
#     ## ==== Challenges end ====
#
#
#     ## ==== Taxonomy begin ====
#     categories = category_label
#     des = category_description
#
#     template = "In this paper, we reviewed existing works and classify them into " + no2word[
#         len(categories)] + " types namely: "
#
#     keywords_des = ""
#     for i in range(len(categories)):
#         if i != len(categories) - 1:
#             keywords_des += categories[i]
#             keywords_des += ', '
#         else:
#             keywords_des += ' and '
#             keywords_des += categories[i]
#
#     keywords_des += "."
#     template += keywords_des
#     types_des = " ".join(des)
#     template += " " + types_des
#     taxonomy = template
#     # ## ==== Taxonomy end ====
#
#     conjunction = "In the following section, we will introduce existing works in each category in detail."
#
#
#     introduction = background.capitalize() + '<br/><br/>' + topic_intro.capitalize() + '<br/><br/>' + challenges.capitalize() \
#                    + '<br/><br/>' + taxonomy.capitalize() + '<br/><br/>' + conjunction
#
#     return introduction


def introGen_supervised(fileID, df_selected, category_label, category_description, survey_sections):
    
    output_intro_text = ""
    
    abs_str_list = [' '.join(abs_i.split(' ')[:300]) for abs_i in df_selected.abstract]
    #abs_str = '\n'.join([' '.join(abs_i.split(' ')[:300]) for abs_i in df_selected.abstract])
    #intro_str = ' '.join([intro_i.split('\n')[0] for intro_i in df_selected.intro])
    #text = abs_str + ' ' + intro_str

    ## ==== classify begin ====
    classifier_dict = MODEL_PATH_DICT
    train_set_name = "CSAbstruct"
    summarizer_name = "bert_base_3sent_classify"
    
    Classifier = Text_Classifier(classifier_dict,train_set_name,summarizer_name)
    background_sent_list,method_sent_list = Classifier.classify_sent(abs_str_list, batch_size=16, input_max_length = 1024)

    background_sent_str=' '.join(background_sent_list)
    method_sent_str=' '.join(method_sent_list)

    ## ==== Background begin ====
    output_background_text = ""
    summarizer_dict = MODEL_PATH_DICT
    
    if "background" in survey_sections.keys():
        section_name = 'background'
        summarizer_name = "asg_pegasus"
        #Summarizer = Text_Summarizer(summarizer_dict).load_model(section_name,summarizer_name)
        Summarizer = Text_Summarizer(summarizer_dict,section_name,summarizer_name)
        output_background_text = Summarizer.summarize(background_sent_str, output_len=300, input_max_length = 1024)
        output_background_text = output_background_text + '\n \n' 
    output_intro_text = output_intro_text + output_background_text

    ## ==== Method begin ====
    output_method_text = ""
    if "method" in survey_sections.keys():
        section_name = 'method'
        summarizer_name = "asg_pegasus"
        Summarizer = Text_Summarizer(summarizer_dict,section_name,summarizer_name)
        output_method_text = Summarizer.summarize(method_sent_str, output_len=300, input_max_length = 1024)
        output_method_text = output_method_text + '\n \n' 
    output_intro_text = output_intro_text + output_method_text

    ## conjunction
    conjunction = ""
    if "reminder" in survey_sections.keys():
        conjunction = "In the following section, we will introduce existing works in each category in detail."
    output_intro_text = output_intro_text + conjunction

    #pdb.set_trace()
    #output_intro_text = output_background_text + '<br/><br/>' + output_method_text + '<br/><br/>' + conjunction
    #introduction = background.capitalize() + '<br/><br/>' + topic_intro.capitalize() + '<br/><br/>' + challenges.capitalize() \
    #               + '<br/><br/>' + taxonomy.capitalize() + '<br/><br/>' + conjunction

    return output_intro_text

def introGen(fileID, df_selected, category_label, category_description, survey_sections):

    introduction = ""
    if "background" in survey_sections.keys():

        ## ==== Background begin ====
        abs_list = '\n'.join([' '.join(i.split(' ')[:50]) for i in df_selected.abstract])
        intro_list = ' '.join([i.split('\n')[0] for i in df_selected.intro])
        text = abs_list + ' ' + intro_list
        background = summarize(text, words=150)
        background = clean_wzy(background)
        ## ==== Background end ====

        introduction += background.capitalize() + '\n \n'

    if "method" in survey_sections.keys():
        import nltk
        from nltk.tokenize import WordPunctTokenizer
        sen_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        whole_text = [i for i in df_selected.abstract] + [i for i in df_selected.intro]
        whole_text = sen_tokenizer.tokenize(' '.join(whole_text))

        ## ==== Problem_def begin ====
        topic = Survey_Topic_dict[fileID]
        topic_sents = []
        for t in topic:
            for sent in whole_text:
                if t in sent.split():
                    topic_sents.append(sent)
        topic_sents = set(topic_sents)
        topic_introduction = ' '.join([i for i in topic_sents])
        topic_intro = summarize(topic_introduction, words=100)
        topic_intro = clean_wzy(topic_intro)
        ## ==== Problem_def end ====

        introduction += topic_intro.capitalize() + '\n \n'

    if "reminder" in survey_sections.keys():
        conjunction = "In the following section, we will introduce existing works in each category in detail."
        introduction += conjunction

    return introduction



def conclusionGen(fileID, category_label):  # Conclusion section generation
    categories = category_label  # return the categories of exisitng works
    template = "In this survey, we conduct a comprehensive overview of " + Survey_dict[
        fileID].lower() + "." + " We provide a taxonomy which groups the researchs of " + Survey_dict[
                   fileID].lower() + " into " + no2word[len(categories)] + " categories: "
    keywords_des = ""
    for i in range(len(categories)):
        if i != len(categories) - 1:
            keywords_des += categories[i]
            keywords_des += ', '
        else:
            keywords_des += ' and '
            keywords_des += categories[i]

    keywords_des += "."
    template += keywords_des
    return template


def methodologyGen(fileID, df_selected, category_label, category_description):
    types, des = category_label, category_description
    refs, ref_des = getReferenceDes(df_selected)

    template = "We reviewed existing works and classify them into " + no2word[
        len(types)] + " types namely: "


    keywords_des = ""
    for i in range(len(types)):
        if i != len(types) - 1:
            keywords_des += types[i]
            keywords_des += ', '
        else:
            keywords_des += ' and '
            keywords_des += types[i]

    keywords_des += "."


    template += keywords_des
    types_des = " ".join(des)
    template += " " + types_des + "\n"

    proceeding = template
    detailed_des = []
    for i, ref_d in enumerate(ref_des):
        tmp_dict = {}
        content = "For " + types[i] + ", there are several existing works. "
        content += " ".join(ref_d) + "\n"

        tmp_dict['subtitle'] = types[i].title()
        tmp_dict['content'] = content
        detailed_des.append(tmp_dict)

    # refs = "\n".join(refs)
    return proceeding, detailed_des


def generateSurvey(fileID, mask):
    abstract = absGen(fileID, mask)
    introduction = introGen(fileID, mask)
    methodology, reference = methodologyGen(fileID, mask)
    conclusion = conclusionGen(fileID, mask)
    survey = abstract + "\n\n" + introduction + "\n\n" + methodology + "\n\n" + conclusion + "\n\n" + reference
    return abstract, introduction, methodology, conclusion, reference, survey


if __name__ == "__main__":
    mask = [1] * 500
    abstract, introduction, methodology, conclusion, reference, survey = generateSurvey("2742488", mask)
    print(survey)
