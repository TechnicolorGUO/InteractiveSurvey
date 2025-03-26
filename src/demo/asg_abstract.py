import os


class AbstractGenerator:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        
    def generate(self, title, intro, mode='lora'):
        if mode == 'lora' or mode == 'test':
            if mode == 'lora':
                self.pipeline.model.set_adapter("abstract")

            system_prompt = f'''You are a helpful assistant that help to generate the abstract of the survey paper given the survey title and survey introduction.'''
            # user_prompt = {"survey_title":survey_title, "claims":cluster_with_claims}
            user_prompt = f'''Help me to generate the abstract of a survey paper given the title: *{title}*, and and the introduction:{intro}'''

            messages = [
                {"role": "system", "content": system_prompt}, 
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content":"Abstract: This survey "}
            ]

            outputs = self.pipeline(
                messages,
                max_new_tokens=4096,
            )
            result = outputs[0]["generated_text"][-1]['content']
            return result
        else:
            raise ValueError('mode not supported')

if __name__ == '__main__':
    from transformers import pipeline
    import torch
    import transformers

    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    Global_pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        token = os.getenv('HF_API_KEY'),
        device_map="auto",
    )
    Global_pipeline.model.load_adapter(peft_model_id = "technicolor/llama3.1_8b_outline_generation", adapter_name="outline")
    Global_pipeline.model.load_adapter(peft_model_id ="technicolor/llama3.1_8b_abstract_generation", adapter_name="abstract")
    Global_pipeline.model.load_adapter(peft_model_id ="technicolor/llama3.1_8b_conclusion_generation", adapter_name="conclusion")
    title = "A Survey of Large Language Models"
    intro = '''L
ANGUAGE is a prominent ability in human beings to
express and communicate, which develops in early
childhood and evolves over a lifetime [3, 4]. Machines,
however, cannot naturally grasp the abilities of understanding and communicating in the form of human language,
unless equipped with powerful artificial intelligence (AI)
algorithms. It has been a longstanding research challenge
to achieve this goal, to enable machines to read, write, and
communicate like humans [5].
Technically, language modeling (LM) is one of the major
approaches to advancing language intelligence of machines.
In general, LM aims to model the generative likelihood
of word sequences, so as to predict the probabilities of
future (or missing) tokens. The research of LM has received
extensive attention in the literature, which can be divided
into four major development stages:
• Statistical language models (SLM). SLMs [6–9] are developed based on statistical learning methods that rose in
the 1990s. The basic idea is to build the word prediction
model based on the Markov assumption, e.g., predicting the
next word based on the most recent context. The SLMs with
a fixed context length n are also called n-gram language
models, e.g., bigram and trigram language models. SLMs
have been widely applied to enhance task performance
in information retrieval (IR) [10, 11] and natural language
processing (NLP) [12–14]. However, they often suffer from
the curse of dimensionality: it is difficult to accurately
estimate high-order language models since an exponential
number of transition probabilities need to be estimated.
Thus, specially designed smoothing strategies such as backoff estimation [15] and Good–Turing estimation [16] have
been introduced to alleviate the data sparsity problem.
• Neural language models (NLM). NLMs [1, 17, 18] characterize the probability of word sequences by neural networks,
e.g., multi-layer perceptron (MLP) and recurrent neural networks (RNNs). As a remarkable contribution, the work in
[1] introduced the concept of distributed representation of
words and built the word prediction function conditioned
on the aggregated context features (i.e., the distributed
word vectors). By extending the idea of learning effective
features for text data, a general neural network approach
was developed to build a unified, end-to-end solution for
various NLP tasks [2]. Furthermore, word2vec [19, 20] was
proposed to build a simplified shallow neural network
for learning distributed word representations, which were
demonstrated to be very effective across a variety of NLP
tasks. These studies have initiated the use of language
models for representation learning (beyond word sequence
modeling), having an important impact on the field of NLP.
• Pre-trained language models (PLM). As an early attempt, ELMo [21] was proposed to capture context-aware
word representations by first pre-training a bidirectional
LSTM (biLSTM) network (instead of learning fixed word
representations) and then fine-tuning the biLSTM network
according to specific downstream tasks. Furthermore, based
on the highly parallelizable Transformer architecture [22]
with self-attention mechanisms, BERT [23] was proposed by
pre-training bidirectional language models with specially
designed pre-training tasks on large-scale unlabeled corpora. These pre-trained context-aware word representations
are very effective as general-purpose semantic features,
which have largely raised the performance bar of NLP
tasks. This study has inspired a large number of follow-up
work, which sets the “pre-training and fine-tuning” learning
paradigm. Following this paradigm, a great number of studies on PLMs have been developed, introducing either different architectures [24, 25] (e.g., GPT-2 [26] and BART [24]) or
improved pre-training strategies [27–29]. In this paradigm, it
often requires fine-tuning the PLM for adapting to different
downstream tasks.
• Large language models (LLM). Researchers find that
scaling PLM (e.g., scaling model size or data size) often
leads to an improved model capacity on downstream tasks
(i.e., following the scaling law [30]). A number of studies
have explored the performance limit by training an ever
larger PLM (e.g., the 175B-parameter GPT-3 and the 540Bparameter PaLM). Although scaling is mainly conducted
in model size (with similar architectures and pre-training
tasks), these large-sized PLMs display different behaviors
from smaller PLMs (e.g., 330M-parameter BERT and 1.5Bparameter GPT-2) and show surprising abilities (called emergent abilities [31]) in solving a series of complex tasks. For
example, GPT-3 can solve few-shot tasks through in-context
learning, whereas GPT-2 cannot do well. Thus, the research
community coins the term “large language models (LLM)”
1
for these large-sized PLMs [32–35], which attract increasing
research attention (See Figure 1). A remarkable application
of LLMs is ChatGPT2
that adapts the LLMs from the GPT
series for dialogue, which presents an amazing conversation
ability with humans. We can observe a sharp increase of the
arXiv papers that are related to LLMs after the release of
ChatGPT in Figure 1.
As discussed before, language model is not a new technical concept specially for LLMs, but has evolved with the
advance of artificial intelligence over the decades. Early language models mainly aim to model and generate text data,
while latest language models (e.g., GPT-4) focus on complex
task solving. From language modeling to task solving, it is an
important leap in scientific thinking, which is the key to
understand the development of language models in the research history. From the perspective of task solving, the four
generations of language models have exhibited different levels of model capacities. In Figure 2, we describe the evolution process of language models in terms of the task solving
capacity. At first, statistical language models mainly assisted
in some specific tasks (e.g., retrieval or speech tasks), in
which the predicted or estimated probabilities can enhance
the performance of task-specific approaches. Subsequently,
neural language models focused on learning task-agnostic
representations (e.g., features), aiming to reduce the efforts
for human feature engineering. Furthermore, pre-trained
language models learned context-aware representations that
can be optimized according to downstream tasks. For the
latest generation of language model, LLMs are enhanced by
exploring the scaling effect on model capacity, which can be
considered as general-purpose task solvers. To summarize,
in the evolution process, the task scope that can be solved
by language models have been greatly extended, and the
task performance attained by language models have been
significantly enhanced.
In the existing literature, PLMs have been widely discussed and surveyed [36–39], while LLMs are seldom reviewed in a systematic way. To motivate our survey, we first
highlight three major differences between LLMs and PLMs.
First, LLMs display some surprising emergent abilities that
may not be observed in previous smaller PLMs. These abilities are key to the performance of language models on complex tasks, making AI algorithms unprecedently powerful
and effective. Second, LLMs would revolutionize the way
that humans develop and use AI algorithms. Unlike small
PLMs, the major approach to accessing LLMs is through
the prompting interface (e.g., GPT-4 API). Humans have to
understand how LLMs work and format their tasks in a way
that LLMs can follow. Third, the development of LLMs no
longer draws a clear distinction between research and engineering. The training of LLMs requires extensive practical
experiences in large-scale data processing and distributed
parallel training. To develop capable LLMs, researchers
have to solve complicated engineering issues, working with
engineers or being engineers.
Nowadays, LLMs are posing a significant impact on
the AI community, and the advent of ChatGPT and GPT-4
leads to the rethinking of the possibilities of artificial general
intelligence (AGI). OpenAI has published a technical article
entitled “Planning for AGI and beyond”, which discusses
the short-term and long-term plans to approach AGI [40],
and a more recent paper has argued that GPT-4 might be
considered as an early version of an AGI system [41]. The
research areas of AI are being revolutionized by the rapid
progress of LLMs. In the field of NLP, LLMs can serve as a
general-purpose language task solver (to some extent), and
the research paradigm has been shifting towards the use
of LLMs. In the field of IR, traditional search engines are
challenged by the new information seeking way through AI
chatbots (i.e., ChatGPT), and New Bing3 presents an initial
attempt that enhances the search results based on LLMs. In
the field of CV, the researchers try to develop ChatGPT-like
vision-language models that can better serve multimodal
dialogues [42–45], and GPT-4 [46] has supported multimodal input by integrating the visual information. This new
wave of technology would potentially lead to a prosperous
ecosystem of real-world applications based on LLMs. For
instance, Microsoft 365 is being empowered by LLMs (i.e.,
Copilot) to automate the office work, and OpenAI supports
the use of plugins in ChatGPT for implementing special
functions.
Despite the progress and impact, the underlying principles of LLMs are still not well explored. Firstly, it is
mysterious why emergent abilities occur in LLMs, instead of
smaller PLMs. As a more general issue, there lacks a deep,
detailed investigation of the key factors that contribute to
the superior abilities of LLMs. It is important to study when
and how LLMs obtain such abilities [47]. Although there are
some meaningful discussions about this problem [31, 47],
more principled investigations are needed to uncover the
“secrets“ of LLMs. Secondly, it is difficult for the research
community to train capable LLMs. Due to the huge demand of computation resources, it is very costly to carry
out repetitive, ablating studies for investigating the effect
of various strategies for training LLMs. Indeed, LLMs are
mainly trained by industry, where many important training
details (e.g., data collection and cleaning) are not revealed
to the public. Thirdly, it is challenging to align LLMs with
human values or preferences. Despite the capacities, LLMs
are also likely to produce toxic, fictitious, or harmful contents. It requires effective and efficient control approaches
to eliminating the potential risk of the use of LLMs [46].
Faced with both opportunities and challenges, it needs
more attention on the research and development of LLMs. In
order to provide a basic understanding of LLMs, this survey
conducts a literature review of the recent advances in LLMs
from four major aspects, including pre-training (how to pretrain a capable LLM), adaptation (how to effectively adapt
pre-trained LLMs for better use), utilization (how to use
LLMs for solving various downstream tasks) and capability
evaluation (how to evaluate the abilities of LLMs and existing
empirical findings). We thoroughly comb the literature and
summarize the key findings, techniques, and methods of
LLMs. For this survey, we also create a GitHub project
website by collecting the supporting resources for LLMs, at
the link https://github.com/RUCAIBox/LLMSurvey. We
are also aware of several related review articles on PLMs
or LLMs [32, 36, 38, 39, 43, 48–54]. These papers either
discuss PLMs or some specific (or general) aspects of LLMs.
Compared with them, we focus on the techniques and
methods to develop and use LLMs and provide a relatively
comprehensive reference to important aspects of LLMs.
The remainder of this survey is organized as follows:
Section 2 introduces the background for LLMs and the evolution of GPT-series models, followed by the summarization
of available resources for developing LLMs in Section 3.
Sections 4, 5, 6, and 7 review and summarize the recent
progress from the four aspects of pre-training, adaptation,
utilization, and capacity evaluation, respectively. Then, Section 8 discusses the practical guide for prompt design,
and Section 9 reviews the applications of LLMs in several
representative domains. Finally, we conclude the survey in
Section 10 by summarizing the major findings and discuss
the remaining issues for future work.
'''


    abstract_generator = AbstractGenerator(Global_pipeline)
    with_lora = abstract_generator.generate(title, intro, mode='lora')
    with_test = abstract_generator.generate(title, intro, mode='test')