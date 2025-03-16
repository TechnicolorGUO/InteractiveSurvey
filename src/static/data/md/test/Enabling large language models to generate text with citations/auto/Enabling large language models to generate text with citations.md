# Enabling Large Language Models to Generate Text with Citations  

Tianyu Gao Howard Yen Jiatong Yu Danqi Chen  

Department of Computer Science & Princeton Language and Intelligence Princeton University {tianyug,hyen,jiatongy,danqic}@cs.princeton.edu  

# Abstract  

Large language models (LLMs) have emerged as a widely-used tool for information seeking, but their generated outputs are prone to hallucination. In this work, our aim is to allow LLMs to generate text with  citations , improving their factual correctness and verifiability. Existing work mainly relies on commercial search engines and human evaluation, making it challenging to reproduce and compare different modeling approaches. We propose  ALCE , the first benchmark for  A utomatic  L LMs’  C itation E valuation. ALCE collects a diverse set of questions and retrieval corpora and requires building end-to-end systems to retrieve supporting evidence and generate answers with citations. We develop automatic metrics along three dimensions—fluency, correctness, and citation quality—and demonstrate their strong correlation with human judgements. Our experiments with state-of-the-art LLMs and novel prompting strategies show that current systems have considerable room for improvement—For example, on the ELI5 dataset, even the best models lack complete citation support 50% of the time. Our analyses further highlight promising future directions, including developing better retrievers, advancing long-context LLMs, and improving the ability to synthesize information from multiple sources.  

# 1 Introduction  

Large language models (LLMs;  Brown et al. ,  2020 ; OpenAI ,  2023 ) have gained increasing popularity as a tool for information seeking. While they generate engaging and coherent responses, their outputs are prone to hallucination and often contain factually incorrect information ( Ji et al. ,  2023 ). This makes it harder for users to trust and verify LLMgenerated outputs without any supporting evidence. In this work, we study a new generation paradigm for LLMs, in which we require LLMs to provide  citations  to one or a few text passages for any statement they generate (Figure  1 ). Incorporating citations brings several benefits: (1) users can easily verify LLMs’ claims with the provided citations; (2) LLMs can generate text that faithfully follows cited passages, which has the promise to improve correctness and alleviate hallucination.  

![](images/bb65b835d9bf1d55f8bf588be42810376c5704e7ea81e09da51112a4c556625c.jpg)  
Figure 1: The task setup of ALCE. Given a question, the system generates text while providing  citing passages  from a large retrieval corpus. Each statement may contain multiple citations (e.g.,  [1][2] ).  

Multiple commercial systems have adopted this paradigm: Bing Chat 2   and perplexity.ai 3   respond to user questions in natural language with references to Web pages.  Nakano et al.  ( 2021 );  Menick et al. ( 2022 ) share a similar motivation, but they mainly experiment with commercial search engines and closed-source models, making their results difficult to evaluate. Retrieval-augmented LMs ( Borgeaud et al. ,  2022 ;  Izacard et al. ,  2022 ) incorporate retrieved passages during both training and inference, but do not guarantee faithfulness to retrieved passages or explicitly provide citations. Additionally, previous studies mostly rely on human evaluation ( Nakano et al. ,  2021 ;  Menick et al. ,  2022 ; Liu et al. ,  2023 ), which is expensive and difficult to reproduce. We argue that the absence of automated evaluation hinders the advances of such systems.  

![](images/c5194c86c88f91ed59db63d71313f5ec371b568c9cb7844f874d7bd9ce94a77d.jpg)  
Table 1: The three datasets used in our ALCE benchmark. These datasets cover a wide range of question types and the corresponding corpora span from Wikipedia to Web-scale document collection.  

We present  ALCE , the first reproducible benchmark for  automatically  evaluating LLMs’ generations with citations. ALCE assumes a naturallanguage question and a retrieval corpus, and requires building end-to-end systems to retrieve relevant passages from the corpus, generate a response to the question, and cite corresponding supporting passages. We compile three datasets that cover different types of questions and corpora— ASQA ( Stelmakh et al. ,  2022 ), QAMPARI ( Rubin et al. ,  2022 ), and ELI5 ( Fan et al. ,  2019 )—as shown in Table  1 . Different from previous benchmarks ( Lee et al. ,  2019 ;  Bohnet et al. ,  2022 ), ALCE evaluates long-text generation, focusing on automatically evaluating citation quality, and allows citing  multiple  passages for individual statements.  

We design automatic evaluation methods in three dimensions:  fluency ,  correctness , and  citation quality . Specifically, we use MAUVE ( Pillutla et al. ,  2021 ) to measure fluency, propose tailored correctness metrics for each dataset, and adopt a natural language inference (NLI) model ( Honovich et al. ,  2022 ) to measure citation quality. We showcase how the three dimensions together contribute to a robust evaluation, preventing systems from exploiting shortcuts. Additionally, we conduct human evaluation and demonstrate a strong correlation with our automatic metrics.  

We experiment on multiple systems with stateof-the-art LLMs and retrievers and also propose novel prompting strategies to synthesize retrieved text into text generation. Although all systems are capable of providing fluent and coherent responses, there remains substantial room for improvement in terms of correctness and citation quality: For example, on the ELI5 dataset, around 50% generations of our ChatGPT and GPT-4 baselines are not fully supported by the cited passages. Additionally, we find that (1) a closed-book model (generating answers without accessing any retrieved documents)  

with post-hoc citing achieves good correctness but much worse citation quality; (2) although interactive retrieval approaches ( Yao et al. ,  2023 ;  Schick et al. ,  2023 ) offer more flexibility in when/what to retrieve, they do not improve the performance on this challenging benchmark; (3) summarizing the retrieved passages in a shorter text improves correctness but not citation quality; (4) reranking multiple generations boosts citation quality measured by human evaluation; (5) incorporating more retrieved passages in context does not help ChatGPT but improves GPT-4 performance.  

Our extensive analyses highlight three major challenges of building LLMs to generate text with citations: (1) the retrieval quality is crucial to the final performance and has substantial room for improvement; (2) LLMs’ limited context window restricts the number of passages they can incorporate; (3) current LLMs struggle to synthesize multiple documents in context without being distracted by irrelevant ones, although better instruction tuning brings significant improvement. These challenges pose promising research directions for developing better systems integrating retrieval and LLMs.  

# 2 Task Setup and Datasets  

Our task is formalized as follows: Given a query q  and a corpus of text passages  D , the system is required to return an output of  n  statements  s 1 , ...,  s n , and each statement  S , which consists  s i cites a list of passages  C i  =  { c i, , c i, , . . . } 4 , where c  ∈D . In this work, we segment LLMs’ output into statements by sentence boundaries.   While LLMs may include sentences that do not require a citation, such as “ I’m happy to help ”, we observe that almost all sentences that LLMs output provide valuable information and require citations, similar to findings in  Liu et al.  ( 2023 ). In this work, citations are enclosed by box brackets such as  [1][2] .  

following previous works on open-domain question We divide the corpus  D  into 100-word passages answering ( Karpukhin et al. ,  2020 ;  Petroni et al. , 2021 ;  Piktus et al. ,  2021 ), in contrast to commercial systems like Bing Chat, which cite entire Web pages. We take 100-word passages because it is easier for humans to verify, and allows for more retrieved passages to fit in LLMs’ limited context.  

We choose QA datasets so that (1) they contain factual questions, in which references are important; (2) questions require long-text answers that cover multiple aspects; (3) answering the questions requires synthesizing multiple sources. We select three datasets (Table  1 ) and introduce them below. See § B  for additional statistics.  

ASQA  ( Stelmakh et al. ,  2022 ) is a long-form factoid dataset. As shown in Figure  1 , each question is an ambiguous question from AmbigQA ( Min et al. , 2020 ) that requires multiple short answers to cover different aspects, and the dataset provides a longform answer that covers all short answers. Since most questions can be answered by Wikipedia, we use the 2018-12-20 Wikipedia snapshot as  D .  

QAMPARI  ( Rubin et al. ,  2022 ) is a factoid QA dataset constructed from Wikipedia, where the answer is a list of entities that are drawn from different passages. Same as ASQA, we use the 2018-12- 20 Wikipedia as the corpus.  

ELI5  ( Fan et al. ,  2019 ) is a long-form QA dataset built on the Reddit forum “Explain Like I’m Five”.   Most ELI5 questions are how/why/what questions that require long answers and multiple passages as evidence. Due to the diverse topics discussed in the questions, we use Sphere ( Piktus et al. , 2021 )—a filtered version of Common Crawl 7 —as the corpus. The ELI5 dataset is widely used in related work due to its challenging nature ( Nakano et al. ,  2021 ;  Menick et al. ,  2022 ;  Liu et al. ,  2023 ).  

We randomly select 1,000 examples from the development set of each dataset for ALCE. Our benchmark primarily assesses the citation capabilities of existing LLMs and does not provide training data, as there are no available examples that provide supervision for citations in these datasets.  

# 3 Automatic Evaluation  

Our benchmark measures the following three dimensions of system responses:  

•  Fluency : whether the model’s generated text is fluent and coherent.   
•  Correctness : whether the answer is accurate and covers all aspects of interest.   
•  Citation quality : whether the answer is well supported by the cited passages and no irrelevant passages are cited.  

In the following, we present automatic metrics for each dimension and discuss why the combination of the three metrics provides a robust evaluation.  

# 3.1 Fluency  

We use MAUVE ( Pillutla et al. ,  2021 ) to evaluate the fluency of the output (§ C ). We deploy MAUVE for ASQA and ELI5 and omit it for QAMPARI, as QAMPARI only requires a list of short answers as the response and LLMs consistently adhere to the format in our experiments. As MAUVE is sensitive to output length and text style, and most LLMs are capable of producing fluent text, we mainly employ it as a sanity check as long as the MAUVE scores are high enough.  

# 3.2 Correctness  

Our objective is to measure the informativeness and utility of the generation to the question.  Liu et al. ( 2023 ) propose to directly evaluate  perceived utility by humans, a process difficult to automate. Therefore, we use correctness—whether the response is accurate compared to a ground truth answer—as a proxy. Evaluating the correctness of long-form generation is a challenging task ( Krishna et al. ,  2021 ), and we describe our strategy for each dataset below. Figure  2  illustrates the metrics and we include additional implementation details in § C .  

For ASQA, we follow Stelmakh et al. (2022)and calculate the recall of correct short answers by checking whether the short answers (provided by the dataset) are exact substrings of the generation ( exact match recall ; EM recall).  

For QAMPARI, we follow Rubin et al. (2022)and calculate the  precision  and  recall  of the model prediction, by checking the exact match to the gold answer list. We add one additional adjustment: considering that users often want to know only a few example answers of the question, our evaluation considers recall to be 100% if the prediction includes at least 5 correct answers ( recall-5 ).  

![](images/7d252461bf77626ee99cb7d04344390d526e6f25c7d0200581de484eaf6e6e24.jpg)  
Figure 2: Evaluation of correctness (details in § 3.2 ).  

Unlike ASQA and QAMPARI, the  ELI5  dataset does not provide short entity answers.  Fan et al. ( 2019 ) use ROUGE for evaluation, which does not reflect the correctness well ( Krishna et al. , 2021 ; § A ). Inspired by works in summarization evaluation ( Zhang and Bansal ,  2021 ;  Kamoi et al. ,  2023 ;  Wang et al. ,  2020 ), we use InstructGPT ( text-davinci-003 ;  Ouyang et al. ,  2022 ) to generate three “sub-claims”. Then we use TRUE 8   ( Honovich et al. ,  2022 ), a T5-11B ( Raf- fel et al. ,  2020 ) model fine-tuned on a collection of natural language inference (NLI) datasets, to check whether the model output entails the sub-claims ( claim recall ). TRUE targets factual correctness and has been used by previous works in similar context ( Bohnet et al. ,  2022 ;  Gao et al. ,  2023 ). We demonstrate that claim recall provides a more accurate measure of correctness than existing metrics (more details in § A ).  

# 3.3 Citation Quality  

We evaluate citation qualities using two metrics: (1) citation recall , which determines if the output is entirely supported by cited passages, and (2)  citation precision , which identifies any irrelevant citations. Although we prioritize citation recall as it entails a well-supported and truthful answer, enhancing precision is crucial for better user satisfaction, reducing the need for human review of extraneous passages. Figure  3  provides an illustrated example. We use the NLI model TRUE ( Honovich et al. , 2022 ) again to automatically examine whether the cited passages entail the model generation. We conduct human evaluation (§ 6 ) to demonstrate strong human correlation of our metric.  

![](images/05dc01de900c8f37c65e32e9861afb74d3215b7c5c822f81a872dd194a95db2d.jpg)  
Figure 3: Evaluation of citation quality (details in § 3.3 ). We use an NLI model to verify whether a statement is supported by its citations.  

Citation recall.  We calculate the citation recall of each statement  (0 or 1) and average over all statements in the model response. For each statement  s i , its citation recall is  1  if and only if there is at least one citation ( C i  ̸ =  ∅ ) and  ϕ ( concat ( C i ) , s i ) = 1 , where  ϕ ( premise ,  hypothesis )  is the NLI model that outputs  1  if the premise entails the hypothesis, and  0  otherwise;  concat ( C i )  concatenates all passages in  C i  together (details in § C ). The NLI evaluation is in accordance with the  attributable to identified sources  (AIS) framework ( Rashkin et al. , 2023 ):  ϕ ( concat ( C i ) , s i ) = 1  implies that  s i  is true based solely on concat ( C i ) .  

Citation precision.  Our citation precision evaluation detects citations that are irrelevant, but it does not require citing a minimal set. We follow this design because human writing often cites redundant sources to enhance credibility; human readers may also appreciate multiple citations, especially when it pertains to critical claims such as medical advice.  

We calculate the citation precision for  each citation  (0 or 1) and average over all citations in the Table 2: An example of our V ANILLA  method. Different colors represent  prompt ,  model generation , and <actions> . We also provide two in-context demonstrations before the test example.  

response. We first define if a citation is “irrelevant”. Intuitively, a citation  c i,j  is “irrelevant” if (a) c i,j  itself cannot support  s i  and (b) removing  c i,j does not affect the rest of the citations to support s i . Formally,  c i,j  is “irrelevant” if and only if  

# (a)  ϕ ( c i,j , s i ) = 0 , AND (b)  ϕ ( concat ( C i  \ { c } ) , s i ) = 1 .  

c i,j  has a precision of 1 if  s i  has recall =1  and c i,j  is not irrelevant. For example (Figure  3 ), when  s 3  cites three references  [2][4][5]  and recall =1 ,  [2]  is “irrelevant” if  ϕ ( [2] , s 3 ) = 0  and ϕ ( [4][5] , s 3 ) = 1 . For condition (b) to work, we set recall =1  as a prerequisite for precision = 1 . Note that this algorithm overlooks the scenario when one citation partially supports the statement. We discuss the details in § E .  

# 3.4 ALCE is Robust to Shortcut Cases  

We showcase how the ALCE evaluation is robust to two possible shortcuts in § D : (1) using the top-1 retrieved passage as the response and citing itself, and (2) using the first two sentences of the top-1 passage. Both cases have almost-perfect citation scores, but (1) has low fluency due to its unnaturally long length compared to human answers, and (2) has low correctness due to low coverage.  

# 4 Modeling  

In this section, we discuss three major modeling components for an ALCE system—retrieval, synthesis, and post-editing.  

# 4.1 Retrieval  

We explore simple, off-the-shelf retrievers. We use dense retrievers for Wikipedia, including GTR ( Ni  

et al. ,  2022 ) and DPR ( Karpukhin et al. ,  2020 ); we use BM25 for Sphere. For each question, we retrieve the top-100 passages.  

# 4.2 Synthesis  

We focus on how to prompt an LLM to interact with the retriever, and synthesize and cite the evidence (without fine-tuning internal parameters). One noteworthy challenge is that existing LLMs all have limited context window and thus can only fit a handful of passages.  

V ANILLA .  We simply provide the model with the topk 9   passages and instruct the model to cite accordingly (Table  2 ). We also use in-context learning ( Brown et al. ,  2020 ) and prepend two demonstrations. The complete instruction is in Table  23 .  

S UMM /S NIPPET .  With a 4K context window, we can at most safely fit  k  = 5  passages. As shown in Figure  4 , top-5 retrieved passages can only cover 56.8% percent of the answers in ASQA.  

To tackle this limitation, we propose to provide summaries  or  snippets  of passages instead of the full text (summaries are abstractive but snippets are spans from passages). We acquire summaries and snippets by prompting ChatGPT with instructions (prompts in Table  25  and  26 ).   Then we replace all passages with summaries/snippets. Summaries or snippets significantly reduce the passage length, allowing for more passages to fit in: for ASQA, they reduce passage length by 6 ×  on average.  

Though S UMM /S NIPPET  allows for more retrieved passages, they are lossy compressions. To alleviate this problem, we propose I NTERACT , an interactive prompting scheme to allow the model to check the full text of certain passages. At each step, the model can execute one of three actions: (1) “ Check: Document [1][2] ” to check the full text of the corresponding documents; (2) “ Output: ” to output a statement of the answer; (3) “ End. ” to end the generation. § C  provides more details.  

I NLINE S EARCH . The above methods all display retrieval results at the beginning. In I NLI - NE S EARCH , we allow LLMs to call “search” during the generation process ( Yao et al. ,  2023 ;  Press et al. ,  2022 ;  Jiang et al. ,  2023 ). At each step, the model can execute one of three actions: “ Search:  

Instruction: ...   
<Retrieve for question “...”> Question: When did US break away from England? Search: Declaration of Independence   
<Search the query among the top-100 passages> Document [1](Title: ...) ...   
Output: The United States ...  [1] . <Remove Document [1] from context>   
Search: Treaty of Paris <Search the query among the top-100 passages> Document [3](Title: ...) ...   
Output: The Treaty of Paris ...  [3] . <Remove Document [3] from context>   
End.  

Table 3: An example of I NLINE S EARCH .  

{query} ” to search among the top-100 passages 11 by using GTR; the “ Output ” and “ End ” actions are the same as I NTERACT . For each “ Search ” action, we display the best retrieved passage in the context. The passage is removed after one action to save context space. Table  3  shows an example.  

C LOSED B OOK .  We also add a simple closedbook baseline, where the model is only prompted with the instruction and the question, without any retrieved passages provided. Consequently, this variant does not cite any evidences.  

# 4.3 Post-editing  

In this section we discuss two strategies for refining the output to further improve its quality.  

R ERANK .  We randomly sample  n sample  = 4  responses for each question, and select the best response using the automatic  citation recall  score. we expect R ERANK  to improve the citation quality.  

P OST C ITE .  For each statement, we find the best matching passage among the top-100 retrieved passages using GTR and cite it. We combine this with C LOSED B OOK  in our experiments.  

# 5 Experiments  

We describe experiment details in § C . We use ChatGPT ( gpt-3.5-turbo-0301 ) with a 4K context window for most main experiments and ablations. We also report results with ChatGPT-16K ( gpt3.5-turbo-16k-0613 ) and GPT-4 ( gpt-4-0613 ; 8K context window). For open-source models, we test LLaMA ( Touvron et al. ,  2023a ) and its instruction-tuned versions, including Alpaca ( Taori et al. ,  2023 ), Vicuna ( Chiang et al. ,  2023 ), and Table 4: Experiments on ASQA. For C LOSED B OOK , we use P OST C ITE  to get citations.  k -psg: putting topk  passages from the retrieval results into the context. Chat-13B and Chat-70B refer to LLaMA-2-Chat.  

![](images/91b81f183967acb11c69646cb696704ed72196f8e0c43c33435db70359f15810.jpg)  

Oasst ( Köpf et al. ,  2023 ). They all have a 2K context window. We use short instructions for LLaMA (Table  24 ) to save context budget. Additionally, we test LLaMA-2-Chat, which were also trained to follow instructions ( Touvron et al. ,  2023b ). These models have a context window of 4K tokens, which allows for 5 passages per question.  

# 5.1 Main Results  

We present the main results on three datasets in Table  4 ,  5 , and  6  respectively (full results in § G.6 ). We first note that all models achieve good fluency scores (except some models on ELI5 mainly due to their longer generations). We summarize the main takeaways from the experiments below.  

V ANILLA  achieves strong performance.  Despite its simplicity, V ANILLA  (putting retrieved passages in context) achieves close-to-the-best performance among all prompting strategies.  

Using summaries or snippets improves correctness.  We see a universal trend that S UMM  or S NIP - PET  improves correctness, though on ASQA and ELI5, such an improvement comes at a cost of citation quality due to the lossy compression. Combining I NTERACT  with S UMM /S NIPPET  does not bring improvement, and we hypothesize that checking the full passages offers limited benefit and current LLMs are not proficient in an interactive usage.  

Retrieving text on the fly does not improve performance.  All datasets show that V ANILLA  outperforms I NLINE S EARCH  on citation quality (and on correctness for ASQA and ELI5). By manually examining the examples, we find that it is challenging to ask detailed questions without seeing any passages. To improve I NLINE S EARCH , one may need to provide more context about the questions in advance or encourage the model to call retrievers with more detailed and diverse queries.  

![](images/eb3d40474c0a30561a7ef42a215eedf048de7e187c470dd0a18af09c03566733.jpg)  
Table 5: Experiments on QAMPARI. “Rec.-5”: we set the recall to be 100% if the prediction includes at least 5 correct answers.  

R ERANK  boosts citation quality.  We observe that R ERANK  leads to consistent improvement in citation quality (on ASQA and ELI5). As the automatic scores may be biased in R ERANK , we also conduct human evaluation (§ 6 ) and verify its effectiveness.  

C LOSED B OOK +P OST C ITE  delivers strong correctness but poor citation quality. C LOSED - B OOK  outperforms V ANILLA  in correctness on ELI5 and QAMPARI, and has only a 2% gap on ASQA. However, C LOSED B OOK  cannot provide any citation; when combined with P OST C ITE , the citation quality remains inadequate. For instance, citation recall of C LOSED B OOK +P OST C ITE  is lower than V ANILLA  by 47% on ASQA.  

To understand why C LOSED B OOK  achieves better correctness and why P OST C ITE  cannot deliver satisfying citation quality, we manually examine model outputs and find that: (1) open-book models are easily distracted by irrelevant passages and generate responses with lower correctness, a phenomenon also observed by  Shi et al.  ( 2023 ); (2) C LOSED B OOK  often generates texts that are correct but not similar to any retrieved passages, making it difficult to match a citation post-hoc.  

![](images/1d0cb0c9dc32a1ef033bc384e5ecce56674b5beae41678b368e3a6d81094777e.jpg)  

Table 6: Experiments on ELI5. We use  claim recall for the correctness evaluation. Chat-13B and Chat-70B refer to LLaMA-2-Chat.  

GPT-4 brings limited improvement but is better at using long context.  We evaluate GPT-4 with V ANILLA  and different numbers of passages (more results in § G.6 ). GPT-4 brings consistent (but limited) improvement on correctness, but often at a cost of citation quality. GPT-4 can also incorporate more passages due to its longer context window, which boosts both correctness and citation quality. On the contrary, including more passages with ChatGPT-16K does not improve the results (Table  7 ), suggesting that processing more passages is non-trivial and GPT-4 is better at synthesizing information from its long context than ChatGPT.  

# 5.2 Comparison of Different LLMs  

Table  7  compares different LLMs on ASQA using V ANILLA  (more results in § G.6 ). Notably, instruction-tuned models (Vicuna-13B and LLaMA-2-Chat) outperform the original LLaMA models in correctness and considerably enhance the citation quality. We observe that while the original LLaMA models are able to copy facts from the context, they struggle with accurately citing the sources or simply do not cite. Notably, the best open-source model, LLaMA-2-70B-Chat, achieves comparable correctness score as the OpenAI models, but still lags behind in citation quality.  

# 5.3 Retrieval Analysis  

The retrieval results play a crucial role to the correctness and the citation quality. Figure  4  presents the retrieval recall@ k  with different datasets and Table 7: Comparison of different LLMs on ASQA (GTR+V ANILLA ). LLaMA-13B and Vicuna-13B have a context limit of 2,048 tokens, and thus can only use a short version of instructions and at most top-3 passages. Chat-13B and Chat-70B refer to LLaMA-2-Chat.  

![](images/4515f0469dd37b69353ceb47e1871c8d84fbc3bd7142defca056d98cb4439741.jpg)  
Figure 4: Retrieval recall@ k  on ASQA ( EM recall ), QAMPARI ( recall-5 ), and ELI5 ( claim recall ). Retrieval recall serves as an upper bound for model performance, and we compare them with two models’ correctness results in the figure (dashed lines): “Vanilla (5-psg)” is ChatGPT V ANILLA  with top-5 passages in context; “Oracle” is the same model except that it uses 5 gold passages (§ G.1 ), whose recall matches Recall@100 on all three datasets.  

![](images/5d2566321ceb46bc95b03613d1e061d27db3fee9457ca774f34bf7bb8baa8e7b.jpg)  

retrievers. As the number of passages increases, retrieval recall steadily improves. Additionally, Figure  4  shows the correctness performance of two models: (1) ChatGPT V ANILLA  with top-5 passages (our primary baseline); (2) an oracle version of the same model employing 5 gold passages (§ G.1 ; the 5 gold passages match the retrieval recall@100). Notably, both models’ correctness lags behind the corresponding retrieval recall (except for ELI5 top-5). The discrepancy suggests that despite the presence of accurate answers in context, LLMs struggle to utilize them in their outputs.  

We compare the impact of different retrievers and different numbers of passages to LLMs. Figure  4  (right) shows that GTR outperforms DPR in both correctness and citation quality, emphasizing the importance of deploying better retrievers. Contrary to the retrieval recall trend in Figure  4 , more passages in context do not yield substantial improvement for ChatGPT. Specifically, correctness plateaus at top-1 passage and citation quality plateaus at top-3. GPT-4 (Table  7 ) exhibits an increasing trend with more passages, but the improvement is not proportional to the retrieval performance. This indicates the limited ability of LLMs in utilizing multiple passages within context.  

# 5.4 Other Ablations  

We provide additional ablations in § G . In summary, we find that (1) using comprehensive instructions enhances the citation quality of instruction-tuned models (§ G.2 ); (2) including at least one demonstration improves the performance (§ G.3 ); (3) finetuned models (FiD;  Izacard and Grave ,  2021 ) with P OST C ITE  lag behind LLMs in both correctness and citation quality and fail to generalize (§ G.4 ).  

# 6 Human Evaluation  

To verify that our automatic evaluation correlates with human judgement, we conduct human evaluation on selected models and request workers to judge model generations on three dimensions similar to Liu et al. (2023)—(1) utility: a 1-to-5 scoreindicating whether the generation helps answer the question; (2) citation recall: the annotator is given a sentence and all passages that the sentence cited, and is asked to judge whether the passages fully support the sentence; (3) citation precision: given a sentence and one of its citations, the annotator is asked to judge whether the citation “fully supports”, “partially supports”, or “does not support” the sentence. Each citation gets a precision score 1 if the output sentence has a citation recall of 1 and this citation at least “partially supports” it. See Appendix  F  for more details.  

Model outputs score high utility. The utility scores do not differ significantly between models, ranging  3 . 7 - 3 . 9  for ASQA and  3 . 5 - 3 . 6  for ELI5. Upon inspection, all tested models are mostly able Table 9: Human citation quality evaluation vs. ALCE citation quality evaluation on ELI5.  

![](images/bc6b2ba1f572703f64dbf4de1ec2ebd1eac30016b0a783c54eba2d29fea89027.jpg)  

Table 8: Human citation quality evaluation vs. ALCE citation quality evaluation on ASQA.   

![](images/6d1523aa6565966cce0c4232641e7bad1a9364131b53a89fbea42ddb65da36b6.jpg)  

to output fluent answers that are related to the question, despite differences in factual correctness.  

Our automatic evaluation of citation quality strongly correlates with human judgements.  As shown in Table  8  (ASQA) and Table  9  (ELI5), the relative rankings induced by human and our automatic metrics are consistent. The absolute citation scores from human and ALCE are very close except for R ERANK  (which uses the automated citation recall for reranking). This suggests that an improvement on ALCE citation metrics translates to improvement on human preferences. Furthermore, the Cohen’s kappa coefficient between human and ALCE suggests substantial agreement for citation recall ( 0 . 698 ) and moderate agreement for citation precision ( 0 . 525 ). We also show in § G.5  that our automatic evaluation achieves high accuracy when treating human annotations as gold labels ( 85 . 1% for citation recall and  77 . 6%  for citation precision).  

# 7 Related Work  

Evaluating citations. Generating text with citations is closely related to attribution.  Rashkin et al.  ( 2023 ) define the “attributable to identified sources” (AIS) score to measure how faithful a generated text is to its sources.  Bohnet et al.  ( 2022 ) apply AIS scores on a single-document short-answer QA dataset.  Honovich et al.  ( 2022 );  Yue et al. ( 2023 ) study automatic evaluations for the AIS score. A concurrent work ( Liu et al. ,  2023 ) conduct human evaluation on commercial generative search engines to examine their citation qualities.  

Scientific citation text generation ( Funkquist et al. ,  2022 ) is a related task to ALCE where the model is provided the papers-to-cite and context and is required to recover the citing text. It is different from ALCE as all citations are provided and the model only needs to perform the summarization.  

Retrieval-augmented LMs.  Many studies have explored augmenting LMs with externally retrieved information.  Guu et al.  ( 2020 );  Borgeaud et al. ( 2022 );  Izacard et al.  ( 2022 ) pre-train language models with retrieved passages, while  Khandelwal et al.  ( 2020 );  Zhong et al.  ( 2022 ) augment LLMs’ output by interpolating it with a  k NN module; though none of them explicitly provide citations to the retrieved sources. Other works prompt or fine-tune LLMs to “retrieve on-the-fly” ( Parisi et al. ,  2022 ;  Schick et al. ,  2023 ;  Shuster et al. ,  2022 ; Jiang et al. ,  2023 ;  Yao et al. ,  2023 ;  Press et al. , 2022 ), which offers flexibility of when and what to search.  Gao et al.  ( 2023 );  He et al.  ( 2022 ) propose to first generate text without accessing external documents and then retrieve relevant documents and revise the generation to be consistent.  

Among previous explorations,  Nakano et al. ( 2021 );  Menick et al.  ( 2022 ) are the closest to our setting, where LLMs are trained to answer questions while providing citations. However, they do not explore retrieval strategies and simply use commercial search engines, which are not reproducible, and their models and training data are closedsource. To the best of our knowledge, we are the first to implement end-to-end systems that retrieve, synthesize, and cite documents with LLMs.  

# 8 Conclusion  

We propose ALCE, the first automatic benchmark for evaluating LLM generations with citations. We deploy automatic metrics to measure fluency, correctness, and citation quality, and verify their efficacy via human evaluation. We explore a variety of strategies for incorporating citations in LLMs and demonstrate that current systems have considerable room for improvement on ALCE.  

Our experiments highlight a number of promising research directions, including (1) enhancing retrieval and refining retrieval integrations in LLMs, (2) developing long-context LLMs, and (3) advancing LLMs’ ability to synthesize multiple sources. What’s even more intriguing is that these research proposals extend beyond the ALCE setup (for example, long-context LLMs have numerous exciting applications), and ALCE can serve as a valuable testbed for their development.  

# Limitations  

Our evaluation still has room for improvement: (1) MAUVE is found to be sensitive to output length and may provide unstable results; (2) for the ELI5’s correctness evaluation, the automatically generated claims may not cover all possible answers due to the open-ended nature of the questions; (3) our citation quality evaluation is limited by the accuracy of the NLI model; for citation precision, the NLI model cannot detect the case of “partially support” and thus leads to a lower citation precision score than the human evaluation.  

Although we believe our curated datasets closely resemble the distribution of real-world user questions, we acknowledge that they do not cover more challenging scenarios, such as multi-hop reasoning, math reasoning, and code completion.  

In our experiments, we focus on prompting LLMs without updating their model weights. Training a model directly to incorporate citations remains challenging due to the lack of supervised data. However, we observe that certain humaninstruction datasets contain examples similar to our task setup. We leave the exploration of training LLMs to generate citations for future work.  

# Acknowledgments  

We appreciate the helpful feedback from the members of the Princeton NLP group. We thank Alexander Wettig, Nelson Liu, Tianyi Zhang, Yu Meng, Sadhika Malladi, Yangsibo Huang, Zhiyuan Zeng, and Dan Friedman for the valuable discussion. We thank Surge AI (especially Anna Folinsky and Edwin Chen) for their support with the human evaluation. Tianyu Gao is supported by an IBM PhD Fellowship. This research is supported by an NSF CAREER award (IIS-2239290), a Sloan Research Fellowship, and Microsoft Azure credits through the “Accelerate Foundation Models Academic Research” Initiative.  

# References  

Bernd Bohnet, Vinh Q Tran, Pat Verga, Roee Aharoni, Daniel Andor, Livio Baldini Soares, Jacob Eisenstein, Kuzman Ganchev, Jonathan Herzig, Kai Hui, et al. 2022.  Attributed question answering: Evaluation and modeling for attributed large language models .  arXiv preprint arXiv:2212.08037 .  

Sebastian Borgeaud, Arthur Mensch, Jordan Hoffmann, Trevor Cai, Eliza Rutherford, Katie Millican, George Bm Van Den Driessche, Jean-Baptiste Lespiau, Bogdan Damoc, Aidan Clark, Diego De Las Casas, Aurelia Guy, Jacob Menick, Roman Ring, Tom Hennigan, Saffron Huang, Loren Maggiore, Chris Jones, Albin Cassirer, Andy Brock, Michela Paganini, Geoffrey Irving, Oriol Vinyals, Simon Osindero, Karen Simonyan, Jack Rae, Erich Elsen, and Laurent Sifre. 2022.  Improving language models by retrieving from trillions of tokens . In  International Conference on Machine Learning (ICML) , volume 162, pages 2206–2240.  

Samuel Bowman, Gabor Angeli, Christopher Potts, and Christopher D Manning. 2015.  A large annotated corpus for learning natural language inference . In Empirical Methods in Natural Language Processing (EMNLP) .  

Tom B Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. 2020.  Language models are few-shot learners . In  Advances in Neural Information Processing Systems (NeurIPS) .  

Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E. Gonzalez, Ion Stoica, and Eric P. Xing. 2023.  Vicuna: An opensource chatbot impressing gpt-4 with 90%\* chatgpt quality .  

Angela Fan, Yacine Jernite, Ethan Perez, David Grangier, Jason Weston, and Michael Auli. 2019.  ELI5: Long form question answering . In  Association for Computational Linguistics (ACL) , pages 3558–3567.  

Martin Funkquist, Ilia Kuznetsov, Yufang Hou, and Iryna Gurevych. 2022.  CiteBench: A benchmark for Scientific Citation Text Generation .  arXiv preprint arXiv:2212.09577 .  

Luyu Gao, Zhuyun Dai, Panupong Pasupat, Anthony Chen, Arun Tejasvi Chaganty, Yicheng Fan, Vincent Y Zhao, Ni Lao, Hongrae Lee, Da-Cheng Juan, et al. 2023.  RARR: Researching and revising what language models say, using language models . In Association for Computational Linguistics (ACL) .  

Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Ming-Wei Chang. 2020.  REALM: Retrievalaugmented language model pre-training . In  International Conference on Machine Learning (ICML) .  

Hangfeng He, Hongming Zhang, and Dan Roth. 2022. Rethinking with retrieval: Faithful large language model inference .  arXiv preprint arXiv:2301.00303 .  

Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, and Yejin Choi. 2020.  The curious case of neural text degeneration . In  International Conference on Learning Representations (ICLR) .  

Or Honovich, Roee Aharoni, Jonathan Herzig, Hagai Taitelbaum, Doron Kukliansy, Vered Cohen, Thomas Scialom, Idan Szpektor, Avinatan Hassidim, and Yossi Matias. 2022.  TRUE: Re-evaluating factual consistency evaluation . In  North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT) , pages 3905–3920.  

Gautier Izacard and Edouard Grave. 2021.  Leveraging passage retrieval with generative models for open domain question answering . In  Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume , pages 874–880, Online. Association for Computational Linguistics.  

Gautier Izacard, Patrick Lewis, Maria Lomeli, Lucas Hosseini, Fabio Petroni, Timo Schick, Jane DwivediYu, Armand Joulin, Sebastian Riedel, and Edouard Grave. 2022. Atlas: Few-shot learning with retrieval augmented language models .  arXiv preprint arXiv:2208.03299 .  

Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan Su, Yan Xu, Etsuko Ishii, Ye Jin Bang, Andrea Madotto, and Pascale Fung. 2023.  Survey of hallucination in natural language generation .  ACM Computing Surveys , 55(12):1–38.  

Zhengbao Jiang, Frank F Xu, Luyu Gao, Zhiqing Sun, Qian Liu, Jane Dwivedi-Yu, Yiming Yang, Jamie Callan, and Graham Neubig. 2023. Active retrieval augmented generation .  arXiv preprint arXiv:2305.06983 .  

Ryo Kamoi, Tanya Goyal, Juan Diego Rodriguez, and Greg Durrett. 2023. WiCE: Real-World En- tailment for Claims in Wikipedia .  arXiv preprint arXiv:2303.01432 .  

Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. 2020.  Dense passage retrieval for opendomain question answering . In  Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP) , pages 6769–6781, Online. Association for Computational Linguistics.  

Urvashi Khandelwal, Omer Levy, Dan Jurafsky, Luke Zettlemoyer, and Mike Lewis. 2020.  Generalization through memorization: Nearest neighbor language models . In  International Conference on Learning Representations (ICLR) .  

Tushar Khot, Ashish Sabharwal, and Peter Clark. 2018. Scitail: A textual entailment dataset from science question answering . In  Conference on Artificial Intelligence (AAAI) , volume 32.  

Kalpesh Krishna, Aurko Roy, and Mohit Iyyer. 2021. Hurdles to progress in long-form question answering . In  Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies , pages 4940–4957, Online. Association for Computational Linguistics.  

Andreas Köpf, Yannic Kilcher, Dimitri von Rütte, Sotiris Anagnostidis, Zhi-Rui Tam, Keith Stevens, Abdullah Barhoum, Nguyen Minh Duc, Oliver Stanley, Richárd Nagyfi, Shahul ES, Sameer Suri, David Glushkov, Arnav Dantuluri, Andrew Maguire, Christoph Schuhmann, Huu Nguyen, and Alexander Mattick. 2023.  Openassistant conversations – democratizing large language model alignment .  arXiv preprint arXiv:2304.07327 .   
Kenton Lee, Ming-Wei Chang, and Kristina Toutanova. 2019.  Latent retrieval for weakly supervised open domain question answering . In  Association for Computational Linguistics (ACL) , pages 6086–6096.   
Nelson F Liu, Tianyi Zhang, and Percy Liang. 2023. Evaluating verifiability in generative search engines . arXiv preprint arXiv:2304.09848 .   
Jacob Menick, Maja Trebacz, Vladimir Mikulik, John Aslanides, Francis Song, Martin Chadwick, Mia Glaese, Susannah Young, Lucy Campbell- Gillingham, Geoffrey Irving, et al. 2022.  Teaching language models to support answers with verified quotes .  arXiv preprint arXiv:2203.11147 .   
Sewon Min, Julian Michael, Hannaneh Hajishirzi, and Luke Zettlemoyer. 2020.  AmbigQA: Answering ambiguous open-domain questions . In  Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP) , pages 5783– 5797, Online. Association for Computational Linguistics.   
Reiichiro Nakano, Jacob Hilton, Suchir Balaji, Jeff Wu, Long Ouyang, Christina Kim, Christopher Hesse, Shantanu Jain, Vineet Kosaraju, William Saunders, et al. 2021.  WebGPT: Browser-assisted questionanswering with human feedback . arXiv preprint arXiv:2112.09332 .   
Jianmo Ni, Chen Qu, Jing Lu, Zhuyun Dai, Gustavo Hernandez Abrego, Ji Ma, Vincent Zhao, Yi Luan, Keith Hall, Ming-Wei Chang, and Yinfei Yang. 2022. Large dual encoders are generalizable retrievers . In Empirical Methods in Natural Language Processing (EMNLP) , pages 9844–9855.   
OpenAI. 2023.  GPT-4 Technical Report .   
Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. 2022.  Training language models to follow instructions with human feedback .  Advances in Neural Information Processing Systems (NeurIPS) , 35:27730– 27744.   
Aaron Parisi, Yao Zhao, and Noah Fiedel. 2022.  TALM: Tool augmented language models .  arXiv preprint arXiv:2205.12255 .   
Fabio Petroni, Aleksandra Piktus, Angela Fan, Patrick  

Yacine Jernite, Vladimir Karpukhin, Jean Maillard, Vassilis Plachouras, Tim Rocktäschel, and Sebastian Riedel. 2021.  KILT: a benchmark for knowledge intensive language tasks . In  Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies , pages 2523–2544, Online. Association for Computational Linguistics.  

Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Dmytro Okhonko, Samuel Broscheit, Gautier Izac- ard, Patrick Lewis, Barlas O˘ guz, Edouard Grave, Wen-tau Yih, et al. 2021.  The Web Is Your Oyster– Knowledge-Intensive NLP against a Very Large Web Corpus .  arXiv preprint arXiv:2112.09924 .  

Krishna Pillutla, Swabha Swayamdipta, Rowan Zellers, John Thickstun, Sean Welleck, Yejin Choi, and Zaid Harchaoui. 2021.  MAUVE: Measuring the gap be- tween neural text and human text using divergence frontiers . In  Advances in Neural Information Processing Systems .  

Ofir Press, Muru Zhang, Sewon Min, Ludwig Schmidt, Noah A Smith, and Mike Lewis. 2022.  Measuring and narrowing the compositionality gap in language models .  arXiv preprint arXiv:2210.03350 .  

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. 2020.  Exploring the limits of transfer learning with a unified text-to-text Transformer .  The Journal of Machine Learning Research (JMLR) , 21(140).  

Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. 2016.  SQuAD: 100,000+ questions for machine comprehension of text . In  Empirical Methods in Natural Language Processing (EMNLP) , pages 2383–2392.  

Hannah Rashkin, Vitaly Nikolaev, Matthew Lamm, Lora Aroyo, Michael Collins, Dipanjan Das, Slav Petrov, Gaurav Singh Tomar, Iulia Turc, and David Reitter. 2023.  Measuring Attribution in Natural Language Generation Models .  Computational Linguistics , pages 1–64.  

Samuel Joseph Amouyal Ohad Rubin, Ori Yoran, Tomer Wolfson, Jonathan Herzig, and Jonathan Berant. 2022.  QAMPARI: An Open-domain Question Answering Benchmark for Questions with Many Answers from Multiple Paragraphs . arXiv preprint arXiv:2205.12665 .  

Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. 2023.  Toolformer: Language models can teach themselves to use tools . arXiv preprint arXiv:2302.04761 .  

Tal Schuster, Adam Fisch, and Regina Barzilay. 2021. Get your vitamin C! robust fact verification with contrastive evidence . In  North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT) , pages 624– 643.  

Freda Shi, Xinyun Chen, Kanishka Misra, Nathan Scales, David Dohan, Ed Chi, Nathanael Schärli, and Denny Zhou. 2023.  Large language models can be easily distracted by irrelevant context . In  International Conference on Machine Learning (ICML) .   
Kurt Shuster, Mojtaba Komeili, Leonard Adolphs, Stephen Roller, Arthur Szlam, and Jason Weston. 2022.  Language models that seek for knowledge: Modular search & generation for dialogue and prompt completion . In  Findings of the Association for Computational Linguistics: EMNLP 2022 , pages 373–393.   
Ivan Stelmakh, Yi Luan, Bhuwan Dhingra, and MingWei Chang. 2022.  ASQA: Factoid questions meet long-form answers . In  Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing , pages 8273–8288, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.   
Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. 2023.  Stanford Alpaca: An Instruction-following LLaMA model .   
James Thorne, Andreas Vlachos, Christos Christodoulopoulos, and Arpit Mittal. 2018. FEVER: a large-scale dataset for fact extraction and VERification . In  North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT) , pages 809–819.   
Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. 2023a. LLaMA: Open and Efficient Foundation Language Models .  arXiv preprint arXiv:2302.13971 .   
Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Di- ana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizen- stein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurelien Ro- driguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. 2023b.  Llama 2: Open foundation and fine-tuned chat models .  

Alex Wang, Kyunghyun Cho, and Mike Lewis. 2020. Asking and answering questions to evaluate the factual consistency of summaries . In  Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics , pages 5008–5020, Online. Association for Computational Linguistics.  

Adina Williams, Nikita Nangia, and Samuel Bowman. 2018.  A broad-coverage challenge corpus for sentence understanding through inference . In  North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT) .  

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik R Narasimhan, and Yuan Cao. 2023. React: Synergizing reasoning and acting in language models . In  International Conference on Learning Representations (ICLR) .  

Xiang Yue, Boshi Wang, Kai Zhang, Ziru Chen, Yu Su, and Huan Sun. 2023.  Automatic evaluation of attribution by large language models .  arXiv preprint arXiv:2305.06311 .  

Shiyue Zhang and Mohit Bansal. 2021.  Finding a balanced degree of automation for summary evaluation . In  Empirical Methods in Natural Language Processing (EMNLP) , pages 6617–6632.  

Yuan Zhang, Jason Baldridge, and Luheng He. 2019. PAWS: Paraphrase adversaries from word scrambling . In  North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT) , pages 1298–1308.  

Zexuan Zhong, Tao Lei, and Danqi Chen. 2022.  Training language models with memory augmentation . In Empirical Methods in Natural Language Processing (EMNLP) , pages 5657–5673.  

# A Generating Claims for ELI5  

We elect not to use ROUGE-L as our main correctness metrics since it does not account for the different ways of expressing the same answer and it can be easily gamed ( Krishna et al. ,  2021 ). We further illustrate this issue in Table  10 . A system can easily achieve high ROUGE-L score by retrieving and returning the top passage from a BM25 index. However, the claims evaluation metric does not reward this approach since the output often lacks different aspects of the answers.  

![](images/a5616a77e8c2ddb32109a76b85ea033b18b355907717d68858dec5cfa38b0c95.jpg)  
Table 10: Comparison between ROUGE-L and claim recall scores on ELI5.  

Instead, we leverage the original answers to generate sub-claims and use them to serve as an estimate of the different aspects of the answers that we expect the model to cover. This approach is inspired by works in summarization evaluation and claim verification ( Zhang and Bansal ,  2021 ;  Kamoi et al. ,  2023 ;  Wang et al. ,  2020 ).  

Specifically, we use  text-davinci-003  to generate the sub-claims. We first manually annotate three question and answer pairs from the original ELI5 training set with 3 sub-claims each. Then, we prompt  text-davinci-003  with these pairs as demonstrations. The full prompt with an example is shown in Table  22 .  

InstructGPT generates coherent and faithful sub-claims. To ensure that the generated subclaims are of good quality, we manually inspect a random sample of 40 answers and their generated sub-claims (totaling to 120 sub-claims). For each sub-claim, we assign a score of  1  if it is relevant to the question and faithful to the facts presented in the ground truth, and  0  otherwise. We found that 112  out of the  120  ( 93 . 33% ) sub-claims received a score of  1 , meaning that our generated sub-claims are of high quality and faithful to the ground truth. Furthermore, the average number of words in the generated sub-claims is  14  words, and they are typically just one sentence long. This is aligned with the intent behind the metric—to capture short factual claims made by the original answer.  

NLI model accurately predicts the entailment of sub-claims.  We further analyze our sub-claim evaluation metrics by checking the error rate of the final prediction of the NLI model. To this end, we first manually annotate the entailment scores between 40 outputs and their sub-claims (in total of 120 pairs; these are the same questions from the previous analysis). We then use the NLI model to obtain the entailment scores for the output and sub-claims. Using the human annotations as the ground truth label, we found that the NLI model achieved an accuracy of  80 . 0% .  

# B Dataset Statistics  

For ASQA, human answers have an average length of 65 words. For QAMPARI, each question has on average 13 answers. For ELI5, human answers have an average length of 131 words.  

# C Implementation Details  

NLI model. We use the version of TRUE model from  https://huggingface.co/google/ t5_xxl_true_nli_mixture , which is trained on SNLI ( Bowman et al. ,  2015 ), MNLI ( Williams et al. ,  2018 ), Fever ( Thorne et al. ,  2018 ), Sci- tail ( Khot et al. ,  2018 ), PAWS ( Zhang et al. ,  2019 ), and VitaminC ( Schuster et al. ,  2021 ). This model uses the following prompt: “ premise: {PREMISE} hypothesis: {} ” and outputs “ 1 ” if the premise entails the hypothesis. We format each passage (when used as premise) by the format of “ Title: {TITLE}\n{TEXT} ” and concatenate all passages with “ \n ” as a separator.  

MAUVE.  When running MAUVE, we concatenate the question and the model output (or human answer) by space. We truncate both the references and the model generations to 100 words, as we found MAUVE results are unstable beyond this length for ELI5 (this is due to that ELI5 has a lot of extremely long human answers).  

Exact match for ASQA and QAMPARI.  Both ASQA and QAMPARI provide aliases for their short answers. We normalize the response and the short answers similarly to  Rajpurkar et al.  ( 2016 ) and report the score with the best-matching aliases. For ASQA,  Stelmakh et al.  ( 2022 ) also propose a QA-based evaluation which we found to be not as stable, and thus we do not report it in our paper.  

Output truncation.  Before evaluation, we truncate model output by new lines, as non-instructiontuned models may generate more content after new lines that are irrelevant.  

I NTERACT .  Empirically, we found that models tend to execute too many consecutive “ check ” actions, so we force the model to always “ output ” after each “ check ”. We limit the maximum number of passages to check as 3 to avoid exceeding the length limit. The full passages are removed from the context after one action to save context space. Table  27  provides an example for I NTERACT .  

Main experiments.  For all experiments except ChatGPT R ERANK , we run each model three times with different seeds and each time we sample two demonstrations from a pool of four. We report the averaged scores for all experiments in the main paper and we report the standard deviations in Appendix  G.6 .  

Decoding methods.  Based on preliminary experiments we choose the following decoding methods: For ChatGPT and GPT-4, we use sampling with temperature  0 . 5 ; for all open-source models, we use Nucleus sampling ( Holtzman et al. ,  2020 ) and set top_p  = 0 . 95 .  

D ALCE Catches Shortcut Cases   

![](images/83713a3856596b23ed641f8ad443b99125ca5df0976e616b2e544a4323ee45f3.jpg)  

Table 11: ASQA cheating cases. “ChatGPT”: the ChatGPT V ANILLA  model with GTR-retrieved top-5 passages. “Top-1 passage”: use the top-1 retrieved passage as the response. “First 2 sents”: use the first 2 sentences of the top-1 retrieved passage.  

Table  11  demonstrates the experiments to show that ALCE is robust to shortcut cases. Using the top-1 passages or first two sentences of the top-1 passages induces almost perfect citation quality, but fluency and correctness are dramatically lower.  

# E Citation Recall Discussion  

Our citation precision evaluation cannot detect a citation that partially supports the statement and hence will falsely penalize it. Consider a statement  s 3  and its citations  [2][4][5] : if  [2]  entails partial information of  s 3  that  [4][5]  also entails, [2]  will be counted as “irrelevant” while it should not be penalized.  Liu et al.  ( 2023 ) conduct human evaluation on citation precision in a different way: For each citation, they ask annotators to judge whether the citation (1) fully support, (2) partially support, or (3) does not support  s i . One citation  c i,j is precise if (a)  c  fully supports  s i  or (b)  C i  fully supports  s i ,  c  partially supports  s i , and no  c  ∈C i alone fully supports  s i . This evaluation solved the corner case we mentioned in the main paper (one citation partially supports the claim but is identified as “irrelevant”). However, it is challenging to conduct such evaluation automatically, as there is no existing model that can judge whether a citation “partially” supports a claim. We also explore prompting ChatGPT to conduct such a task, which yields poor results. We defer it to future work to collect supervised data to train a better  ϕ  that can detect “partial support”.  

# F Human Evaluation  

We employ Surge AI ( https://www.surgehq. ai/ ) for our human evaluation. The average pay to workers is 20 USD per hour. We randomly sample 100 examples from ASQA and ELI5 and annotate outputs of selected models: ChatGPT V ANILLA , ChatGPT R ERANK , and Vicuna-13B V ANILLA .  

# F.1 Utility  

To check if the model output is useful to downstream users, we measure the utility of the response S . We first show the query  q  and model response S ment with the statement "The response is a helpful  to the worker and ask them to rate their agreeand informative answer to the query" on a Likert scale of 1-5, corresponding to  Strongly Disagree , Disagree ,  Neutral ,  Agree , and  Strongly Agree .  

# F.2 Citation Recall  

The annotators are shown the question  q , the statement  s i , and all of its citations  C i , and they rate if the joint set of citations fully support the statement (recall=1) or if they do not support all the claims (recall=0). We calculate the overall recall score for the generation by taking an average of all the statements’ recall scores.  

# F.3 Citation Precision  

We show the question  q  and a pair of a statement s i  and one of its citation  c  ∈C i  to the annotator. We ask the annotator if the citation  fully supports , partially supports , or  does not support  the factual claims in  s i . Citation  c i,j  has a citation precision of  1  if  s i  has a recall of  1 , and  c i,j  fully or partially supports  s i . Finally, we take an average of precision scores of all citations in the statement obtain the citation precision score.  S  to  

![](images/a0e2f105644f4255bdbcb2d62b4707733845a6b1a0e19f95d532cd750a9a7bae.jpg)  

Table 12: Retrieval results for ASQA (EM recall).   
Table 13: Retrieval results for QAMPARI (recall-5).   

![](images/65cb5b76e7e0a9ded2e3902ed7dc8affb14063bc308a7e9136b0fdf4dd42c7ea.jpg)  

# G More Experiments  

# G.1 Retrieval Analysis  

Oracle.  Since the original datasets do not contain gold passages at the same granularity level as our setting (100-word passages), we approximate gold passages by running the following algorithm on the top-100 retrieved passages. We first calculate the recall score for each passage. Then, we sort the passages using their recall score and take the top 5 passages as our initial oracle set. Finally, we iterate through all passages that were not initially in the oracle set and try to replace the passages in the oracle set in a greedy fashion: we calculate the change in the recall score of the oracle set for every possible replacement and proceed with the replacement that results in the largest recall improvement. The set of 5 oracle passages were able to match the recall scores of the top-100 retrieved passages.  

Detailed retrieval results.  We show detailed retrieval results in Tables  12 ,  13 , and  14 .  

# G.2 Effect of Instructions  

Table  15  shows results of using a full instruction (Table  23 ) and a short version of the instruction (Table  24 ). We see that the full version induces stronger correctness and citation recall, while the two instructions lead to similar citation precision.  

![](images/76618c9ef02f16b730fa65a2328918f1a0594ee2de622a03f87c88ea9d16ec66.jpg)  

Table 14: Retrieval results for ELI5 (claim recall).   
Table 15: Effect of different instructions on ASQA.   

![](images/587e7e2c5011728358aea9632c9e5ffbbc9be54cffa0e9f0aece00d79d8932d4.jpg)  

# G.3 Effect of Demonstrations  

Table  16  shows results on effect of different numbers of demonstrations. We see that numbers of demonstrations do not affect ChatGPT’s correctness but using at least one demonstration ensures high citation recall. For the original LLaMA model, Table  16  shows the trend that more demonstrations lead to better performance.  

# G.4 Fine-tuned Models  

To better understand the differences between finetuned models and prompted large language models, we train state-of-the-art question answering model, Fusion-in-Decoder (FiD;  Izacard and Grave  ( 2021 )), and evaluate it in conjunction with P OST C ITE . Due to the lack of training data with citation annotation, we first train a T5-base FiD model for 5 epochs on the ASQA training set with a batch size of 64 and a learning rate of 1e-4. During evaluation, we use P OST C ITE  to add citations to the output. We also use  k  = 5  passages during both training and evaluation of the FiD model.  

Then, we evaluate this model on both ASQA (in-domain) and ELI5 (out-of-domain), and the results can be found in Tables  17  and  18 . Note that this is not a direct comparison, as ALCE assumes only evaluation data available and uses only fewshot data for prompting. As the results show, the FiD baseline still significantly lags behind prompting ChatGPT in both correctness and citation quality (even though it is trained on 4000+ examples). When tested on another dataset (ELI5), FiD performs even worse, showing that it is challenging to solve the problem by fine-tuning a small pretrained model.  

![](images/b43bb15a88a6665346c0eded4f613f15fd239d39cae970455153e8900d1397b1.jpg)  

Table 16: Different demonstrations on ASQA.   

![](images/a0fb5a46ab3fa72f07fdf833b6684294efc90ccac2eee1c22cfc3efeeea1f180.jpg)  

Table 17: Comparison of Fusion-in-Decoder with ChatGPT on ASQA. Both models use top-5 GTR passages.  

# G.5 More Human Evaluation  

We evaluate the accuracy of our automatic metrics by treating the human annotations as gold labels. For citation recall, ALCE achieves an accuracy of 85 . 1% ; for citation precision, ALCE has an accuracy of  77 . 6% . Regarding detecting insufficient citations, ALCE has a recall of  82 . 3%  and a precision of  84 . 2% ; regarding detecting “irrelevant” citations, ALCE has a recall of  75 . 6%  and a precision of  66 . 1% —ALCE is effective in detecting “irrelevant” citations, but due to the limitation of the NLI model (cannot detect “partial support”), it has a relatively high false positive rate.  

# G.6 Main Results  

We show full results of our experiments along with the standard deviation in Tables  19 ,  20 , and  21 . We repeat all experiments with three different random seeds. However, for ChatGPT R ERANK , we use only one seeded run since each run repeats the generation step four times, and more experiments would incur significant costs. Similarly, due to the cost of running ChatGPT-16K and GPT-4, we only use one seeded run for each model.  

# G.7 Open-source Models  

In addition to the open-source models discussed in the main text, we also show the results of LLaMA7B, Alpaca-7B, Vicuna-7B, LLaMA-33B, Oasst33B, and Stable Beluga 2 in the Tables  19 ,  20 , and 21 . For selected models, we also tested them using approaches beyond V ANILLA .  

Although open-source models generally lag behind the state-of-the-art models (i.e. GPT-4) in both correctness and citation quality, the largest instruction-following models (i.e. LLaMA-2-70BChat and Stable Beluga 2) can sometimes achieve competitive correctness to the SoTA.  

Table 18: Comparison of Fusion-in-Decoder with ChatGPT on ELI5. Both models use top-5 GTR passages.   

![](images/49ddc2088102e53184237b5b0ebaf78dd24aee3a2e0c4c6540e3ac48047fbb0f.jpg)  

Furthermore, we found that the open-source models follow a similar trend between different approaches as ChatGPT. Specifically, using S UMM and S NIPPET  improves correctness and R ERANK boosts citation quality.  

# H Prompts  

We show detailed prompts used in our paper in Tables  23 ,  24 ,  25 ,  26 ,  27 ,  28 , and  29 .  

# IExamples  

In Tables  30  and  31  we show some examples of questions and model generated outputs.  

Table 19: ASQA full results.   

![](images/5d488c4ef13016b577c46e2cd159972d0ff2bc1680857114bbcdeea098002df5.jpg)  

Table 20: QAMPARI full results.   

![](images/1e4a3004512f623010880bbebd6a606e3ff5fb4bee5a85d2b71dcb71a91e0516.jpg)  

![](images/6ee2220f81e4962f9bc95b7751ee1e9ed90a0743924b04958f50dec2cb45b65f.jpg)  
Table 21: ELI5 full results.  

Read the original question and passage, and generate 3 additional claims that are supported by the passage and answer the question.  

Original question: What’s the difference between Shia vs. Sunni Islam?   
Passage: The main difference between Shia and Sunni Muslim is related to ideological heritage and issues of leadership. This difference is first formed after the death of the Prophet Muhammad in 632 A.D. The ideological practice of the Sunni branch strictly follows Prophet Muhammad and his   
teachings, while the Shia branch follows Prophet Muhammad’s son-in-law Ali. Nowadays, Sunni and Shia are the major branches of Islam.   
Claim 1: The major branches of Islam are Sunni and Shia. Claim 2: Prophet Muhammad died in 632 A.D.   
Claim 3: The ideological practice of the Sunni branch strictly follows Prophet Muhammad and his teachings.  

Original question: What causes Bi-polar disorder?  

Passage: Bipolar disorder is an emotional disorder that causes extreme mood swings between excitement and depression. The spectrum of mood swing may span from days to months. We are still not certain of the exact factors that cause such disorder, but genetics is considered a major factor. Claim 1: One symptom of Bi-polar disorder is extreme mood swings between excitement and depression. Claim 2: Genetics could be one of the major factors that causes Bi-polar disorder. Claim 3: The mood swing from Bi-polar disorder can last days to months.  

Original question: How do we hear differences in sound besides volume and pitch?  

Passage: Pitch refers to the frequency of soundwave, and volumn refers to the amplitude of the soundwave. Besides volumn and pitch, we can also tell the difference between sounds based on the tone of sound. For example, we can differentiate the sound of different instruments based on the tone of the sounds.   
Claim 1: Volume of sound is the amplitude of the soundwave. Claim 2: Pitch is the frequency of soundwave.   
Claim 3: We can use the tone of the sounds to differentiate the sound of different instruments. Original question: How are we able to discern whether a sound is coming from in front of us or   
behind us? Passage: There are multiple explanations for why we can localize sounds. One explanation is that   
sounds travelling to the corresponding side of one’s ear will be slightly louder. Another explanation is that there is a slight difference in the hitting time to one’s left and right ear   
based on the sound’s direction. However, these explanation means that when a sound is exactly in front of someone or exactly behind someone, he or she can not tell the difference.   
Claim 1:  We can localize sounds by recognizing that the sound travelling to the corresponding side of one’s ear will be slightly louder.   
Claim 2: We can also localize sounds by recognizing the difference in hitting time to one’s left and right ear based on the sound’s direction.   
Claim 3: We cannot tell the difference between a sound that is exactly in front of us or exactly behind us.  

Table 22: Prompt used to generate the sub-claims for ELI5 questions.  Blue text is model generation.  Brown text is the ELI5 example that we want to generate sub-claims for.  We construct the prompt by manually writing the sub-claims for three questions from the training set.  

Summarize the following document within 50 words with the question of interest "{QUESTION}" Return "irrelevant" if the document is irrelevant to the question. Try to keep all the important dates, numbers, and names.  

Title: {TITLE} Text: {TEXT} Summary:  

Table 25: Prompts for S UMM .  

Given the follow passage and the question "{QUESTION}", extract a useful span from the passage that can answer the question. Resolve all the coreference issues to make the extracted span understandable standalone. If the passage is not helpful for answering the question, return "irrelevant".  

Title: {TITLE} Text: {TEXT} Extracted span:  

Table 26: Prompts for S NIPPET .  

Instruction: Write an accurate, engaging, and concise answer for the given question using only the provided search results and cite them properly. Use an unbiased and journalistic tone. Always cite for any factual claim.  

You are provided summaries/snippets of the search results. You can use "Check: Document [1][2]" to check the corresponding full documents (you should only check relevant documents and you can at most check 3 documents at a time) and use "Output:" to output a sentence in the answer. In the answer, cite properly by using [1][2][3]. Cite at least one document and at most three documents in each sentence. If multiple documents support the sentence, only cite a minimum sufficient subset of the documents. Use "End" to end the generation.  

<Retrieve for question “...”>   
<Get summaries/snippets for the passages and delete those that are “irrelevant”>   
Document [1](Title: ...) {SUMMARY OR SNIPPET} Question: When did US break away from England? Check: Document [1][2]   
Document [1] {FULL TEXT} Document [2] {FULL TEXT}   
Output: The United States ...  [1]  ...  [2] . <Remove the full text of [1][2] from context> Check: Document [3]   
Document [3] {FULL TEXT} Output: The Treaty of Paris ...  [3] .   
<Remove the full text of [3] from context> End.  

Table 27: An example for I NTERACT .  

Instruction: Write an accurate, engaging, and concise answer for the given question using only the provided search results and cite them properly. Use an unbiased and journalistic tone.   
Always cite for any factual claim. You can use "Search: key words" to check the most relevant document’s full text and use   
"Output:" to output a sentence in the answer. In the answer, cite properly by using [1][2][3]. Cite at least one document and at most three documents in each sentence. If multiple documents support the sentence, only cite a minimum sufficient subset of the documents. Use "End" to end the generation.  

Table 28: Instruction for I NLINE S EARCH .  

Instruction: Write an accurate, engaging, and concise answer for the given question. Use an unbiased and journalistic tone.  

Instruction: Write an accurate, engaging, and concise answer for ...  

Document [1](Title: How to Treat and Prevent Food Poisoning - MsPrepper) just a typical gastro upset. Salmonella is most commonly caused by eating undercooked or raw foods like eggs or meat. You know how your mom always warned you not to eat raw cookie dough? This is why. Most people do eat cookie dough and they are fine, but salmonella is a risk. If you do contract salmonella, you could start to feel bad within in a couple of hours after eating contaminated food, and sometimes it could take a day or two. Common symptoms are nausea and vomiting, loose stools (sometimes bloody), flu like symptoms, and stomach cramps. To treat  

Document [2](Title: FDA Issues Warning About Eating Raw Cookie Dough, But Not For Salmonella Risks) FDA Issues Warning About Eating Raw Cookie Dough, But Not For Salmonella Risks Used to licking the spoon or placating yourself with full-on chunks of raw cookie dough? The Food and Drug Administration issued a warning on Tuesday that strongly advises against continuing the habit. The agency asserted that consuming raw batter of any kind, whether for bread, cookies or pizza, could make a person sick. While you may have been warned in the past against eating raw dough due to the risk of contracting salmonella from raw eggs, the FDA is citing raw flour as the culprit for a  

Document [3](Title: It’s Probably OK to Eat Raw Cookie Dough — As Long As You’re Smart About It - The Crux - Very Top Secret Information) First, when most people think about health risks and cookie dough, they think about raw egg. Eggs can be contaminated with salmonella bacteria, and food safety recommendations encourage people to cook eggs until the white and yolk are firm in order to kill any bacteria. However, anyone making cookies can do things to reduce this risk by using pasteurized egg products. When my kids and I make cookie dough, we never use regular eggs. Instead, we use shell eggs that have been pasteurized to kill any harmful bacteria without actually cooking the egg itself. (A great public health innovation, if  

Document [4](Title: How Dangerous Is It to Eat Raw Cookie Dough? | Men’s Health) Can Eating Raw Cookie Dough Really Make You Sick? Scientists reveal the truth about this supposedly dangerous delicacy By Katherine Dempsey There are few things more tempting in life than eyeing a bowl of cookie dough and deciding whether or not to stick your finger in for a scoop. It’s a bit like playing Russian roulette. You could get lucky and enjoy the delicious dough without conseqence, but there’s always the risk of getting serously sick with a food-borne illness. That’s because multiple ingredients within the dough could be contaminated by pathogens such as Salmonella and E. coli, says Soohyoun  

Document [5](Title: How Dangerous Is It to Eat Raw Cookie Dough? | Men’s Health) is usually pasteurized, so it’s not likely that the egg would make you sick. However, other ingredients in the dough could potentially harbor pathogens. Experts say that a prime suspect in a 2009 E. coli outbreak linked to prepackaged cookie dough was actually the flour. Nuts and chocolate have also been linked to Salmonella outbreaks. Bottom line: You’re better off skipping raw cookie dough. At least you’ll have killer cookies as a consolation. The article How Bad Is It To Eat Raw Cookie Dough? originally ran on Prevention.com The 9 Foods Most Likely to Make You Sick Mmm, Just Don’t Question: Why is it bad to eat cookie dough for risk of salmonella but things like Cookie Dough Bites are ok?  

Answer:  Raw cookie dough is not recommended to be eaten due to the risk of salmonella  [1][2] . Eating raw flour is also a risk for food poisoning  [2] . The egg in cookie dough is usually pasteurized, so it’s not likely that the egg would make you sick, but other ingredients in the dough could potentially harbor pathogens  [4][5] . However, prepackaged cookie dough like Cookie Dough Bites is safe to eat because the dough is made with pasteurized egg products and heat-treated flour  [2][3] . .  

Ground truth claims:   
1. Cookie Dough Bites are safe to eat since they do not contain eggs. 2. Salmonella is a bacteria found in eggs.   
3. Eating raw cookie dough carries the risk of salmonella infection.  

Instruction: Write an accurate, engaging, and concise answer for ...  

Document [1](Title: Is Snapchat really worth \$19 billion? - CSMonitor.com) reporting that the Los Angeles-based company is aiming to raise \$500 million at a valuation of \$16 billion to \$19 billion, making it the third most highly valued tech start-up backed by venture capitalists. The Chinese handset maker Xiaomi is valued at \$45 billion, while Uber is estimated to be valued at about \$40 billion, according to data from CB Insights. Read MoreVC investment hits \$86B thanks to Uber, Xiaomi Snapchat was valued at \$10 billion in August, according to a Dow Jones report. Some of its investors from previous rounds include Benchmark, Lightspeed Venture Partners and Kleiner Perkins Caufield Document [2](Title: What Are Venture Capital Investments? – DollarsAndSense.my) Ever wondered how highly valued technology giants like Google and Facebook were able to grow so fast and pay their employees so well in such a short amount of time, or how still growing start-ups like Uber are able to lose 1.2 billion US dollars in just the first half of this year alone and still command a valuation upwards of 50 billion US dollars? The answer lies with a special category of investment activity known as venture capital. Venture capitalists are professional investors who invest in a number of highly scalable high-risk technology ventures hoping to make a multi-fold  

Document [3](Title: Opinion | What Dara Khosrowshahi Must Do to Save Uber - The New York Times) at a discount. These are troubling signs. Every start-up must one day fulfill the market’s demand that it turn a profit, but Uber has never figured out how to do that. While ride sharing in some form will probably survive, it’s more likely that without some drastic changes, Uber won’t be around in three to five years. Mr. Khosrowshahi must avoid the mistakes of his predecessor by accepting that “pivots” (Silicon Valley-speak for the desperate changes troubled companies make to reassure their venture capitalist funders) are not the answer. None of the pivots Mr. Kalanick tried — like on-demand delivery  

Document [4](Title: Snapchat raising funding round at \$19 billion valuation: Report) Snapchat raising funding round at \$19 billion valuation: Report The mobile messaging app Snapchat is said to be seeking funding at a \$19 billion valuation, according to a report, citing a person with knowledge of the matter. Bloomberg is reporting that the Los Angeles-based company is aiming to raise \$500 million at a valuation of \$16 billion to \$19 billion, making it the third most highly valued tech start-up backed by venture capitalists. The Chinese handset maker Xiaomi is valued at \$45 billion, while Uber is estimated to be valued at about \$40 billion, according to data from CB Insights. Document [5](Title: Unicorns And Wall Street | MoneyTips) not start showing profits soon. Even Uber, with its massive \$62-billion-plus valuation, will test investor’s patience if it does not start showing signs of a profit in the next few years. Cracks are starting to show in terms of pre-IPO devaluations. Dropbox and Snapchat have seen a nearly one-quarter drop in their pre-IPO valuations, while Square’s pre-IPO value has dropped by approximately one-third. Larger investors and venture capitalists are sending a message to the unicorns: Outline a solid path to profits and start producing them soon, or expect your valuation to be slashed. Diving ahead with an IPO is not  

...  

Question: How are firms like snapchat, uber etc valued so highly while still not making a profit? Do venture capitalists not expect some form of repayment within a number of years?  

Answer:  Venture capitalists invest in a number of highly scalable high-risk technology ventures hoping to make a multi-fold return on their investment in a short amount of time  [2] . Highly valued companies like Snapchat, Uber, and Xiaomi are able to grow so fast and command high valuations because they have the potential to generate significant revenue in the future  [2][4] . Venture capitalists expect a return on their investment within a number of years, but they also invest in the long-term potential of the company  [2] . It is important to note that every start-up must eventually turn a profit, but some start-ups like Uber have not yet figured out how to do that  [3][5] .  