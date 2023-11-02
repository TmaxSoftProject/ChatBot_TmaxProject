import os 
import openai
from konlpy.tag import Okt
import re
import pandas as pd
openai.api_key = "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"
from gensim.models.word2vec import Word2Vec
from gensim.models import word2vec
import nltk
import gensim
import gensim.models as g





t = Okt()
tokenized=[]
tok_sentence_forVectorize=[]
def tokenize(df):
    for sentences in df:  #### 전처리및 토크나이징
        
        sentences=str(sentences)
        sentences=sentences.replace('\n','')
        sentences=sentences.replace(',','')
        sentences=sentences.replace('"','')
        sentences=sentences.replace("'",'')
        sentences=sentences.replace('-','')
        sentences=sentences.replace('(','')
        sentences=sentences.replace(')','')
        sentences=sentences.replace('.','')
        #print(sentences)
        contents_tokens=t.morphs(sentences)
        tokenized.append(contents_tokens)
    
    return tokenized



def tokenize_to_fullsents(df):
    for sentences in df:  #### 전처리및 토크나이징 후 유사도 비교를 위해 다시붙이기
        
        sentences=str(sentences)
        sentences=sentences.replace('\n','')
        sentences=sentences.replace(',','')
        sentences=sentences.replace('"','')
        sentences=sentences.replace("'",'')
        sentences=sentences.replace('-','')
        sentences=sentences.replace('(','')
        sentences=sentences.replace(')','')
        sentences=sentences.replace('.','')
        sentences=sentences.replace('{','')
        sentences=sentences.replace('}','')
        sentences=sentences.replace('`','')


        #print(sentences)
        contents_tokens=t.morphs(sentences)
        tokenized.append(contents_tokens)

    

    for content in tokenized:
        sentence=''
        for word in content:
            sentence=sentence+' '+word
        

        tok_sentence_forVectorize.append(sentence)

    return tok_sentence_forVectorize
# print(tok_sentence_forVectorize[0])


########
    
###타겟질문의 상위3개 문맥을 고려한 키워드와, 타겟질문의 답변의 문맥을 고려한 상위 3개 키워드를 비교하여 동일하게 일치되지 않은 키워드를 추천중요 키워드로 선정한다

def keyword_extractor_Qanswer(tokenizinedList):
    idx=0 #### gpt로 키워드 뽑는 부분

    answer=str(tokenizinedList[idx]) #토큰화된것중에 타겟질문에 대한 해답을 index로 반환해주는 idx
    context='' #gpt 시스템에서는 context가 존재하지않는거 같다. 혹은 context로 어떤것을 넣어줘야할지는 미정.
    prompt=context+answer+' 다음 문장의 핵심 키워드들을 문맥을 참고하여 중요한 순서대로 dict자료형으로 뽑아줘'
    completion = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo",
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role":"user","content":prompt}
        ]
    )


    keyword_dict=completion.choices[0].message.content
    # print(type(keyword_dict))
    keyword_dict=eval(keyword_dict) #str->dict type difference

    print(completion.choices[0].message.content)
    return keyword_dict




####### 10/26~ 키워드 비교 작업후 추천키워드에 핵심이 될 키워드 선정() 1차:질문의 키워드랑 겹치는 키워드(포함관계가 성립되는) 제거######
def define_recomm_keyword1(key_ans,key_qes):
    
    keyword_list=[]
    for i in range(len(key_qes)):
        for j in range(len(key_ans)-1,-1,-1):
            if key_qes[i][0] in key_ans[j][0]:
                del key_ans[j] # **error index out of range에러 =>> 해결
    
    for item in key_ans:
        a=item[0].replace(' ','')
        keyword_list.append(a)
    
    return keyword_list

    #2차 겹치는거 제거후 질문과 가장 유사도가 작은 키워드 추출 하위 3개(word2vec활용해서)10/27 10/30~
    # ***[errror] gpt api에서 뽑은 키워드 워딩과 word2vec모델에서의 토크나이징된 워딩이 서로일치하지 않아 
    # word2vec모델의 메소드에서 key 에러가 났다 
    # -> gpt api에서는 두 단어를 합친 키워드를 사용했으므로, 합친단어를 공백을 제거하고 토큰하나로 취급해 
    # 추가적으로 model에 넣어주는 선(w2v.wv.vocabulary list 에 넣어주는 선)에서 해결





    #2차 겹치는거 제거후 질문과 가장 유사도가 높은 키워드 추출  상위3개(word2vec.wv.similarity활용해서)10/27 10/30~
def define_recomm_keyword2(key_ans,key_qes,model):

    for i in range(len(key_ans)):  #answer 키워드 w2v vocabulary 리스트 삽입
        new_word=key_ans[i]
        model.wv.index_to_key.append(new_word)
        model.wv.key_to_index[new_word] = len(model.wv.index_to_key) - 1
        model.build_vocab([new_word], update=True)
        model.save("reccmodel2")
    

    for i in range(len(key_qes)):  #question 키워드 w2v vocabulary 리스트 삽입
        new_word=key_qes[i]
        model.wv.index_to_key.append(new_word)
        model.wv.key_to_index[new_word] = len(model.wv.index_to_key) - 1
        model.build_vocab([new_word], update=True)
        model.save("reccmodel2")
    
    

    key_ans_score_list=[]
    final_keyword_list=[]
    for i in range(len(key_qes)):
        for j in range(len(key_ans)):
            key_ans_score_list.append(model.wv.similarity(key_qes[i][0],key_ans[j][0]))#남은 answer keyword list와 
    
    indexed_list = list(enumerate(key_ans_score_list))
    sorted_list = sorted(indexed_list, key=lambda x: x[1], reverse=True)
    sorted_list = sorted_list[:3] # 내림차순 정리후 유사도가 가장 높은 상위3개 select
    final_keyword_list = [key_ans[item[0]%(len(key_qes)-1)] for item in sorted_list]
        

    return final_keyword_list


##### 추출한 최종키워드와 답변컬럼 df_a 와의 유사도가 가장높은 상위 n개 문장선택후, 그에 대응되는 질문 추출(최종)
### gpt api로 새롭게 생성한 질문이여도 낫뱃인듯





##### 


#####  main  ###############

df=pd.read_csv("후속질문추천시스템연구/JEUS_application-client_final_DB(문단)_0705_new_eng.csv",encoding='utf-8',header=None)
df_q=df[2]
df_q=df_q[:680]
df_a=df[3] #답변데이터 추출
df_a=df_a[:680]  

df_q=df_q.values.tolist() #리스트로 시작(매개변수)
df_a=df_a.values.tolist()

df_a=tokenize(df_a)
df_q=tokenize(df_q)

 # 미리 정의해둔 모델을 불러옴(제우스 메뉴얼 + 질 + 답 토크나이징 데이터 학습시킨 word2vec모델)


key_ans=list(keyword_extractor_Qanswer(df_a).items())
key_qes=list(keyword_extractor_Qanswer(df_q).items())
recomm_keyword_1=define_recomm_keyword1(key_ans,key_qes)
tokens=df_a+df_q+recomm_keyword_1
model=word2vec.Word2Vec(tokens,min_count=1)
recomm_keyword_2=define_recomm_keyword2(recomm_keyword_1,key_qes,model)
print(recomm_keyword_2)



    


    
    




