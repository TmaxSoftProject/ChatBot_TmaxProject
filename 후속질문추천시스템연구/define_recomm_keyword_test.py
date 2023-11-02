# string_A = "클라이언트 컨테이너"
# string_B = "클라이언트"

# if string_B.lower() in string_A.lower():
#     print("string_B는 string_A에 포함됩니다.")
# else:
#     print("string_B는 string_A에 포함되지 않습니다.")
from gensim.models.word2vec import Word2Vec
from gensim.models import word2vec
import pandas as pd
from konlpy.tag import Okt
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
    # 추가적으로 model에 넣어주는 선에서 해결
    
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
    
    key_ans_list=[]
    final=[]
    for i in range(len(key_qes)):
        for j in range(len(key_ans)):
            key_ans_list.append(model.wv.similarity(key_qes[i][0],key_ans[j][0]))#여기서 판별함수를 어떤걸 사용할지
    
    indexed_list = list(enumerate(key_ans_list)) #error부분
    sorted_list = sorted(indexed_list, key=lambda x: x[1], reverse=True)
    sorted_list = sorted_list[-3:]
    final = [key_ans[item[0]%(len(key_qes)-1)] for item in sorted_list]
    #final2 = [key_ans[i] for i in final]
        

    return final

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

#model=word2vec.Word2Vec.load("reccmodel")
# vocab = model.wv.vocab
# sorted(vocab, key=vocab.get, reverse=True)[:30]
dict1={
   "애플리케이션 클라이언트": "JEUS 서버와 별도의 JVM에서 수행되는 standalone 클라이언트",
   "JEUS": "클라이언트 컨테이너를 사용하여 Jakarta EE 환경에서 애플리케이션 호출 및 서비스 제공",
   "클라이언트 컨테이너": "Naming Service, Scheduler, Security 등의 JEUS 서비스 사용",
   "JEUS 클라이언트 라이브러리": "JNDI, Security 등의 서비스 사용 가능하지만 Dependency Injection, JEUS Scheduler 등의 서비스는 사용 불가",
   "Jakarta EE 스펙": "더 자세한 내용 확인 가능",
   "JEUS XML 스키마": "jeusclientdd.xml로 참고 가능"
}
dict2={ 
    "JEUS": None,
    "클라이언트 라이브러리": None,
    "서비스": None,
    "이용": None
}
dict1=list(dict1.items())
dict2=list(dict2.items())
# a=len(dict1)
# for i in range(len(dict2)):
#     for j in range(len(dict1)-1,-1,-1):
#         if dict2[i][0] in dict1[j][0]:
#             print('delete1')
#             print(j)
#             del dict1[j] # **error
#             print(dict1)

df=pd.read_csv("후속질문추천시스템연구/JEUS_application-client_final_DB(문단)_0705_new_eng.csv",encoding='utf-8',header=None)


df_q=df[2]
df_q=df_q[:680]
df_a=df[3] #답변데이터 추출
df_a=df_a[:680]  

df_q=df_q.values.tolist() #리스트로 시작(매개변수)
df_a=df_a.values.tolist()

df_a=tokenize(df_a)
df_q=tokenize(df_q)
lis=define_recomm_keyword1(dict1,dict2)
print(lis)
tokens=[]
tokens=df_a+df_q+lis
print(tokens[-3:])
# print(tokens) ##manual DB까지 포함시키면 오히려 단어 유사도를 잘 못 측정하는거 같아서, 질답 DB만 토크나이징 해서 학습시켰다. 
model=word2vec.Word2Vec(tokens,min_count=1)
model_name="reccmodel2"
model.save(model_name)
model=word2vec.Word2Vec.load(model_name)
final=define_recomm_keyword2(lis,dict2,model)


print(final)
print(lis[i] for i in final)
#rint(lis[2],lis[1],lis[0])


# new_word = "애플리케이션클라이언트"
# model.wv.index_to_key.append(new_word)
# model.wv.key_to_index[new_word] = len(model.wv.index_to_key) - 1
# model.build_vocab([new_word], update=True)
# model.save(model_name)
# #model.train([new_word], total_examples=model.corpus_count, epochs=model.epochs)

# voc=model.wv.index_to_key
# # voc=voc+list
# if "애플리케이션클라이언트" in voc:
#     print("1")
    
# print(voc)

# print(model.wv.most_similar('애플리케이션클라이언트'))
# # # print(a)

# # l=[1,2,3,4,5,6,7,8,9]
# # b=l[-3:]
# # print(b)