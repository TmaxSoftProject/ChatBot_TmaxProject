
import pandas as pd
from gensim.models.word2vec import Word2Vec
from gensim.models import word2vec
import nltk
import gensim
import gensim.models as g
from konlpy.tag import Okt








t=Okt()

tokenized=[]
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
        sentences=sentences.replace('{','')
        sentences=sentences.replace('}','')
        sentences=sentences.replace('`','')

        #print(sentences)
        contents_tokens=t.morphs(sentences)
        tokenized.append(contents_tokens)
    
    return tokenized

df=pd.read_csv("후속질문추천시스템연구/JEUS_application-client_final_DB(문단)_0705_new_eng.csv",encoding='utf-8',header=None)


df_q=df[2]
df_q=df_q[:680]
df_a=df[3] #답변데이터 추출
df_a=df_a[:680]  

df_q=df_q.values.tolist() #리스트로 시작(매개변수)
df_a=df_a.values.tolist()

df_a=tokenize(df_a)
df_q=tokenize(df_q)
tokens=df_a+df_q  ##manual DB까지 포함시키면 오히려 단어 유사도를 잘 못 측정하는거 같아서, 질답 DB만 토크나이징 해서 학습시켰다. 
model=word2vec.Word2Vec(tokens,min_count=1)
model_name="reccmodel"
model.save(model_name)
model=word2vec.Word2Vec.load("reccmodel")
print(model.wv.similarity('애플리케이션','서비스')) #manual DB 포함 0.09876247 미포함 0.16866803
list=model.wv.most_similar('애플리케이션') #most similar 보다 내가 직접 비교하는게 성능이 더 좋을것같다
#print(type(model.wv.most_similar('애플리케이션'
print(list)

