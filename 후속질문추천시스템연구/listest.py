import pandas as pd
from konlpy.tag import Okt
t= Okt()
#df=['자동차도장기능사, 실무교육, 생산설비의, 자동화로, 없다, 없다, 없다, 건설현장...','건축전기설비기술사, 건설, 수주가, 없어서, 없다, 매타기, 드라이버, 가위, C...','건축전기설비기술사, 신축, 건설경기가, 좋지, 않아서, 없다, 건축설계표, 오토캐...']
tokenized=[]
#df=pd.read_csv("JEUS_application-client_final_DB(문단)_0705_new_eng.csv",encoding='utf-8',header=None)
df=pd.read_csv("JEUS_application-client_final_DB(문단)_0705_new_eng.csv",encoding='utf-8',header=None)
df_a=df[2]
print(df_a)
df_a=df_a[:10]
print(df_a)
# df_a=df_a[:10]  
contents=df_a.values.tolist()
for sentences in contents:  #### 전처리및 토크나이징
        sentences=str(sentences)
        sentences=sentences.replace(',','')
        sentences=sentences.replace('"','')
        sentences=sentences.replace("'",'')
        sentences=sentences.replace('-','')
        sentences=sentences.replace('(','')
        sentences=sentences.replace(')','')

        #print(sentences)
        contents_tokens=t.morphs(sentences)
        print(contents_tokens)
        print('\n xxx')
        tokenized.append(contents_tokens)
print(tokenized)