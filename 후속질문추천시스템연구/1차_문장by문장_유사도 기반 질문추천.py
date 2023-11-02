import konlpy
from konlpy.tag import Okt
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import scipy as sp

##### 1. 벡터 방법 tf-idf 방법   ######v


#### 형태소 분석기
t=Okt()

#### data slicing(한글질문 위주, 100개만 잘라서 테스트)
df=pd.read_csv("JEUS_application-client_final_DB(문단)_0705_new_eng.csv",encoding='utf-8',header=None)
df=df[2]
df=df[:100]  
contents=df.values.tolist()
#print(len(contents))
#print(df.head(3))



#### 형태소 단위로 tokenize
contents_tokens=[t.morphs(row) for row in contents]
print('토큰나이징\n', contents_tokens[5])



#### 형태소로 나눈 토큰을 띄어쓰기로 구분후 한문장으로 붙이기
tok_sentence_forVectorize=[]

for content in contents_tokens:
    sentence=''
    for word in content:
        sentence=sentence+' '+word
    

    tok_sentence_forVectorize.append(sentence)
print(tok_sentence_forVectorize[0])
#### 타겟 문장 설정하고, 현재 질문 DB에 동일한 질문이 있을시 질문DB에서 해당 질문을 삭제한다.
target_q=tok_sentence_forVectorize[8]

print("타겟 문장:\n", target_q)
print("\n\n------------------")
#print('원래문장\n',df[0:1],'\n형태소토큰단위로 쪼개진 문장\n', tok_sentence_forVectorize[0])


#### tf-idf vectorize
vectorizer = TfidfVectorizer(min_df=1, decode_error='ignore')
vec=vectorizer.fit_transform(tok_sentence_forVectorize)
num_samples, num_features =vec.shape
#print(num_samples, num_features)

#### 이제 타겟이 될 질문을 정한다. 타겟문장의 번호로 설정(이거는 무작위로 번호 인덱스)

target_q=[target_q] #임베딩하기위해 리스트 변환
target_q_vec =vectorizer.transform(target_q)
print("타겟 질문의 임베딩 결과: \n", target_q_vec)
print('\n\n\n\n---------------------------------------')


####  단순 벡터 내적 거리계산 함수 정의
def dist_raw(v1,v2):
    delta = v1 - v2
    return sp.linalg.norm(delta.toarray())





#### 최적의 유사도를 가진 질문을 찾아보자[일단 best score1개만] (한글형태소 기반 토크나이징, tf-idf 벡터화(임베딩), norm메소드를 이용한 벡터거리 계산 이용)
best_q=None
best_dist=65535
best_i=[]
dis_list=[]

for i in range(0,num_samples):
    if i != 8: #제거한 문장 제외
        post_vec=vec.getrow(i)

        dis=dist_raw(post_vec, target_q_vec)
        print("==Post %i with dist=%.3f : %s" % (i,dis,contents[i]))

        if dis < best_dist:
            best_dist=dis
            best_i.append(i)


print('\n\n\n\n----------------------------------------------------------')
print("Best recommendation question is %i -> %s, dist = %.3f" % (best_i[-1],contents[best_i[-1]],best_dist))
print("Best recommendation question is %i -> %s, dist = %.3f" % (best_i[-2],contents[best_i[-2]],best_dist))
print("Best recommendation question is %i -> %s, dist = %.3f" % (best_i[-3],contents[best_i[-3]],best_dist))
#best_i 리스트에서 최상위 n개의 score를 가진 관련질문을 뽑을 수 있다.




####### 2.코사인 유사도로 구하기 ######

