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
print(type(dict1))

dict=str(dict1)
print(type(dict))
dict=eval(dict)
print(type(dict))
print(len(dict))
l=list(dict.items())
print(l[0][0])
print(l[0][1])
print(len(l))
del l[0]
print(l)

a=[1,2,3]
del a[0]
print(a)
b=a+l
print(b)