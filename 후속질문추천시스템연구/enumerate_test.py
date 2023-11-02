# 원본 리스트
original_list = [9, 2, 5, 1, 7, 4, 8, 3, 6]

# 내림차순으로 정렬하되, 원래의 인덱스를 기억할 수 있도록 튜플로 구성
indexed_list = list(enumerate(original_list))

# 내림차순으로 정렬
sorted_list = sorted(indexed_list, key=lambda x: x[1], reverse=True)

# 정렬된 리스트 출력
print(sorted_list)

# 정렬된 리스트에서 원래의 인덱스 가져오기
sorted_indices = [item[0] for item in sorted_list]

# 원래의 인덱스 출력
print(sorted_indices)
