# 프로젝트 목적

---

- Dizzycode 프로젝트 중 일부 기능(방 추천 기능)을 별도로 담당하는 서버 제작

# 기술 스택 및 사용목적

---

- FAISS ⇒ 밀집 벡터 유사도 검사 및 클러스터링
- Sentence Transformer(LaBSE) ⇒ Language-agnostic한 문장 임베딩 기능 활용
- Flask ⇒ 서버 구성

# 주요 기능

---

1. 방 추가 시 이를 room_data에 추가하고 room_name을 임베딩하여 FAISS index에 저장함
2. 방 삭제 시 해당 room_data를 삭제하고, 임베딩 된 내용을 FAISS index에서 삭제함
3. 주어진 키워드와 의미상 가장 가까운 k개의 방 이름(room_name)을 검색하고, 해당 방의 식별자(room_id) 및 distance(키워드와 room_name 벡터 간의 거리)와 함께 반환함 

# 의존성

---

```
# OS나 CPU에 따라 오류가 발생할 수 있음에 주의
torch==2.3.1+cpu

sentence-transformers
faiss-cpu
flask
numpy
```

# 기타

---

- FAISS index와 room_data는 모두 pickle 형태로 저장 및 관리함
