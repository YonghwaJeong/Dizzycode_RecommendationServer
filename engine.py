from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
from flask import Flask, request, jsonify

# 다국어 모델 사용
model = SentenceTransformer('sentence-transformers/LaBSE')
d = model.get_sentence_embedding_dimension()

index = faiss.IndexFlatL2(d)
# {room_id: (room_name, faiss_index)} 구조
room_data = {}  

# Pickle 생성 혹은 로드 메서드
def load_data():
    global index, room_data
    try:
        with open('faiss_index.pkl', 'rb') as f:
            index = pickle.load(f)
    except FileNotFoundError:
        index = faiss.IndexFlatL2(d)

    try:
        with open('room_data.pkl', 'rb') as f:
            room_data = pickle.load(f)
    except FileNotFoundError:
        room_data = {}

# 전역으로 관리되는 index, room_data를 pickle에 저장하는 메서드
def save_data():
    with open('faiss_index.pkl', 'wb') as f:
        pickle.dump(index, f)
    with open('room_data.pkl', 'wb') as f:
        pickle.dump(room_data, f)

app = Flask(__name__)
load_data()

# 검색 기능 (query = keyword / k = topN)
@app.route('/search', methods=['POST'])
def search():
    data = request.json
    query = data.get('query')
    k = data.get('k', 20)
    
    # 요청 본문에 query(keyword)가 없는 경우
    if not query:
        return jsonify({"status": "error", "message": "키워드가 존재하지 않습니다."}), 400

    try:
        # 모델을 사용해 주어진 keyword의 벡터를 계산하고, 벡터 간 거리가 가까운 k개 항목의 (거리, faiss_index) 튜플 반환
        query_vector = model.encode([query]).astype('float32')
        distances, indices = index.search(query_vector, k)

        results = []
        for idx, i in enumerate(indices[0]):
            # faiss_index의 위치와 room_id를 매핑하여 결과 반환
            room_id = [key for key, value in room_data.items() if value[1] == i][0]
            room_name = room_data[room_id][0]

            # roomId, keyword와 거리, roomName을 반환
            results.append({
                "roomId": room_id,
                "distance": float(distances[0][idx]),
                "roomName": room_name
            })

        return jsonify(results)

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# 방 추가에 따른 room_data, faiss_index(sentence vector) 추가
@app.route('/add_room', methods=['POST'])
def add_room():
    data = request.json
    room_name = data.get('roomName')
    room_id = data.get('roomId')

    # 요청 본문이 적절하지 않은 경우
    if not room_name or not room_id:
        return jsonify({"status": "error", "message": "Room name and ID are required"}), 400

    try:
        room_vector = model.encode([room_name]).astype('float32')
        # 현재 인덱스 개수를 기반으로 새 위치 지정
        faiss_index = index.ntotal
        room_data[room_id] = (room_name, faiss_index)
        index.add(room_vector)

        save_data()

        return jsonify({"status": "success"})
    
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# 방 삭제에 따른 room_data, faiss_index(sentence vector) 삭제
@app.route('/delete_room', methods=['POST'])
def delete_room():
    data = request.json
    room_id = data.get('roomId')

    # 요청 본문이 적절하지 않은 경우
    if not room_id:
        return jsonify({"status": "error", "message": "Room ID is required"}), 400

    try:
        if room_id in room_data:
            # room_id와 대응되는 faiss_index, room_data를 삭제
            faiss_index = room_data[room_id][1]
            index.remove_ids(np.array([faiss_index], dtype='int64'))
            del room_data[room_id]

            save_data()
            
            return jsonify({"status": "success"})
        else:
            return jsonify({"status": "error", "message": "Room ID not found"}), 404
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000)