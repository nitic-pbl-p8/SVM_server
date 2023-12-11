import requests
import json
import os
from dotenv import load_dotenv

# .envファイルの内容を読み込見込む
load_dotenv()

QUERY ="A patterned, bifold wallet with a brown and tan circular design and a dark brown leather trim and clasp."

# os.environを用いて環境変数を表示させます
API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L12-v2"
headers = {"Authorization": "Bearer " + os.environ['HUGGIN_FACE_KEY']}

with open('../data/case.json') as f:
    docs = json.load(f)

input_sentences = []
for i,doc in enumerate(docs):
    input_sentences.append(doc['title'])


def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

output = query({
    "inputs": {
        "source_sentence": QUERY,
        "sentences": input_sentences
    },
})
ans = dict(zip(input_sentences, output))
ans = sorted(ans.items(), key=lambda x: x[1], reverse=True)  # 変更
ans = dict((x, y) for x, y in ans)  # 変更
print(ans)

# 以下で結果を表示
print(f"Query: {QUERY}")
print("Rank: Title Similarity")
for i, (key, value) in enumerate(ans.items()):
    print(f'{i+1}: {key} {value}')
