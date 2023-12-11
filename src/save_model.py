import json
from sklearn import svm
import pickle

# データベースの読み込み
with open('../data/ans.json') as f:
    docs = json.load(f)

datas = []
targets = []

for doc in docs['true']:
    datas.append([doc['cos'],doc['date']])
    targets.append(0)
   
for doc in docs['false']:
    datas.append([doc['cos'],doc['date']])
    targets.append(1)


clf = svm.SVC(gamma="scale")
clf.fit(datas, targets)

test_data = [[ 0.8530104756355286,  160]]

print(clf.predict(test_data))

with open('../data/model.pickle', mode='wb') as f:
    pickle.dump(clf,f,protocol=2)