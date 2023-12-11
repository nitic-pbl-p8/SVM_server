import pickle

# モデルのオープン
with open('../data/model.pickle', mode='rb') as f:
    clf = pickle.load(f)

# 評価データ
test_data = [[ 0.8530104756355286,  160]]

ans = clf.predict(test_data)
#　モデルを用いた予測


if ans == 0:
    print("True")
if ans == 1:
    print("False")
