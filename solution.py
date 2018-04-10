# coding=utf-8
import sys
import jieba
import nltk
import jieba.analyse


file = 'tmt.txt'
with open(file, 'r', encoding='utf8') as fr:
    s = fr.read()
tmt_list = jieba.cut(s, cut_all=True)

file = 'food.txt'
with open(file, 'r', encoding='utf8') as fr:
    s = fr.read()
food_list = jieba.cut(s, cut_all=True)

file = 'eng.txt'
with open(file, 'r', encoding='utf8') as fr:
    s = fr.read()
eng_list = jieba.cut(s, cut_all=True)

train_list = ([(tmt_word, 'tmt') for tmt_word in tmt_list]
              + [(food_word, 'food') for food_word in food_list]
              + [(eng_word, 'eng') for eng_word in eng_list])


food_freq_list = nltk.FreqDist(food_list)
tmt_freq_list = nltk.FreqDist(tmt_list)
eng_freq_list = nltk.FreqDist(eng_list)
def freq(s, t):
    if t == 'tmt':
        return {s: tmt_freq_list[s]}
    if t == 'food':
        return {s: food_freq_list[s]}
    if t == 'eng':
        return {s: eng_freq_list[s]}


feature_set = [(freq(word, t), t) for (word, t) in train_list]

classifier = nltk.NaiveBayesClassifier.train(feature_set)
#
# print(classifier.classify({'混凝土': 0}))
# classifier.show_most_informative_features(10)

file = 'test.txt'
test_list = []
with open(file, 'r', encoding='utf8') as fr:
    for l in fr:
        test_list.append(l)

eng = 0
food = 0
tmt = 0
for t in test_list:
    line_list = jieba.analyse.textrank(t, topK=1, withWeight=True, allowPOS=('ns', 'n', 'vn', 'v'))
    s = (classifier.classify({line_list[0][0]: 0}))
    if s == 'eng':
        eng += 1
    elif s == 'food':
        food += 1
    else:
        tmt += 1
sum = eng + food + tmt
print({'eng': eng/sum, 'food': food/sum, 'tmt': tmt/sum})




