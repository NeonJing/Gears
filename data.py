import random
import nltk
import string
import pandas as pd
import csv
import time
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.tree import Tree
from negate import Negator
from aip import AipNlp
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from rouge import Rouge

# 替换为您的百度翻译 API 密钥信息
APP_ID = '20230713001742746'
API_KEY = 'http://api.fanyi.baidu.com/api/trans/vip/translate'
SECRET_KEY = 'jo9vA0LQoOt4IOM7CuKr'

# 创建百度翻译客户端对象
client = AipNlp(APP_ID, API_KEY, SECRET_KEY)

# 导入数据集
df = pd.read_csv("E:\\desktop\\gears\\Gen_review_factuality_train_1.csv", encoding="gbk")

feedback = df["feedback"]
l = len(feedback)
nouns = list()
pid = df["pid"]
doc = df["doc"]
label = df["label"]
feedback_new = list()
pid_new = list()
doc_new = list()
label_new = list()




# 获取反馈中的所有名词
def get_nouns(sentence):
    tokens = word_tokenize(sentence)
    tokens = [w.lower() for w in tokens]
    exclude = set(string.punctuation)
    stripped = [''.join([ch for ch in w if ch not in exclude]) for w in tokens]
    words = [word for word in stripped if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]
    tags = nltk.pos_tag(words)
    nouns = [word for word, pos in tags if (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS')]
    return nouns

def rouge_score(sentence1, sentence2):
    # 使用rouge库来计算Rouge score
    rouge = Rouge()
    # 将两个句子都转换为小写，以保持一致性
    # Rouge score返回的是一个字典，包含多个 Rouge score 指标，如 Rouge-1、Rouge-2、Rouge-L 等
    # 这里取 Rouge-L score 作为相似度度量
    scores = rouge.get_scores(sentence1.lower(), sentence2.lower())
    rouge_l_score = scores[0]['rouge-l']['f']
    return rouge_l_score

for i in range(l):
    if get_nouns(str(feedback[i])):
        nouns.append(random.choice(get_nouns(str(feedback[i]))))


class TextTransformation(object):
    def __init__(self):
        super(TextTransformation, self).__init__()
        self.nouns = nouns
        self.get_nouns = get_nouns
        self.negator = Negator()
        self.pid_new = []
        self.doc_new = []
        self.feedback_new = []
        self.label_new = []


    def __len__(self):
                # 定义 __len__ 方法，返回 feedback_new 列表的长度
                return len(self.feedback_new)

    def __getitem__(self, index):
                # 定义 __getitem__ 方法，使 TextTransformation 对象可以像列表一样通过索引访问
                return self.feedback_new[index]

    def negate(self, sentence):
        if self.negator.negate_sentence(sentence):
            return self.negator.negate_sentence(sentence)

    def get_adjective(self, sentence):
        lemmatizer = WordNetLemmatizer()
        tokens = word_tokenize(sentence)
        tokens = [w.lower() for w in tokens]
        exclude = set(string.punctuation)
        stripped = [''.join([ch for ch in w if ch not in exclude]) for w in tokens]
        words = [word for word in stripped if word.isalpha()]
        from nltk.corpus import stopwords
        stop_words = set(stopwords.words('english'))
        words = [w for w in words if not w in stop_words]

        # 获取单词的定义并检查词性是否为形容词
        candidates = [word for word in words if lemmatizer.lemmatize(word, pos=wn.ADJ) == word]
        return candidates

    def replace_adjectives(self, sentence):
        adjectives = self.get_adjective(sentence)
        if adjectives:
            replaced = random.choice(adjectives)
            antonyms = [lemma.antonyms()[0].name() for synset in wn.synsets(replaced, pos=wn.ADJ) for lemma in
                        synset.lemmas() if lemma.antonyms()]
            if antonyms:
                antonym = random.choice(antonyms)
                new_sentence = sentence.replace(replaced, antonym)
                return new_sentence
        return None

    def noun_swap(self, sentence):
        if self.get_nouns(str(sentence)):
            noun_replaced = random.choice(self.get_nouns(str(sentence)))
            new_sentence = sentence.replace(noun_replaced, random.choice(nouns))
            return new_sentence

    def duplicate(self, sentence):
        duplicated = random.choice(word_tokenize(sentence))
        new_sentence = sentence.replace(duplicated, duplicated + " " + duplicated)
        return new_sentence

    def delete(self, sentence):
        deleted = random.choice(word_tokenize(sentence))
        new_sentence = sentence.replace(deleted, "")
        return new_sentence


    def translate_cn_back(self, sentence):
        result = client.lexer(sentence)
        if 'items' in result and len(result['items']) > 0:
            trans = result['items'][0]['formal']
            back_trans = client.lexer(trans)
            if 'items' in back_trans and len(back_trans['items']) > 0:
                return back_trans['items'][0]['formal']
        return ""

    def translate_de_back(self, sentence):
        result = client.lexer(sentence)
        if 'items' in result and len(result['items']) > 0:
            trans = result['items'][0]['formal']
            back_trans = client.lexer(trans)
            if 'items' in back_trans and len(back_trans['items']) > 0:
                return back_trans['items'][0]['formal']
        return ""

    def translate_fr_back(self, sentence):
        result = client.lexer(sentence)
        if 'items' in result and len(result['items']) > 0:
            trans = result['items'][0]['formal']
            back_trans = client.lexer(trans)
            if 'items' in back_trans and len(back_trans['items']) > 0:
                return back_trans['items'][0]['formal']
        return ""

    def translate_es_back(self, sentence):
        result = client.lexer(sentence)
        if 'items' in result and len(result['items']) > 0:
            trans = result['items'][0]['formal']
            back_trans = client.lexer(trans)
            if 'items' in back_trans and len(back_trans['items']) > 0:
                return back_trans['items'][0]['formal']
        return ""

    def translate_ru_back(self, sentence):
        result = client.lexer(sentence)
        if 'items' in result and len(result['items']) > 0:
            trans = result['items'][0]['formal']
            back_trans = client.lexer(trans)
            if 'items' in back_trans and len(back_trans['items']) > 0:
                return back_trans['items'][0]['formal']
        return ""

    def unigram_noising(self, sentence, noise_prob=0.1):
        tokens = word_tokenize(sentence)
        noised_tokens = []
        for token in tokens:
            if random.random() < noise_prob:
                if random.random() < 0.5:
                    continue
                else:
                    if self.nouns:
                        noised_tokens.append(random.choice(self.nouns))
            else:
                noised_tokens.append(token)
        noised_sentence = ' '.join(noised_tokens)
        return noised_sentence

    def InstanceCrossoverAugmentation(self, sentence1, sentence2, mixup_alpha=0.5):
        # 获取两个句子的单词列表
        tokens1 = word_tokenize(sentence1)
        tokens2 = word_tokenize(sentence2)
        # 随机生成一个混合比例，用于控制混合程度
        mixup_ratio = random.uniform(0, mixup_alpha)
        # 混合两个句子的单词列表
        mixed_tokens = tokens1[:int(len(tokens1) * mixup_ratio)] + tokens2[int(len(tokens2) * mixup_ratio):]
        mixed_sentence = ' '.join(mixed_tokens)
        return mixed_sentence

    def add_similar_feedback_pair(feedback_list):
        def calculate_rouge_score(reference, candidate):
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            scores = scorer.score(reference, candidate)
            return scores

        def select_similar_feedback(feedback_list):
            min_diff = float('inf')
            most_similar_pair = (0, 0)

            for i in range(len(feedback_list)):
                for j in range(i + 1, len(feedback_list)):
                    ref_feedback = feedback_list[i]
                    cand_feedback = feedback_list[j]
                    scores = calculate_rouge_score(ref_feedback, cand_feedback)
                    avg_rouge_score = (scores['rouge1'][2] + scores['rouge2'][2] + scores['rougeL'][2]) / 3

                    # 更新最小差异度和对应的两个feedback索引
                    if avg_rouge_score < min_diff:
                        min_diff = avg_rouge_score
                        most_similar_pair = (i, j)

            return most_similar_pair

        # 使用Rouge score找到一对相似的反馈
        similar_pair = select_similar_feedback(feedback_list)
        index_1, index_2 = similar_pair

        # 获取选定的两个反馈文本
        selected_feedback_1 = feedback_list[index_1]
        selected_feedback_2 = feedback_list[index_2]

        # 将新的数据条目添加到相应的列表中
        pid_new.append(pid[index_1])
        doc_new.append(doc[index_1])
        feedback_new.append(selected_feedback_2)  # 将相似的反馈添加到对方的feedback列表中
        label_new.append(label[index_1])

        # 将选定的两个反馈文本再次添加到相应的列表中
        pid_new.append(pid[index_2])
        doc_new.append(doc[index_2])
        feedback_new.append(selected_feedback_1)
        label_new.append(label[index_2])

    def convert_to_passive(self, sentence):
        tokens = word_tokenize(sentence)
        tagged_tokens = nltk.pos_tag(tokens)

        # 寻找是否已经包含了助动词或情态动词
        has_auxiliary = any(tag in ['MD', 'VBZ', 'VBP', 'VBD', 'VBN'] for _, tag in tagged_tokens)

        # 只有当句子中没有助动词或情态动词时，才进行主被动语态转换
        if not has_auxiliary:
            main_verb = None
            subject = None
            for i, (word, tag) in enumerate(tagged_tokens):
                if tag == 'VB':  # 找到动词
                    main_verb = word
                    if i > 0 and tagged_tokens[i - 1][1] == 'DT':  # 查找前面是否有冠词
                        subject = tagged_tokens[i - 2][0] + ' ' + tagged_tokens[i - 1][0]
                    else:
                        subject = tagged_tokens[i - 1][0]
                    break

            if main_verb and subject:
                passive_sentence = f"{subject} is {main_verb} by {subject}"
                return passive_sentence
            else:
                return sentence
        else:
            return sentence

    def convert_to_active(self, sentence):
        tokens = word_tokenize(sentence)
        tagged_tokens = nltk.pos_tag(tokens)

        # 寻找是否已经包含了助动词或情态动词
        has_auxiliary = any(tag in ['MD', 'VBZ', 'VBP', 'VBD', 'VBN'] for _, tag in tagged_tokens)

        # 只有当句子中没有助动词或情态动词时，才进行主动语态转换
        if not has_auxiliary:
            # 找到被动语态结构（主语是 nsubjpass，谓词是 VBN）
            main_verb = None
            subject = None
            for i, (word, tag) in enumerate(tagged_tokens):
                if tag == 'VBN':  # 找到过去分词
                    main_verb = word
                    if i > 1 and tagged_tokens[i - 1][1] == 'IN' and tagged_tokens[i - 2][1] == 'DT':  # 查找是否有冠词和介词
                        subject = tagged_tokens[i - 3][0]
                    elif i > 0 and tagged_tokens[i - 1][1] == 'DT':  # 查找是否有冠词
                        subject = tagged_tokens[i - 2][0] + ' ' + tagged_tokens[i - 1][0]
                    else:
                        subject = tagged_tokens[i - 1][0]
                    break

            # 如果找到了被动语态结构，转换为主动语态
            if main_verb and subject:
                active_sentence = f"{subject} {main_verb} {subject}"
                return active_sentence
            else:
                return sentence
        else:
            return sentence




# 创建 TextTransformation 对象
TextTransformation = TextTransformation()



# 对每个反馈文本进行处理和增强，并生成新的数据
for i in range(l):
    if (i % 10 == 0):
        time.sleep(30)
    # 保留原始数据
    pid_new.append(pid[i])
    doc_new.append(doc[i])
    feedback_new.append(feedback[i])
    label_new.append(label[i])

    # 进行否定处理
    if TextTransformation.negate(feedback[i]):
        pid_new.append(pid[i])
        doc_new.append(doc[i])
        new_feedback = TextTransformation.negate(feedback[i])
        feedback_new.append(new_feedback)
        # 对标签进行处理：0 变为 1，1 变为 0
        if label[i] == 0:
            label_new.append(1)
        elif label[i] == 1:
            label_new.append(0)

    # 进行名词替换
    if TextTransformation.noun_swap(feedback[i]):
        pid_new.append(pid[i])
        doc_new.append(doc[i])
        new_feedback = TextTransformation.noun_swap(feedback[i])
        feedback_new.append(new_feedback)
        # 对标签保持不变
        label_new.append(label[i])

    # 进行形容词反义词替换
    if TextTransformation.replace_adjectives(feedback[i]):
        pid_new.append(pid[i])
        doc_new.append(doc[i])
        new_feedback = TextTransformation.replace_adjectives(feedback[i])
        feedback_new.append(new_feedback)
        # 对标签进行处理：0 变为 1，1 变为 0
        if label[i] == 0:
            label_new.append(1)
        elif label[i] == 1:
            label_new.append(0)

    # 添加Unigram Noising
    if random.random() < 0.5:
        if feedback[i]:
            noised_feedback = TextTransformation.unigram_noising(feedback[i], noise_prob=0.1)
            pid_new.append(pid[i])
            doc_new.append(doc[i])
            feedback_new.append(noised_feedback)
            label_new.append(label[i])

    if random.random() < 0.5:
        random_index = random.randint(0, l - 1)
        mixed_feedback = TextTransformation.InstanceCrossoverAugmentation(feedback[i], feedback[random_index], mixup_alpha=0.5)
        pid_new.append(pid[i])
        doc_new.append(doc[i])
        feedback_new.append(mixed_feedback)
        # 对标签进行混合
        mixed_label = (label[i] + label[random_index]) / 2
        # 定义混合后的标签
        if mixed_label < 0.5:
            mixed_label = 0
        else:
            mixed_label = 1
        label_new.append(mixed_label)

        # 转换为被动语态
        passive_sentence = TextTransformation.convert_to_passive(feedback[i])
        if passive_sentence:
            pid_new.append(pid[i])
            doc_new.append(doc[i])
            feedback_new.append(passive_sentence)
            label_new.append(label[i])

        # 转换为主动语态
        active_sentence = TextTransformation.convert_to_active(passive_sentence)
        if active_sentence:
            pid_new.append(pid[i])
            doc_new.append(doc[i])
            feedback_new.append(active_sentence)
            label_new.append(label[i])

        # 找到不同pid的feedback，通过Rouge score来判断两者的相似度，将别人的feedback加入到自己的feedback
        for j in range(i + 1, l):
            if pid[i] != pid[j]:
                # 计算Rouge score
                rouge_l_score = rouge_score(feedback[i], feedback[j])
                # 判断相似度较高的两个feedback，将别人的feedback加入到自己的feedback
                if rouge_l_score >= 0.5:
                    pid_new.append(pid[i])
                    doc_new.append(doc[i])
                    feedback_new.append(feedback[j])  # 将别人的feedback作为新的数据条目加入
                    label_new.append(label[i])




# 将数据写入新的 CSV 文件
d = len(label_new)
with open('E:\\desktop\\gears\\Gen_review_factuality_train_create.csv', 'w', newline='', encoding="UTF-8") as file:
    writer = csv.writer(file)
    writer.writerow(['pid', 'doc', 'feedback', 'label'])
    for i in range(l):
        writer.writerow([pid[i], doc[i], feedback[i], label[i]])
    for i in range(d):
        if feedback_new[i]:
            # 在新的 feedback 和 doc 之间添加逗号，将它们分隔开
            writer.writerow([pid_new[i], doc_new[i], feedback_new[i], label_new[i]])