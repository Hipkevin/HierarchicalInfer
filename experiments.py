import pandas as pd
import json

from tqdm import tqdm
from itertools import combinations
from sklearn.cluster import KMeans
from typing import Tuple
from kneed import KneeLocator

# screen the co-occurrence, keep those co-occurrence >= N keywords
N_CONST = 15

# subsumption threshold (relaxed)
ALPHA_CONST = 0.5

# klink threshold
T_CONST = 0.15

def camel(word):
    return  ''.join([item.capitalize() for item in word.lower().split(' ')])

def cso_camel(word):
    return  ''.join([item.capitalize() for item in word.lower().split('_')])

def test(y_hat, y_relation):
    y_relation = set(y_relation)
    y_hat = set(y_hat)
    print(f"y: {len(y_relation)}\ny_hat: {len(y_hat)}\ncorrect: {len(y_relation & y_hat)}")

def subsumption_exp(screened_co_occurrence_list):
    y_hat_relation = []
    for word_tuple, weight in screened_co_occurrence_list:
        w1, w2 = word_tuple
        if w1 in vocab or w2 in vocab:
            if word_freq[w1] >= word_freq[w2]:
                x = w1
                y = w2
            else:
                x = w2
                y = w1

            p_x_y = weight / word_freq[y]
            p_y_x = weight / word_freq[x]

            # print(f'{x} ~ {y}, P(x|y) = {p_x_y}, P(y|x) = {p_y_x}, co-occurrence = {weight}, x({word_freq[x]}) y({word_freq[y]})')
            if p_x_y >= ALPHA_CONST and p_y_x < 1:
                y_hat_relation.append((x, y))

    return y_hat_relation

# klink
import numpy as np


def cosine_sim(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-6)


def longest_common_subsequence(s1: str, s2: str) -> int:
    m, n = len(s1), len(s2)

    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[m][n]


def klink_exp(screened_co_occurrence_list, vocab):
    co_occurrence_matrix = np.zeros((len(vocab), len(vocab)))
    word2id = word2id = {w: idx for idx, w in enumerate(vocab)}
    for words, weight in screened_co_occurrence_list:
        w1, w2 = words
        i, j = word2id[w1], word2id[w2]
        co_occurrence_matrix[i][j] = weight
        co_occurrence_matrix[j][i] = weight

    y_hat_relation = []
    for words, weight in tqdm(screened_co_occurrence_list):
        w1, w2 = words
        i, j = word2id[w1], word2id[w2]
        if w1 in vocab or w2 in vocab:
            if word_freq[w1] >= word_freq[w2]:
                x = w1
                y = w2
            else:
                x = w2
                y = w1

            p_x_y = weight / word_freq[y]
            p_y_x = weight / word_freq[x]

            c_x_y = cosine_sim(co_occurrence_matrix[i, :], co_occurrence_matrix[j, :])
            n_x_y = longest_common_subsequence(x, y)

            l_x_y = (p_x_y - p_y_x) * c_x_y * (1 + n_x_y)

            # print(f'{x} ~ {y}\nL(x, y) = {l_x_y}, P(x|y) = {p_x_y}, P(y|x) = {p_y_x}, cosine = {c_x_y}, LCS = {n_x_y}, co-occurrence = {weight}, x({word_freq[x]}) y({word_freq[y]})')
            if l_x_y > T_CONST:
                y_hat_relation.append((x, y))

    return y_hat_relation


import requests


def relation_infer(w1, w2, prompt, model_name):
    base_url = "http://ip"
    response = requests.post(f"{base_url}:11434/api/chat", json={
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "stream": False
    }, headers={
        "Content-Type": "application/json",
    })

    # 检查响应状态码
    if response.status_code == 200:
        response_data = response.json()
        return response_data['message']['content']
    else:
        print(f"ERROR, code: {response.status_code}")
        print(response.text)


def LLM_exp(screened_co_occurrence_list, model_name, prompt):
    y_hat_relation = []
    loop = tqdm(screened_co_occurrence_list, desc=f'LLM-{model_name}')
    for w1, w2 in loop:
        res = relation_infer(w1, w2, prompt, model_name)
        if res == '1':
            y_hat_relation.append((w1, w2))
        elif res == '0':
            y_hat_relation.append((w2, w1))

    return y_hat_relation


def get_emb_batch(w, model_name):
    t = list(w)
    base_url="http://ip"
    response = requests.post(f"{base_url}:11434/api/embed", json={
      "model": model_name,
      "input": t
      }, headers={
        "Content-Type": "application/json",
    })

    # 检查响应状态码
    if response.status_code == 200:
        response_data = response.json()
        return np.array(response_data['embeddings'])
    else:
        print(f"ERROR, code: {response.status_code}")
        print(response.text)


def knee_point_check(series):
    kneed_check = KneeLocator(range(len(series)), series, curve="convex", direction="decreasing", online=True)
    return min(kneed_check.all_elbows)


def get_cluster_by_kmeans(X, n_clusters_range=range(2, 15)) -> Tuple[KMeans, list]:
    sse_list = []
    model_dict = {}
    for k in tqdm(n_clusters_range):
        kmeans = KMeans(n_clusters=k, n_init='auto', random_state=42)
        kmeans.fit(X)

        model_dict[k] = kmeans
        sse_list.append(kmeans.inertia_)

    k = knee_point_check(sse_list) + 2
    print(k)

    return model_dict[k], sse_list


if __name__ == '__main__':
    # data process
    columns = ["DE", "WC", "PY"]
    data = pd.read_csv('cs.tsv', sep='\t')[columns].dropna()

    content = []
    for idx, line in data.iterrows():
        de_list = list(set([camel(item) for item in line["DE"].strip().split('; ')]))
        content.append(de_list)

    # keywords frequency and co-occurrence
    word_freq = {}
    for item in content:
        for w in item:
            word_freq[w] = word_freq.get(w, 0) + 1

    co_relation = list()
    for line in tqdm(content):
        co_relation.extend(combinations(line, 2))

    co_occurrence = {}
    for co_line in tqdm(co_relation):
        w1, w2 = co_line[0], co_line[1]

        if (w1, w2) not in co_occurrence:

            if (w1, w2) in co_occurrence:
                co_occurrence[(w2, w1)] += 1
            else:
                co_occurrence[(w1, w2)] = 1

        else:
            co_occurrence[(w1, w2)] += 1

    co_occurrence_list = list(co_occurrence.items())
    screened_co_occurrence_list = [item for item in co_occurrence_list if item[1] >= N_CONST]

    vocab = set()
    for item in screened_co_occurrence_list:
        w1, w2 = item[0]
        vocab.update({w1})
        vocab.update({w2})

    # standard
    openalex_y_relation = []
    cs_concepts = json.loads(open('cs_concepts.json', 'r').read())
    for idx, node in cs_concepts.items():

        if node['parent_ids'] and camel(node['display_name']) in vocab:
            for parent_name in node['parent_display_names']:
                if camel(parent_name) in vocab:
                    openalex_y_relation.append((camel(parent_name), camel(node['display_name'])))

    cso_y_relation = []
    cso = pd.read_csv('CSO.3.4.1.csv', header=None)
    for idx, line in cso.iterrows():
        relation = line[1]
        if "superTopicOf" in relation:
            parent = cso_camel(line[0].strip('>').split('/')[-1])
            child = cso_camel(line[2].strip('>').split('/')[-1])

            if parent in vocab and child in vocab:
                cso_y_relation.append((parent, child))

    # experiments
    y_hat_relation = subsumption_exp(screened_co_occurrence_list)
    test(y_hat_relation, openalex_y_relation)
    test(y_hat_relation, cso_y_relation)

    y_hat_relation = klink_exp(screened_co_occurrence_list, vocab)
    test(y_hat_relation, openalex_y_relation)
    test(y_hat_relation, cso_y_relation)

    model_name = "qwen2.5:72b"
    prompt = f"Hypernymy and hyponymy are the semantic relations between a generic term (hypernym) and a more specific term (hyponym). Determine the hierarchical relationship between two words based on subject classification. Answer 1 if {w1} is the superordinate of {w2}, 0 if {w2} is the superordinate of {w1}, or -1 if there is no superordinate relationship between the two.Do not output any text other than 1, 0, and -1."
    y_hat_relation = LLM_exp([item[0] for item in screened_co_occurrence_list], model_name, prompt)
    test(y_hat_relation, openalex_y_relation)
    test(y_hat_relation, cso_y_relation)

