from itertools import combinations
from tqdm import tqdm
from math import sqrt
import numpy as np

def cosine_sim(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-6)

def get_frequency(content, freq_threshold=10):
    """
    pre-process for keywords set

    :param content: content -> List[List[str]]
    :param freq_threshold:
    :return:
    """

    word_freq = {}
    for line in tqdm(content):
        line = list(set(line))
        for w in line:
            word_freq[w] = word_freq.get(w, 0) + 1

    vocab = set([w for w, freq in word_freq.items() if freq >= freq_threshold])

    return word_freq, vocab

def get_co_occurrence(content, vocab):
    """
    co-occurrence in author keywords

    :param content: source keywords list
    :param vocab: screened by frequency
    :return:
    """

    co_relation = list()
    for line in tqdm(content):
        line = list(set(line) & vocab)
        co_relation.extend(combinations(line, 2))

    # 统计共现关系（无向边）
    word2id = {w: idx for idx, w in enumerate(vocab)}
    co_occurrence = {}
    for co_line in tqdm(co_relation):
        w1, w2 = co_line[0], co_line[1]

        i = word2id[w1]
        j = word2id[w2]

        if (i, j) not in co_occurrence:

            if (j, i) in co_occurrence:
                co_occurrence[(j, i)] += 1
            else:
                co_occurrence[(i, j)] = 1

        else:
            co_occurrence[(i, j)] += 1

    return co_occurrence

def subsumption_infer(co_occurrence, word_freq, vocab, alpha=0.8):
    """
    P(x|y) = P(x,y) / P(y) = num(x,y) / num(y)
    P(y|x) = P(x,y) / P(x) = num(x,y) / num(x)

    conditions:
    (1) P(x|y) >= alpha
    (2) P(y|x) < 1

    :param co_occurrence:
    :param word_freq:
    :param vocab:
    :param alpha:
    :return:
    """
    id2word = {idx: w for idx, w in enumerate(vocab)}

    y_hat_relation = []
    for idx_tuple, weight in tqdm(co_occurrence.items()):

        i, j = idx_tuple
        if word_freq[id2word[i]] >= word_freq[id2word[j]]:
            x = i
            y = j
        else:
            x = j
            y = i

        p_x_y = weight / word_freq[id2word[y]]
        p_y_x = weight / word_freq[id2word[x]]

        if p_x_y >= alpha and p_y_x < 1:
            y_hat_relation.append((id2word[x], id2word[y]))
            # print(f'{id2word[x]} ~ {id2word[y]}, P(x|y) = {p_x_y}, P(y|x) = {p_y_x}, co-occurrence = {weight}, x({word_freq[id2word[x]]}) y({word_freq[id2word[y]]})')

    return y_hat_relation

def longest_common_subsequence(s1: str, s2: str) -> int:
    """
    计算两个字符串之间的最长公共子序列的长度。
    :param s1: 第一个字符串
    :param s2: 第二个字符串
    :return: 最长公共子序列的长度
    """
    m, n = len(s1), len(s2)
    # 创建一个二维数组来存储最长公共子序列的长度。
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # 自底向上地构建dp数组
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1  # 若当前字符相同，则将这一对字符加入到LCS中
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])  # 否则，考虑不包括s1[i-1]或s2[j-1]的情况

    # dp[m][n] 包含了s1[0..m-1]和s2[0..n-1]的最长公共子序列的长度
    return dp[m][n]

def sparse_cosine(x_idx, y_idx, co_occurrence):
    x_co_dict = {}
    y_co_dict = {}

    for idx_tuple, weight in co_occurrence.items():
        i, j = idx_tuple

        if x_idx in (i, j):
            if x_idx == i:
                x_co_dict[j] = weight
            else:
                x_co_dict[i] = weight

        if y_idx in (i, j):
            if y_idx == i:
                y_co_dict[j] = weight
            else:
                y_co_dict[i] = weight

    public_key = set(x_co_dict.keys()) & set(y_co_dict.keys())
    x_dot_y = 0
    for p_k in public_key:
        x_dot_y += x_co_dict[p_k] * y_co_dict[p_k]

    norm_x = 0
    for _, weight in x_co_dict.items():
        norm_x += weight ** 2
    norm_x = sqrt(norm_x)

    norm_y = 0
    for _, weight in y_co_dict.items():
        norm_y += weight ** 2
    norm_y = sqrt(norm_y)

    return x_dot_y / (norm_x * norm_y)

def klink_l(co_occurrence, word_freq, vocab, t=0.2):
    """
    L(x,y) = (P(y|x) - P(x|y) ) * c(x,y) * (1 + N(x,y))

    :param co_occurrence:
    :param word_freq:
    :param vocab:
    :param t:
    :return:
    """

    id2word = {idx: w for idx, w in enumerate(vocab)}
    idx2weight = {}
    for idx_tuple, _ in co_occurrence.items():
        i, j = idx_tuple
        idx2weight[i] = idx2weight.get(i, []) + [idx_tuple]
        idx2weight[j] = idx2weight.get(j, []) + [idx_tuple]

    y_hat_relation = []
    for idx_tuple, weight in tqdm(co_occurrence.items()):
        i, j = idx_tuple
        if word_freq[id2word[i]] >= word_freq[id2word[j]]:
            x = i
            y = j
        else:
            x = j
            y = i

        p_x_y = weight / word_freq[id2word[y]]
        p_y_x = weight / word_freq[id2word[x]]

        # c_x_y = sparse_cosine(x, y, co_occurrence)
        x_v, y_v = np.zeros(len(vocab)), np.zeros(len(vocab))
        for idx1, idx2 in idx2weight[x]:
            if x == idx1:
                x_v[idx2] = co_occurrence[(idx1, idx2)]
            else:
                x_v[idx1] = co_occurrence[(idx1, idx2)]

        for idx1, idx2 in idx2weight[y]:
            if y == idx1:
                y_v[idx2] = co_occurrence[(idx1, idx2)]
            else:
                y_v[idx1] = co_occurrence[(idx1, idx2)]

        c_x_y = cosine_sim(x_v, y_v)

        n_x_y = longest_common_subsequence(id2word[i], id2word[j])

        l_x_y = (p_y_x - p_x_y) * c_x_y * (1 + n_x_y)

        if l_x_y > t:
            y_hat_relation.append((id2word[x], id2word[y]))

    return y_hat_relation