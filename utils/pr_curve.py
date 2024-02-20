import numpy as np
from tqdm import tqdm
import math
import configs
from functions.hashing import get_hamm_dist
from functions.metrics import preprocess_for_calculate_mAP
from scipy.spatial.distance import cdist
from sklearn.metrics import auc
import matplotlib.pyplot as plt

draw_range = [1, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500,
              9000, 9500, 10000]


# def pr_curve(rF, qF, rL, qL, draw_range=draw_range):
#     #  https://blog.csdn.net/HackerTom/article/details/89425729
#     #  Detail usage: https://github.com/swuxyj/DeepHash-pytorch/blob/master/utils/precision_recall_curve.py 
#     n_query = qF.shape[0]
#     Gnd = (np.dot(qL, rL.transpose()) > 0).astype(np.float32)
#     Rank = np.argsort(get_hamm_dist(qF, rF))
#     P, R = [], []
#     for k in tqdm(draw_range, disable=configs.disable_tqdm):
#         p = np.zeros(n_query)
#         r = np.zeros(n_query)
#         for it in range(n_query):
#             gnd = Gnd[it]
#             gnd_all = np.sum(gnd)
#             if gnd_all == 0:
#                 continue
#             asc_id = Rank[it][:k]
#             gnd = gnd[asc_id]
#             gnd_r = np.sum(gnd)
#             p[it] = gnd_r / k
#             r[it] = gnd_r / gnd_all
#         P.append(np.mean(p))
#         R.append(np.mean(r))
#     return P, R

def pr_curve(qF, rF, qL, rL, what=1, topK=-1):
    """Input:
    what: {0: cosine 距离, 1: Hamming 距离}
    topK: 即 mAP 中的 position threshold，只取前 k 个检索结果，默认 `-1` 是全部，见 [3]

    qF: 查询样本的特征向量，形状为 (n_query, feature_dim)，其中 n_query 是查询样本的数量，feature_dim 是特征向量的维度。
    rF: 数据库样本的特征向量，形状为 (n_database, feature_dim)，其中 n_database 是数据库样本的数量。
    qL: 查询样本的标签，形状为 (n_query, label_dim)，其中 label_dim 是标签的维度。
    rL: 数据库样本的标签，形状为 (n_database, label_dim)
    """
    rF, qF = preprocess_for_calculate_mAP(rF, qF, ternarization=None, 
                                distance_func='hamming', zero_mean=False)

    qF = qF.cpu().numpy()
    rF = rF.cpu().numpy()
    qL = qL.cpu().numpy()
    rL = rL.cpu().numpy()

    

    n_query = qF.shape[0]
    if topK == -1 or topK > rF.shape[0]:  # top-K 之 K 的上限
        topK = rF.shape[0]
        topks = [1, 50, 100, 200, 300, 400, 500, 600, 
            800, 1000, 1200, 1600, 2000, 3000, 4000, 
            5000, 7000, 10000, 15000, 20000, 25000, 32322]
        save_name = 'AUC_all.png'
    else:
        topks = [k for k in range(1, topK + 1, (topK + 1)//21)]
        save_name = 'AUC_topk.png'
    #print(topK,rF.shape)#  32322,64  FOR AWA2
    
    # print([np.argmax(x) for x in rL])

    Gnd = (np.dot(qL, rL.T) > 0).astype(np.float32) #n_query x n_database
    #print(Gnd, Gnd.shape)
    if what == 0:
        Rank = np.argsort(cdist(qF, rF, 'cosine'))
    else:
        Rank = np.argsort(cdist(qF, rF, 'hamming'))
        #对于每个 qF 中的数据点，这些索引表示与其距离最近的 rF 中的数据点。这些索引按照从最小距离到最大距离的顺序排列。
    #print(Rank, Rank.shape)  1500, 5788
    # Rank = np.argsort(get_hamm_dist(qF, rF))
    # print(Rank, Rank.shape)
    #topK = 1000
    P, R = [], []
    # for k in tqdm(range(1, topK + 1, (topK + 1)//21)):  # 枚举 top-K 之 K 取21左右个点近似，加快求解
    

    for k in topks:#手动设置点，让结果更均匀
    # for k in tqdm(range(draw_range)):
        # ground-truth: 1 vs all
        #k = max(1, int((k)**0.5))
        p = np.zeros(n_query)  # 各 query sample 的 Precision@R
        r = np.zeros(n_query)  # 各 query sample 的 Recall@R
        for it in range(n_query):  # 枚举 query sample
            gnd = Gnd[it] #当前查询数据和所有检索数据类别标签是否一样的向量
            gnd_all = np.sum(gnd)  # 整个被检索数据库中的相关样本数
            if gnd_all == 0:
                print("warning: no same label in db")
                continue
            asc_id = Rank[it][:k] #当前查询数据topk距离的检索id

            gnd = gnd[asc_id]
            gnd_r = np.sum(gnd)  # top-K 中的相关样本数

            p[it] = gnd_r / k #
            # p[it] = gnd_r / min(k,gnd_all) # 分母应该不超过相关的总数，否则结果极低
            r[it] = gnd_r / gnd_all

        P.append(np.mean(p))
        R.append(np.mean(r))

    fig = plt.figure(figsize=(5, 5))
    plt.plot(R, P, marker='s')  # 第一个是 x，第二个是 y
    plt.grid(True)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.legend()
    plt.savefig(save_name)
    
    print(P)
    print(R)
    # bb
    pr_auc = auc(R,P)
    #print(pr_auc)
    return pr_auc