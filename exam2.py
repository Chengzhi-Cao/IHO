import torch

# tensor_0 = torch.arange(3, 12).view(3, 3)
# print(tensor_0)

# index = torch.tensor([[2, 1, 0]])
# tensor_1 = tensor_0.gather(0, index)
# print(tensor_1)


from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor import Meteor
from pycocoevalcap.rouge import Rouge
from pycocoevalcap.cider import Cider

# scorers = [
#     (Bleu(4), ["BLEU_1", "BLEU_2", "BLEU_3", "BLEU_4"]),
#     (Meteor(), "METEOR"),
#     (Cider(),'CIDEr'),
#     (Rouge(), "ROUGE_L")
# ]




import numpy as np

mat = np.random.randint(low=0, high=10, size=(3, 30)).flatten()
mat2 = np.random.randint(low=0, high=5, size=(3, 30)).flatten()

score = f1_score(mat, mat2,average='macro')
print('score=',score)

# from sklearn.metrics import precision_score, recall_score, f1_score
# y_true = [0, 1, 0, 0, 1, 0, 1]
# y_pred = [0, 1, 0, 0, 0, 1, 0]
# # 计算二分类情况下的average = 'macro' 'micro' 'binary'
# # 二分类情况下，也能用macro和micro，但一般用binary
# f1_score(y_true, y_pred,average='macro')
# precision_score(y_true, y_pred, average='macro')
# recall_score(y_true, y_pred, average='macro')



y_true = [1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4,5]
y_pred = [1, 1, 1, 0, 0, 2, 2, 3, 3, 3, 4, 3, 4, 3, 0]

y_true = np.array(y_true)
y_pred = np.array(y_pred)
print(f1_score(y_true,y_pred,average='macro'))
#>>> 0.615384615385