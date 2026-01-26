import torch
import torch.nn as nn


class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask):
        # input=[1,44,761],
        # target=[1,44], 
        # mask=[1,44]
        
        # truncate to the same size
        target = target[:, :input.size(1)]  # target=[1,44]
        mask = mask[:, :input.size(1)]      # mask=[1,44]
        output = -input.gather(2, target.long().unsqueeze(2)).squeeze(2) * mask # output=[1,44]. 
        #gather函数的解释https://blog.csdn.net/qq_44665283/article/details/138576187?fromshare=blogdetail&sharetype=blogdetail&sharerId=138576187&sharerefer=PC&sharesource=qq_40776179&sharefrom=from_link。有点复杂
        # https://blog.csdn.net/weixin_46707326/article/details/120424556?fromshare=blogdetail&sharetype=blogdetail&sharerId=120424556&sharerefer=PC&sharesource=qq_40776179&sharefrom=from_link
        output = torch.sum(output) / torch.sum(mask)

        return output


def compute_loss(output, reports_ids, reports_masks):
    criterion = LanguageModelCriterion()
    loss = criterion(output, reports_ids[:, 1:], reports_masks[:, 1:]).mean()
    return loss