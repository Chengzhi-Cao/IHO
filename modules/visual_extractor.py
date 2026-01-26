import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

class VisualExtractor(nn.Module):
    def __init__(self, args):
        super(VisualExtractor, self).__init__()
        self.visual_extractor = args.visual_extractor
        self.pretrained = args.visual_extractor_pretrained
        model = getattr(models, self.visual_extractor)(pretrained=self.pretrained)
        modules = list(model.children())[:-2]
        self.model = nn.Sequential(*modules)
        self.avg_fnt = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0)

    def forward(self, images):
        patch_feats = self.model(images)    # patch_feats=[1,2048,7,7]
        avg_feats = self.avg_fnt(patch_feats).squeeze().reshape(-1, patch_feats.size(1))    # avg_feats=[1,2048]
        batch_size, feat_size, _, _ = patch_feats.shape # batch_size=1, feat_size=2048
        patch_feats = patch_feats.reshape(batch_size, feat_size, -1).permute(0, 2, 1)   # patch_feats=[1,49,2048]
        return patch_feats, avg_feats





#######################################################################################################

class HGNN(nn.Module):
    def __init__(self, in_ch, n_out):
        super(HGNN, self).__init__()
        self.conv = nn.Linear(in_ch, n_out)
        self.bn = nn.BatchNorm1d(n_out)

    def forward(self, x, G):
        residual = x
        x = self.conv(x)
        x = G.matmul(x)
        x = F.relu(self.bn(x.permute(0,2,1).contiguous())).permute(0,2,1).contiguous() + residual
        return x

class HGNN_layer(nn.Module):
    """
        Writen by Shaocong Mo,
        College of Computer Science and Technology, Zhejiang University,
    """

    def __init__(self, in_ch, node = None, K_neigs=None, kernel_size=5, stride=2):
        super(HGNN_layer, self).__init__()
        self.HGNN = HGNN(in_ch, in_ch)
        self.K_neigs = K_neigs

        self.local_H = self.local_kernel(node, kernel_size=kernel_size, stride=stride)

    def forward(self, x):


        B, N, C = x.shape
        topk_dists, topk_inds, ori_dists, avg_dists = self.batched_knn(x, k=self.K_neigs[0])
        H = self.create_incidence_matrix(topk_dists, topk_inds, avg_dists)
        Dv = torch.sum(H, dim=2, keepdim=True)
        alpha = 1.
        Dv = Dv * alpha
        max_k = int(Dv.max())
        _topk_dists, _topk_inds, _ori_dists, _avg_dists = self.batched_knn(x, k=max_k - 1)
        top_k_matrix = torch.arange(max_k)[None, None, :].repeat(B, N, 1).to(x.device)
        range_matrix = torch.arange(N)[None, :, None].repeat(1, 1, max_k).to(x.device)
        new_topk_inds = torch.where(top_k_matrix >= Dv, range_matrix, _topk_inds).long()
        new_H = self.create_incidence_matrix(_topk_dists, new_topk_inds, _avg_dists)
        local_H = self.local_H.repeat(B,1,1).to(new_H.device)

        _H = torch.cat([new_H,local_H],dim=2)
        _G = self._generate_G_from_H_b(_H)# _H=[8,196,340]

        x = self.HGNN(x, _G)# _G=[8,196,196]
        return x

    @torch.no_grad()
    def _generate_G_from_H_b(self, H, variable_weight=False):
        """
        calculate G from hypgraph incidence matrix H
        :param H: hypergraph incidence matrix H
        :param variable_weight: whether the weight of hyperedge is variable
        :return: G
        """
        bs, n_node, n_hyperedge = H.shape


        # the weight of the hyperedge

        W = torch.ones([bs, n_hyperedge], requires_grad=False, device=H.device)
        # the degree of the node
        DV = torch.sum(H, dim=2)
        # the degree of the hyperedge

        DE = torch.sum(H, dim=1)


        invDE = torch.diag_embed((torch.pow(DE, -1)))
        DV2 = torch.diag_embed((torch.pow(DV, -0.5)))
        W = torch.diag_embed(W)
        HT = H.transpose(1, 2)



        if variable_weight:
            DV2_H = DV2 @ H
            invDE_HT_DV2 = invDE @ HT @ DV2
            return DV2_H, W, invDE_HT_DV2
        else:

            G = DV2 @ H @ W @ invDE @ HT @ DV2

            return G


    @torch.no_grad()
    def pairwise_distance(self, x):
        """
        Compute pairwise distance of a point cloud.
        Args:
            x: tensor (batch_size, num_points, num_dims)
        Returns:
            pairwise distance: (batch_size, num_points, num_points)
        """
        with torch.no_grad():
            x_inner = -2 * torch.matmul(x, x.transpose(2, 1))
            x_square = torch.sum(torch.mul(x, x), dim=-1, keepdim=True)
            return x_square + x_inner + x_square.transpose(2, 1)



    @torch.no_grad()
    def batched_knn(self, x, k=1):

        ori_dists = self.pairwise_distance(x)
        avg_dists = ori_dists.mean(-1, keepdim=True)
        topk_dists, topk_inds = ori_dists.topk(k + 1, dim=2, largest=False, sorted=True)

        return topk_dists, topk_inds, ori_dists, avg_dists

    @torch.no_grad()
    def create_incidence_matrix(self, top_dists, inds, avg_dists, prob=False):
        B, N, K = top_dists.shape
        weights = self.weights_function(top_dists, avg_dists, prob)
        incidence_matrix = torch.zeros(B, N, N, device=inds.device)

        batch_indices = torch.arange(B)[:, None, None].to(inds.device)  # shape: [B, 1, 1]
        pixel_indices = torch.arange(N)[None, :, None].to(inds.device)  # shape: [1, N, 1]

        incidence_matrix[batch_indices, pixel_indices, inds] = weights

        return incidence_matrix.permute(0,2,1).contiguous()



    @torch.no_grad()
    def weights_function(self, topk_dists, avg_dists, prob):
        if prob:
            # Chai's weight function
            topk_dists_sq = topk_dists.pow(2)
            normalized_topk_dists_sq = topk_dists_sq / avg_dists
            weights = torch.exp(-normalized_topk_dists_sq)
        else:
            weights = torch.ones(topk_dists.size(), device=topk_dists.device)
        return weights

    @torch.no_grad()
    def local_kernel(self, size, kernel_size=3, stride=1):
        inp = torch.arange(size * size, dtype=torch.float).reshape(size, size)[None, None, :, :]

        inp_unf = torch.nn.functional.unfold(inp, kernel_size=(kernel_size, kernel_size), stride=stride).squeeze(
            0).transpose(0, 1).long()

        edge, node = inp_unf.shape
        matrix = torch.arange(edge)[:, None].repeat(1, node).long()

        H_local = torch.zeros((size * size, edge))
        H_local[inp_unf, matrix] = 1.

        return H_local

class HyperNet(nn.Module):
    def __init__(self, channel, node = 28, kernel_size=3, stride=1, K_neigs = None):
        super(HyperNet, self).__init__()
        self.HGNN_layer = HGNN_layer(channel, node = node, kernel_size=kernel_size, stride=stride, K_neigs=K_neigs)

    def forward(self, x):# x=[8,512,28,28]

        b,c,w,h = x.shape
        x = x.view(b,c,-1).permute(0,2,1).contiguous()# x=[8,784,512]
        x = self.HGNN_layer(x)  # x=[8,784,512]
        x = x.permute(0,2,1).contiguous().view(b,c,w,h)

        return x    # x=[8,512,28,28]
    
class HyperEncoder_my(nn.Module):
    def __init__(self,channel = [32],_node=28):
        super(HyperEncoder_my, self).__init__()

        kernel_size  = 3
        stride = 1
        self.HGNN_layer1 = HyperNet(channel[0], node=_node, kernel_size=kernel_size, stride=stride, K_neigs=[1])

    def forward(self, x):   # x = [8,512,28,28]
        feature1 = self.HGNN_layer1(x)
        return feature1
        # feature1=[8,512,28,28]

class VisualExtractor_hyper_graph(nn.Module):
    def __init__(self, args):
        super(VisualExtractor_hyper_graph, self).__init__()
        self.visual_extractor = args.visual_extractor
        self.pretrained = args.visual_extractor_pretrained
        model = getattr(models, self.visual_extractor)(pretrained=self.pretrained)
        modules = list(model.children())[:-2]
        self.model = nn.Sequential(*modules)

        modules2 = list(model.children())[:-4]
        self.model2 = nn.Sequential(*modules2)

        modules3 = list(model.children())[:-6]
        self.model3 = nn.Sequential(*modules3)

        modules4 = list(model.children())[:-8]
        self.model4 = nn.Sequential(*modules4)
        

        self.avg_fnt = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0)

        nodes = 7
        channel_num = 2048
        # inout = torch.randn((1,channel_num,nodes,nodes)).cuda()
        self.hypergraph = HyperEncoder_my(_node=nodes,channel=[channel_num])
        
    def forward(self, images,images_id=None,image_index=None):
        patch_feats = self.model(images)    # patch_feats=[1,2048,7,7]

        patch_feats2 = self.model2(images)  # [1,512,28,28]
        patch_feats3 = self.model3(images)  # [1,64,56,56]
        patch_feats4 = self.model4(images)  # [1,64,112,112]

        file_path = '/data/chengzhicao/Medical_Report/R2Gen-main_my/visual/{}/{}'.format(images_id[0],image_index)
        print('images_id[0]=',images_id[0])
        for i in range(len(patch_feats4[0,:,0,0])):
            if i % 2 == 0:
                _fea = patch_feats4[0,i,:,:].cpu().data.numpy()
                _a = np.clip(_fea, 0, 1) # 将numpy数组约束在[0, 1]范围内
                trans_prob_mat = (_a.T/np.sum(_a, 1)).T
                df = pd.DataFrame(trans_prob_mat)
                plt.figure()
                ax = sns.heatmap(df, cmap='jet', cbar=False)
                plt.xticks(alpha=0)
                plt.tick_params(axis='x', width=0)
                plt.yticks(alpha=0)
                plt.tick_params(axis='y', width=0)
                plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
                plt.margins(0, 0)
                save_visual_path = os.path.join(file_path,'patch_feats4') 
                # _save_path = os.path.join(save_visual_path)
                if not os.path.exists(save_visual_path):
                    os.makedirs(save_visual_path)
                output_path = os.path.join(save_visual_path,'feat{}.jpg'.format(i))
                plt.savefig(output_path, transparent=True)  


        for i in range(len(patch_feats3[0,:,0,0])):
            if i % 2 == 0:
                _fea = patch_feats3[0,i,:,:].cpu().data.numpy()
                _a = np.clip(_fea, 0, 1) # 将numpy数组约束在[0, 1]范围内
                trans_prob_mat = (_a.T/np.sum(_a, 1)).T
                df = pd.DataFrame(trans_prob_mat)
                plt.figure()
                ax = sns.heatmap(df, cmap='jet', cbar=False)
                plt.xticks(alpha=0)
                plt.tick_params(axis='x', width=0)
                plt.yticks(alpha=0)
                plt.tick_params(axis='y', width=0)
                plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
                plt.margins(0, 0)
                save_visual_path = os.path.join(file_path,'patch_feats3') 
                # _save_path = os.path.join(save_visual_path)
                if not os.path.exists(save_visual_path):
                    os.makedirs(save_visual_path)
                output_path = os.path.join(save_visual_path,'feat{}.jpg'.format(i))
                plt.savefig(output_path, transparent=True)  


        for i in range(len(patch_feats2[0,:,0,0])):
            if i % 2 == 0:
                _fea = patch_feats2[0,i,:,:].cpu().data.numpy()
                _a = np.clip(_fea, 0, 1) # 将numpy数组约束在[0, 1]范围内
                trans_prob_mat = (_a.T/np.sum(_a, 1)).T
                df = pd.DataFrame(trans_prob_mat)
                plt.figure()
                ax = sns.heatmap(df, cmap='jet', cbar=False)
                plt.xticks(alpha=0)
                plt.tick_params(axis='x', width=0)
                plt.yticks(alpha=0)
                plt.tick_params(axis='y', width=0)
                plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
                plt.margins(0, 0)
                save_visual_path = os.path.join(file_path,'patch_feats2') 
                # _save_path = os.path.join(save_visual_path)
                if not os.path.exists(save_visual_path):
                    os.makedirs(save_visual_path)
                output_path = os.path.join(save_visual_path,'feat{}.jpg'.format(i))
                plt.savefig(output_path, transparent=True)  

        patch_feats = self.hypergraph(patch_feats)

        avg_feats = self.avg_fnt(patch_feats).squeeze().reshape(-1, patch_feats.size(1))    # avg_feats=[1,2048]
        batch_size, feat_size, _, _ = patch_feats.shape # batch_size=1, feat_size=2048
        patch_feats = patch_feats.reshape(batch_size, feat_size, -1).permute(0, 2, 1)   # patch_feats=[1,49,2048]
        return patch_feats, avg_feats