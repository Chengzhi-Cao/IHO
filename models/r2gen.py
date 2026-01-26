import torch
import torch.nn as nn
import numpy as np

from modules.visual_extractor import VisualExtractor,VisualExtractor_hyper_graph
from modules.encoder_decoder import EncoderDecoder


class R2GenModel(nn.Module):
    def __init__(self, args, tokenizer):
        super(R2GenModel, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.visual_extractor = VisualExtractor(args)
        self.encoder_decoder = EncoderDecoder(args, tokenizer)
        if args.dataset_name == 'iu_xray':
            self.forward = self.forward_iu_xray
        else:
            self.forward = self.forward_mimic_cxr

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def forward_iu_xray(self, images, targets=None, mode='train'):# images=[1,2,3,224,224]
        att_feats_0, fc_feats_0 = self.visual_extractor(images[:, 0])# att_feats_0=[1,49,2048],fc_feats_0=[1,2048]
        att_feats_1, fc_feats_1 = self.visual_extractor(images[:, 1])# att_feats_1=[1,49,2048],fc_feats_1=[1,2048]
        
        
        fc_feats = torch.cat((fc_feats_0, fc_feats_1), dim=1)# fc_feats=[1,4096]
        att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)# att_feats=[1,98,2048]


        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')# targets=[1,37]
        elif mode == 'sample':
            output, _ = self.encoder_decoder(fc_feats, att_feats, mode='sample')
        else:
            raise ValueError
        return output       # output=[1,44,761]总共有44个单词

    def forward_mimic_cxr(self, images, targets=None, mode='train'):
        att_feats, fc_feats = self.visual_extractor(images)
        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
        elif mode == 'sample':
            output, _ = self.encoder_decoder(fc_feats, att_feats, mode='sample')
        else:
            raise ValueError
        return output






class R2GenModel_hyper(nn.Module):
    def __init__(self, args, tokenizer):
        super(R2GenModel_hyper, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.visual_extractor = VisualExtractor_hyper_graph(args)
        self.encoder_decoder = EncoderDecoder(args, tokenizer)
        if args.dataset_name == 'iu_xray':
            self.forward = self.forward_iu_xray
        else:
            self.forward = self.forward_mimic_cxr

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def forward_iu_xray(self, images, targets=None, mode='train',images_id=None):# images=[1,2,3,224,224]
        att_feats_0, fc_feats_0 = self.visual_extractor(images[:, 0],images_id=images_id,image_index=0)# att_feats_0=[1,49,2048],fc_feats_0=[1,2048]
        att_feats_1, fc_feats_1 = self.visual_extractor(images[:, 1],images_id=images_id,image_index=1)# att_feats_1=[1,49,2048],fc_feats_1=[1,2048]
        
        
        fc_feats = torch.cat((fc_feats_0, fc_feats_1), dim=1)# fc_feats=[1,4096]
        att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)# att_feats=[1,98,2048]


        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')# targets=[1,37]
        elif mode == 'sample':
            output, _ = self.encoder_decoder(fc_feats, att_feats, mode='sample')
        else:
            raise ValueError
        return output       # output=[1,44,761]总共有44个单词

    def forward_mimic_cxr(self, images, targets=None, mode='train'):
        att_feats, fc_feats = self.visual_extractor(images)
        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
        elif mode == 'sample':
            output, _ = self.encoder_decoder(fc_feats, att_feats, mode='sample')
        else:
            raise ValueError
        return output

