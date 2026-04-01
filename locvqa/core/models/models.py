# Project:
#   Localized Questions in VQA
# Description:
#   Model definition file for VQA
# Author:
#   Sergio Tascon-Morales, Ph.D. Student, ARTORG Center, University of Bern

import torch
import torch.nn as nn
from torchvision import transforms
from .components import image, text, attention, fusion, classification

from .Hypergraph_model import HyperEncoder_my

from .gcn import GraphConvolution

class VQA_Base(nn.Module):
    # base class for simple VQA model
    def __init__(self, config, vocab_words, vocab_answers):
        super().__init__()
        self.visual_feature_size = config['visual_feature_size']
        self.question_feature_size = config['question_feature_size']
        self.pre_visual = config['pre_extracted_visual_feat']
        self.use_attention = config['attention']
        self.number_of_glimpses = config['number_of_glimpses']
        self.visual_size_before_fusion = self.visual_feature_size # 2048 by default, changes if attention
        # Create modules for the model

        # if necesary, create module for offline visual feature extraction
        if not self.pre_visual:
            self.image = image.get_visual_feature_extractor(config)

        # create module for text feature extraction
        self.text = text.get_text_feature_extractor(config, vocab_words)

        # if necessary, create attention module
        if self.use_attention:
            self.visual_size_before_fusion = self.number_of_glimpses*self.visual_feature_size
            self.attention_mechanism = attention.get_attention_mechanism(config)
        else:
            self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))

        # create multimodal fusion module
        self.fuser, fused_size = fusion.get_fuser(config['fusion'], self.visual_size_before_fusion, self.question_feature_size)

        # create classifier
        self.classifer = classification.get_classfier(fused_size, config)


    def forward(self, v, q, m): #* For simplicity, VQA_Base receives the mask as an input, but it is not used
        # if required, extract visual features from visual input 
        if not self.pre_visual:
            v = self.image(v) # [B, 2048, 14, 14]

        # l2 norm
        v = v / (v.norm(p=2, dim=1, keepdim=True).expand_as(v) + 1e-8)
        
        # extract text features
        q = self.text(q)

        # if required, apply attention
        if self.use_attention:
            v = self.attention_mechanism(v, q) # should apply attention too
        else:
            v = self.avgpool(v).squeeze_() # [B, 2048]

        # apply multimodal fusion
        fused = self.fuser(v, q)

        # apply MLP
        x = self.classifer(fused)

        return x

# VQA Models

class VQA_MaskRegion(VQA_Base):
    # First model for region-based VQA, with single mask. Input image is multiplied with the mask to produced a masked version which is sent to the model as normal
    # A.k.a. VQARS_1
    def __init__(self, config, vocab_words, vocab_answers):
        # call mom
        super().__init__(config, vocab_words, vocab_answers)

    # override forward method to accept mask
    def forward(self, v, q, m):
        # if required, extract visual features from visual input 
        if self.pre_visual:
            raise ValueError("This model does not allow pre-extracted features")
        else:
            v = self.image(torch.mul(v, m)) # [B, 2048, 14, 14]   MASK IS INCLUDED HERE

        # l2 norm
        v = v / (v.norm(p=2, dim=1, keepdim=True).expand_as(v) + 1e-8)

        # extract text features
        q = self.text(q)

        # if required, apply attention
        if self.use_attention:
            v = self.attention_mechanism(v, q) # should apply attention too
        else:
            v = self.avgpool(v).squeeze_(dim=-1).squeeze_(dim=-1) # [B, 2048]

        # apply multimodal fusion
        fused = self.fuser(v, q)

        # apply MLP
        x = self.classifer(fused)

        return x


class VQA_IgnoreMask(VQA_Base):
    # First model for region-based VQA, with single mask, but the mask is totally ignored. This model measures the ability of the system to answer without masks
    # A.k.a. VQARS_2
    def __init__(self, config, vocab_words, vocab_answers):
        # call mom
        super().__init__(config, vocab_words, vocab_answers)

    # override forward method to accept mask
    def forward(self, v, q, m):
        # if required, extract visual features from visual input 
        if self.pre_visual:
            raise ValueError("This model does not allow pre-extracted features")
        else:
            v = self.image(v) # [B, 2048, 14, 14]   MASK IS INCLUDED HERE

        # l2 norm
        v = v / (v.norm(p=2, dim=1, keepdim=True).expand_as(v) + 1e-8)

        # extract text features
        q = self.text(q)

        # if required, apply attention
        if self.use_attention:
            v = self.attention_mechanism(v, q) # should apply attention too
        else:
            v = self.avgpool(v).squeeze_() # [B, 2048]

        # apply multimodal fusion
        fused = self.fuser(v, q)

        # apply MLP
        x = self.classifer(fused)

        return x


class VQARS_3(VQA_Base):
    # Same as model VQARS_1 but mask is resized, flattened and then injected into the multimodal fusion
    def __init__(self, config, vocab_words, vocab_answers):
        # call mom
        super().__init__(config, vocab_words, vocab_answers)

        # create fuser to include reshaped, vectorized mask
        self.fuser2, fused_size2 = fusion.get_fuser(config['fusion'], self.visual_size_before_fusion + self.question_feature_size, 14*14)

        # correct classifier
        self.classifer = classification.get_classfier(fused_size2, config)

    # override forward method to accept mask
    def forward(self, v, q, m):
        # if required, extract visual features from visual input 
        if self.pre_visual:
            raise ValueError("This model does not allow pre-extracted features")
        else:
            v = self.image(v) # [B, 2048, 14, 14]   

        # resize mask
        m = transforms.Resize(14)(m) #should become size (B,14,14)
        m = m.view(-1, 14*14)

        # extract text features
        q = self.text(q)

        # if required, apply attention
        if self.use_attention:
            v = self.attention_mechanism(v, q) # should apply attention too
        else:
            v = self.avgpool(v).squeeze_() # [B, 2048]

        # apply multimodal fusion
        fused = self.fuser(v, q)
        fused = self.fuser2(fused, m)

        # apply MLP
        x = self.classifer(fused)

        return x


class VQARS_4(VQA_Base):
    # Model that requires attention. Mask is used to mask the attention maps of the attention mechanism. 
    def __init__(self, config, vocab_words, vocab_answers):
        if not config['attention']:
            raise ValueError("This model requires attention. Please set <attention> to True in the config file")

        # call mom
        super().__init__(config, vocab_words, vocab_answers)

        # replace attention mechanism
        self.attention_mechanism = attention.get_attention_mechanism(config, special='Att1')

    # override forward method to accept mask
    def forward(self, v, q, m):
        # if required, extract visual features from visual input 
        if self.pre_visual:
            raise ValueError("This model does not allow pre-extracted features")
        else:
            v = self.image(v) 

        # extract text features
        q = self.text(q)

        # resize mask
        m = transforms.Resize(14)(m) #should become size (B,1,14,14)
        m = m.view(m.size(0),-1, 14*14) # [B,1,196]

        # if required, apply attention
        if self.use_attention:
            v = self.attention_mechanism(v, m, q) 
        else:
            raise ValueError("This model requires attention")

        # apply multimodal fusion
        fused = self.fuser(v, q)

        # apply MLP
        x = self.classifer(fused)

        return x


class VQARS_5(VQA_Base):
    # Alternative model that considers mask as an additional channel along with the input image
    def __init__(self, config, vocab_words, vocab_answers):
        # call mom
        super().__init__(config, vocab_words, vocab_answers)
        
        # re-define first layer of visual feature extractor so that it admits 4 channels as input instead of 3
        self.image.net_base.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        modules = list(self.image.net_base.children())[:-2] # ignore avgpool layer and classifier
        self.image.extractor = nn.Sequential(*modules)
        for p in self.image.extractor.parameters():
            p.requires_grad = False 
    # override forward method to accept mask
    def forward(self, v, q, m):
        # if required, extract visual features from visual input 
        if self.pre_visual:
            raise ValueError("This model does not allow pre-extracted features")
        else:
            v = self.image(torch.cat((v, m), 1)) # [B, 2048, 14, 14]   MASK IS INCLUDED HERE

        # extract text features
        q = self.text(q)

        # if required, apply attention
        if self.use_attention:
            v = self.attention_mechanism(v, q) # should apply attention too
        else:
            v = self.avgpool(v).squeeze_() # [B, 2048]

        # apply multimodal fusion
        fused = self.fuser(v, q)

        # apply MLP
        x = self.classifer(fused)

        return x


class VQARS_6(VQA_Base):
    # Same as base class. Had to override forward because of number of arguments passed in training functions
    def __init__(self, config, vocab_words, vocab_answers):
        if not config['attention']:
            raise ValueError("This model requires attention. Please set <attention> to True in the config file")

        # call mom
        super().__init__(config, vocab_words, vocab_answers)

        # replace attention mechanism
        self.attention_mechanism = attention.get_attention_mechanism(config, special='Att2')

    # override forward method to accept mask
    def forward(self, v, q, m):
        # if required, extract visual features from visual input 
        if self.pre_visual:
            raise ValueError("This model does not allow pre-extracted features")
        else:
            v = self.image(v) # [B, 2048, 14, 14]   

        # extract text features
        q = self.text(q)

        # resize mask
        m = transforms.Resize(14)(m) #should become size (B,1,14,14)
        m = m.view(m.size(0),-1, 14*14) # [B,1,196]

        # if required, apply attention
        if self.use_attention:
            v = self.attention_mechanism(v, m, q) 
        else:
            raise ValueError("This model requires attention")

        # apply multimodal fusion
        fused = self.fuser(v, q)

        # apply MLP
        x = self.classifer(fused)

        return x


class VQA_LocalizedAttention(VQA_Base):
    # Model that requires attention. Mask is used to mask the attention maps of the attention mechanism. 
    # A.k.a. VQARS_7
    def __init__(self, config, vocab_words, vocab_answers):
        if not config['attention']:
            raise ValueError("This model requires attention. Please set <attention> to True in the config file")

        # call mom
        super().__init__(config, vocab_words, vocab_answers)

        # replace attention mechanism
        self.attention_mechanism = attention.get_attention_mechanism(config, special='Att3')

    # override forward method to accept mask
    def forward(self, v, q, m):
        # if required, extract visual features from visual input 
        if not self.pre_visual:
            v = self.image(v) 

        # l2 norm
        v = v / (v.norm(p=2, dim=1, keepdim=True).expand_as(v) + 1e-8)

        # extract text features
        q = self.text(q)

        # resize mask
        m = transforms.Resize(14)(m) #should become size (B,1,14,14)
        m = m.view(m.size(0),-1, 14*14) # [B,1,196]

        # if required, apply attention
        if self.use_attention:
            v = self.attention_mechanism(v, m, q) 
        else:
            raise ValueError("This model requires attention")

        # apply multimodal fusion
        fused = self.fuser(v, q)

        # apply MLP
        x = self.classifer(fused)

        return x

class VQARS_8(VQA_Base):
    # Model that requires attention. Mask is used to mask the attention maps of the attention mechanism. 
    # A.k.a. VQARS_7
    def __init__(self, config, vocab_words, vocab_answers):
        if not config['attention']:
            raise ValueError("This model requires attention. Please set <attention> to True in the config file")

        # call mom
        super().__init__(config, vocab_words, vocab_answers)

        # replace attention mechanism
        self.attention_mechanism = attention.get_attention_mechanism(config, special='Att5')

    # override forward method to accept mask
    def forward(self, v, q, m):
        # if required, extract visual features from visual input 
        if not self.pre_visual:
            v = self.image(v) 

        # l2 norm
        v = v / (v.norm(p=2, dim=1, keepdim=True).expand_as(v) + 1e-8)

        # extract text features
        q = self.text(q)

        # resize mask
        m = transforms.Resize(14)(m) #should become size (B,1,14,14)
        m = m.view(m.size(0),-1, 14*14) # [B,1,196]

        # if required, apply attention
        if self.use_attention:
            v = self.attention_mechanism(v, m, q) 
        else:
            raise ValueError("This model requires attention")

        # apply multimodal fusion
        fused = self.fuser(v, q)

        # apply MLP
        x = self.classifer(fused)

        return x


class VQARS_9(VQA_Base):
    # Model that requires attention. Mask is used to mask the attention maps of the attention mechanism. 
    # A.k.a. VQARS_7
    def __init__(self, config, vocab_words, vocab_answers):
        if not config['attention']:
            raise ValueError("This model requires attention. Please set <attention> to True in the config file")

        # call mom
        super().__init__(config, vocab_words, vocab_answers)

        # replace attention mechanism
        self.attention_mechanism = attention.get_attention_mechanism(config, special='Att6')

    # override forward method to accept mask
    def forward(self, v, q, m):
        # if required, extract visual features from visual input 
        if not self.pre_visual:
            v = self.image(v) 

        # l2 norm
        v = v / (v.norm(p=2, dim=1, keepdim=True).expand_as(v) + 1e-8)

        # extract text features
        q = self.text(q)

        # resize mask
        m = transforms.Resize(14)(m) #should become size (B,1,14,14)
        m = m.view(m.size(0),-1, 14*14) # [B,1,196]

        # if required, apply attention
        if self.use_attention:
            v = self.attention_mechanism(v, m, q) 
        else:
            raise ValueError("This model requires attention")

        # apply multimodal fusion
        fused = self.fuser(v, q)

        # apply MLP
        x = self.classifer(fused)

        return x

class VQA_LocalizedAttentionScale(VQA_Base):
    # Model that requires attention. Mask is used to mask the attention maps of the attention mechanism. 
    # Added feature: scale the attention in the region by abs(max(ouside) - max(inside)).
    # A.k.a. VQARS_7
    def __init__(self, config, vocab_words, vocab_answers):
        if not config['attention']:
            raise ValueError("This model requires attention. Please set <attention> to True in the config file")

        # call mom
        super().__init__(config, vocab_words, vocab_answers)

        # replace attention mechanism
        self.attention_mechanism = attention.get_attention_mechanism(config, special='Att4')

    # override forward method to accept mask
    def forward(self, v, q, m):
        # if required, extract visual features from visual input 
        if not self.pre_visual:
            v = self.image(v) 

        # l2 norm
        v = v / (v.norm(p=2, dim=1, keepdim=True).expand_as(v) + 1e-8)

        # extract text features
        q = self.text(q)

        # resize mask
        m = transforms.Resize(14)(m) #should become size (B,1,14,14)
        m = m.view(m.size(0),-1, 14*14) # [B,1,196]

        # if required, apply attention
        if self.use_attention:
            v = self.attention_mechanism(v, m, q) 
        else:
            raise ValueError("This model requires attention")

        # apply multimodal fusion
        fused = self.fuser(v, q)

        # apply MLP
        x = self.classifer(fused)

        return x
##################################################################################
##################################################################################
##################################################################################
##################################################################################




##################################################################################
##################################################################################
##################################################################################
##################################################################################
import torch.nn.functional as F

class VQA_LocalizedAttention_hypergraph(VQA_Base):
    # Model that requires attention. Mask is used to mask the attention maps of the attention mechanism. 
    # A.k.a. VQARS_7
    def __init__(self, config, vocab_words, vocab_answers):
        if not config['attention']:
            raise ValueError("This model requires attention. Please set <attention> to True in the config file")

        # call mom
        super().__init__(config, vocab_words, vocab_answers)

        # replace attention mechanism
        self.attention_mechanism = attention.get_attention_mechanism(config, special='Att3')
        self.hypergraph = HyperEncoder_my(_node=14,channel=[2048])
        
        self.hypergraph1 = HyperEncoder_my(_node=7,channel=[2048])
        self.hypergraph3 = HyperEncoder_my(_node=28,channel=[2048])
        
        
    # override forward method to accept mask
    def forward(self, v, q, m): # v=[64,3,448,448], q=[64,12],m=[64,1,488,488]
        # if required, extract visual features from visual input 
        if not self.pre_visual:
            v = self.image(v)   # v=[64,2048,14,14]

        # l2 norm
        v = v / (v.norm(p=2, dim=1, keepdim=True).expand_as(v) + 1e-8)
        # v1 = F.interpolate(v, scale_factor=0.5, mode='nearest') # v1=[64,2048,7,7]
        # v3 = F.interpolate(v, scale_factor=2, mode='nearest')   # v3=[64,2048,7,7]

        # extract text features
        q = self.text(q)

        # resize mask
        m = transforms.Resize(14)(m) #should become size (B,1,14,14)
        m = m.view(m.size(0),-1, 14*14) # [B,1,196]
        # v=[64,2048,14,14], m=[64,1,196], q=[64,1024]
        
        # v1 = self.hypergraph1(v1)
        # v1 = F.interpolate(v1, scale_factor=2, mode='nearest')
        # v3 = self.hypergraph3(v3)
        # v3 = F.interpolate(v3, scale_factor=0.5, mode='nearest')
        v = self.hypergraph(v)
        
        # v = v + v1 + v3
        # v = v + v1
        # if required, apply attention
        if self.use_attention:
            v = self.attention_mechanism(v, m, q) # v=[64,4096]
        else:
            raise ValueError("This model requires attention")

        # apply multimodal fusion
        fused = self.fuser(v, q)# fuse=[64,5120]

        # apply MLP
        x = self.classifer(fused)#

        return x    # x=[64,1]




class VQA_LocalizedAttention_hypergraph_layer3(VQA_Base):
    # Model that requires attention. Mask is used to mask the attention maps of the attention mechanism. 
    # A.k.a. VQARS_7
    def __init__(self, config, vocab_words, vocab_answers):
        if not config['attention']:
            raise ValueError("This model requires attention. Please set <attention> to True in the config file")

        # call mom
        super().__init__(config, vocab_words, vocab_answers)

        # replace attention mechanism
        self.attention_mechanism = attention.get_attention_mechanism(config, special='Att3')
        self.hypergraph = HyperEncoder_my(_node=14,channel=[2048])
        
        self.hypergraph1 = HyperEncoder_my(_node=7,channel=[2048])
        self.hypergraph3 = HyperEncoder_my(_node=28,channel=[2048])
        
        
    # override forward method to accept mask
    def forward(self, v, q, m): # v=[64,3,448,448], q=[64,12],m=[64,1,488,488]
        # if required, extract visual features from visual input 
        if not self.pre_visual:
            v = self.image(v)   # v=[64,2048,14,14]

        # l2 norm
        v = v / (v.norm(p=2, dim=1, keepdim=True).expand_as(v) + 1e-8)
        v1 = F.interpolate(v, scale_factor=0.5, mode='nearest') # v1=[64,2048,7,7]
        v3 = F.interpolate(v, scale_factor=2, mode='nearest')   # v3=[64,2048,7,7]

        # extract text features
        q = self.text(q)

        # resize mask
        m = transforms.Resize(14)(m) #should become size (B,1,14,14)
        m = m.view(m.size(0),-1, 14*14) # [B,1,196]
        # v=[64,2048,14,14], m=[64,1,196], q=[64,1024]
        
        v1 = self.hypergraph1(v1)
        v1 = F.interpolate(v1, scale_factor=2, mode='nearest')
        v3 = self.hypergraph3(v3)
        v3 = F.interpolate(v3, scale_factor=0.5, mode='nearest')
        v = self.hypergraph(v)
        
        v = v + v1 + v3
        # v = v + v1
        # if required, apply attention
        if self.use_attention:
            v = self.attention_mechanism(v, m, q) # v=[64,4096]
        else:
            raise ValueError("This model requires attention")

        # apply multimodal fusion
        fused = self.fuser(v, q)# fuse=[64,5120]

        # apply MLP
        x = self.classifer(fused)#

        return x    # x=[64,1]



class VQA_LocalizedAttention_hypergraph_layer2(VQA_Base):
    # Model that requires attention. Mask is used to mask the attention maps of the attention mechanism. 
    # A.k.a. VQARS_7
    def __init__(self, config, vocab_words, vocab_answers):
        if not config['attention']:
            raise ValueError("This model requires attention. Please set <attention> to True in the config file")

        # call mom
        super().__init__(config, vocab_words, vocab_answers)

        # replace attention mechanism
        self.attention_mechanism = attention.get_attention_mechanism(config, special='Att3')
        self.hypergraph = HyperEncoder_my(_node=14,channel=[2048])
        
        self.hypergraph1 = HyperEncoder_my(_node=7,channel=[2048])
        # self.hypergraph3 = HyperEncoder_my(_node=28,channel=[2048])
        
        
    # override forward method to accept mask
    def forward(self, v, q, m): # v=[64,3,448,448], q=[64,12],m=[64,1,488,488]
        # if required, extract visual features from visual input 
        if not self.pre_visual:
            v = self.image(v)   # v=[64,2048,14,14]

        # l2 norm
        v = v / (v.norm(p=2, dim=1, keepdim=True).expand_as(v) + 1e-8)
        v1 = F.interpolate(v, scale_factor=0.5, mode='nearest') # v1=[64,2048,7,7]
        v3 = F.interpolate(v, scale_factor=2, mode='nearest')   # v3=[64,2048,7,7]

        # extract text features
        q = self.text(q)

        # resize mask
        m = transforms.Resize(14)(m) #should become size (B,1,14,14)
        m = m.view(m.size(0),-1, 14*14) # [B,1,196]
        # v=[64,2048,14,14], m=[64,1,196], q=[64,1024]
        
        v1 = self.hypergraph1(v1)
        v1 = F.interpolate(v1, scale_factor=2, mode='nearest')
        # v3 = self.hypergraph3(v3)
        # v3 = F.interpolate(v3, scale_factor=0.5, mode='nearest')
        v = self.hypergraph(v)
        
        # v = v + v1 + v3
        v = v + v1
        # if required, apply attention
        if self.use_attention:
            v = self.attention_mechanism(v, m, q) # v=[64,4096]
        else:
            raise ValueError("This model requires attention")

        # apply multimodal fusion
        fused = self.fuser(v, q)# fuse=[64,5120]

        # apply MLP
        x = self.classifer(fused)#

        return x    # x=[64,1]








class VQA_LocalizedAttention_hypergraph_CNN(VQA_Base):
    # Model that requires attention. Mask is used to mask the attention maps of the attention mechanism. 
    # A.k.a. VQARS_7
    def __init__(self, config, vocab_words, vocab_answers):
        if not config['attention']:
            raise ValueError("This model requires attention. Please set <attention> to True in the config file")

        # call mom
        super().__init__(config, vocab_words, vocab_answers)

        # replace attention mechanism
        self.attention_mechanism = attention.get_attention_mechanism(config, special='Att3')
        
        self.cnn = nn.Conv2d(in_channels=2048,out_channels=2048,kernel_size=1)
        
    # override forward method to accept mask
    def forward(self, v, q, m): # v=[64,3,448,448], q=[64,12],m=[64,1,488,488]
        # if required, extract visual features from visual input 
        if not self.pre_visual:
            v = self.image(v)   # v=[64,2048,14,14]

        # l2 norm
        v = v / (v.norm(p=2, dim=1, keepdim=True).expand_as(v) + 1e-8)
        # v1 = F.interpolate(v, scale_factor=0.5, mode='nearest') # v1=[64,2048,7,7]
        # v3 = F.interpolate(v, scale_factor=2, mode='nearest')   # v3=[64,2048,7,7]

        # extract text features
        q = self.text(q)

        # resize mask
        m = transforms.Resize(14)(m) #should become size (B,1,14,14)
        m = m.view(m.size(0),-1, 14*14) # [B,1,196]
        # v=[64,2048,14,14], m=[64,1,196], q=[64,1024]
        
        v = self.cnn(v)
        
        # if required, apply attention
        if self.use_attention:
            v = self.attention_mechanism(v, m, q) # v=[64,4096]
        else:
            raise ValueError("This model requires attention")

        # apply multimodal fusion
        fused = self.fuser(v, q)# fuse=[64,5120]

        # apply MLP
        x = self.classifer(fused)#

        return x    # x=[64,1]





class VQA_LocalizedAttention_hypergraph_trans(VQA_Base):
    # Model that requires attention. Mask is used to mask the attention maps of the attention mechanism. 
    # A.k.a. VQARS_7
    def __init__(self, config, vocab_words, vocab_answers):
        if not config['attention']:
            raise ValueError("This model requires attention. Please set <attention> to True in the config file")

        # call mom
        super().__init__(config, vocab_words, vocab_answers)

        # replace attention mechanism
        self.attention_mechanism = attention.get_attention_mechanism(config, special='Att3')

        self.cnn = nn.Conv2d(in_channels=2048,out_channels=2048,kernel_size=1)
        
        self.trans = nn.MultiheadAttention(
                        embed_dim = 49,
                        num_heads = 7,
                        bias=True,
                        batch_first=True)

        
    # override forward method to accept mask
    def forward(self, v, q, m): # v=[64,3,448,448], q=[64,12],m=[64,1,488,488]
        # if required, extract visual features from visual input 
        if not self.pre_visual:
            v = self.image(v)   # v=[64,2048,14,14]

        # l2 norm
        v = v / (v.norm(p=2, dim=1, keepdim=True).expand_as(v) + 1e-8)

        # extract text features
        q = self.text(q)

        # resize mask
        m = transforms.Resize(14)(m) #should become size (B,1,14,14)
        m = m.view(m.size(0),-1, 14*14) # [B,1,196]
        # v=[64,2048,14,14], m=[64,1,196], q=[64,1024]
        
        
        v = F.interpolate(v, scale_factor=0.5, mode='nearest')
        
        v = v.view(v.size(0),v.size(1),-1)
        v = self.trans(v,v,v)
        v = v[0].view(v[0].size(0),v[0].size(1),7,7)
        v = F.interpolate(v, scale_factor=2, mode='nearest')

        # if required, apply attention
        if self.use_attention:
            v = self.attention_mechanism(v, m, q) # v=[64,4096]
        else:
            raise ValueError("This model requires attention")

        # apply multimodal fusion
        fused = self.fuser(v, q)# fuse=[64,5120]

        # apply MLP
        x = self.classifer(fused)#

        return x    # x=[64,1]






import numpy as np

class VQA_LocalizedAttention_hypergraph_gcn(VQA_Base):
    # Model that requires attention. Mask is used to mask the attention maps of the attention mechanism. 
    # A.k.a. VQARS_7
    def __init__(self, config, vocab_words, vocab_answers):
        if not config['attention']:
            raise ValueError("This model requires attention. Please set <attention> to True in the config file")

        # call mom
        super().__init__(config, vocab_words, vocab_answers)

        # replace attention mechanism
        self.attention_mechanism = attention.get_attention_mechanism(config, special='Att3')

        # self.cnn = nn.Conv2d(in_channels=2048,out_channels=2048,kernel_size=1)
        
        # self.trans = nn.MultiheadAttention(
        #                 embed_dim = 49,
        #                 num_heads = 7,
        #                 bias=True,
        #                 batch_first=True)
        
        self.gcn = GraphConvolution(49, 49)
        self.adj = torch.zeros(16, 2048, 2048) + torch.Tensor(np.eye(2048, k=-1) + np.eye(2048) + np.eye(2048, k=1))


        
    # override forward method to accept mask
    def forward(self, v, q, m): # v=[64,3,448,448], q=[64,12],m=[64,1,488,488]
        # if required, extract visual features from visual input 
        if not self.pre_visual:
            v = self.image(v)   # v=[64,2048,14,14]

        # l2 norm
        v = v / (v.norm(p=2, dim=1, keepdim=True).expand_as(v) + 1e-8)

        # extract text features
        q = self.text(q)

        # resize mask
        m = transforms.Resize(14)(m) #should become size (B,1,14,14)
        m = m.view(m.size(0),-1, 14*14) # [B,1,196]
        # v=[64,2048,14,14], m=[64,1,196], q=[64,1024]
        
        
        v = F.interpolate(v, scale_factor=0.5, mode='nearest')
        
        v = v.view(v.size(0),v.size(1),-1)
        v = self.gcn(v,self.adj.cuda())
        v = v.view(v.size(0),v.size(1),7,7)
        v = F.interpolate(v, scale_factor=2, mode='nearest')

        # if required, apply attention
        if self.use_attention:
            v = self.attention_mechanism(v, m, q) # v=[64,4096]
        else:
            raise ValueError("This model requires attention")

        # apply multimodal fusion
        fused = self.fuser(v, q)# fuse=[64,5120]

        # apply MLP
        x = self.classifer(fused)#

        return x    # x=[64,1]