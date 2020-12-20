import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from torch.autograd import Variable
from genotypes import PRIMITIVES
from genotypes import Genotype

img_shape = (3, 32, 32)

class MixedOp(nn.Module):

    def __init__(self, C, stride, switch, p):
        super(MixedOp, self).__init__()
        self.m_ops = nn.ModuleList()
        self.p = p
        for i in range(len(switch)):
            if switch[i]:
                primitive = PRIMITIVES[i]
                op = OPS[primitive](C, stride, False)
                if 'pool' in primitive:
                    op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
                if isinstance(op, Identity) and p > 0:
                    op = nn.Sequential(op, nn.Dropout(self.p))
                self.m_ops.append(op)
                
    def update_p(self):
        for op in self.m_ops:
            if isinstance(op, nn.Sequential):
                if isinstance(op[0], Identity):
                    op[1].p = self.p
                    
    def forward(self, x, weights):
        return sum(w * op(x) for w, op in zip(weights, self.m_ops))


class Cell(nn.Module):

    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev, switches, p):
        super(Cell, self).__init__()
        self.reduction = reduction
        self.p = p
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        self._steps = steps
        self._multiplier = multiplier

        self.cell_ops = nn.ModuleList()
        switch_count = 0
        for i in range(self._steps):
            for j in range(2+i):
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(C, stride, switch=switches[switch_count], p=self.p)
                self.cell_ops.append(op)
                switch_count = switch_count + 1
    
    def update_p(self):
        for op in self.cell_ops:
            op.p = self.p
            op.update_p()

    def forward(self, s0, s1, weights):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        states = [s0, s1]
        offset = 0
        for i in range(self._steps):
            s = sum(self.cell_ops[offset+j](h, weights[offset+j]) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)

        return torch.cat(states[-self._multiplier:], dim=1)


class Network(nn.Module):

    def __init__(self, C, num_classes, layers, criterion, steps=4, multiplier=4, stem_multiplier=3, switches_normal=[], switches_reduce=[], p=0.0):
        super(Network, self).__init__()
        self._C = C
        self._num_classes = num_classes
        self.label_embedding = nn.Embedding(10, 50)
        self._layers = layers
        self._criterion = criterion
        self._steps = steps
        self._multiplier = multiplier
        self.p = p

        # self.input_conv = torch.nn.Conv2d(3, 128, kernel_size=4, stride=2, padding=1)

        # self.label_conv = torch.nn.Conv2d(1, 128, kernel_size=4, stride=2, padding=1)

        # torch.nn.init.normal(self.input_conv.weight, mean=0.0, std=0.02)
        # torch.nn.init.constant(self.input_conv.bias, 0.0)

        # torch.nn.init.normal(self.label_conv.weight, mean=0.0, std=0.02)
        # torch.nn.init.constant(self.label_conv.bias, 0.0)
        
        self.switches_normal = switches_normal
        switch_ons = []
        for i in range(len(switches_normal)):
            ons = 0
            for j in range(len(switches_normal[i])):
                if switches_normal[i][j]:
                    ons = ons + 1
            switch_ons.append(ons)
            ons = 0
        self.switch_on = switch_ons[0]

        C_curr = stem_multiplier*C
        
        self.network_head = nn.Sequential(
          nn.Linear(50, 32*32),
          nn.BatchNorm1d(32*32),
          nn.LeakyReLU(0.2, inplace=True)
        )

        self.stem = nn.Sequential(
            nn.Conv2d(4, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )
    
        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers//3, 2*layers//3]:
                C_curr *= 2
                reduction = True
                cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, switches_reduce, self.p)
            else:
                reduction = False
                cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, switches_normal, self.p)
#            cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, switches)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier*C_curr
            # print(C_prev)
        
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        # self.classifier = nn.Linear(C_prev, num_classes)
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(C_prev, 1),
            nn.Sigmoid()
        )

        self._initialize_alphas()

    def forward(self, input, labels):
        input = input.view(input.size(0),3,32,32)
        labels = self.label_embedding(labels)
        head = self.network_head(labels)
        head = head.view(-1,1,32,32)
        cat = torch.cat((input, head), 1)
        
        # print(cat.size())
        # print(labels.size())
        # labels = labels.view(labels.size(0), -1, 1, 1)
        # head = self.network_head(labels)
        # labels = self.network_head(labels)
        # head_in = self.network_head(labels)
        # head_in = head_in.view(-1,1,32,32)
        # print(input.size())
        # print(head_in.size())
        # h1 = self.input_conv(input)
        # h2 = self.label_conv(head_in)
        # x = torch.cat([h1, h2], 1)
        # print(x.size())
        # head_in = self.network_head(cat)
        # head_in = head_in.view(-1,4,32,32)
        # head_in = torch.cat((input, head_in), dim=1)
        s0 = s1 = self.stem(cat)
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                if self.alphas_reduce.size(1) == 1:
                    weights = F.softmax(self.alphas_reduce, dim=0)
                else:
                    weights = F.softmax(self.alphas_reduce, dim=-1)
            else:
                if self.alphas_normal.size(1) == 1:
                    weights = F.softmax(self.alphas_normal, dim=0)
                else:
                    weights = F.softmax(self.alphas_normal, dim=-1)
            s0, s1 = s1, cell(s0, s1, weights)
        out = self.global_pooling(s1)
        out = out.view(out.size(0), -1)
        # print(out.size())
        logits = self.classifier(out)
        # print(logits.size())
        return logits

    def update_p(self):
        for cell in self.cells:
            cell.p = self.p
            cell.update_p()
    
    def _loss(self, input, target):
        logits = self(input)
        return self._criterion(logits, target) 

    def _initialize_alphas(self):
        k = sum(1 for i in range(self._steps) for n in range(2+i))
        num_ops = self.switch_on
        self.alphas_normal = nn.Parameter(torch.FloatTensor(1e-3*np.random.randn(k, num_ops)))
        self.alphas_reduce = nn.Parameter(torch.FloatTensor(1e-3*np.random.randn(k, num_ops)))
        self._arch_parameters = [
            self.alphas_normal,
            self.alphas_reduce,
        ]
    
    def arch_parameters(self):
        return self._arch_parameters
