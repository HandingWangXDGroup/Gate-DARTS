import imp
import math
import random
from statistics import mean
from sys import stderr
from types import TracebackType
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from torch.autograd import Variable
from genotypes import PRIMITIVES
from genotypes import Genotype

from model import NetworkCIFAR
from thop import profile
from functools import partial
from utils import parse2gene

SearchControllerConf = {
    'noise_darts': {
     'noise_type': 'N',
     'noise_scheduler': 'cosine_anne',
     'base_lr': 1.0,
     'T_max': 50
     },
    'reweight': False,
    'random_search': {
      'num_identity': 2,
      'num_arch': 8,
      'flops_threshold': None
    }
}

DEBUG_CNT = 0

def _alpha_loss(alphas):
  def entropy(mat):
     return (-(mat.log2()*mat).sum(dim=-1)).mean()
  alphas = alphas.softmax(dim=-1)
  return entropy(alphas)


  
class Gate(nn.Module):
  def __init__(self, channel, n_op, la=1.0) -> None:
    super().__init__()
    self.fc = nn.Linear(channel,n_op)
    self.la = la
  
  def set_la(self,la):
    self.la = la
  
  def forward(self,x,is_train=True):
    B,C,H,W = x.shape
    # x = x.detach()
    x = x.mean(dim=(-1,-2))
    out = self.fc(x)
    return out

class MixedOp(nn.Module):
  def __init__(self, C, stride,top, is_first=False,la=1.):
    super(MixedOp, self).__init__()
    self._ops = nn.ModuleList()
    self.stride = stride
    self.is_first = is_first
    self.la = la
    self.top = top

    for primitive in PRIMITIVES:
      op = OPS[primitive](C, stride, False)
      if 'pool' in primitive:
        op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
      self._ops.append(op)
    if is_first:
      self.gate = Gate(C,len(self._ops))
  
  def set_la(self,la):
    if self.is_first:
      self.gate.set_la(la)
  def arch_parameters(self):
    if self.is_first:
      return list(self.gate.parameters())
    else:
      return list([])
  
  def model_parameters(self):
    if not self.is_first: return list(self.parameters())
    else:
      model_parameters = list([])
      for op in self._ops:
        model_parameters+=list(op.parameters())
      return model_parameters


  def forward(self, x, weights, epoch=None,is_train=True):
    gate_value = self.gate(x,is_train) if self.is_first else weights
    gate_value_cp = torch.clone(gate_value)
    B,C,H,W = x.shape


    # 2. topk+softmax
    alphas, indices = gate_value.topk(self.top)
    gate_value = torch.full_like(gate_value,-float('inf')).scatter_(1,indices,0) + gate_value
    gate_value = gate_value.softmax(dim=-1)
    alphas = gate_value.gather(1,indices)
    out = torch.zeros((B,C,H,W), device=x.device) if self.stride == 1 else torch.zeros((B,C,H//2,W//2),device=x.device) 
    for i_op in range(len(self._ops)):
      ids = (indices==i_op).any(dim=-1)
      if ids.any():
        op = self._ops[i_op]
        
        out[ids]+= op(x[ids])*(alphas[indices==i_op].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))
      

    return out, gate_value_cp
    # return sum(w * self.noise_identity(x, epoch) if isinstance(op, NoiseIdentity) and self.training else w * op(x) for w, op in zip(weights, self._ops))


class Cell(nn.Module):

  def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev, top,is_first=False):
    super(Cell, self).__init__()
    self.reduction = reduction
    self.is_first = is_first
    self.top = top

    if reduction_prev:
      self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
    else:
      self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
    self._steps = steps
    self._multiplier = multiplier

    self._ops = nn.ModuleList()

    for i in range(self._steps):
      for j in range(2+i):
        stride = 2 if reduction and j < 2 else 1
        op = MixedOp(C, stride,top, is_first)
        self._ops.append(op)
  
  def set_la(self,la):
    if self.is_first:
      for op in self._ops:
        op.set_la(la)
  
  def model_parameters(self):
    model_parameters  = list([])
    model_parameters += list(self.preprocess0.parameters()) + list(self.preprocess1.parameters())
    for op in self._ops:
      model_parameters += op.model_parameters()
    return model_parameters

  def arch_parameters(self):
    arch_parameters = list([])
    for op in self._ops:
      arch_parameters+=op.arch_parameters()
    return arch_parameters

  def forward(self, s0, s1, weights, epoch=None, is_train=True, is_alpha=False):
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)

    states = [s0, s1]
    offset = 0
    weights = [] if weights is None  else weights
    alpha_loss = 0.
    for i in range(self._steps):
      s = []
      for j , h in enumerate(states):
          state, gate_value = self._ops[offset+j](h,weights[offset+j],epoch,is_train) if not self.is_first else self._ops[offset+j](h,None,epoch,is_train)
          s.append(state)
          if self.is_first:
            # alpha loss
            if is_alpha:
              alpha_loss+= _alpha_loss(gate_value)
            weights.append(gate_value.detach())
      s = sum(s)
      offset += len(states)
      states.append(s)

    return torch.cat(states[-self._multiplier:], dim=1), weights, alpha_loss/len(self._ops)

class Network(nn.Module):
  def __init__(self, C, num_classes, layers, criterion,top, steps=4, multiplier=4, stem_multiplier=3,la=1.):
    super(Network, self).__init__()
    self._C = C
    self._num_classes = num_classes
    self._layers = layers
    self._criterion = criterion
    self._steps = steps
    self._multiplier = multiplier
    self._reweight = SearchControllerConf['reweight']
    self.la = la 
    self.top = top

    C_curr = stem_multiplier*C
    self.stem = nn.Sequential(
      nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
      nn.BatchNorm2d(C_curr)
    )

    C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
    self.cells = nn.ModuleList()
    reduction_prev = False
    for i in range(layers):
      if i in [layers//3, 2*layers//3]:
        C_curr *= 2
        reduction = True
      else:
        reduction = False
      if i in [0,layers//3]:
        cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev,top = top,is_first=True)
      else:
        cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev,top=top)
      reduction_prev = reduction
      self.cells += [cell]

      C_prev_prev, C_prev = C_prev, multiplier*C_curr

    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_prev, num_classes)

    self._initialize_alphas()
  
  def set_la(self,la):
    self.la = la
    for cell in self.cells:
      cell.set_la(la)
  
  def get_la(self):
    return self.la

  def model_parameters(self):
    model_parameters = list([])
    model_parameters += list(self.stem.parameters())+list(self.global_pooling.parameters())+list(self.classifier.parameters())
    for cell in self.cells:
      model_parameters+=cell.model_parameters()
    return model_parameters
  
  def arch_parameters(self):
    arch_parameters = list([])
    for cell in self.cells:
      arch_parameters+=cell.arch_parameters()
    return arch_parameters
    
  def new(self):
    model_new = Network(self._C, self._num_classes, self._layers, self._criterion).cuda()
    for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
        x.data.copy_(y.data)
    return model_new

  def forward(self, input, epoch=None,is_train=True,is_alpha=False):
    s0 = s1 = self.stem(input)
    alpha_loss_total = 0.
    for i, cell in enumerate(self.cells):
      if cell.reduction:
        state, weights,alpha_loss = cell(s0, s1, self.alphas_reduce, epoch, is_train,is_alpha=is_alpha)
        if cell.is_first: self.alphas_reduce = weights
        s0, s1 = s1, state
      else:
        state, weights,alpha_loss = cell(s0, s1, self.alphas_normal, epoch,is_train,is_alpha=is_alpha)
        if cell.is_first: self.alphas_normal = weights
        s0, s1 = s1, state
      alpha_loss_total+=alpha_loss
      
    out = self.global_pooling(s1)
    logits = self.classifier(out.view(out.size(0), -1))
    self.alphas_normal_prev = self.alphas_normal
    self.alphas_reduce_prev = self.alphas_reduce
    self.alphas_reduce = None
    self.alphas_normal = None
    return logits, alpha_loss_total/2

  def _loss(self, input, target, epoch):
    logits = self(input, epoch)
    return self._criterion(logits, target)

  def _initialize_alphas(self):
    """Initialize the architecture parameter: alpha
    """
    self.alphas_normal  = None
    self.alphas_reduce = None
    self.alphas_normal_prev = None
    self.alphas_reduce_prev = None
    self._arch_parameters = [self.alphas_normal, self.alphas_reduce]

  def genotype(self):
    if self.alphas_normal_prev is None or self.alphas_reduce_prev is None: return None
    gene_normal = parse2gene(self.alphas_normal_prev, self._reweight,self._steps)
    gene_reduce = parse2gene(self.alphas_reduce_prev, self._reweight,self._steps)

    concat = range(2+self._steps-self._multiplier, self._steps+2)
    genotype = Genotype(
      normal=gene_normal, normal_concat=concat,
      reduce=gene_reduce, reduce_concat=concat
    )
    return genotype

  def update_history(self):
    mm = 0
    last_id = 1
    node_id = 0
    weights1 = self.alphas_normal
    weights2 = self.alphas_reduce
    if weights1 is None or weights2 is None: return
    k, num_ops = weights1.shape
    for i in range(k):
      for j in range(num_ops):
        self.alphas_normal_history['edge: {}, op: {}'.format((node_id, mm), PRIMITIVES[j])].append(float(weights1[i][j]))
        self.alphas_reduce_history['edge: {}, op: {}'.format((node_id, mm), PRIMITIVES[j])].append(float(weights2[i][j]))
      if mm == last_id:
        mm = 0
        last_id += 1
        node_id += 1
      else:
        mm += 1

  def random_generate(self):

    num_skip_connect = SearchControllerConf['random_search']['num_identity']
    num_arch = SearchControllerConf['random_search']['num_arch']
    flops_threshold = SearchControllerConf['random_search']['flops_threshold']

    """Random generate the architecture"""
    # k = 2 + 3 + 4 + 5 = 14
    k = sum(1 for i in range(self._steps) for n in range(2+i))
    num_ops = len(PRIMITIVES)

    self.random_arch_list = []
    for ai in range(num_arch):
      seed = random.randint(0, 1000)
      torch.manual_seed(seed)
      while True:
        self.alphas_normal = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=False)
        self.alphas_reduce = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=False)
        arch = self.genotype()
        # if the skip connect meet num_skip_connect
        op_names, indices = zip(*arch.normal)
        cnt = 0
        for name, index in zip(op_names, indices):
          if name == 'skip_connect':
            cnt += 1
        if cnt == num_skip_connect:
          # the flops threshold
          model = NetworkCIFAR(36, 10, 20, True, arch, False)
          flops, params = profile(model, inputs=(torch.randn(1, 3, 32, 32),), verbose=False)
          if flops / 1e6 >= flops_threshold:
            self.random_arch_list += [('arch_' + str(ai), arch)]
            break
          else:
            continue

    return self.random_arch_list