import os
from sre_constants import RANGE_UNI_IGNORE
import sys
import time
import glob
from genotypes import Genotype
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from model_search import Network, SearchControllerConf
from architect import Architect
from search_dataset import SearchDataset
import math
from genotypes import PRIMITIVES

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='/home/work/dataset/cifar/', help='location of the data corpus')
parser.add_argument("--data_name", type=str, default="cifar10")
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--noise_darts', action='store_true', default=False, help='use noise_darts')
parser.add_argument('--noise_type', type=str, default='N', help='noise type (default gaussian N(0,1)')
parser.add_argument('--max_step', type=int, default=30, help='T_max')
parser.add_argument('--random_search', action='store_true', default=False, help='use single level')
parser.add_argument('--num_identity',  type=int, default=2,  help='the number of skip connect (when random search is true)')
parser.add_argument('--num_arch',  type=int, default=8,  help='the number of architecture (when random search is true)')
parser.add_argument('--flops_threshold',  type=int, default=500,  help='the flops of architecture random generate')
parser.add_argument('--reweight', action='store_true', default=False, help='reweight the two operation with edge')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP-Original', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_lr_gamma', type=float, default=0.9, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--infer',action='store_true', default=False)
parser.add_argument('--flops',action='store_true',default=False)
parser.add_argument('--resume',type=str)
parser.add_argument('--top',type=int,default=2,help="the number of selected ops")
args = parser.parse_args()

args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

types = {"cifar10":10, "cifar100":100}

CIFAR_CLASSES = types[args.data_name]
print("num classes: ",CIFAR_CLASSES)

def main():
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  np.random.seed(args.seed)
  torch.cuda.set_device(0)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  logging.info('gpu device = %d' % args.gpu)
  logging.info("args = %s", args)

  criterion = nn.CrossEntropyLoss()

  # """Noise Darts"""
  # if args.noise_darts:
  #   SearchControllerConf['noise_darts']['noise_type'] = args.noise_type
  #   SearchControllerConf['noise_darts']['T_max'] = args.max_step
  # else:
  #   SearchControllerConf['noise_darts'] = None

  # """Random Darts"""
  # if args.random_search:
  #   SearchControllerConf['random_search']['num_identity'] = args.num_identity
  #   SearchControllerConf['random_search']['num_arch'] = args.num_arch
  #   SearchControllerConf['random_search']['flops_threshold'] = args.flops_threshold
  # else:
  #   SearchControllerConf['random_search'] = None

  # """Reweight Darts"""
  # SearchControllerConf['reweight'] = args.reweight
  
  model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion,top=args.top)
  if args.resume:
    model.load_state_dict(torch.load(args.resume))
    print("load model from resume")
    
  model = model.cuda()
  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  if args.flops:
    from fvcore.nn import FlopCountAnalysis
    tensor = (torch.randn(args.batch_size,3,32,32),)
    flops = FlopCountAnalysis(model,tensor)
    import IPython; import sys
    IPython.embed();sys.exit()
    print("Flops:",flops.total())
    exit(0)

  if args.random_search:
    genotype_list = model.random_generate()
    logging.info('genotype list = %s', genotype_list)
    logging.info('generate done!')
    sys.exit(0)

  model_optimizer = torch.optim.SGD(
      model.model_parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay)

  arch_optimizer = torch.optim.Adam(model.arch_parameters(),
        lr=args.arch_learning_rate, betas=(0.9, 0.999), weight_decay=args.arch_weight_decay)

  train_transform, valid_transform = utils._data_transforms_cifar10(args)
  if args.data_name=='cifar10':
    train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
    test_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)
  elif args.data_name=='cifar100':
    train_data = dset.CIFAR100(root=args.data, train=True, download=True, transform=train_transform)
    test_data = dset.CIFAR100(root=args.data, train=False, download=True, transform=valid_transform)
  
 
  #  split "train_data" to "train_data"+"valid_data"
  num_train = len(train_data)
  indices = list(range(num_train))
  split = int(np.floor(args.train_portion * num_train))

  search_data = SearchDataset(train_data,indices,split)
  
  search_queue = torch.utils.data.DataLoader(
    search_data, batch_size = args.batch_size, shuffle=True,
    pin_memory=True, num_workers=2
  )
  test_queue = torch.utils.data.DataLoader(
    test_data,batch_size=args.batch_size,shuffle=False,
    pin_memory=True, num_workers=2
  )

  if args.infer:
    print("batch_size: ",args.batch_size)
    generate_genotype(test_queue,model)
    sys.exit(0)

  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        model_optimizer, float(args.epochs), eta_min=args.learning_rate_min)

  base_la= (args.batch_size * 2 // len(PRIMITIVES))+1
  max_la = args.batch_size
  for epoch in range(args.epochs):
    scheduler.step()
    lr = scheduler.get_lr()[0]

    # model.set_la(cur_la)
    logging.info('epoch %d lr %e la %e', epoch, lr, model.get_la())

    genotype = model.genotype()
    logging.info('genotype = %s', genotype)

    model.update_history()

    # training and search the model
    train_acc, train_obj, arch_acc, arch_obj = train(search_queue, model, criterion, model_optimizer,arch_optimizer, lr, epoch)
    logging.info('train_acc %f, %f'%(train_acc, arch_acc))

    # validation the model
    valid_acc, valid_obj = infer(test_queue, model, criterion)
    logging.info('valid_acc %f', valid_acc)

    utils.save(model, os.path.join(args.save, 'weights_{}.pt'.format(epoch+1)))

def train(search_queue, model, criterion, model_optimizer, arch_optimizer, lr, epoch=None):
  model_objs = utils.AvgrageMeter()
  model_top1 = utils.AvgrageMeter()
  model_top5 = utils.AvgrageMeter()
  arch_objs = utils.AvgrageMeter()
  alpha_objs = utils.AvgrageMeter()
  arch_top1 = utils.AvgrageMeter()
  arch_top5 = utils.AvgrageMeter()

  for step, (input, target, input_search, target_search) in enumerate(search_queue):
    model.train()
    n = input.size(0)
    input = Variable(input, requires_grad=False).cuda(non_blocking=True)
    target = Variable(target, requires_grad=False).cuda(non_blocking=True)
    input_search = Variable(input_search, requires_grad=False).cuda(non_blocking=True)
    target_search = Variable(target_search, requires_grad=False).cuda(non_blocking=True)

    # update model
    model_optimizer.zero_grad()
    logits,_ = model(input, epoch)
    loss = criterion(logits, target)
    loss.backward()
    nn.utils.clip_grad_norm(model.model_parameters(), args.grad_clip)
    model_optimizer.step()

    model_prec1, model_prec5 = utils.accuracy(logits, target, topk=(1, 5))
    model_objs.update(loss.item(), n)
    model_top1.update(model_prec1.item(), n)
    model_top5.update(model_prec5.item(), n)

    #update arch
    arch_optimizer.zero_grad()

    logits,alpha_loss = model(input_search, epoch,is_alpha=True)
    arch_loss = criterion(logits, target_search)
    total_loss = 0.3*alpha_loss+arch_loss if (epoch>30) else arch_loss

    total_loss.backward()
    arch_optimizer.step()
    
    arch_prec1, arch_prec5 = utils.accuracy(logits, target_search, topk=(1, 5))
    arch_objs.update(arch_loss.item(), n)
    alpha_objs.update(alpha_loss.item(),n)
    arch_top1.update(arch_prec1.item(), n)
    arch_top5.update(arch_prec5.item(), n)

    if step % args.report_freq == 0:
      logging.info('train %03d loss: %e top1: %f top5: %f', step, model_objs.avg, model_top1.avg, model_top5.avg)
      logging.info('val %03d loss: %e alpha_loss: %e top1: %f top5: %f', step, arch_objs.avg, alpha_objs.avg, arch_top1.avg, arch_top5.avg)

  return model_top1.avg, model_objs.avg, arch_top1.avg, arch_objs.avg

def infer(valid_queue, model, criterion):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()

  with torch.no_grad():
    for step, (input, target) in enumerate(valid_queue):
      input = Variable(input, volatile=True).cuda(non_blocking=True)
      target = Variable(target, volatile=True).cuda(non_blocking=True)

      logits,_ = model(input)
      loss = criterion(logits, target)

      prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
      n = input.size(0)
      objs.update(loss.item(), n)
      top1.update(prec1.item(), n)
      top5.update(prec5.item(), n)

      if step % args.report_freq == 0:
        logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg

def generate_genotype(valid_queue, model):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()
  steps=4
  multiplier=4
  stem_multiplier=3
  
  weights_normal = []
  weights_reduce = []
  count = 0
  with torch.no_grad():
    for step, (input, target) in enumerate(valid_queue):
      input = Variable(input, volatile=True).cuda(non_blocking=True)
      target = Variable(target, volatile=True).cuda(non_blocking=True)

      logits,_ = model(input,is_train=False)
      alphas_normal = model.alphas_normal_prev
      alphas_reduce = model.alphas_reduce_prev
      if len(weights_normal) == 0:
        weights_normal.extend(alphas_normal)
      else: 
        for  i in range(len(weights_normal)):
          weights_normal[i] = torch.cat([weights_normal[i], alphas_normal[i]])
      if len(weights_reduce) == 0:
        weights_reduce.extend(alphas_reduce)
      else:
        for i in range(len(weights_reduce)):
          weights_reduce[i] = torch.cat([weights_reduce[i],alphas_reduce[i]])
      count+=input.shape[0]
      if count>=1000:
        break
    print(len(weights_normal))
    print(weights_normal[0].shape)
    gene_normal = utils.parse2gene(weights_normal, SearchControllerConf['reweight'],steps)
    gene_reduce = utils.parse2gene(weights_reduce, SearchControllerConf['reweight'],steps)
    
    concat = range(2+steps-multiplier, steps+2)
    genotype = Genotype(
      normal=gene_normal, normal_concat=concat,
      reduce=gene_reduce, reduce_concat=concat
    )
    print(genotype)
    return genotype

if __name__ == '__main__':
  main()
