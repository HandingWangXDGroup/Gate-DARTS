export CUDA_VISIBLE_DEVICES=2

cd moedarts

## train single
# python train.py --auxiliary --cutout --arch MoEDARTS --batch_size 96 --epoch 600

# bi_level train
python train_search_bi_level.py --data /data/Fanliang/data --epochs 50 --data_name cifar10 --learning_rate 0.025 --batch_size 128 --save 'EXP-bilevel'

## bi_level genotypes 
# python train_search_bi_level.py --data /data/Fanliang/data --data_name cifar10 --learning_rate 0.005 --batch_size 256 --infer --save 'EXP-lr_0005_alw_10_bilevel' --resume search-EXP-bilevel-20230504-124449/weights_50.pt
#