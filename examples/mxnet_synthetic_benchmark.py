"Benchmark script for compression algorithm in MXNet training"
import mxnet as mx
import horovod.mxnet as hvd
import numpy as np

import argparse
import logging
import math
import os
import timeit

from gluoncv.model_zoo import get_model
from mxnet import autograd, gluon, lr_scheduler
from mxnet.io import DataBatch, DataIter

# Benchmark settings
parser = argparse.ArgumentParser(description='MXNet Synthetic Benchmark',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--low-precision-allreduce', type=str, default='bf16',
                    choices=['fp16', 'bf16'], help='use low precision 16-bit compression during allreduce')

parser.add_argument('--model', type=str, default='resnet50_v1b',
                    help='model to benchmark')
parser.add_argument('--hybrid', action='store_true', default=False,
                    help='whether to use hybridized model')
parser.add_argument('--use-pretrained', action='store_true', default=False,
                    help='load pretrained model weights (default: False)')
parser.add_argument('--dtype', type=str, default='float32',
                    help='data type for training (default: float32)')

parser.add_argument('--batch-size', type=int, default=32,
                    help='input batch size')

parser.add_argument('--num-warmup-batches', type=int, default=5,
                    help='number of warm-up batches that don\'t count towards benchmark')
parser.add_argument('--num-batches-per-iter', type=int, default=10,
                    help='number of batches per benchmark iteration')
parser.add_argument('--num-iters', type=int, default=50,
                    help='number of benchmark iterations')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

opt = parser.parse_args()

if (opt.no_cuda and opt.low_precision_allreduce == 'fp16') or \
    (not opt.no_cuda and opt.low_precision_allreduce == 'bf16'):
    raise NotImplementedError("Currently, %s is not supported on %s platform for MXNet framework" % (opt.low_precision_allreduce, "CPU" if opt.no_cuda else "GPU"))

opt.no_cuda = True

hvd.init()
local_rank = hvd.local_rank()

# set context
context = mx.cpu(local_rank) if args.no_cuda else mx.gpu(local_rank)

# Get model from GluonCV model zoo
# https://gluon-cv.mxnet.io/model_zoo/index.html
kwargs = {'ctx': context,
          'pretrained': args.use_pretrained}

net = get_model(args.model, **kwargs)
net.cast(args.dtype)

# Create initializer
initializer = mx.init.Xavier(rnd_type='gaussian', factor_type="in",
                             magnitude=2)

# Hybridize and initialize model
if opt.hybrid:
    net.hybridize(static_alloc=True, static_shape=True)
net.initialize(initializer, ctx=context)

# Horovod: fetch and broadcast parameters
params = net.collect_params()
if params is not None:
    hvd.broadcast_parameters(params, root_rank=0)

# Create optimizer
optimizer_params = {}
# if args.dtype == 'float16':
    # optimizer_params['multi_precision'] = True
opt = mx.optimizer.create('sgd', **optimizer_params)

# Create loss function and train metric
loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()

# Horovod: create DistributedTrainer, a subclass of gluon.Trainer
compressor = hvd.Compression.bf16 if opt.low_precision_allreduce == 'bf16' else hvd.Compression.none
trainer = hvd.DistributedTrainer(params, opt, compressor)

# dummy data
data = mx.nd.random.uniform(opt.batch_size, 3, 224, 224)
label = mx.nd.random.randint(0, 1000, opt.batch_size, ctx=context)

# 
def benchmark_step():
    with autograd.record():
        output = net(data.astype(args.dtype, copy=False))
        loss = loss_fn(output, label)
    loss.backward()
    trainer.step(opt.batch_size)


def log(s, nl=True):
    if hvd.rank() != 0:
        return
    print(s, end='\n' if nl else '')


log('Model: %s' % args.model)
log('Batch size: %d' % args.batch_size)
device = 'GPU' if not args.no_cuda else 'CPU'
log('Number of %ss: %d' % (device, hvd.size()))

# Warm-up
log('Running warmup...')
timeit.timeit(benchmark_step, number=args.num_warmup_batches)

# Benchmark
log('Running benchmark...')
img_secs = []
for x in range(args.num_iters):
    time = timeit.timeit(benchmark_step, number=args.num_batches_per_iter)
    img_sec = args.batch_size * args.num_batches_per_iter / time
    log('Iter #%d: %.1f img/sec per %s' % (x, img_sec, device))
    img_secs.append(img_sec)

# Results
img_sec_mean = np.mean(img_secs)
img_sec_conf = 1.96 * np.std(img_secs)
log('Img/sec per %s: %.1f +-%.1f' % (device, img_sec_mean, img_sec_conf))
log('Total img/sec on %d %s(s): %.1f +-%.1f' %
    (hvd.size(), device, hvd.size() * img_sec_mean, hvd.size() * img_sec_conf))
