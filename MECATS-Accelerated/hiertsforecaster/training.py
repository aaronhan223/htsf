import logging
from datetime import datetime
from operator import le
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data.dataloader import DataLoader
from hiertsforecaster.preprocess.hierarchical import TreeNodes
from hiertsforecaster.preprocess import utils
import torchvision
import torchvision.transforms as transforms
import torch.multiprocessing as mp


logger = logging.getLogger('MECATS.training')


def fit_gn(gpu, data, gn_params, parallel_params, optimizers, gns, name, window, split_point, device, experts, valid_range, preds, final_weights, childs=None):
    logger.info('Train gating network at node {}'.format(name))
    rank = parallel_params.nr * parallel_params.n_gpu + gpu
    dist.init_process_group(backend='nccl', init_method='env://', world_size=parallel_params.world_size, rank=rank)
    torch.cuda.set_device(gpu)

    MSE, regularization = nn.MSELoss().to(device + str(gpu)), False
    gn, optimizer, data = gns[name].to(device + str(gpu)), optimizers[name], data[name]
    gn = nn.parallel.DistributedDataParallel(gn, device_ids=[gpu])
    lr_scheduler, early_stopping = utils.LRScheduler(optimizer), utils.EarlyStopping()
    set_range = range(window, split_point)

    if childs is not None:
        gn_input = utils.prepare_data(set_range, data[childs], window)
        regularization = True
    else:
        gn_input = utils.prepare_data(set_range, data, window)
    
    # Don't set batch size too large (>128) for short time series.
    dataset = utils.GatingDataset(gn_input)
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=parallel_params.world_size, rank=rank)
    train_loader = DataLoader(dataset=dataset, batch_size=gn_params.batch_size, num_workers=0, shuffle=False, pin_memory=True, sampler=train_sampler, drop_last=True)
    preds = preds[name].to(device + str(gpu))
    y = torch.tensor(data[valid_range].to_list(), dtype=torch.float32)
    y = torch.unsqueeze(y, 0).to(device + str(gpu))
    all_weights = torch.zeros((gn_params.num_epochs, len(experts)))

    start = datetime.now()
    for epoch in range(gn_params.num_epochs):
        epoch_loss, count = 0., 0
        epoch_weights = torch.zeros((1, len(experts)), dtype=torch.float32, requires_grad=False)
        for train_batch in train_loader:
            if regularization:
                batch = torch.unsqueeze(train_batch[:, :, 0], 2)
            else:
                batch = train_batch
            
            batch = batch.to(device + str(gpu))
            weights = gn(batch)
            weighted_preds = torch.matmul(weights, preds)
            optimizer.zero_grad()

            if regularization:
                loss = MSE(weighted_preds, y) + get_reg(childs, gns, train_batch, device + str(gpu), preds.shape[1], gn_params.Lambda)
            else:
                loss = MSE(weighted_preds, y)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.data
            epoch_weights += weights.cpu().data
            count += 1

        epoch_loss /= count
        all_weights[epoch] = (epoch_weights / count)[0]
        if epoch % 1 == 0:
            logger.info('Epoch [{}] | Node {} | MSE {}'.format(epoch, name, epoch_loss))
        if gn_params.lr_schedule:
            lr_scheduler(epoch_loss)
        if gn_params.early_stop:
            early_stopping(epoch_loss)
            if early_stopping.early_stop:
                break
    if gpu == 0:
        logger.info("Training complete in: " + str(datetime.now() - start))
    final_weights[name] = all_weights[-1, :].numpy()
    return final_weights

    if self.plot and not self.config.MecatsParams.unit_test:
        level = TreeNodes(self.nodes, name=name).get_levels()
        legend = True if name == '0' else False
        utils.plot_weights(all_weights.numpy(), gn_params.num_epochs, level, name, legend, self.experts)


def train_78751(gpu, params, model, optimizer, data, name, window, split_point):
    rank = params.nr * params.n_gpu + gpu
    dist.init_process_group(backend='nccl', init_method='env://', world_size=params.world_size, rank=rank)
    torch.manual_seed(0)
    torch.cuda.set_device(gpu)
    model.to('cuda:' + str(gpu))
    batch_size = 100
    criterion = nn.CrossEntropyLoss().to('cuda:' + str(gpu))
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

    # ---------test------------
    data = data[name]
    set_range = range(window, split_point)
    gn_input = utils.prepare_data(set_range, data, window)
    dataset = utils.GatingDataset(gn_input)
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=params.world_size, rank=rank)
    print('dataset length', len(dataset))
    train_loader = DataLoader(dataset=dataset, batch_size=2, num_workers=0, shuffle=False, pin_memory=True, sampler=train_sampler, drop_last=True)
    # ---------test------------
    # train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=params.world_size, rank=rank)
    # train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
    #                                             batch_size=batch_size,
    #                                             shuffle=False,
    #                                             num_workers=0,
    #                                             pin_memory=True,
    #                                             sampler=train_sampler,
    #                                             drop_last=True)

    start = datetime.now()
    total_step = len(train_loader)
    for epoch in range(6):
        for i, (images, labels) in enumerate(train_loader):
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 100 == 0 and gpu == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, 6, i + 1, total_step, loss.item()))
    if gpu == 0:
        print("Training complete in: " + str(datetime.now() - start))


def get_reg(preds, childs, gns, train_batch, device, h, Lambda=.5):
    reg = torch.zeros((1, h), requires_grad=False, device=device)
    for i, node in enumerate(childs):
        weights = gns[node](torch.unsqueeze(train_batch[:, :, i], 2).to(device))
        preds = preds[node]
        if i == 0:
            reg += torch.matmul(weights, preds)
        else:
            reg -= torch.matmul(weights, preds)
    return Lambda * torch.sum(torch.pow(reg, 2))
