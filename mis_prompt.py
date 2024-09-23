import argparse
import os
import yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path
import json
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from models.PMPL import PMPL
from models.tokenization_bert import BertTokenizer

import utils
from dataset import create_dataset, create_sampler, create_loader
from scheduler import create_scheduler
from optim import create_optimizer
from sklearn.metrics import f1_score, accuracy_score
from sklearn.manifold import TSNE
import numpy as np
import warnings
import os

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
# os.environ["WANDB_API_KEY"] = YOUR_API_KEY
os.environ["WANDB_MODE"] = "offline"

def tsne_visualize(source_feature: torch.Tensor, target_feature: torch.Tensor, target_feature1: torch.Tensor, 
              filename: str, source_color='r', target_color='b', target_color1='g'):
    """
    Visualize features from different domains using t-SNE.

    Args:
        source_feature (tensor): features from source domain in shape :math:`(minibatch, F)`
        target_feature (tensor): features from target domain in shape :math:`(minibatch, F)`
        filename (str): the file name to save t-SNE
        source_color (str): the color of the source features. Default: 'r'
        target_color (str): the color of the target features. Default: 'b'

    """
    source_feature = source_feature.numpy()
    target_feature = target_feature.numpy()
    target_feature1 = target_feature1.numpy()
    features = np.concatenate([source_feature, target_feature, target_feature1], axis=0)

    # map features to 2-d using TSNE
    X_tsne = TSNE(n_components=2, random_state=42).fit_transform(features)

    # domain labels, 1 represents source while 0 represents target
    domains = np.concatenate((np.ones(len(source_feature)), np.zeros(len(target_feature))))
    domains = np.concatenate((domains, np.zeros(len(target_feature1))+2))

    # visualize using matplotlib
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=domains, cmap=col.ListedColormap([target_color, source_color, target_color1]), s=20)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(filename)

def train(model, data_loader, optimizer, tokenizer, epoch, warmup_steps, device, scheduler, config, args, total_step, max_epoch, label_dict):
    # train
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 500
    step_size = 100 
    warmup_iterations = warmup_steps * step_size
    temp_labels=None
    for i, data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        total_step += 1
        # for i,data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        #     images, text, targets, missing_type = data['image'], data['text'], data['label'], data['missing_type']
        if args.dataset == 'mmimdb':
            (images, text, labels, text_labels, image_labels, information_label, img_id) = data
        elif args.dataset == 'crisismmd':
            (images, text, labels, text_labels, image_labels, information_label, img_id) = data
        elif args.dataset == 'hfir':
            (images, text, labels, text_labels, image_labels, information_label, img_id) = data
        labels = labels.to(device, non_blocking=True)

        if args.setting != 'multimodal' and epoch == 0 and args.dataset != 'mmimdb':
            text_labels, image_labels = F.one_hot(labels.to(device, non_blocking=True).to(torch.int64), args.class_num), F.one_hot(labels.to(device, non_blocking=True).to(torch.int64), args.class_num)
            temp_labels = F.one_hot(labels.to(device, non_blocking=True).to(torch.int64), args.class_num)
        elif args.setting != 'multimodal' and epoch == 0:
            text_labels, image_labels = labels.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            temp_labels = labels.to(device, non_blocking=True)    
        elif epoch == 0 :
            text_labels, image_labels = text_labels.to(device, non_blocking=True), image_labels.to(device, non_blocking=True)
        elif args.setting != 'multimodal':
            text_labels = []
            image_labels = []
            for i in range(len(img_id)):
                id = img_id[i]
                text_labels.append(torch.tensor(label_dict['text'][int(id)]).unsqueeze(0))
                image_labels.append(torch.tensor(label_dict['img'][int(id)]).unsqueeze(0))
            text_labels = torch.cat(text_labels,dim=0).to(device, non_blocking=True)
            image_labels = torch.cat(image_labels,dim=0).to(device, non_blocking=True)
            if args.dataset == 'mmimdb':
                temp_labels = labels.to(device, non_blocking=True)
            else:
                temp_labels = F.one_hot(labels.to(device, non_blocking=True).to(torch.int64), args.class_num)
        else:
            text_labels, image_labels = text_labels.to(device, non_blocking=True), image_labels.to(device, non_blocking=True)


        information_label = information_label.to(device, non_blocking=True)
        # if args.type != 'text':
        images = images.to(device, non_blocking=True)

        text_inputs = tokenizer(text, padding='longest', max_length=512, truncation=True, return_tensors="pt").to(
            device)
 
        if epoch < 0:
            ret = model(images, text_inputs, labels, text_labels, image_labels, information_label, epoch, use_caloss=False, Train=True,temp_labels=temp_labels)
        elif args.use_ca_loss:
            ret = model(images, text_inputs, labels, text_labels, image_labels, information_label, epoch, use_caloss=True, Train=True,temp_labels=temp_labels)
        else:
            ret = model(images, text_inputs, labels, text_labels, image_labels, information_label, epoch, use_caloss=False, Train=True,temp_labels=temp_labels)
        loss = ret['loss'] + args.beta*(ret['image_cls_loss']+ret['text_cls_loss'])
        if args.setting != 'multimodal':
            for i in range(len(img_id)):
                id = img_id[i]
                label_dict['text'][int(id)] = ret['text_labels'][i]
                label_dict['img'][int(id)] = ret['image_labels'][i]
 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(loss=loss.item())

        if epoch == 0 and i % step_size == 0 and i <= warmup_iterations:
            scheduler.step(i // step_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.4f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}, total_step, label_dict

@torch.no_grad()
def evaluate(model, data_loader, tokenizer, epoch, device, config, args, validation_type='test'):
    # test
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")

    header = 'Evaluation:'
    print_freq = 100

    end_labels = None
    preds = None


    informative = []
    informative_label = []
    no_informative = []
    no_informative_label = []

    for i, data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if args.dataset == 'mmimdb':
            (images, text, labels, text_labels, image_labels, information_label, img_id) = data
        elif args.dataset == 'crisismmd':
            (images, text, labels, text_labels, image_labels, information_label, img_id) = data
        elif args.dataset == 'hfir':
            (images, text, labels, text_labels, image_labels, information_label, img_id) = data
        labels = labels.to(device, non_blocking=True)
        information_label = information_label.to(device, non_blocking=True)
        images = images.to(device, non_blocking=True)
        text_labels, image_labels = text_labels.to(device, non_blocking=True), image_labels.to(device, non_blocking=True)

        text_inputs = tokenizer(text, padding='max_length', max_length=512, truncation=True, return_tensors="pt").to(
            device)
        ret = model(images, text_inputs, labels, text_labels, image_labels, information_label, epoch, use_caloss=True, Train=False)
        

        if args.dataset == 'crisismmd' or args.dataset == 'hfir':
            prediction = ret['logits']
        elif args.dataset == 'mmimdb':
            prediction = ret['logits']
            prediction = torch.tensor(prediction > args.shreshold, dtype=int)

        if end_labels == None:
            if args.model == 'pipeline':
                end_labels = torch.tensor(labels, dtype=int).cpu()
            else:
                end_labels = torch.tensor(labels, dtype=int).cpu()
               
            preds = prediction.cpu()
            if args.type == 'train' and args.model != 'pipeline' and args.dataset != 'mmimdb':
                for i in range(len(information_label)):
                    if information_label[i] == 1:
                        informative_label.append(labels[i].squeeze().unsqueeze(0))     
                        informative.append(prediction[i].squeeze().unsqueeze(0))
                    else:
                        no_informative_label.append(labels[i].squeeze().unsqueeze(0))     
                        no_informative.append(prediction[i].squeeze().unsqueeze(0))
        else:
            end_labels = torch.cat([end_labels, torch.tensor(labels, dtype=int).cpu()], dim=0)

            preds = torch.cat([preds, prediction.cpu()], dim=0)
            if args.type == 'train' and args.model != 'pipeline' and args.dataset != 'mmimdb':
                for i in range(len(information_label)):
                    if information_label[i] == 1:
                        informative_label.append(labels[i].squeeze().unsqueeze(0))     
                        informative.append(prediction[i].squeeze().unsqueeze(0))
                    else:
                        no_informative_label.append(labels[i].squeeze().unsqueeze(0))     
                        no_informative.append(prediction[i].squeeze().unsqueeze(0))

    if args.type == 'train' and args.model != 'pipeline' and args.dataset != 'mmimdb':
        informative_label = torch.cat(informative_label, dim=0)
        informative = torch.cat(informative, dim=0)
        no_informative_label = torch.cat(no_informative_label, dim=0)
        no_informative = torch.cat(no_informative, dim=0)
    if args.dataset == 'crisismmd' or args.dataset == 'hfir':
        print("It's All!")
        f1_mi = f1_score(end_labels.cpu(), torch.argmax(preds.cpu(), -1), average='micro')
        f1_ma = f1_score(end_labels.cpu(), torch.argmax(preds.cpu(), -1), average='macro')
        f1_we = f1_score(end_labels.cpu(), torch.argmax(preds.cpu(), -1), average='weighted')
        acc = accuracy_score(end_labels.cpu(), torch.argmax(preds.cpu(), -1))
        metrics = {'f1_mi':f1_mi, 'f1_ma':f1_ma, 'f1_we':f1_we, 'acc':acc}
        print('f1_mi',f1_mi)
        print('f1_ma',f1_ma)
        print('f1_we',f1_we)
        print('acc',acc)
    if args.dataset == 'mmimdb':
        print("It's All!")
        f1_ma = f1_score(end_labels.cpu(), preds.cpu(), average='macro')
        f1_mi = f1_score(end_labels.cpu(), preds.cpu(), average='micro')
        f1_we = f1_score(end_labels.cpu(), preds.cpu(), average='weighted')
        acc = accuracy_score(end_labels.cpu(), preds.cpu())
        metrics = {'f1_mi':f1_mi, 'f1_ma':f1_ma, 'f1_we':f1_we, 'acc':acc}
        print('f1_mi',f1_mi)
        print('f1_ma',f1_ma)
        print('f1_we',f1_we)
        print('acc',acc)
    if args.type == 'train' and args.model != 'pipeline' and args.dataset != 'mmimdb':      
        print("\n It's information!")
        f1_mi = f1_score(informative_label.cpu(), torch.argmax(informative.cpu(), -1), average='micro')
        f1_ma = f1_score(informative_label.cpu(), torch.argmax(informative.cpu(), -1), average='macro')
        f1_we = f1_score(informative_label.cpu(), torch.argmax(informative.cpu(), -1), average='weighted')
        acc = accuracy_score(informative_label.cpu(), torch.argmax(informative.cpu(), -1))
        metrics['inform_mi'] = f1_mi
        metrics['inform_ma'] = f1_ma
        metrics['inform_we'] = f1_we
        metrics['inform_acc'] = acc
        print('f1_mi',f1_mi)
        print('f1_ma',f1_ma)
        print('f1_we',f1_we)
        print('acc',acc)

        print("\n It's no_informative!")
        f1_mi = f1_score(no_informative_label.cpu(), torch.argmax(no_informative.cpu(), -1), average='micro')
        f1_ma = f1_score(no_informative_label.cpu(), torch.argmax(no_informative.cpu(), -1), average='macro')
        f1_we = f1_score(no_informative_label.cpu(), torch.argmax(no_informative.cpu(), -1), average='weighted')
        acc = accuracy_score(no_informative_label.cpu(), torch.argmax(no_informative.cpu(), -1))
        metrics['not_inform_mi'] = f1_mi
        metrics['not_inform_ma'] = f1_ma
        metrics['not_inform_we'] = f1_we
        metrics['not_inform_acc'] = acc
        print('f1_mi',f1_mi)
        print('f1_ma',f1_ma)
        print('f1_we',f1_we)
        print('acc',acc)

    labels = {'labels':end_labels, 'preds':preds}

    return {k: "{:.4f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}, metrics, labels


def main(args, config):
    utils.init_distributed_mode(args)

    device = torch.device(args.device)

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    #### Dataset ####
    print("Creating dataset")
    datasets = create_dataset(args.dataset, config, args)

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        samplers = create_sampler(datasets, [True, False, False], num_tasks, global_rank)

    print(args.dataset)
    if args.dataset != 'hfir':
        samplers = [None, None, None]
        train_loader, val_loader, test_loader = create_loader(datasets, samplers,
                                                                batch_size=[args.batch_size] + [
                                                                    config['batch_size_test']] * 2,
                                                                num_workers=[8, 8, 8], is_trains=[True, False, False],
                                                                collate_fns=[None, None, None])
    else:
        samplers = [None, None]
        train_loader, test_loader = create_loader(datasets, samplers,
                                                        batch_size=[args.batch_size] +
                                                            [config['batch_size_test']],
                                                        num_workers=[8, 8], is_trains=[True, False],
                                                        collate_fns=[None, None])

    tokenizer = BertTokenizer.from_pretrained(f'../models/{args.text_encoder}')

    #### Model ####
    print("Creating model")

    model =  PMPL(args, config)

    total_step = 0
    if args.evaluate:
        checkpoint = torch.load(args.load_path, map_location='cpu')
        state_dict = checkpoint['model']

        msg = model.load_state_dict(state_dict, strict=False)
        print('load checkpoint from %s' % args.load_path)
        print(msg)

    model = model.to(device)

    arg_opt = utils.AttrDict(config['optimizer'])
    optimizer = create_optimizer(arg_opt, model)
    arg_sche = utils.AttrDict(config['schedular'])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)

    max_epoch = config['schedular']['epochs']
    warmup_steps = config['schedular']['warmup_epochs']
    best = 0
    best_epoch = 0

    print("Start training")
    start_time = time.time()
    label_dict = {'text':{},'img':{}}
    for epoch in range(0, max_epoch):
        if epoch - best_epoch > 10:
            break
        if not args.evaluate:
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)
            train_stats, total_step, label_dict = train(model, train_loader, optimizer, tokenizer, epoch, warmup_steps, device, lr_scheduler,
                                config, args, total_step, max_epoch, label_dict)
        if args.dataset != 'hfir':
            val_stats, val_metrics, val_labels = evaluate(model, val_loader, tokenizer, epoch, device, config, args, validation_type='valid')
        test_stats, test_metrics, test_labels = evaluate(model, test_loader, tokenizer, epoch, device, config, args, validation_type='test')
        if utils.is_main_process():
            if args.evaluate:
                if args.dataset != 'hfir':
                    log_stats = {**{f'val_{k}': v for k, v in val_stats.items()},
                                 **{f'test_{k}': v for k, v in test_stats.items()},
                                 **{f'test_{k}': v for k, v in test_metrics.items()},
                                 'epoch': epoch,
                                 'best_epoch':best_epoch,
                                 }
                else:
                    log_stats = {**{f'test_{k}': v for k, v in test_stats.items()},
                                 f'test_{metrics_name}': test_f1,
                                 'epoch': epoch,
                                 'best_epoch':best_epoch,
                                 }

                with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                    f.write(json.dumps(log_stats) + "\n")
            else:
                if args.dataset != 'hfir':
                    if float(val_metrics['acc']) >= best:
                        save_obj = {
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'lr_scheduler': lr_scheduler.state_dict(),
                            'config': config,
                            'epoch': epoch,
                        }
                        torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best.pth'))
                        for k, v in test_labels.items():
                            torch.save(v, os.path.join(args.output_dir, f'{k}.pth'))

                        best = float(val_metrics['acc'])
                        best_epoch = epoch                  
                                            
                    log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                                 **{f'val_{k}': v for k, v in val_stats.items()},
                                 **{f'val_{k}': v for k, v in val_metrics.items()},
                                 **{f'test_{k}': v for k, v in test_stats.items()},
                                 **{f'test_{k}': v for k, v in test_metrics.items()},
                                 'epoch': epoch,
                                 'best_epoch':best_epoch,
                                 }


                    with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                        f.write(json.dumps(log_stats) + "\n")


                else:
                    if float(test_metrics['acc']) >= best:
                        save_obj = {
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'lr_scheduler': lr_scheduler.state_dict(),
                            'config': config,
                            'epoch': epoch,
                        }
                        torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best.pth'))

                        for k, v in test_labels.items():
                            torch.save(v, os.path.join(args.output_dir, f'{k}.pth'))

                        best = float(test_metrics['acc'])
                        best_epoch = epoch


                    log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                                 **{f'test_{k}': v for k, v in test_stats.items()},
                                 **{f'test_{k}': v for k, v in test_metrics.items()},
                                 'epoch': epoch,
                                 'best_epoch':best_epoch,
                                 }
                    
                    with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                        f.write(json.dumps(log_stats) + "\n")


        if args.evaluate:
            break
        lr_scheduler.step(epoch + warmup_steps + 1)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    if utils.is_main_process():
        with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
            f.write("best epoch: %d" % best_epoch)

        with open(os.path.join(args.output_dir, "args_file.txt"), 'w') as f:
            json.dump(args.__dict__, f, indent=2)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs')
    parser.add_argument('--output_dir', default='output/')
    parser.add_argument('--load_path', default='')
    parser.add_argument('--text_encoder', default='bert-base-uncased')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=False, type=bool)

    parser.add_argument('--prompt_length', default=16, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--type', default='train', choices=['text', 'image', 'both','train', 'tsne'], type=str)
    parser.add_argument('--class_num', default=24, choices=[23, 3, 102, 8, 7], type=int)
    parser.add_argument('--dataset', default='mmimdb', choices=['hfir','mmimdb', 'crisismmd'], type=str)
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('--hidden_size', default=768, type=int)
    parser.add_argument('--mmdlayer', default=0, type=int)
    parser.add_argument('--ca_loss', action='store_true')
    parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument('--train_dataset', default='train.txt', type=str)
    parser.add_argument('--test_dataset', default='test.txt', type=str)
    parser.add_argument('--dev_dataset', default='dev.txt', type=str)
    parser.add_argument('--alpha', default='0.7', type=float)
    parser.add_argument('--beta', default='0.7', type=float)
    parser.add_argument('--memory_length', default=20, type=int)
    parser.add_argument('--shreshold', default=0.5, type=float)

    # args for pmf model
    parser.add_argument('--n_encoder', default=4, type=int)
    parser.add_argument('--n_fusion', default=4, type=int)
    parser.add_argument('--n_trans', default=4, type=int)
    parser.add_argument('--mlp_hidden_sz', default=-1, type=int)
    parser.add_argument('--n_fusion_layers', default=10, type=int)
    parser.add_argument('--file_path', default='only_mm', type=str)
    parser.add_argument('--args_file', default='', type=str)

    parser.add_argument('--use_adapter', action='store_true')
    parser.add_argument('--use_cls', action='store_false')
    parser.add_argument('--use_gate', action='store_true')
    parser.add_argument('--use_layer_gate', action='store_false')
    parser.add_argument('--use_ca_loss', action='store_false')
    parser.add_argument('--all_cat', action='store_false')
    parser.add_argument('--use_prompt', action='store_false')
    parser.add_argument('--setting', default='multimodal', type=str)

    args = parser.parse_args()

    if args.args_file != '':
        with open(f'{args.args_file}', 'r') as f:
            args.__dict__.update(json.load(f))

    config = yaml.load(open(os.path.join(args.config, args.dataset+'.yaml'), 'r'), Loader=yaml.Loader)
    config['optimizer']['lr']=args.lr
    config['schedular']['lr']=args.lr
    config['train_file']=f'../dataset/{args.dataset}/'+args.train_dataset
    config['val_file']=f'../dataset/{args.dataset}/'+args.dev_dataset
    config['test_file']=f'../dataset/{args.dataset}/'+args.test_dataset

    args.output_dir = args.output_dir + f'{args.dataset}/{args.file_path}/' + f'{args.seed}_{args.lr}_{args.prompt_length}_{args.mlp_hidden_sz}_{args.n_fusion_layers}_{args.beta}_{args.seed}_{args.two_gate}_{args.use_adapter}_{args.use_layer_gate}_{args.use_ca_loss}_{args.all_cat}_{args.use_prompt}_{args.model}'

    main(args, config)