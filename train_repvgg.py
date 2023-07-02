#! /usr/bin/env python3
import os, sys, time, random, numpy as np
from DKD import dkd_loss
import torch, torch.nn as nn, torchaudio, model.classifier as classifiers
from model.repvgg import RepVGGStatsPool_
from torch.utils.data import DataLoader
from dataset import WavDataset
from tools.utils import get_eer, get_lr, change_lr
import torch.nn.functional as F
from config.config_repvgg import Config
from torch.nn import DataParallel
import model.MultiLevelTransfer as MLTmodel


def main():
    opt = Config()
    SEED = opt.seed
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu

    # training dataset
    train_dataset = WavDataset(opt=opt, train_mode=True)
    train_loader = DataLoader(train_dataset,
                      num_workers=opt.workers,
                      batch_size=opt.batch_size,
                      shuffle=True,
                      pin_memory=True)

    # validation dataset
    val_dataset = WavDataset(opt=opt, train_mode=False)
    val_dataloader = DataLoader(val_dataset, num_workers=opt.workers, pin_memory=True, batch_size=1)
    # RepVGG
    teacher_model = RepVGGStatsPool_(in_planes=opt.in_planes,embedding_size=opt.embd_dim).cuda()
    student_model = RepVGGStatsPool_(in_planes=opt.in_planes,embedding_size=opt.embd_dim).cuda()

    classifier = getattr(classifiers, opt.classifier)(opt.embd_dim, len(opt.spk2int),
                                      device_id=[i for i in range(len(opt.gpu.split(',')))],
                                      m=opt.angular_m, s=opt.angular_s).cuda() # arcface
    # epochs, start_epoch = opt.epochs, opt.start_epoch
    # logs = open('exp/train.out', 'w')
    
    os.system('mkdir -p exp/%s' % opt.save_dir)
    
    epochs, start_epoch = opt.epochs, opt.start_epoch
    if start_epoch != 0:
        print('Load exp/%s/model_%d.pkl' % (opt.save_dir, start_epoch-1))
        checkpoint = torch.load('exp/%s/model_%d.pkl' % (opt.save_dir, start_epoch-1))
        student_model.load_state_dict(checkpoint['model'])
        if opt.load_classifier:
            classifier.load_state_dict(checkpoint['classifier'])
        logs = open('exp/%s/train.out' % opt.save_dir, 'a')
    else:
        logs = open('exp/%s/train.out' % opt.save_dir, 'w')
        logs.write(str(student_model) + '\n' + str(classifier) + '\n')
    
    criterion = nn.CrossEntropyLoss().cuda()
    
    optimizer = torch.optim.Adam(list(student_model.parameters()) + list(classifier.parameters()),
                                 lr=opt.lr, betas=(0.9, 0.999), 
                                 eps=1e-08, weight_decay=2e-5,
                                 amsgrad=False)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [15,30,40], gamma=0.1, last_epoch=-1)
    batch_per_epoch = len(train_loader)
    lr_lambda = lambda x: opt.lr / (batch_per_epoch * opt.warm_up_epoch) * (x + 1)
    
    student_model = DataParallel(student_model)

    for epoch in range(start_epoch, epochs):
        student_model.train()
        classifier.train()
        end = time.time()
        for i, (feats, _, key) in enumerate(train_loader):
            data_time = time.time() - end
            if epoch < opt.warm_up_epoch:
                change_lr(optimizer, lr_lambda(len(train_loader) * epoch + i))
    
            feats, key = feats.cuda(), key.cuda()

            logits_student = classifier(student_model(feats), key)
            logits_teacher = classifier(teacher_model(feats), key)

            dkd_ls = dkd_loss(
                logits_student,
                logits_teacher,
                key,
                1,
                2.0,
                4,
                )
            
            loss_ce = criterion(logits_student, key)
            
            loss = dkd_ls + loss_ce

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            output_pre = np.argmax(logits_student.data.cpu().numpy(), axis=1)
            acc = np.mean((output_pre == key.cpu().numpy()).astype(int))

            batch_time = time.time() - end
            end = time.time()
            
            logs.write('Epoch [%d][%d/%d]\t ' % (epoch, i+1, len(train_loader)) + 
                       'Length %d\t' % (feats.shape[2]) +
                       'Time [%.3f/%.3f]\t' % (batch_time, data_time) +
                       'Loss %.4f\t' % (loss.data.item()) +
                       'Accuracy %3.3f\t' % (acc*100) +
                       'LR %.6f\n' % get_lr(optimizer))
            logs.flush()

        save_model('exp/%s' % opt.save_dir, epoch, student_model, classifier, optimizer, scheduler)
        # strongly recommend the following validate code when implement finetuning step.
        eer, cost = validate(student_model,val_dataloader,epoch,opt)
        logs.write('Epoch %d\t  lr %f\t  EER %.4f\t  cost %.4f\n'
                   % (epoch, get_lr(optimizer), eer*100, cost))
        scheduler.step()
        
def validate(model,val_dataloader,epoch,opt):
    model.eval()
    embd_dict={}
    with torch.no_grad():
        for j, (feat, _, utt) in enumerate(val_dataloader):
            outputs = model(feat.cuda())
            for i in range(len(utt)):
                embd_dict[utt[i]] = outputs[i,:].cpu().numpy()
    eer,_, cost,_ = get_eer(embd_dict, trial_file='%s/trails_dev_mic' % opt.val_dir)
    # np.save('exp/%s/test_%s.npy' % (opt.save_dir, epoch),embd_dict)
    return eer, cost

def save_model(chk_dir, epoch, model, classifier, optimizer, scheduler):
    torch.save({'model': model.module.state_dict(),
            'classifier': classifier.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
            }, os.path.join(chk_dir, 'model_%d.pkl' % epoch))

if __name__ == '__main__':
    main()

