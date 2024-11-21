import torch
import numpy as np
from torch import nn
import copy
from utils import loss,dataset
from os.path import join
import argparse
from torch.utils.data import DataLoader,random_split
from Model import ResNet
from Model import RCBAM
from Model import ConvNext_V2
import wandb
from utils.transform import bacteria_train_transform  # 确保导入数据增强模块

max_acc_val_RCBAM = 0                   #记录验证集RCBAM的最大准确率
max_acc_val_ResNet = 0                  #记录验证集ResNet的最大准确率  
max_acc_val_ConvNext_V2 = 0             #记录验证集ConvNext_V2的最大准确率
      
acc_max_model_RCBAM = 0                 #记录训练集RCBAM的最大准确率
acc_max_model_ResNet = 0                #记录训练集ResNet的最大准确率
acc_max_model_ConvNext_V2 = 0           #记录训练集ConvNext_V2的最大准确率



def train_model(args,model):            #训练模型

    wandb.init(
        # set the wandb project where this run will be logged
        project="bac-id-trains",

        # track hyperparameters and run metadata
        config={
        "learning_rate": args.lr,
        "architecture": args.arc,
        "dataset": args.train_dir,
        "epochs": args.epochs,
        }
    )

    data = torch.load(join(args.train_dir,'train_data.pt'))
    label = torch.load(join(args.train_dir,'train_label.pt'))
    
    print("data shape:", data.shape)
    load_dataset = dataset.PointsFolder(data,label)
    train_len=int(len(data)*args.ratio)
    val_len=int(len(data)-train_len)
    train_dataset,val_dataset = random_split(dataset=load_dataset,lengths=[train_len,val_len],generator=torch.Generator().manual_seed(1))
    train_loader = DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True)
    val_loader   = DataLoader(val_dataset,batch_size=args.batch_size,shuffle=True)

    [RCBAM_model, ResNet_model, ConvNext_V2_model] = model                  #获取模型 初始化参数
    args.criterion = nn.CrossEntropyLoss()                                  #初始化损失函数
    args.DML_criterion = loss.DistillationLoss(0.25)                        #初始化蒸馏损失函数
    args.optimizer_RCBAM = torch.optim.Adam(RCBAM_model.parameters(), lr=0.000138, weight_decay=0.00001, betas=(0.5, 0.99))             #初始化优化器
    args.schduler_RCBAM = torch.optim.lr_scheduler.StepLR(args.optimizer_RCBAM, step_size=10, gamma=0.1)                                #初始化学习率调度器
    args.optimizer_ResNet = torch.optim.Adam(ResNet_model.parameters(), lr=0.0002, weight_decay=0.00001, betas=(0.5, 0.99))             #初始化优化器
    args.schduler_ResNet = torch.optim.lr_scheduler.StepLR(args.optimizer_ResNet, step_size=10, gamma=0.1)                              #初始化学习率调度器
    args.optimizer_ConvNext_V2 = torch.optim.Adam(ConvNext_V2_model.parameters(), lr=0.00008, weight_decay=0.00001, betas=(0.5, 0.99))  #初始化优化器
    args.schduler_ConvNext_V2 = torch.optim.lr_scheduler.StepLR(args.optimizer_ConvNext_V2, step_size=10, gamma=0.1)                    #初始化学习率调度器
    
    if args.load:
        print('--load Model!')
        RCBAM_model.load_state_dict(torch.load(join(args.checkpoints_dir,'RCBAM.pth')))
        ResNet_model.load_state_dict(torch.load(join(args.checkpoints_dir,'ResNet.pth')))
        ConvNext_V2_model.load_state_dict(torch.load(join(args.checkpoints_dir,'ConvNext_V2.pth')))
        
    RCBAM_model.to(args.device)
    ResNet_model.to(args.device)
    ConvNext_V2_model.to(args.device)
    for epoch in range(args.epochs):
        running_loss_RCMAB = 0.0
        running_loss_ResNet = 0.0
        running_loss_ConvNext_V2 = 0.0

        running_acc_RCMAB = 0.0
        running_acc_ResNet = 0.0
        running_acc_ConvNext_V2 = 0.0

        RCBAM_model.train()
        ResNet_model.train()
        ConvNext_V2_model.train()
    
        for X_train, y_train in train_loader:
            X_train = X_train.to(args.device)
            y_train = y_train.to(args.device)

            X_train = X_train.to(torch.float32)
            y_train = y_train.to(torch.long)

            args.optimizer_RCBAM.zero_grad()
            args.optimizer_ResNet.zero_grad()
            args.optimizer_ConvNext_V2.zero_grad()

           
            output_RCBAM = RCBAM_model(X_train)
            output_ResNet = ResNet_model(X_train)
            output_ConvNext_V2 = ConvNext_V2_model(X_train)

            loss_RCBAM_cla = args.criterion(output_RCBAM, y_train)
            loss_ResNet_cla = args.criterion(output_ResNet, y_train)
            loss_ConvNext_V2_cla = args.criterion(output_ConvNext_V2, y_train)
            
            loss_RCBAM = args.DML_criterion(output_RCBAM, loss_RCBAM_cla, output_ConvNext_V2) / 3
            loss_ResNet = args.DML_criterion(output_ResNet, loss_ResNet_cla, output_RCBAM) / 3
            loss_ConvNext_V2 = args.DML_criterion(output_ConvNext_V2, loss_ConvNext_V2_cla, output_RCBAM) / 3

            (loss_RCBAM).backward(retain_graph=True)
            args.optimizer_RCBAM.step()

            loss_ResNet.backward(retain_graph=True)
            args.optimizer_ResNet.step()

            loss_ConvNext_V2.backward(retain_graph=True)
            args.optimizer_ConvNext_V2.step()

            running_loss_RCMAB += loss_RCBAM.item()
            running_loss_ResNet += loss_ResNet.item()
            running_loss_ConvNext_V2 += loss_ConvNext_V2.item()

            _, pred_RCBAM = torch.max(output_RCBAM.data, 1)
            _, pred_ResNet = torch.max(output_ResNet.data, 1)
            _, pred_ConvNext_V2 = torch.max(output_ConvNext_V2.data, 1)

            running_acc_RCMAB += (pred_RCBAM == y_train).sum().item()
            running_acc_ResNet += (pred_ResNet == y_train).sum().item()
            running_acc_ConvNext_V2 += (pred_ConvNext_V2 == y_train).sum().item()
        print('Epoch [%d/%d], Loss_RCBAM: %.4f, Loss_ResNet: %.4f, Loss_ConvNext_V2: %.4f, Acc_RCBAM: %.4f, Acc_ResNet: %.4f, Acc_ConvNext_V2: %.4f'
                % (epoch + 1, args.epochs, running_loss_RCMAB / len(train_dataset), running_loss_ResNet / len(train_dataset), running_loss_ConvNext_V2 / len(train_dataset), running_acc_RCMAB / len(train_dataset), running_acc_ResNet / len(train_dataset), running_acc_ConvNext_V2 / len(train_dataset)))
        
        # log metrics to wandb
        wandb.log({'Loss_train/RCBAM': running_loss_RCMAB / len(train_dataset),
                   'Loss_train/ResNet': running_loss_ResNet / len(train_dataset),
                   'Loss_train/ConvNext_V2': running_loss_ConvNext_V2 / len(train_dataset),
                   'Accuracy_train/RCBAM': running_acc_RCMAB / len(train_dataset),
                   'Accuracy_train/ResNet': running_acc_RCMAB / len(train_dataset),
                   'Accuracy_train/ConvNext_V2': running_acc_ConvNext_V2 / len(train_dataset)
                    }, step = epoch)

        args.schduler_RCBAM.step()
        args.schduler_ResNet.step()
        args.schduler_ConvNext_V2.step()

        acc_max_model_RCBAM,acc_max_model_ResNet,acc_max_model_ConvNext_V2 = evaluate(args,model,epoch,val_loader,val_dataset)

        # 每10个 epoch 保存模型
        if epoch % 10 == 9:
            torch.save(RCBAM_model.state_dict(), join(args.checkpoints_dir, f'RCBAM_epoch_{epoch+1}.pth'))
            torch.save(ResNet_model.state_dict(), join(args.checkpoints_dir, f'ResNet_epoch_{epoch+1}.pth'))
            torch.save(ConvNext_V2_model.state_dict(), join(args.checkpoints_dir, f'ConvNext_V2_epoch_{epoch+1}.pth'))

    return [acc_max_model_RCBAM, acc_max_model_ResNet, acc_max_model_ConvNext_V2]


def evaluate(args,model,epoch,test_loader,test_dataset):
    [RCBAM_model, ResNet_model, ConvNext_V2_model] = model
    
    RCBAM_model.eval()
    ResNet_model.eval()
    ConvNext_V2_model.eval()
    testing_correct_RCBAM = 0
    testing_correct_ResNet = 0
    testing_correct_ConvNext_V2 = 0
    testing_correct_sum = 0
    for X_test, y_test in test_loader:

        X_test = X_test.to(args.device)
        y_test = y_test.to(args.device)

        X_test = X_test.to(torch.float32)
        y_test = y_test.to(torch.long)

        with torch.no_grad():
            output_RCBAM = RCBAM_model(X_test)
            output_ResNet = ResNet_model(X_test)
            output_ConvNext_V2 = ConvNext_V2_model(X_test)

        out = (output_RCBAM + output_ResNet + output_ConvNext_V2) / 3

        _, pred_RCBAM = torch.max(output_RCBAM.data, 1)
        _, pred_ResNet = torch.max(output_ResNet.data, 1)
        _, pred_ConvNext_V2 = torch.max(output_ConvNext_V2.data, 1)
        _, pred = torch.max(out.data, 1)

        testing_correct_RCBAM += (pred_RCBAM == y_test).sum().item()
        testing_correct_ResNet += (pred_ResNet == y_test).sum().item()
        testing_correct_ConvNext_V2 += (pred_ConvNext_V2 == y_test).sum().item()
        testing_correct_sum += (pred == y_test).sum().item()
        

    acc_RCBAM = testing_correct_RCBAM / len(test_dataset)
    acc_ResNet = testing_correct_ResNet / len(test_dataset)
    acc_ConvNext_V2 = testing_correct_ConvNext_V2 / len(test_dataset)
    acc = testing_correct_sum / len(test_dataset)
    
    print('Epoch [%d/%d], Acc_RCBAM: %.4f, Acc_ResNet: %.4f, Acc_ConvNext_V2: %.4f, Acc: %.4f'
            % (epoch + 1, args.epochs, acc_RCBAM, acc_ResNet, acc_ConvNext_V2, acc))

    # 使用wandb记录测试集的准确率, 横轴为epoch
    wandb.log({
                'Accuracy_test/RCBAM': acc_RCBAM,
                'Accuracy_test/ResNet': acc_ResNet,
                'Accuracy_test/ConvNext_V2': acc_ConvNext_V2,
                'Accuracy_test/Ensemble': acc
    }, step = epoch)

    global max_acc_val_RCBAM,max_acc_val_ResNet,max_acc_val_ConvNext_V2
    global acc_max_model_RCBAM,acc_max_model_ResNet,acc_max_model_ConvNext_V2
    if acc_RCBAM > max_acc_val_RCBAM:
        max_acc_val_RCBAM = acc_RCBAM
        acc_max_model_RCBAM = copy.deepcopy(RCBAM_model)
    
    if acc_ResNet > max_acc_val_ResNet:
        max_acc_val_ResNet = acc_ResNet
        acc_max_model_ResNet = copy.deepcopy(ResNet_model)
    
    if acc_ConvNext_V2 > max_acc_val_ConvNext_V2:
        max_acc_val_ConvNext_V2 = acc_ConvNext_V2
        acc_max_model_ConvNext_V2 = copy.deepcopy(ConvNext_V2_model)
    return acc_max_model_RCBAM,acc_max_model_ResNet,acc_max_model_ConvNext_V2

        

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
    parser.add_argument('--epochs',type=int, default=100, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default= 0.0001, help='learning rate')
    parser.add_argument('--device', type=int, default=0, help='GPU ID')
    parser.add_argument('--train_dir', type=str, default='./data2048/train',help='save model with max accuracy')
    parser.add_argument('--ratio', type=float, default=0.9,help='the ratio of split train and val data')
    parser.add_argument('--load',action='store_true',help='load pretrained model')
    parser.add_argument('--checkpoints_dir',type=str,default='./checkpoints',help='save model with max accuracy')
    parser.add_argument('--arc', type=str, default='CM-Net', help='name of the model architecture to be used')
    args = parser.parse_args()
    
    RCBAM_model = RCBAM.ResNet8_RCBAM()
    ResNet_model = ResNet.ResNet_8()
    ConvNext_V2_model = ConvNext_V2.ConvNext_T()
    model = [RCBAM_model, ResNet_model, ConvNext_V2_model]
    
    acc_max_RCBAM, acc_max_ResNet, acc_max_ConvNext_V2=train_model(args,model)
    torch.save(acc_max_RCBAM.state_dict(), join(args.checkpoints_dir, 'RCBAM.pth'))
    torch.save(acc_max_ResNet.state_dict(), join(args.checkpoints_dir, 'ResNet.pth'))
    torch.save(acc_max_ConvNext_V2.state_dict(), join(args.checkpoints_dir, 'ConvNext_V2.pth'))
    wandb.save(join(args.checkpoints_dir, 'RCBAM.pth'))
    wandb.save(join(args.checkpoints_dir, 'ResNet.pth'))
    wandb.save(join(args.checkpoints_dir, 'ConvNext_V2.pth'))
    wandb.finish()

