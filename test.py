import torch
import argparse
import numpy as np
from utils import dataset
from Model import ResNet
from Model import RCBAM
from Model import ConvNext_V2
from tqdm import tqdm
from torch.utils.data import DataLoader
from os.path import join
import wandb



def infer(args, model):
    wandb.init(project="cmnet-test")
    [RCBAM_model, ResNet_model, ConvNext_V2_model] = model
    RCBAM_model.eval()                  # 评估模式进行评估和推理任务，禁用dropout和BN，不进行任何参数更新
    ResNet_model.eval()
    ConvNext_V2_model.eval()
    testing_correct_RCBAM = 0
    testing_correct_ResNet = 0
    testing_correct_ConvNext_V2 = 0
    testing_correct_sum = 0

    if args.load:
        # # 使用wandb保存模型
        # wandb.save(join(args.checkpoints_dir, 'RCBAM.pth'))
        # wandb.save(join(args.checkpoints_dir, 'ResNet.pth'))
        # wandb.save(join(args.checkpoints_dir, 'ConvNext_V2.pth'))
        RCBAM_model.load_state_dict(torch.load(join(args.checkpoints_dir, 'RCBAM.pth')))               #加载存储的模型权重
        ResNet_model.load_state_dict(torch.load(join(args.checkpoints_dir, 'ResNet.pth')))
        ConvNext_V2_model.load_state_dict(torch.load(join(args.checkpoints_dir, 'ConvNext_V2.pth')))

    RCBAM_model.to(args.device)
    ResNet_model.to(args.device)
    ConvNext_V2_model.to(args.device)

    data = torch.load(join(args.test_dir, 'test_data.pt'))
    label = torch.load(join(args.test_dir, 'test_label.pt'))
    test_dataset = dataset.PointsFolder(data, label)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)            # 测试数据集,不急看

    for X_test, y_test in tqdm(test_loader):
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

    wandb.log({
        "Acc_RCBAM": acc_RCBAM,
        "Acc_ResNet": acc_ResNet,
        "Acc_ConvNext_V2": acc_ConvNext_V2,
        "Acc_Ensemble": acc
    })
    print('Testing: Acc_RCBAM: %.4f, Acc_ResNet: %.4f, Acc_ConvNext_V2: %.4f, Acc: %.4f'
          % (acc_RCBAM, acc_ResNet, acc_ConvNext_V2, acc))
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the predict')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
    parser.add_argument('--device', type=int, default=0, help='GPU ID')
    parser.add_argument('--test_dir', type=str, default='./data2048/test', help='save model with max accuracy')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='save model with max accuracy')
    parser.add_argument('--load', action='store_true', help='save model with max accuracy')
    args = parser.parse_args()
    print('args.load', args.load)
    print('args.batch_size', args.batch_size)
    RCBAM_model = RCBAM.ResNet8_RCBAM()
    ResNet_model = ResNet.ResNet_8()
    ConvNext_V2_model = ConvNext_V2.ConvNext_T()
    model = [RCBAM_model, ResNet_model, ConvNext_V2_model]

    infer(args, model)
