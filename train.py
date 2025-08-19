import os
import argparse
import numpy as np
import torch
import torch.optim as opt
import torch.nn.functional as F
from dataloader import get_loader,test_dataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from SAM2DEGNet import SAM2DEGNet
import logging


parser = argparse.ArgumentParser("SAM2-DEGNet")
parser.add_argument("--hiera_path", type=str, default='sam2_hiera_large.pt',
                    help="path to the sam2 pretrained hiera")
parser.add_argument('--trainsize', type=int, default=352, help='training dataset size')
parser.add_argument('--train_root', type=str, default='Dataset/TrainDataset/',
                        help='the training rgb images root')
parser.add_argument('--val_root', type=str, default='./Dataset/TestDataset/COD10K/',
                        help='the test rgb images root')
parser.add_argument('--save_path', type=str, default='./trainpth/',
                    help="path to store the checkpoint")
parser.add_argument("--epoch", type=int, default=80,
                    help="training epochs")
# parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--lr", type=float, default=0.0006, help="learning rate")
parser.add_argument("--batch_size", default=6, type=int)
parser.add_argument("--weight_decay", default=5e-4, type=float)
args = parser.parse_args()



def dice_loss(predict, target):
    smooth = 1
    p = 2
    valid_mask = torch.ones_like(target)
    predict = predict.contiguous().view(predict.shape[0], -1)
    target = target.contiguous().view(target.shape[0], -1)
    valid_mask = valid_mask.contiguous().view(valid_mask.shape[0], -1)
    num = torch.sum(torch.mul(predict, target) *
                    valid_mask, dim=1) * 2 + smooth
    den = torch.sum((predict.pow(p) + target.pow(p))
                    * valid_mask, dim=1) + smooth
    loss = 1 - num / den
    return loss.mean()

def dda_loss(pred, mask):
    a = torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask) + 1e-6
    b = torch.abs(F.avg_pool2d(mask, kernel_size=51, stride=1, padding=25) - mask) + 1e-6
    c = torch.abs(F.avg_pool2d(mask, kernel_size=61, stride=1, padding=30) - mask) + 1e-6
    d = torch.abs(F.avg_pool2d(mask, kernel_size=27, stride=1, padding=13) - mask) + 1e-6
    e = torch.abs(F.avg_pool2d(mask, kernel_size=21, stride=1, padding=10) - mask) + 1e-6
    alph = 1.75

    fall = a ** (1.0 / (1 - alph)) + b ** (1.0 / (1 - alph)) + c ** (1.0 / (1 - alph)) + d ** (
                1.0 / (1 - alph)) + e ** (1.0 / (1 - alph))
    a1 = ((a ** (1.0 / (1 - alph)) / fall) ** alph) * a
    b1 = ((b ** (1.0 / (1 - alph)) / fall) ** alph) * b
    c1 = ((c ** (1.0 / (1 - alph)) / fall) ** alph) * c
    d1 = ((d ** (1.0 / (1 - alph)) / fall) ** alph) * d
    e1 = ((e ** (1.0 / (1 - alph)) / fall) ** alph) * e

    weight = 1 + 5 * (a1 + b1 + c1 + d1 + e1)

    dwbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    dwbce = (weight * dwbce).sum(dim=(2, 3)) / weight.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weight).sum(dim=(2, 3))
    union = ((pred + mask) * weight).sum(dim=(2, 3))
    dwiou = 1 - (inter + 1) / (union - inter + 1)

    return (dwbce + dwiou).mean()

def val(test_loader, model, epoch, save_path):
    """
    验证模型性能，并自动保存最佳 MAE 结果的模型
    """
    global best_mae, best_epoch
    model.eval()
    # with torch.no_grad():
    mae_sum = 0
    for i in range(test_loader.size):
        image, gt, name, img_for_post = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)  # 归一化
        image = image.cuda()

        res = model(image)
        res = F.upsample(res[2], size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        mae_sum += np.sum(np.abs(res - gt)) / (gt.shape[0] * gt.shape[1])  # 计算 MAE

    mae = mae_sum / test_loader.size  # 计算平均 MAE
    print(f'Epoch: {epoch}, MAE: {mae:.4f}, Best MAE: {best_mae:.4f}, Best Epoch: {best_epoch}.')

    # 如果当前 MAE 更小，则保存模型
    if mae < best_mae:
        best_mae = mae
        best_epoch = epoch
        torch.save(model.state_dict(), save_path + 'Net_epoch_best_{}.pth'.format(epoch))
        print(f"🔥 Save best model at epoch {epoch} with MAE: {mae:.4f}")

    logging.info(f'[Val Info]: Epoch: {epoch} MAE: {mae:.4f} Best Epoch: {best_epoch} Best MAE: {best_mae:.4f}')


def main(args):
    device = torch.device("cuda")
    model = SAM2DEGNet(args.hiera_path)
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")
    optim = opt.AdamW([{"params":model.parameters(), "initia_lr": args.lr}], lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optim, 80, eta_min=1.0e-8)
    os.makedirs(args.save_path, exist_ok=True)

    val_loader = test_dataset(image_root=args.val_root + 'Imgs/',
                              gt_root=args.val_root + 'GT/',
                              testsize=args.trainsize)

    for epoch in range(args.epoch):
        for i, (images, gts, edges) in enumerate(train_loader, start=1):
            images = images.cuda()
            gts = gts.cuda()
            edges = edges.cuda()

            optim.zero_grad()
            lateral_map_3, lateral_map_2, lateral_map_1, edge_map, coarse_map, lateral_map_4, e4, e3, e2, e1 = model(images)
            loss4 = dda_loss(lateral_map_4, gts)
            loss3 = dda_loss(lateral_map_3, gts)
            loss2 = dda_loss(lateral_map_2, gts)
            loss1 = dda_loss(lateral_map_1, gts)
            losse = dice_loss(edge_map, edges)
            lossc = dda_loss(coarse_map, gts)
            losse3 = dice_loss(e3, edges) * 0.25
            losse2 = dice_loss(e2, edges) * 0.5
            losse1 = dice_loss(e1, edges)
            loss = loss3 + loss2 + loss1 * 2 + 4 * losse + lossc + loss4 + losse3 + losse2 + losse1

            loss.backward()
            optim.step()
            if i % 50 == 0:
                print("epoch:{}-{}: loss:{}".format(epoch + 1, i + 1, loss.item()))
                
        scheduler.step()
        if (epoch+1) % 5 == 0 or (epoch+1) == args.epoch:
            torch.save(model.state_dict(), os.path.join(args.save_path, 'SAM2-UNet-%d.pth' % (epoch + 1)))
            print('[Saving Snapshot:]', os.path.join(args.save_path, 'SAM2-UNet-%d.pth'% (epoch + 1)))


        val(val_loader, model, epoch, save_path)



# best_mae = 1
best_fmeasure = 0.0
best_epoch = 0
save_path = args.save_path
train_loader = get_loader(image_root=args.train_root + 'Imgs/',
                              gt_root=args.train_root + 'GT/',
                              edge_root=args.train_root + 'Edge/',
                              batchsize=args.batch_size,
                              trainsize=args.trainsize,
                              num_workers=8)
total_step = len(train_loader)

if __name__ == "__main__":
    # seed_torch(1024)
    main(args)