import argparse
from dataloader import PannukeDataset
from utils import collate_func
from infer.utils import *
# from models.xjnuseg import TransNuSeg
from loss.loss import *

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
num_classes = 2
nuc_classes = 5

IMG_SIZE = 256


def main():
    '''
    model_type:  default: transnuseg
    alpha: ratio of the loss of nuclei mask loss, dafault=0.3
    beta: ratio of the loss of normal edge segmentation, dafault=0.35
    gamma: ratio of the loss of cluster edge segmentation, dafault=0.35
    sharing_ratio: ratio of sharing proportion of decoders, default=0.5
    random_seed: set the random seed for splitting dataset
    dataset: Radiology(grayscale) or Histology(rgb), default=Histology
    num_epoch: number of epoches
    lr: learning rate
    model_path: if used pretrained model, put the path to the pretrained model here
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', required=True, default="transnuseg",
                        help="declare the model type to use, currently only support input being transnuseg")
    parser.add_argument("--alpha", required=True, default=0.3, help="coeffiecient of the weight of nuclei mask loss")
    parser.add_argument("--beta", required=True, default=0.35, help="coeffiecient of the weight of normal edge loss")
    parser.add_argument("--gamma", required=True, default=0.35, help="coeffiecient of the weight of cluster edge loss")
    parser.add_argument("--sharing_ratio", required=True, default=0.5, help=" ratio of sharing proportion of decoders")
    parser.add_argument("--num_workers", type=int, default=4, help="random seed")
    parser.add_argument("--batch_size", type=int, required=True, help="batch size")
    parser.add_argument("--dataset", required=True, default="Histology", help="Histology, Radiology")
    parser.add_argument("--num_epoch", type=int, required=True, help='number of epoches')
    parser.add_argument("--lr", required=True, help="learning rate")
    parser.add_argument("--model_path", default=None, help="the path to the pretrained model")
    parser.add_argument('--train_fold', type=int, default=3)
    parser.add_argument('--val_fold', type=int, default=2)
    parser.add_argument('--test_fold', type=int, default=1)
    parser.add_argument('--output_stride', type=int, default=4)
    parser.add_argument('--data_root', type=str, default='/root/autodl-tmp/transnuseg/pannuke')
    parser.add_argument('--seed', type=int, default=10)

    args = parser.parse_args()

    model_type = args.model_type
    dataset = args.dataset

    alpha = float(args.alpha)
    beta = float(args.beta)
    gamma = float(args.gamma)
    sharing_ratio = float(args.sharing_ratio)
    batch_size = int(args.batch_size)
    num_epoch = int(args.num_epoch)
    base_lr = float(args.lr)

    if dataset == "Radiology":
        channel = 1
        IMG_SIZE = 512
    elif dataset == "Histology":
        channel = 3
        IMG_SIZE = 256

    else:
        print("Wrong Dataset type")
        return 0

    model = TransNuSeg(img_size=IMG_SIZE)
    if args.model_path is not None:
        try:
            model.load_state_dict(torch.load(args.model_path))
        except Exception as err:
            print("{} In Loading previous model weights".format(err))

    # 使用双卡
    if torch.cuda.device_count() > 1:
        print("使用{}张卡".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)
    model.to(device)

    now = datetime.now()
    create_dir('./log')
    logging.basicConfig(filename='./log/log_{}_{}_{}.txt'.format(model_type, dataset, str(now)), level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(
        "Batch size : {} , epoch num: {}, alph: {}, beta : {}, gamma: {}, sharing_ratio = {}".format(batch_size,
                                                                                                     num_epoch, alpha,
                                                                                                     beta, gamma,
                                                                                                     sharing_ratio))

    train_dataset = PannukeDataset(data_root=args.data_root, seed=args.seed, is_train=True, fold=args.train_fold,
                                   output_stride=args.output_stride)
    trainloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
                             drop_last=True, num_workers=args.num_workers, persistent_workers=True,
                             collate_fn=collate_func, pin_memory=True)

    valid_dataset = PannukeDataset(data_root=args.data_root, seed=args.seed, is_train=False, fold=args.val_fold,
                                   output_stride=args.output_stride)
    valloader = DataLoader(dataset=valid_dataset, batch_size=args.batch_size, shuffle=False,
                           drop_last=True, num_workers=args.num_workers, persistent_workers=True,
                           collate_fn=collate_func, pin_memory=True)
    test_dataset = PannukeDataset(data_root=args.data_root, seed=args.seed, is_train=False, fold=args.test_fold,
                                  output_stride=args.output_stride)
    testloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False,
                            drop_last=True, num_workers=args.num_workers, persistent_workers=True,
                            collate_fn=collate_func, pin_memory=True)
    dataloaders = {"train": trainloader, "valid": valloader, "test": testloader}
    dataset_sizes = {"train": len(trainloader), "valid": len(valloader), "test": len(testloader)}

    val_loss = []
    train_loss = []
    lr_lists = []

    ce_loss1 = CrossEntropyLoss()
    dice_loss1 = DiceLoss(num_classes)
    ce_loss2 = CrossEntropyLoss()
    dice_loss2 = DiceLoss(num_classes)
    ce_loss3 = CrossEntropyLoss()
    dice_loss3 = DiceLoss(num_classes)
    dice_loss_dis = DiceLoss(num_classes)
    focal_loss = LocalFocal()

    optimizer = optim.Adam(model.parameters(), lr=base_lr)

    best_loss = 100
    best_epoch = 0

    for epoch in range(num_epoch):
        # early stop, if the loss does not decrease for 50 epochs
        if epoch > best_epoch + 50:
            break
        for phase in ['train', 'valid']:
            running_loss = 0
            running_loss_wo_dis = 0
            running_loss_seg = 0
            running_loss_nor = 0
            running_loss_clu = 0
            running_loss_dis = 0
            s = time.time()  # start time for this epoch
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()

            for i, d in enumerate(dataloaders[phase]):

                img, target_cate, target_edge, target_ins = d['image'], d['cate_labels'], d['edge_labels'], d['ins_labels']

                img = img.float()
                img = img.to(device)
                target_ins = torch.tensor(np.array(target_ins))
                target_edge = torch.tensor(np.array(target_edge))
                target_cate = torch.tensor(np.array([item.cpu().detach().numpy() for item in target_cate])).cuda()
                semantic_seg_mask = target_ins.to(device)
                normal_edge_mask = target_edge.to(device)
                point_mask = target_cate.to(device)

                # print('img shape ',img.shape)
                # print('semantic_seg_mask shape ',semantic_seg_mask.shape)

                output1, output2, output3 = model(img)

                loss_seg = 0.4 * ce_loss1(output1, semantic_seg_mask.long()) + 0.6 * dice_loss1(output1,
                                                                                                semantic_seg_mask.float(),
                                                                                                softmax=True)
                loss_nor = 0.4 * ce_loss2(output2, normal_edge_mask.long()) + 0.6 * dice_loss2(output2,
                                                                                               normal_edge_mask.float(),
                                                                                               softmax=True)
                loss_point = focal_loss(output3, point_mask)
                # print("loss_seg {}, loss_nor {}, loss_clu {}".format(loss_seg,loss_nor,loss_clu))
                if epoch < 10:
                    ratio_d = 1
                elif epoch < 20:
                    ratio_d = 0.7
                elif epoch < 30:
                    ratio_d = 0.4
                # elif epoch < 40:
                #     ratio_d = 0.1
                # # elif epoch >= 40:
                # #     ratio_d = 0
                # else:
                #     ratio_d = 0

                ### calculating the distillation loss
                m = torch.softmax(output1, dim=1)
                m = torch.argmax(m, dim=1)
                # m = m.squeeze(0)
                m = m.cpu().detach().numpy()

                b = torch.argmax(torch.softmax(output2, dim=1), dim=1)

                b2 = b.cpu().detach().numpy()
                # print('b2 shape',b2.shape)

                c = torch.argmax(torch.softmax(output3, dim=1), dim=1)
                pred_edge_1 = edge_detection(m.copy(), channel)
                pred_edge_1 = torch.tensor(pred_edge_1).to(device)
                pred_edge_2 = output2
                pred_edge_2[pred_edge_2 < 0] = 0

                # print("pred_edge_1 shape ",pred_edge_1.shape)
                # print("pred_edge_2 shape ",pred_edge_2.shape)
                dis_loss = dice_loss_dis(pred_edge_2, pred_edge_1.float())

                ### calculating total loss
                loss = alpha * loss_seg + beta * loss_nor + gamma * loss_point + ratio_d * dis_loss

                running_loss += loss.item()
                running_loss_wo_dis += (
                            alpha * loss_seg + beta * loss_nor + gamma * loss_point).item()  ## Loss without distillation loss
                running_loss_seg += loss_seg.item()  ## Loss for nuclei segmantation
                running_loss_nor += loss_nor.item()
                running_loss_clu += loss_point.item()
                running_loss_dis += dis_loss.item()
                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            e = time.time()
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_loss_wo_dis = running_loss_wo_dis / dataset_sizes[phase]  ## Epoch Loss without distillation loss
            epoch_loss_seg = running_loss_seg / dataset_sizes[phase]  ## Epoch Loss for nuclei segmantation
            epoch_loss_nor = running_loss_nor / dataset_sizes[phase]
            epoch_loss_clu = running_loss_clu / dataset_sizes[phase]
            epoch_loss_dis = running_loss_dis / dataset_sizes[phase]
            logging.info('Epoch {},: loss {}, {},time {}'.format(epoch + 1, epoch_loss, phase, e - s))
            logging.info(
                'Epoch {},: loss without distillation {}, {},time {}'.format(epoch + 1, epoch_loss_wo_dis, phase,
                                                                             e - s))
            logging.info('Epoch {},: loss seg {}, {},time {}'.format(epoch + 1, epoch_loss_seg, phase, e - s))
            logging.info('Epoch {},: loss nor {}, {},time {}'.format(epoch + 1, epoch_loss_nor, phase, e - s))
            logging.info('Epoch {},: loss clu {}, {},time {}'.format(epoch + 1, epoch_loss_clu, phase, e - s))
            logging.info('Epoch {},: loss dis {}, {},time {}'.format(epoch + 1, epoch_loss_dis, phase, e - s))

            if phase == 'train':
                train_loss.append(epoch_loss)
            else:
                val_loss.append(epoch_loss)

            if phase == 'valid' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_epoch = epoch + 1
                # best_model_wts = copy.deepcopy(model.state_dict())
                best_model_wts = copy.deepcopy(model.state_dict())
                logging.info("Best val loss {} save at epoch {}".format(best_loss, epoch + 1))

    draw_loss(train_loss, val_loss, str(now))

    create_dir('./saved')
    torch.save(best_model_wts, '/root/autodl-tmp/transnuseg/saved/model_epoch:{}_valloss:{}_{}.pt'.format(best_epoch, best_loss, str(now)))
    logging.info(
        'Model saved. at {}'.format('/root/autodl-tmp/transnuseg/saved/model_epoch:{}_valloss:{}_{}.pt'.format(best_epoch, best_loss, str(now))))

    model.load_state_dict(best_model_wts)
    model.eval()

    dice_acc_val = 0
    dice_loss_val = DiceLoss(num_classes)

    with torch.no_grad():
        for i, d in enumerate(testloader, 0):
            img, target_cate, target_edge, target_ins = d['image'], d['cate_labels'], d['edge_labels'], d['ins_labels']
            # semantic_seg_mask2 = semantic_seg_mask.cpu().detach().numpy()
            # normal_edge_mask2 = normal_edge_mask.cpu().detach().numpy()
            # cluster_edge_mask2 = cluster_edge_mask.cpu().detach().numpy()
            # img = img.unsqueeze(0)
            img = img.float()
            img = img.to(device)
            target_ins = torch.tensor(np.array(target_ins))
            target_ins = target_ins.to(device)

            # semantic_seg_mask = semantic_seg_mask.unsqueeze(0).float()

            output1, output2, output3 = model(img)
            d_l = dice_loss_val(output1, target_ins.float(), softmax=True)
            dice_acc_val += 1 - d_l.item()

    print(dice_acc_val, dataset_sizes['test'])
    logging.info("dice_acc {}".format(dice_acc_val / dataset_sizes['test']))


if __name__ == '__main__':
    main()
