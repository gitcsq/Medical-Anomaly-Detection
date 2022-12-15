import argparse
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score, average_precision_score

import sys
sys.path.append('..')
from util.datasets import *
import stage_2.s2_models_mae as s2_models_mae
import stage_2.s2_dataset as s2_dataset
from stage_2.s2_models_mae import cls_token_model, fc_model


def prepare_model(chkpt_dir, patch_num, args):
    # build model
    in_chans = 3 if args.dataset_name=='mvtec' else 1

    model = eval(args.model_name)(in_chans=in_chans, num_patches=patch_num, depth=args.depth, k=args.mlp_k)
    # load model
    checkpoint = torch.load(chkpt_dir)
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    return model


def generate_test_label(dataloader, model, source):
    """
    这是阶段二，模式1train_eval调用的函数
    输入：
        dataloader：
        model：
        source："img" or "latent"

    输出：auc，ap
    """
    device = torch.device("cuda")
    loss_all = []
    label_all = []
    loss_0_all = []
    wrong_file_name = []
    first = True
    average_0_loss = 0
    for batch_idx, d in enumerate(dataloader):
        (x, truth, data_pos, label_pos, test_label, file_name) = d
        x = x.to(device, non_blocking=True)
        truth = truth.to(device)
        data_pos = data_pos.to(device)
        label_pos = label_pos.to(device)

        model = model.to(device)
        if source == 'img':  # 如果是
            loss = model.forward_test_img(x, truth, data_pos, label_pos)
        else:
            # source是latent，训练数据的pos_embed是在generate_latent的时候加的，因此在第二阶段训练时不用再加。
            # 只用把label的pos_embed给cls_token （因为cls_token最后会成为label的预测）
            loss = model.forward_test(x, truth, label_pos)

        loss_all.append(loss.item())
        label_all.append(int(test_label.numpy()))
        if int(test_label.numpy()) == 0:
            loss_0_all.append(loss.item())
        else:
            if first:
                average_0_loss = np.mean(loss_0_all)
                print(average_0_loss)
                first = False

    loss_all = loss_all / np.max(np.array(loss_all))
    auc = roc_auc_score(label_all, loss_all)
    ap = average_precision_score(label_all, loss_all)
    print('auc: {}, ap: {}'.format(auc, ap))
    return auc, ap


def main(train_dir_name, test_dir_name, args):
    """
    这是阶段二，模式二test调用的函数
    输入：
        train_dir_name：数据集，要从这个文件夹提取数据
        test_dir_name：同上
        args：第二阶段主函数参数
    """
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    dataset_test = s2_dataset.latent_dataset(train_dir_name, test_dir_name, mode='test')
    patch_num = dataset_test.num_patches
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=1,
        shuffle=False,
        drop_last=False
    )
    model = prepare_model(args.s2_chkpt, patch_num=patch_num, args=args)
    auc, ap = generate_test_label(data_loader_test, model, args.source)
    save_results(args, auc, ap)


def save_results(args, auc, ap):
    if os.path.exists(args.output_dir + "result.xlsx"):
        df = pd.read_excel(args.output_dir + "result.xlsx")
        df_new = pd.DataFrame([(args.s1_chkpt, args.model_name, args.win_size, args.depth, args.mlp_k, args.epoch, auc, ap)],
                              columns=['s1_chkpt', 'model_name', 'win_size', 'depth', 'k', 'epoch', 'auc', 'ap'])
        df = pd.concat([df, df_new], ignore_index=True)
        print(df)
        df.to_excel(args.output_dir + "result.xlsx", index=False)
    else:
        df = pd.DataFrame([(args.s1_chkpt, args.model_name, args.win_size, args.depth, args.mlp_k, args.epoch, auc, ap)],
                          columns=['s1_chkpt', 'model_name', 'win_size', 'depth', 'k', 'epoch', 'auc', 'ap'])
        df.to_excel(args.output_dir + "result.xlsx", index=False)

