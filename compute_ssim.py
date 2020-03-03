from network import SRNDeblurNet
from log import TensorBoardX
from utils import *
import train_config as config
from data import Dataset,TestDataset
from tqdm import tqdm
from time import time
import sys
import os
from pytorch_ssim import *

log10 = np.log(10)
MAX_DIFF = 2

def compute_loss(  db256 , db128 , db64 , batch  ):

    # 256

    ssim_loss = ssim((db256+1)/2, (batch['label256']+1)/2 )

    mse1 = mse(db256, batch['label256'])

    #psnr
    psnr = 10*torch.log( MAX_DIFF**2 / mse1 ) / log10
    return  {'psnr':psnr,'ssim':ssim_loss}


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    tb = TensorBoardX(config_filename='train_config.py', sub_dir=config.train['sub_dir'])
    log_file = open('{}/{}'.format(tb.path, 'train.log'), 'w')

    val_img_list = open(config.train['train_img_list'], 'r').read().strip().split('\n')


    val_dataset = Dataset(val_img_list)

    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=config.train['val_batch_size'], shuffle=True,
                                                 drop_last=True, num_workers=2, pin_memory=True)

    mse = torch.nn.MSELoss().cuda()
    net = torch.nn.DataParallel(SRNDeblurNet(xavier_init_all=config.net['xavier_init_all'])).cuda()

    assert config.train['optimizer'] in ['Adam', 'SGD']
    if config.train['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=config.train['learning_rate'],
                                     weight_decay=config.loss['weight_l2_reg'])
    if config.train['optimizer'] == 'SGD':
        optimizer = torch.optim.SGD(net.parameters(), lr=config.train['learning_rate'],
                                    weight_decay=config.loss['weight_l2_reg'], momentum=config.train['momentum'],
                                    nesterov=config.train['nesterov'])

    last_epoch = -1

    if config.train['resume'] is not None:
        last_epoch = load_model(net, config.train['resume'], epoch=config.train['resume_epoch'])

    if config.train['resume_optimizer'] is not None:
        _ = load_optimizer(optimizer, net, config.train['resume_optimizer'], epoch=config.train['resume_epoch'])
        assert last_epoch == _

    val_loss_log_list = []

    t = time()

    best_val_psnr = 0

    with torch.no_grad():
        for step, batch in tqdm(enumerate(val_dataloader), total=len(val_dataloader), file=sys.stdout,
                                desc='validating'):
            for k in batch:
                batch[k] = batch[k].cuda(non_blocking=True)
                batch[k].requires_grad = False
            db256, db128, db64 = net(batch['img256'], batch['img128'], batch['img64'])
            loss = compute_loss(db256, db128, db64, batch)

            for k in loss:
                loss[k] = float(loss[k].cpu().detach().numpy())
            val_loss_log_list.append({k: loss[k] for k in loss})


        val_loss_log_dict = {k: float(np.mean([dic[k] for dic in val_loss_log_list])) for k in
                             val_loss_log_list[0]}

        val_loss_log_list.clear()

        tt = time()
        log_msg = ""

        log_msg += "  | val : "
        for idx, k_v in enumerate(val_loss_log_dict.items()):
            k, v = k_v
            if k == 'acc':
                log_msg += "{} {:.3%} {}".format(k, v, ',')
            else:
                log_msg += "{} {:.5f} {}".format(k, v, ',' if idx < len(val_loss_log_list) - 1 else '')
        tqdm.write(log_msg, file=sys.stdout)
        sys.stdout.flush()
        log_file.write(log_msg + '\n')
        log_file.flush()
        t = time()


    # plt.plot(train_loss, color='b')
    # plt.plot(val_loss, color='r')
    # plt.savefig('loss.jpg')