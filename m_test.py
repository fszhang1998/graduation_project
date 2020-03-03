from network import SRNDeblurNet
from utils import *
from tqdm import tqdm
import sys
import torch
from PIL import Image
import scipy.misc
import imageio
class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_list):
        super(type(self), self).__init__()
        self.img_list = img_list
        self.long=1280
        self.wide=720
        self.to_tensor = transforms.ToTensor()

    def crop_resize_totensor(self, img):
        img1 = img
        img2 = img1.resize((self.long // 2, self.wide // 2), resample=Image.BILINEAR)
        img3 = img2.resize((self.long // 4, self.wide // 4), resample=Image.BILINEAR)
        return self.to_tensor(img1), self.to_tensor(img2), self.to_tensor(img3)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        # filename processing
        blurry_img_name = self.img_list[idx]

        blurry_img = Image.open('./test/' + blurry_img_name)

        img1, img2, img3 = self.crop_resize_totensor(blurry_img)
        batch = {'img1': img1, 'img2': img2, 'img3': img3}
        for k in batch:
            batch[k] = batch[k] * 2 - 1.0#in range [-1,1]

        return batch

def print_photo(db, step, num):
    db = db.data.numpy()
    temp = db[0]
    temp = temp.transpose(1, 2, 0)
    # scipy.misc.imsave('output' + epoch + '_' + str(step) + '.png', temp)
    imageio.imwrite('output' + epoch + '_' + str(step + 1) + num + '.png', temp)
if __name__ == "__main__":



    val_img_list = open('./test.list', 'r').read().strip().split('\n')

    val_dataset = Dataset(val_img_list)

    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False,
                                                 drop_last=True, num_workers=2, pin_memory=True)

    net = SRNDeblurNet()
    epoch = '2799'
    net.load_state_dict(torch.load('./save/SRNDeblurNet_epoch'+epoch+'.pth'))


    with torch.no_grad():
        first_val = False
        for step, batch in tqdm(enumerate(val_dataloader), total=len(val_dataloader), file=sys.stdout,
                                desc='validating'):
            for k in batch:
                batch[k].requires_grad = False
            db256, db128, db64 = net(batch['img1'], batch['img2'], batch['img3'])


            print_photo(db256,step,'256')
            print_photo(db128, step, '128')
            print_photo(db64, step, '64')


