from io import BytesIO
import os
import lmdb
from PIL import Image


def FFHQ2PNG(input_dir,output_dir):
    env = lmdb.open(input_dir,max_readers=32,readonly=True,lock=False,readahead=False,meminit=False)

    if not env:
        raise IOError('Cannot open lmdb dataset', input_dir)

    length=0
    with env.begin(write=False) as txn:
        length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

    resolution = 256
    print('length=',length)
    
    with env.begin(write=False) as txn:
        for idx in range(length):
            key = f'{resolution}-{str(idx).zfill(5)}'.encode('utf-8')
            img_bytes = txn.get(key)

            buffer = BytesIO(img_bytes)
            img = Image.open(buffer)
            imageName = ot_dir=os.path.join(output_dir, str(idx)+'.png')
            img.save(imageName)


if __name__ == '__main__':
    input_dir='/user36/code/diffusion/DF-OT/exp/datasets/FFHQ/'
    output_dir='/user36/code/diffusion/DF-OT/exp/datasets/FFHQ/png/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    FFHQ2PNG(input_dir,output_dir)