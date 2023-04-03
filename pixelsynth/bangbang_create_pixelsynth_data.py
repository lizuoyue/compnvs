from PIL import Image
import os, glob

if __name__ == '__main__':

    for name in ['westminster_abbey', 'pantheon_exterior', 'sacre_coeur','trevi_fountain', 'notre_dame_front_facade']:
        abbr = name.split('_')[0]

        with open(f'bangbang/{name}/{abbr}.tsv', 'r') as f:
            lines = f.readlines()[1:]
            d = {'train': set(), 'test':  set()}
            for line in lines:
                filename, _, split, _ = line.strip().split('\t')
                if split in d:
                    d[split].add(filename.replace('.jpg', ''))
                else:
                    pass

        # os.system(f'mkdir -p {name}/dense/distill_train/rgb')
        # files = sorted(glob.glob(f'{name}/dense/distill/*.jpg'))
        # for file in files:
        #     img_id = os.path.basename(file).replace('_distill.jpg', '')
        #     if img_id in d['train']:
        #         img = Image.open(file)
        #         # w, h = img.size
        #         # if w < h:
        #         #     img = img.rotate(90, expand=True)
        #         img.save(f'{name}/dense/distill_train/rgb/{img_id}.jpg')


        # os.system(f'mkdir -p {name}/dense/distill_val/rgb')
        # files = sorted(glob.glob(f'{name}/dense/images/*.jpg'))
        # for file in files:
        #     img_id = os.path.basename(file).replace('.jpg', '') # _distill
        #     if img_id in d['test']:
        #         print(name, 'has val')
        #         img = Image.open(file)
        #         # w, h = img.size
        #         # if w < h:
        #         #     img = img.rotate(90, expand=True)
        #         img.save(f'{name}/dense/distill_val/rgb/{img_id}.jpg')
    
        files = sorted(glob.glob(f'bangbang/{name}/dense/distill_test_mask/*.jpg'))#[:-1]
        for file in files:
            bsnm = os.path.basename(file)
            os.system(f'cp bangbang/{name}/dense/distill_val/rgb/{bsnm} bangbang/{name}/dense/distill_test/rgb/{bsnm}')

            # mask = Image.open(file)
            # w, h = mask.size
            # if h > w:
            #     mask.rotate(90, expand=True).save(file)
