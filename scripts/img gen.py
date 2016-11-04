from PIL import Image
import os


def get_cropped_img(img):
    if img.size[0] is not img.size[1]:
        new_size = 256
        new_img = img.crop(
            (
                img.size[0] / 2 - new_size / 2,
                img.size[1] / 2 - new_size / 2,
                img.size[0] / 2 + new_size / 2,
                img.size[1] / 2 + new_size / 2
            )
        )
    return new_img

img_src = "H:\\Entwicklung\\ConvNN\\Thesis\\Thesis_Bilder\\raws\\places2 bsp"
cols = []
categories = 5
num_per_cat = 4
for i in range(categories):
    cols.append([])
for f in os.listdir(img_src):
    try:
        if int(f[0]) < 10:
            img = Image.open(os.path.join(img_src, f))
            if f.__contains__("(4)"):
                cols[int(f[0])].append(img)
            else:
                img = get_cropped_img(img)
                cols[int(f[0])].append(img)
            print '{}: {}'.format(f, img.size)
    except:
        print 'no img: '+f


comp_im = Image.new("RGB",
                    (256 * categories + (categories - 1) * 3,
                     256 * (num_per_cat - 1) + (num_per_cat - 1) * 3 + 50),
                    "white")
offset_x = 0
for col in cols:
    offset_y = 0
    for img in col:
        comp_im.paste(img,
                      (offset_x, offset_y))

        offset_y += 256+3
    offset_x += 256 + 3
#comp_im_big = comp_im.resize((400, 400))
comp_im.save(os.path.join(img_src, 'big.png'))
