import argparse
import json
import os
import os.path
import xml.etree.ElementTree as et

import cv2
import numpy as np


def collectpts(pts):
    '''Convert to list of pairs of floats'''
    return [(float(x), float(y))
            for x, y in [s.split(',') for s in pts.split(';')]]


def extract_xml(anns):
    '''Parse xml annotation, output csv table and masks'''
    res = []
    assert (isinstance(anns, list))
    for fn in anns:
        tree = et.parse(f'Datasetforsegmentation/{fn}.xml')
        r = tree.getroot()
        mydir = r.find('meta/task/name').text
        anns = []
        for ch in r:
            if ch.tag == 'image':
                imfile = ch.attrib['name']
                for poly in ch:
                    assert (poly.tag == 'polygon')
                    pts = collectpts(poly.attrib['points'])
                    for attr in poly:
                        # image/polygon/attribute(sex,body_shape_guessed,main_orientation,completely_in_frame,type)
                        assert (attr.tag == 'attribute')
                        if attr.attrib['name'] == 'type':
                            species = attr.text
                    anns.append((imfile, species, pts))
        res.append((mydir, anns))
    return res


def rename(fn):
    return fn.replace(' ', '-')


class Data:

    def __init__(self, conf, mypath):
        self.config = conf
        self.mypath = mypath

    def get(self):
        '''Download and upack the data'''
        if os.path.exists('images'):
            print(
                'The "images" directory exists already - skipping data download!'
            )
            return

        if os.path.exists(os.path.basename(datasrc)):
            print('Data already downloaded - skipping')
        else:
            os.system(f'scp {datasrc} .')
        os.system(f'unzip -u {os.path.basename(datasrc)}')

        os.mkdir("images")
        os.mkdir("instance_masks")
        i = 0
        with open('annotations.csv', 'w') as f:
            for mydir, anns in extract_xml(self.config['annotations']):
                print(mydir, len(anns))
                # todo: copy images, create masks, output annotations to f
                for im, spec, pts in anns:
                    imname = rename(im)
                    i += 1
                    maskname = f'{imname}_{spec}_{i}.png'
                    xs = [x for (x, y) in pts]
                    ys = [y for (x, y) in pts]
                    bbox = (min(xs), min(ys), max(xs), max(ys))
                    f.write(f'{imname}\t{spec}\t{bbox}\t{maskname}\n')
                    os.system(
                        f'cp "Datasetforsegmentation/{mydir}/{im}" images/{imname}'
                    )
                    tmpimg = cv2.imread(f'images/{imname}')
                    H, W, C = tmpimg.shape
                    mask = np.zeros((H, W))
                    p = np.array(pts, dtype=np.int32)
                    cv2.fillPoly(mask, [p], 1)
                    cv2.imwrite(f'instance_masks/{maskname}', mask * 255)


if __name__ == '__main__':

    # Read in arguments from user
    parser = argparse.ArgumentParser(
        description='We need the default paths for things!')
    parser.add_argument('--config_path',
                        type=str,
                        default='default_config.json',
                        help='The path to the config. Should be a .json file')
    parser.add_argument('--raw_data_path',
                        type=str,
                        default='/data/data_for_segmentation',
                        help='The path to where the raw data is stored')
    parser.add_argument(
        '--converted_data_path',
        type=str,
        default='/data/converted_data/',
        help='The path to where the converted data should be stored')

    args = parser.parse_args()

    # Read in our data config from json
    with open(args.config_path) as json_file:
        config = json.load(json_file)

    # d = Data(defaultconf, '')
    # d.get()
