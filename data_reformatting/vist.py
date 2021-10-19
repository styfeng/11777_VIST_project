import os.path as osp
import json
import numpy as np
import math
import glob
from collections import defaultdict
from datetime import datetime
from pprint import pprint
from imageio import imread
from PIL import Image
import matplotlib.pyplot as plt

class Story_in_Sequence:
    def __init__(self, images_dir, annotations_dir, splits=None, retain_all_captions=False, max_stories=None):
        """
        The vist_dir should contain images and annotations, which further contain train/val/test.
        We will load train/val/test together on default and add split in albums, and make mapping.
        - albums  = [{id, title, vist_label, description, img_ids, story_ids}]
        - images  = [{id, album_id, datetaken, title, text, tags}]
        - sents   = [{id, story_id, album_id, img_id, order, original_text, text, length}]
        - stories = [{id, story_id, album_id, sent_ids, img_ids}]
        """
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir

        # Load annotations and add splits to each album
        if not splits:
            splits = ['train', 'val', 'test']
        sis = {'images': [], 'albums': [], 'annotations': []}
        for split in splits:
            b = datetime.now()
            info = json.load(open(osp.join(annotations_dir, 'sis', split+'.story-in-sequence.json')))
            print('sis\'s [%s] loaded. It took %.2f seconds.' % (split, (datetime.now() - b).total_seconds()))
            for album in info['albums']:
                album['split'] = split
            sis['albums'] += info['albums']
            sis['images'] += info['images']
            sis['annotations'] += info['annotations']
        dii = {'annotations': []}
        for split in splits:
            b = datetime.now()
            info = json.load(open(osp.join(annotations_dir, 'dii', split+'.description-in-isolation.json')))
            print('dii\'s [%s] loaded. It took %.2f seconds.' % (split, (datetime.now() - b).total_seconds()))
            dii['annotations'] += info['annotations']

        sents_sis = []
        for ann in sis['annotations']:
            # sent = {album_id, img_id, story_id, order, text, original_text, }
            sent = ann[0].copy()
            sent['id'] = sent.pop('storylet_id')
            sent['order'] = sent.pop('worker_arranged_photo_order')
            sent['img_id'] = sent.pop('photo_flickr_id')
            sent['length'] = len(sent['text'].split())  # add length property
            sents_sis += [sent]
        sents_dii = []; image_to_sents_dii = defaultdict(list); image_to_sent_ids_dii = defaultdict(list)
        for i, ann in enumerate(dii['annotations']):
            # sent = {album_id, img_id, order, text, original_text, }
            sent = ann[0].copy()
            sent['order'] = sent.pop('photo_order_in_story')
            sent['img_id'] = sent.pop('photo_flickr_id')
            sent['id'] = str(i)
            sents_dii += [sent]
            image_to_sent_ids_dii[sent['img_id']] += [sent['id']]
            image_to_sents_dii[sent['img_id']] += [sent['text']]

        # make mapping
        print('Make mapping ...')
        self.Albums = {album['id']: album for album in sis['albums']}
        self.Images = {img['id']: img for img in sis['images']}
        self.Sents = {sent['id']: sent for sent in sents_sis}
        self.Captions = {sent['id']: sent for sent in sents_dii}
        
        # album_id -> img_ids
        album_to_img_ids = {}
        for img in sis['images']:
            album_id = img['album_id']
            img_id = img['id']
            album_to_img_ids[album_id] = album_to_img_ids.get(album_id, []) + [img_id]
        def getDateTime(img_id):
            x = self.Images[img_id]['datetaken']
            return datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
        for album_id, img_ids in album_to_img_ids.items():
            img_ids.sort(key=getDateTime)

        # story_id -> sent_ids
        story_to_sent_ids = {}
        for sent_id, sent in self.Sents.items():
            story_id = sent['story_id']
            story_to_sent_ids[story_id] = story_to_sent_ids.get(story_id, []) + [sent_id]
        def get_order(sent_id):
            return self.Sents[sent_id]['order']
        for story_id, sent_ids in story_to_sent_ids.items():
            sent_ids.sort(key=get_order)

        # album_id -> story_ids
        album_to_story_ids = {}
        for story_id, sent_ids in story_to_sent_ids.items():
            sent = self.Sents[sent_ids[0]]
            album_id = sent['album_id']
            album_to_story_ids[album_id] = album_to_story_ids.get(album_id, []) + [story_id]

        # add to albums (and self.Albums)
        for album in sis['albums']:
            album['img_ids'] = album_to_img_ids[album['id']]
            album['story_ids'] = album_to_story_ids[album['id']]

        # make Stories: {story_id: {id, album_id, sent_ids, img_ids}}
        self.Stories = {story_id: {'id': story_id, 
                                   'sent_ids': sent_ids, 
                                   'sent_orders': [self.Sents[sent_id]['order'] for sent_id in sent_ids], 
                                   'sent_texts': [self.Sents[sent_id]['text'] for sent_id in sent_ids], 
                                   'captions': [image_to_sents_dii[self.Sents[sent_id]['img_id']] for sent_id in sent_ids], 
                                   'img_ids': [self.Sents[sent_id]['img_id'] for sent_id in sent_ids],
                                   'album_id': self.Sents[sent_ids[0]]['album_id']} 
            for story_id, sent_ids in story_to_sent_ids.items()}

        print('Mapping for [Albums][Images][Stories][Sents] done.')

        # fix data dictionaries by removing non-exisiting albums and stories
        self.fix_data_dicts(retain_all_captions, max_stories)

        # back to albums, images, stories, sents, 
        self.albums = list(self.Albums.values())
        self.images = list(self.Images.values())
        self.stories = list(self.Stories.values())
        self.sents = list(self.Sents.values())
        self.captions = list(self.Captions.values())


    def fix_data_dicts(self, retain_all_captions=False, max_stories=None):
        story_ids = list(self.Stories.keys())
        num_stories = 0
        for story_id in story_ids:
            if max_stories is not None and num_stories == max_stories:
                del self.Stories[story_id]
                continue
            story = self.Stories[story_id]
            sent_ids = story['sent_ids']
            album = self.Albums[story['album_id']]
            bad_sent_ids = []; bad_img_ids = []
            for i, sent_id in enumerate(sent_ids):
                img_id = self.Sents[sent_id]['img_id']
                img_file = osp.join(self.images_dir, album['split'], img_id + '.jpg')
                if not osp.isfile(img_file): bad_sent_ids.append(sent_id); bad_img_ids.append(img_id)
            if bad_sent_ids or (retain_all_captions and not all(story['captions'])):
                # for sent_id in bad_sent_ids: del self.Sents[sent_id]
                # for img_id in bad_img_ids:
                #     if img_id in self.Images: del self.Images[img_id]
                #     try: self.Albums[story['album_id']]['img_ids'].remove(img_id)
                #     except: pass
                # self.Albums[story['album_id']]['story_ids'].remove(story_id)
                del self.Stories[story_id]
            else:
                num_stories += 1
        if max_stories is not None and num_stories == max_stories:
            print(f'Reached max number of stories: {max_stories}')
        if retain_all_captions: print(f'Retained {num_stories}/{len(story_ids)} stories with captions.')
        else: print(f'Retained {num_stories}/{len(story_ids)} stories.')

    def read_img(self, img_file):
        img_content = imread(img_file)
        if len(img_content.shape) == 2:
            img_content = np.tile(img_content[:,:,np.newaxis], (1,1,3))
        img_content = Image.fromarray(img_content).resize(size=(224, 224))
        return img_content

    def show_story(self, story_id, show_image=True, show_img_ids=True, show_sents=True, show_captions=True):
        story = self.Stories[story_id]
        sent_ids = story['sent_ids']
        if show_image:
            plt.figure()
            for i, sent_id in enumerate(sent_ids):
                img_id = self.Sents[sent_id]['img_id']
                img = self.Images[img_id]
                album_id = img['album_id']
                split = self.Albums[album_id]['split']
                img_file = osp.join(self.images_dir, split, img_id + '.jpg')
                img_content = self.read_img(img_file)
                ax = plt.subplot(1, len(sent_ids), i+1)
                ax.imshow(img_content)
                ax.axis('off')
                ax.set_title(str(img_id)+'\n'+img['datetaken'][5:])
            plt.show()
        if show_img_ids:
            print('Image IDs:', '\n', story['img_ids'])
        if show_sents:
            print('Story of Images-in-Sequence (SIS):')
            for order,img_id,text in zip(story['sent_orders'], story['img_ids'], story['sent_texts']):
                print(' %s: img_id[%s], %s' % (order, img_id, text))
        if show_captions:
            print('Description of Images-in-Isolation (DII):')
            for order,img_id,captions in zip(story['sent_orders'], story['img_ids'], story['captions']):
                for k,caption in enumerate(captions):
                    print(' %s.%s: img_id[%s], %s' % (order, k, img_id, caption))


class Description_in_Isolation:
    def __init__(self, images_dir, annotations_dir, splits=None):
        """
        The vist_dir should contain images and annotations, which futher contain train/val/test
        We will load train/val/test together on default and add split in albums, and make mapping.
        - albums  = [{id, title, vist_label, description, img_ids}]
        - images  = [{id, album_id, datetaken, title, text, tags}]
        - sents   = [{id, album_id, img_id, order, original_text, text}], order no use???
        """
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir

        # Load annotations and add splits to each album
        if not splits:
            splits = ['train', 'val', 'test']
        dii = {'images': [], 'albums': [], 'annotations': []}
        for split in splits:
            b = datetime.now()
            info = json.load(open(osp.join(annotations_dir, 'dii', split+'.description-in-isolation.json')))
            print('dii\'s [%s] loaded. It took %.2f seconds.' % (split, (datetime.now() - b).total_seconds()))
            for album in info['albums']:
                album['split'] = split
            dii['albums'] += info['albums']
            dii['images'] += info['images']
            dii['annotations'] += info['annotations']

        sents = []
        for i, ann in enumerate(dii['annotations']):
            # sent = {album_id, img_id, order, text, original_text, }
            sent = ann[0].copy()
            sent['order'] = sent.pop('photo_order_in_story')
            sent['img_id'] = sent.pop('photo_flickr_id')
            sent['id'] = str(i)
            sents += [sent]

        # make mapping
        print('Make mapping ...')
        self.Albums = {album['id']: album for album in dii['albums']}
        self.Images = {img['id']: img for img in dii['images']}
        self.Sents  = {sent['id']: sent for sent in sents}

        # album_id -> img_ids
        album_to_img_ids = {}
        for img in dii['images']:
            album_id = img['album_id']
            img_id = img['id']
            album_to_img_ids[album_id] = album_to_img_ids.get(album_id, []) + [img_id]
        def getDateTime(img_id):
            x = self.Images[img_id]['datetaken']
            return datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
        for album_id, img_ids in album_to_img_ids.items():
            img_ids.sort(key=getDateTime)
        # add img_ids to albums (and self.Albums)
        for album in dii['albums']:
            album['img_ids'] = album_to_img_ids[album['id']]

        # img_id -> sent_ids
        img_to_sent_ids = {}
        for sent in sents:
            img_id = sent['img_id']
            img_to_sent_ids[img_id] = img_to_sent_ids.get(img_id, []) + [sent['id']]
        # add sent_ids to images
        for img_id in img_to_sent_ids.keys():
            self.Images[img_id]['sent_ids'] = img_to_sent_ids[img_id]
        for img_id, img in self.Images.items():
            if 'sent_ids' not in img:
                img['sent_ids'] = []

        print('Mapping for [Albums][Images][Sents] done.')

        # back to albums, images, stories, sents
        self.albums = self.Albums.values()
        self.images = self.Images.values()
        self.sents = self.Sents.values()

    def read_img(self, img_file):
        img_content = imread(img_file)
        if len(img_content.shape) == 2:
            img_content = np.tile(img_content[:,:,np.newaxis], (1,1,3))
        img_content = Image.fromarray(img_content).resize(size=(224, 224))
        return img_content

    def show_imgs_with_sents(self, img_ids, show_image=True):
        if show_image:
            plt.figure()
            cols = 5
            rows = math.ceil(len(img_ids) / float(cols))
            for i, img_id in enumerate(img_ids):
                img = self.Images[img_id]
                album = self.Albums[img['album_id']]
                img_file = osp.join(self.images_dir, album['split'], img_id + '.jpg')
                img_content = self.read_img(img_file)
                ax = plt.subplot(rows, cols, i+1)
                ax.imshow(img_content)
                ax.axis('off')
                ax.set_title(str(img_id)+'\n'+img['datetaken'][5:])
            plt.show()

        for i, img_id in enumerate(img_ids):
            sent_ids = self.Images[img_id]['sent_ids']
            for k, sent_id in enumerate(sent_ids):
                sent = self.Sents[sent_id]
                print('%s.%s: img_id[%s], %s' % (sent['order'], k, sent['img_id'], sent['text']))

