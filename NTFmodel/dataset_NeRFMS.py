"""
MX-Font
Copyright (c) 2021-present NAVER Corp.
MIT license
"""
from pathlib import Path
from itertools import chain
import random
from PIL import Image

import torch

from base.dataset import BaseTrainDataset, BaseDataset, sample, render, read_font

class NeRFTrainDataset(BaseTrainDataset):
    def __init__(self, data_dir, chars, source_path, transform=None,
                 n_in_s=3, n_in_c=3, extension="png"):
        super().__init__(data_dir, chars, transform, extension)

        self.key_char_dict, self.char_key_dict = self.filter_chars()

        self.keys = sorted(self.key_char_dict)
        self.chars = sorted(set.union(*map(set, self.key_char_dict.values())))
        self.data_list = [(_key, _char) for _key, _chars in self.key_char_dict.items() for _char in _chars]
        self.n_in_s = n_in_s
        self.n_in_c = n_in_c
        self.n_chars = len(self.chars) 
        self.n_fonts = len(self.keys)   

        # source_path = " "
        self.source = read_font(source_path)
        self.transform = transform

    def filter_chars(self):
        char_key_dict = {}
        for char, keys in self.char_key_dict.items():
            num_keys = len(keys)
            if num_keys > 1:
                char_key_dict[char] = keys
            else:
                pass

        filtered_chars = set(char_key_dict)
        key_char_dict = {}
        for key, chars in self.key_char_dict.items():
            key_char_dict[key] = list(set(chars).intersection(filtered_chars))

        return key_char_dict, char_key_dict

    def render_from_source(self, char):
        img = render(self.source, char)
        img = self.transform(img)
        return img

    def __getitem__(self, index):
        key, char = self.data_list[index]
        fidx = self.keys.index(key)
        cidx = self.chars.index(char)

        trg_img = self.get_img(key, char)

        style_chars = set(self.key_char_dict[key]).difference({char})
        style_chars = sample(list(style_chars), self.n_in_s)
        style_imgs = torch.stack([self.get_img(key, c) for c in style_chars])

        char_keys = set(self.char_key_dict[char]).difference({key})
        char_keys = sample(list(char_keys), self.n_in_c)

        char_imgs = torch.stack([self.render_from_source(char) for k in char_keys])
        char_fids = [self.keys.index(_k) for _k in char_keys]

        ret = {
            "trg_imgs": trg_img,
            "trg_fids": torch.LongTensor([fidx]),
            "trg_cids": torch.LongTensor([cidx]),
            "style_imgs": style_imgs,
            "style_fids": torch.LongTensor([fidx]).repeat(self.n_in_s),
            "char_imgs": char_imgs,
            "char_fids": torch.LongTensor(char_fids)
        }

        return ret

    def __len__(self):
        return len(self.data_list)

    @staticmethod
    def collate_fn(batch):
        _ret = {}
        for dp in batch:
            for key, value in dp.items():
                saved = _ret.get(key, [])
                _ret.update({key: saved + [value]})

        ret = {
            "trg_imgs": torch.stack(_ret["trg_imgs"]),
            "trg_fids": torch.cat(_ret["trg_fids"]),
            "trg_cids": torch.cat(_ret["trg_cids"]),
            "style_imgs": torch.stack(_ret["style_imgs"]),
            "style_fids": torch.stack(_ret["style_fids"]),
            "char_imgs": torch.stack(_ret["char_imgs"]),
            "char_fids": torch.stack(_ret["char_fids"])
        }

        return ret