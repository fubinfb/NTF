"""
not need to modify the config file, only change the path of model and output
CUDA_VISIBLE_DEVICES=4 python inferencev4.py cfgs/CDF/eval.yaml cfgs/data/eval/chn_ttf.yaml --model CDF --weight /home/fubin/FFG/result/cdfv6_resume3/checkpoints/last.pth --result_dir ./result/CDFv6Gen
FFG-benchmarks
Copyright (c) 2021-present NAVER Corp.
MIT license
"""
import json
import argparse
from pathlib import Path
from itertools import chain
from sconf import Config
from PIL import Image
import random

import torch
from torchvision import transforms

from base.dataset import render, read_font, get_filtered_chars, sample
from base.utils import save_tensor_to_image, load_reference
import os
import scipy.io as io
import matplotlib.pyplot as plt

import cv2

from skimage import morphology
import time

import numpy as np

TRANSFORM = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])


def setup_eval_config(args, left_argv={}):
    default_config_path = Path(args.config_paths[0]).parent / "default.yaml"
    # cfg = Config(*args.config_paths)
    # default_config_path = Path(args.config_paths[0]).parent / "default.yaml"
    cfg = Config(*args.config_paths,
                 default=default_config_path)
    print(cfg)
    cfg.argv_update(left_argv)

    if cfg.dset.test.ref_chars is not None:
        ref_chars = json.load(open(cfg.dset.test.ref_chars))
        if args.n_ref is not None:
            ref_chars = sample(ref_chars, args.n_ref)
        cfg.dset.test.ref_chars = ref_chars

    if cfg.dset.test.gen_chars is not None:
        cfg.dset.test.gen_chars = json.load(open(cfg.dset.test.gen_chars))

    args.result_dir = Path(args.result_dir)
    args.model = args.model.lower()

    from NTFmodel.models import NTFGenerator
    infer_func = infer_CDF

    source_path = cfg.dset.test.source_path

    source_ext = cfg.dset.test.source_ext

    infer_args = {
        "source_path": source_path,
        "source_ext": source_ext,
    }

    return args, cfg, NTFGenerator, infer_func, infer_args

def infer_CDF(gen, save_dir, source_path, source_ext, gen_chars, key_ref_dict, load_img, batch_size=32, return_img=False):
    # print("gen.parameters()")
    # print(gen.parameters())
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    genimgs_save_dir = Path(os.path.join(save_dir, "genimgs"))
    genimgs_save_dir.mkdir(parents=True, exist_ok=True)
    gtimgs_save_dir = Path(os.path.join(save_dir, "gtimgs"))
    gtimgs_save_dir.mkdir(parents=True, exist_ok=True)

    if source_ext == "ttf":
        source = read_font(source_path)
        gen_chars = get_filtered_chars(source) if gen_chars is None else gen_chars

        def read_source(char):
            return render(source, char)
    else:
        source = Path(source_path)
        gen_chars = [p.stem for p in source.glob(f"*.{source_ext}")] if gen_chars is None else gen_chars

        def read_source(char):
            impath = source / f"{char}.png"
            return Image.open(str(impath))
    
    key_gen_dict = {k: gen_chars for k in key_ref_dict}

    outs = {}


    for key, gchars in key_gen_dict.items():

        (genimgs_save_dir / key).mkdir(parents=True, exist_ok=True)
        (gtimgs_save_dir / key).mkdir(parents=True, exist_ok=True)

        ref_chars = key_ref_dict[key] # the ref_chars defined in the ref_chars.json
        ref_imgs = torch.stack([TRANSFORM(load_img(key, c)) for c in ref_chars]).cuda() # for font img
        ref_batches = torch.split(ref_imgs, batch_size)


        iter = 0
        for batch in ref_batches:
            style_fact = gen.infer_styleencode(batch)
            if iter == 0:
                style_facts = style_fact
            else:
                style_facts = torch.cat((style_facts, style_fact), dim=0)
            iter = iter + 1

        style_facts = torch.mean(style_facts, dim=0, keepdim=True)


        for char in gchars:
            source_img = TRANSFORM(read_source(char)).unsqueeze(0).cuda()

            source_img = torch.cat((source_img, source_img, source_img), dim=0)

            char_code1 = gen.infer_encoder(source_img)


            out, img0 = gen.decode(style_facts, char_code1)

            ##################################################################################
            out = torch.squeeze(out,dim=0)
            out = out.detach().cpu()


            ######## save generated imgs for comparison
            genpath = genimgs_save_dir / key / f"{char}.png"
            save_tensor_to_image(out, genpath) 

            gt_img = load_img(key, char)
            gtpath = gtimgs_save_dir / key / f"{char}.png"
            gt_img.save(gtpath, format="png")

    return outs

def load_model(args, cfg, gen_model):
    # g_kwargs = cfg.get('gen', {})
    # print(g_kwargs)
    # gen = gen_model(**g_kwargs).cuda()
    gen = gen_model().cuda()
    print(args.weight)
    weight = torch.load(args.weight)
    print(weight["step"])
    if "generator_ema" in weight:
        weight = weight["generator_ema"]
    gen.load_state_dict(weight)
    gen.eval()

    return gen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_paths", nargs="+", help="path to config.yaml")
    parser.add_argument("--model", help="one of (DM, LF, MX, FUNIT)")
    parser.add_argument("--weight", help="path to weight to evaluate.pth")
    parser.add_argument("--result_dir", help="path to save the result file")
    parser.add_argument("--n_ref", type=int, default=None, help="number of reference characters to use")
    parser.add_argument("--seed", type=int, default=1504, help="path to save the result file")
    args, left_argv = parser.parse_known_args()
    args, cfg, gen_model, infer_func, infer_args = setup_eval_config(args, left_argv)
    gen = load_model(args, cfg, gen_model)

    random.seed(args.seed)

    data_dir = cfg.dset.test.data_dir

    extension = cfg.dset.test.extension
    ref_chars = cfg.dset.test.ref_chars
 
    key_ref_dict, load_img = load_reference(data_dir, extension, ref_chars)

    infer_func(gen=gen,
               save_dir=args.result_dir,
               gen_chars=cfg.dset.test.gen_chars,
               key_ref_dict=key_ref_dict,
               load_img=load_img,
               **infer_args)


if __name__ == "__main__":
    main()