# -*- encoding: utf-8 -*-
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import argparse
from sat.model.mixins import CachedAutoregressiveMixin

from utils.chat import chat
from models.cogvlm_model import CogVLMModel
from utils.language import llama2_tokenizer, llama2_text_processor_inference
from utils.vision import get_image_processor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_length",
                        type=int,
                        default=2048,
                        help='max length of the total sequence')
    parser.add_argument("--top_p",
                        type=float,
                        default=0.4,
                        help='top p for nucleus sampling')
    parser.add_argument("--top_k",
                        type=int,
                        default=1,
                        help='top k for top k sampling')
    parser.add_argument("--temperature",
                        type=float,
                        default=.8,
                        help='temperature for sampling')
    parser.add_argument("--english",
                        action='store_true',
                        help='only output English')
    parser.add_argument("--version",
                        type=str,
                        default="chat",
                        help='version to interact with')
    parser.add_argument("--from_pretrained",
                        type=str,
                        default="cogvlm-chat",
                        help='pretrained ckpt')
    parser.add_argument("--local_tokenizer",
                        type=str,
                        default="lmsys/vicuna-7b-v1.5",
                        help='tokenizer path')
    parser.add_argument("--no_prompt",
                        action='store_true',
                        help='Sometimes there is no prompt in stage 1')
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--n-workers",
                        type=int,
                        default=1,
                        help="how many workers in total to complete this task")
    parser.add_argument("--worker-idx",
                        type=int,
                        default=0,
                        help="worker index")
    args = parser.parse_args()
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    parser = CogVLMModel.add_model_specific_args(parser)
    args = parser.parse_args()

    # load model
    model, model_args = CogVLMModel.from_pretrained(
        args.from_pretrained,
        args=argparse.Namespace(deepspeed=None,
                                local_rank=rank,
                                rank=rank,
                                world_size=world_size,
                                model_parallel_size=world_size,
                                mode='inference',
                                skip_init=True,
                                use_gpu_initialization=True
                                if torch.cuda.is_available() else False,
                                device='cuda',
                                **vars(args)),
        overwrite_args={'model_parallel_size': world_size}
        if world_size != 1 else {})
    model = model.eval()
    from sat.mpu import get_model_parallel_world_size
    assert world_size == get_model_parallel_world_size(
    ), "world size must equal to model parallel size for cli_demo!"

    tokenizer = llama2_tokenizer(args.local_tokenizer,
                                 signal_type=args.version)
    image_processor = get_image_processor(model_args.eva_args["image_size"][0])

    model.add_mixin('auto-regressive', CachedAutoregressiveMixin())

    text_processor_infer = llama2_text_processor_inference(
        tokenizer, args.max_length, model.image_length)

    # configs -- please update the data folder and data list
    data_dir = "/workspace/data/coco_stuff164k/"
    data_list = "/workspace/repos/lseg/datasets/coco/train2017.txt"

    # get image filepaths
    image_paths = []
    with open(data_list, "r", encoding="utf8") as list_file:
        for line in list_file:
            line = line.strip("\n")
            try:
                image_path, _ = line.split(" ")
            except:  # pylint: disable=bare-except
                image_path = line

            image_path = os.path.join(data_dir, image_path)
            image_paths.append(image_path)

    # get the chuncked index
    import numpy as np

    def get_assigned_workload(n_chuncks, chunck_idx, all_index):
        if n_chuncks > 1 and chunck_idx >= 0:
            chunked_array = np.array_split(all_index, n_chuncks)
            assigned_index_list = list(chunked_array[chunck_idx])
            return assigned_index_list
        else:
            return all_index

    n_samples = len(image_paths)
    all_index = np.arange(n_samples)
    index_list = get_assigned_workload(args.n_workers, args.worker_idx,
                                       all_index)

    # main loop
    import re
    import json

    def get_img_id(image_path):
        img_extension_matches = re.findall(r"[^.]*$", image_path)
        if img_extension_matches:
            img_extention = img_extension_matches[0]
        else:
            return None, None

        img_id_pattern = r"/(\d+)\." + img_extention
        img_id_match = re.search(img_id_pattern, image_path)  # find image_id
        if img_id_match:
            img_id = img_id_match.group(1)
            return img_id, img_extention
        else:
            return None, None

    query = "Describe this image in detail. In your description, specifically mention ALL VISIBLE parts of each object in the image."
    inference_dict = {}
    from tqdm import tqdm
    for data_index in tqdm(index_list):
        image_path = image_paths[data_index]
        img_id, _ = get_img_id(image_path)
        img_id = int(img_id)

        history = None
        cache_image = None
        response, history, cache_image = chat(
            image_path,
            model,
            text_processor_infer,
            image_processor,
            query,
            history=history,
            image=cache_image,
            max_length=args.max_length,
            top_p=args.top_p,
            temperature=args.temperature,
            top_k=args.top_k,
            invalid_slices=text_processor_infer.invalid_slices,
            no_prompt=args.no_prompt)

        if img_id in inference_dict:
            inference_dict[img_id].append(response)
        else:
            inference_dict[img_id] = [response]

    # write to disk - please update the output filepath
    out_json = {"annotations": []}
    for img_id in inference_dict:
        for caption in inference_dict[img_id]:
            out_json["annotations"].append({
                "image_id": int(img_id),
                "caption": caption
            })
    output_dict_path = "/workspace/data/coco_stuff164k/cogvlm_captions_train2017_" + str(
        args.worker_idx) + ".json"
    with open(output_dict_path, "w") as f:
        json.dump(out_json, f)


if __name__ == "__main__":
    main()
