"""
Usage:
python offline_batch_inference_vlm.py --model-path Qwen/Qwen2-VL-7B-Instruct --chat-template=qwen2-vl
"""

import argparse
import dataclasses
import time

import torch
from PIL import Image

import sglang as sgl
from sglang.srt.conversation import chat_templates
from sglang.srt.server_args import ServerArgs


def main(
    server_args: ServerArgs,
):
    vlm = sgl.Engine(**dataclasses.asdict(server_args))

    conv = chat_templates[server_args.chat_template].copy()
    image_token = conv.image_token

    # image_url = "/root/test_img/example_image.png"

    uji_url = "/root/test_img/uji.jpg"

    hoshi_url_large = "/root/test_img/keihann.jpg"

    # img = Image.open(image_url)
    uji = Image.open(uji_url)
    hoshi = Image.open(hoshi_url_large)

    prompt = f"What's in this image?\n{image_token}"
    prompt_2 = f"What's in this image is a people in this image?\n{image_token}"
    # pdb.set_trace()
    sampling_params = {
        "temperature": 0.001,
        "max_new_tokens": 30,
    }

    output = vlm.generate(
        prompt=prompt,
        image_data=[hoshi],
        sampling_params=sampling_params,
    )

    # output_1 = vlm.generate(
    #     prompt=prompt_2,
    #     image_data=[uji],
    #     sampling_params=sampling_params,
    # )

    print("===============================")
    print(f"Prompt: {prompt}")
    print(f"Generated text: {output['text']}")
    print("warmup end")

    torch.cuda.synchronize()

    print("======profiling start=====")
    start_time = time.time()
    run_num = 1
    for run_idx in range(run_num):
        out_contents = vlm.generate(
            prompt=prompt,
            image_data=[uji],
            sampling_params=sampling_params,
        )

    torch.cuda.synchronize()
    end_time = time.time()
    during_time = (end_time - start_time) * (1000 / run_num)
    print("prefill time for this image is {} ms \n".format(during_time))

    vlm.shutdown()


# The __main__ condition is necessary here because we use "spawn" to create subprocesses
# Spawn starts a fresh program every time, if there is no __main__, it will run into infinite loop to keep spawning processes from sgl.Engine
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    args = parser.parse_args()
    # pdb.set_trace()
    server_args = ServerArgs.from_cli_args(args)
    main(server_args)
