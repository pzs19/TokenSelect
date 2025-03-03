"""Launch the inference server."""

import argparse

from omegaconf import OmegaConf
from sglang.srt.server_args import ServerArgs


def patch_model(config):
    import sys
    sys.path.append(".")
    from patcher.token_retrieval import patch

    patch(
        rope_base=config.model.rope_base,
        rope_scale=config.model.rope_scale,
        rope_model="ROPE_LLAMA",
        max_n_tokens=config.model.max_n_tokens,
        n_init=config.model.n_init,
        n_local=config.model.n_local,
        top_k=config.model.top_k,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sgl-conf-file", type=str, default="")
    ServerArgs.add_cli_args(parser)
    args = parser.parse_args()
    server_args = ServerArgs.from_cli_args(args)

    if args.sgl_conf_file:
        sgl_conf = OmegaConf.load(open(args.sgl_conf_file))
        patch_model(sgl_conf)

    from sglang.srt.server import launch_server

    launch_server(server_args)
