import argparse
import logging
import socket

import torch

from slim.model import SLIMModel
from slim.serving.websocket_server import WebsocketPolicyServer


def main(args) -> None:
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    policy = SLIMModel.from_checkpoint(args.checkpoint)
    if args.bf16:
        policy = policy.to(torch.bfloat16)
    policy = policy.to(args.device).eval()

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    # start websocket server
    server = WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        idle_timeout=args.idle_timeout,
        metadata={"model": "SLIM"},
    )
    logging.info("server running ...")
    server.serve_forever()


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--idle-timeout", type=int, default=1800)
    parser.add_argument("--seed", type=int, default=42)
    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    parser = build_argparser()
    args = parser.parse_args()
    main(args)
