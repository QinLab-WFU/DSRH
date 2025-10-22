from argparse import Namespace

from DCHMT_CM.network import DCHMT


def build_model(args: Namespace):
    if args.backbone != "clip":
        raise NotImplementedError(f"not support: {args.backbone}")
    net = DCHMT(args.n_bits)
    return net.to(args.device)


if __name__ == "__main__":
    pass
