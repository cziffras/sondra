"""
uv run python -m scripts.sweep pretrain --seeds 0
uv run python -m scripts.sweep finetune --seeds 0
uv run python -m scripts.sweep finetune --encoders <tags> --seeds 0,1,2
uv run python -m scripts.sweep frugal --encoders <tags> --seeds 0,1,2
uv run python -m scripts.sweep summary
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from tabulate import tabulate

import yaml

BASELINE_CONFIG = "baseline_segformer"
PRETRAINING_CONFIG = "contrastive_segformer"
WANDB_PROJECT = "sondra-revised"
LOGDIR = Path("logs/sweep")
METRICS = ("test_overall_accuracy", "test_mean_iou", "test_macro_f1")
FROZEN = {"model.freeze_encoder": True}


@dataclass
class Run:
    config: str
    tag: str
    seed: int
    overrides: dict = field(default_factory=dict)
    nepochs: int | None = None


def load_config(name):
    return yaml.safe_load(Path("configs", f"{name}.yaml").read_text())


def execute(run):
    logdir = LOGDIR / run.tag / f"seed{run.seed}"
    result_path = logdir / "result.json"
    if result_path.exists():
        print(f"/!\\ {run.tag} seed {run.seed} already done")
        return

    config = load_config(run.config)
    config["seed"] = run.seed
    config["logging"]["logdir"] = str(logdir)
    config["wandb"] = config.get("wandb", {}) | {
        "project": WANDB_PROJECT,
        "name": f"{run.tag}-seed{run.seed}",
    }
    if run.nepochs:
        config["nepochs"] = run.nepochs
    for key, value in run.overrides.items():
        *parents, last = key.split(".")
        section = config
        for parent in parents:
            section = section[parent]
        section[last] = value

    logdir.mkdir(parents=True, exist_ok=True)
    config_path = logdir / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    print(f"{run.tag} seed {run.seed}: training", flush=True)
    start = time.monotonic()
    command = [sys.executable, "-m", "src.torchtmpl.main", str(config_path), "train", "novis"]
    exit_code = subprocess.run(command, env=os.environ | {"WANDB_RUN_GROUP": run.tag}).returncode
    if exit_code:
        print(f"{run.tag} seed {run.seed}: failed with exit code {exit_code}", flush=True)
        return

    # main() logs each attempt into a new <model>_<n> directory
    run_dir = max(logdir.glob("*_*"), key=lambda path: int(path.name.rsplit("_", 1)[1]))
    result = {
        "tag": run.tag,
        "seed": run.seed,
        "commit": git_commit(),
        "minutes": round((time.monotonic() - start) / 60, 1),
        "checkpoint": str(run_dir / "best_model.pt"),
    }
    metrics = run_dir / "test_metrics.json"
    if metrics.exists():
        result |= json.loads(metrics.read_text())
    result_path.write_text(json.dumps(result, indent=2))


def supervised(tag, seed, overrides, args, nepochs=None, scratch=False):
    # a baseline trained from scratch does not start from the same point
    # and might need a greater learning rate
    lr = args.lr_scratch if scratch else args.lr
    if lr is not None:
        overrides = overrides | {"optimizer.params.lr": lr}
    return Run(BASELINE_CONFIG, tag, seed, overrides, nepochs)


def pretrained_encoders():
    return sorted({path.parts[-3] for path in LOGDIR.glob("pretrain-*/seed*/result.json")})


def encoder_checkpoint(encoder, seed):
    """The pre-training of that seed, or else of the lowest seed done."""
    done = sorted(
        (LOGDIR / encoder).glob("seed*/result.json"), key=lambda path: int(path.parent.name[4:])
    )
    if not done:
        raise SystemExit(f"No completed pre-training in {LOGDIR / encoder}")
    same_seed = [path for path in done if path.parent.name == f"seed{seed}"]
    return json.loads((same_seed or done)[0].read_text())["checkpoint"]


def pretrain(args):
    return [
        Run(
            PRETRAINING_CONFIG,
            f"pretrain-{loss}-lambda{lambda_reg:g}",
            seed,
            {"loss.name": loss, "model.lambda_reg": lambda_reg},
            args.nepochs,
        )
        for loss, lambda_reg, seed in product(args.losses, args.lambdas, args.seeds)
    ]


def finetune(args):
    runs = []
    for seed in args.seeds:
        runs.append(supervised("baseline", seed, {}, args, args.nepochs, scratch=True))
    for encoder, seed in product(args.encoders or pretrained_encoders(), args.seeds):
        weights = {"model.pretrained_weights": encoder_checkpoint(encoder, seed)}
        tag = encoder.replace("pretrain-", "finetune-", 1)
        runs.append(supervised(tag, seed, weights, args, args.nepochs))
    return runs


def frugal(args):
    full_epochs = args.nepochs or load_config(BASELINE_CONFIG)["nepochs"]
    runs = []
    for fraction, seed in product(args.fractions, args.seeds):
        prefix = f"frac{fraction:g}-"
        data = {"data.train_fraction": fraction, "valid_every": round(1 / fraction)}
        nepochs = round(full_epochs / fraction)
        if "scratch" in args.modes:
            runs.append(supervised(prefix + "baseline", seed, data, args, nepochs, scratch=True))
        if "random" in args.modes:
            runs.append(supervised(prefix + "frozen-random", seed, data | FROZEN, args, nepochs))
        for encoder in args.encoders:
            weights = data | {"model.pretrained_weights": encoder_checkpoint(encoder, seed)}
            if "frozen" in args.modes:
                tag = prefix + encoder.replace("pretrain-", "frozen-", 1)
                runs.append(supervised(tag, seed, weights | FROZEN, args, nepochs))
            if "finetune" in args.modes:
                tag = prefix + encoder.replace("pretrain-", "finetune-", 1)
                runs.append(supervised(tag, seed, weights, args, nepochs))
    return runs


def summary():
    by_tag = defaultdict(list)
    for path in LOGDIR.glob("*/seed*/result.json"):
        result = json.loads(path.read_text())
        if "test_mean_iou" in result:
            by_tag[result["tag"]].append(result)

    headers = ["experiment", "seeds", "OA", "mIoU", "macro F1"]
    data_to_print = []
    for tag, results in sorted(by_tag.items()):
        cells = [mean_std([result[metric] for result in results]) for metric in METRICS]
        data_to_print.append([tag, len(results), *cells])
    print(tabulate(data_to_print, headers=headers))


def mean_std(values):
    spread = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{statistics.mean(values):.2f} ± {spread:.2f}"


def git_commit():
    head = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"])
    return f"{head}-dirty" if dirty else head


def comma_list(cast):
    return lambda text: [cast(value) for value in text.split(",") if value]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("pretrain", "finetune", "frugal", "summary"))
    parser.add_argument(
        "--losses", type=comma_list(str), default="NTXentKLUnif,NTXentLearnableTemp"
    )
    parser.add_argument("--lambdas", type=comma_list(float), default="0,0.03,0.3,3")
    parser.add_argument("--seeds", type=comma_list(int), default="0")
    parser.add_argument("--nepochs", type=int, help="default: the config's")
    parser.add_argument("--encoders", type=comma_list(str), default="", help="pre-training tags")
    parser.add_argument("--fractions", type=comma_list(float), default="0.1,0.25,0.5")
    parser.add_argument("--modes", type=comma_list(str), default="scratch,random,frozen,finetune")
    parser.add_argument("--lr", type=float, help="runs from a pre-trained or frozen encoder")
    parser.add_argument("--lr-scratch", type=float, help="baselines")
    args = parser.parse_args()

    if args.command == "summary":
        summary()
        return
    commands = {"pretrain": pretrain, "finetune": finetune, "frugal": frugal}
    for run in commands[args.command](args):
        execute(run)


if __name__ == "__main__":
    main()
