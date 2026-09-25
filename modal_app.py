"""
Runs the experiments on Modal (https://modal.com), always from the repository root:

    modal run modal_app.py::smoke                   # one epoch of each kind of run, end to end
    modal run --detach modal_app.py::pretrain       # the contrastive pre-trainings alone
    modal run --detach modal_app.py::campaign       # baselines, and fine-tunings of the encoders
    modal run modal_app.py::summary                 # mean ± std over seeds, per experiment

A run is `python -m src.torchtmpl.main <config> train novis` in a GPU container, on a copy
of one of the configs where only the seed, the paths and a few keys change. What must
outlive the containers sits on two Volumes (persistent disks):

    sondra-data    the ALOS-2 scene and its labels, downloaded once from IETR
    sondra-runs    one directory per run: config.yaml, checkpoint, test metrics, result.json

Spending: at most MAX_GPUS runs at once, each stopped after TIMEOUT_HOURS, and a failed run
is never retried. The hard cap is the Workspace budget, on https://modal.com/settings/usage.
"""

import json
import os
import shutil
import statistics
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import modal

image = (
    modal.Image.debian_slim(python_version="3.13")
    .uv_sync()
    .add_local_dir("src", "/root/sondra/src")
    .add_local_dir("configs", "/root/sondra/configs")
)
app = modal.App("sondra", image=image)

with image.imports():  
    import yaml

data_volume = modal.Volume.from_name("sondra-data", create_if_missing=True)
runs_volume = modal.Volume.from_name("sondra-runs", create_if_missing=True)

CODE_DIR = Path("/root/sondra")
DATA_DIR = Path("/data/SAN_FRANCISCO_ALOS2")  
LOCAL_DATA_DIR = Path("/tmp/SAN_FRANCISCO_ALOS2") 
RUNS_DIR = Path("/runs")  

ARCHIVE_URL = (
    "https://ietr-lab.univ-rennes1.fr/polsarpro-bio/san-francisco/dataset/SAN_FRANCISCO_ALOS2.zip"
)
LABELS_URL = "https://raw.githubusercontent.com/liuxuvip/PolSF/master/SF-ALOS2/SF-ALOS2-label2d.png"

BASELINE_CONFIG = "baseline_segformer"
PRETRAINING_CONFIG = "contrastive_segformer"
WANDB_PROJECT = "sondra-revamp"
METRICS = ("test_overall_accuracy", "test_mean_iou", "test_macro_f1", "test_kappa_score")

# benchmarked on local rtx4060 an epoch takes ~1hour same on an L4
MAX_GPUS = 6
TIMEOUT_HOURS = 4

# dowloading from the IETR might be a bit long 
@app.function(volumes={"/data": data_volume}, memory=2 * 1024, timeout=3 * 60 * 60)
def fetch_data():
    labels = DATA_DIR / "SF-ALOS2-label2d.png"
    if labels.exists():
        return

    archive = Path("/tmp/SAN_FRANCISCO_ALOS2.zip")
    print(f"Downloading {ARCHIVE_URL}")
    with urllib.request.urlopen(ARCHIVE_URL) as source, archive.open("wb") as target:
        shutil.copyfileobj(source, target, 16 * 1024 * 1024)
    with zipfile.ZipFile(archive) as zipped:
        zipped.extractall(DATA_DIR.parent)  # the archive holds SAN_FRANCISCO_ALOS2/
    with urllib.request.urlopen(LABELS_URL) as source:
        labels.write_bytes(source.read())
    data_volume.commit()


@app.function(
    gpu="L4",
    cpu=6.0,  # the 4 DataLoader workers of the configs, and the training process
    memory=16 * 1024,
    volumes={"/data": data_volume, "/runs": runs_volume},
    secrets=[modal.Secret.from_name("wandb")],
    timeout=TIMEOUT_HOURS * 60 * 60,
    max_containers=MAX_GPUS,
)
def train(config_name: str, tag: str, seed: int, overrides: dict, commit: str) -> dict:
    """
    One run of src.torchtmpl.main on configs/<config_name>.yaml, logged into
    /runs/<tag>/seed<seed> and into the W&B group <tag>.
    """
    runs_volume.reload()  # checkpoints committed by other containers since this one started
    if not LOCAL_DATA_DIR.exists():  # the ALOS reader seeks line by line: faster on local disk
        shutil.copytree(DATA_DIR, LOCAL_DATA_DIR)

    logdir = RUNS_DIR / tag / f"seed{seed}"
    config_path = Path("/tmp/config.yaml")  # main() copies it into the run directory
    config_path.write_text(yaml.safe_dump(make_config(config_name, seed, logdir, overrides)))

    start = time.monotonic()
    subprocess.run(
        [sys.executable, "-m", "src.torchtmpl.main", str(config_path), "train", "novis"],
        cwd=CODE_DIR,
        env=os.environ | {"WANDB_RUN_GROUP": tag},
        check=True,
    )

    run_dir = latest_run(logdir)
    result = {
        "tag": tag,
        "seed": seed,
        "commit": commit,
        "minutes": round((time.monotonic() - start) / 60, 1),
        "checkpoint": str(run_dir / "best_model.pt"),
    }
    test_metrics = run_dir / "test_metrics.json"  # supervised runs only
    if test_metrics.exists():
        result |= json.loads(test_metrics.read_text())

    (logdir / "result.json").write_text(json.dumps(result, indent=2))
    runs_volume.commit()
    return result


@app.function(volumes={"/runs": runs_volume}, timeout=24 * 60 * 60)
def run_campaign(losses: list[str], seeds: list[int], commit: str) -> list[dict]:
    """
    Launches the baselines and the missing pre-trainings, then each fine-tuning once its
    pre-training is done. A fine-tuning starts from the encoder pre-trained with the same
    seed, so the spread over seeds covers both stages.
    """
    runs_volume.reload()
    baselines = [launch_baseline(seed, commit) for seed in seeds]
    pending = {
        (loss, seed): launch_pretraining(loss, seed, commit)
        for loss in losses
        for seed in seeds
        if pretrained_checkpoint(loss, seed) is None
    }

    finetunings = []
    for loss in losses:
        for seed in seeds:
            if (loss, seed) in pending:
                checkpoint = pending[loss, seed].get()["checkpoint"]
            else:
                checkpoint = pretrained_checkpoint(loss, seed)
            finetunings.append(launch_finetuning(loss, seed, checkpoint, commit))

    return [run.get() for run in baselines + finetunings]


@app.function(volumes={"/runs": runs_volume})
def summarize() -> str:
    """Mean ± standard deviation over seeds of the test metrics, one line per experiment."""
    runs_volume.reload()
    by_tag = {}
    for path in RUNS_DIR.glob("*/seed*/result.json"):  # smoke runs sit one level deeper
        result = json.loads(path.read_text())
        if METRICS[0] in result:  # pre-trainings have no test metrics
            by_tag.setdefault(result["tag"], []).append(result)

    lines = ["| experiment | seeds | OA | mIoU | macro F1 | kappa |", "|---|---|---|---|---|---|"]
    for tag, results in sorted(by_tag.items()):
        cells = []
        for metric in METRICS:
            values = [result[metric] for result in results]
            spread = statistics.stdev(values) if len(values) > 1 else 0.0
            cells.append(f"{statistics.mean(values):.2f} ± {spread:.2f}")
        lines.append(f"| {tag} | {len(results)} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ------------------------------- RUN DIRECTORIES -------------------------------


def make_config(name: str, seed: int, logdir: Path, overrides: dict) -> dict:
    """
    configs/<name>.yaml with the seed, the container paths and `overrides` applied.
    Override keys are dotted paths into the config, e.g. {"loss.name": "NTXentKLUnif"}.
    """
    config = yaml.safe_load((CODE_DIR / "configs" / f"{name}.yaml").read_text())
    config["seed"] = seed
    config["data"]["root_dir"] = str(LOCAL_DATA_DIR)
    config["logging"]["logdir"] = str(logdir)
    config.setdefault("wandb", {})["project"] = WANDB_PROJECT

    for dotted_key, value in overrides.items():
        *parents, key = dotted_key.split(".")
        section = config
        for parent in parents:
            section = section[parent]
        section[key] = value
    return config


def latest_run(logdir: Path) -> Path:
    """main() logs into <logdir>/<model class>_<n>, n growing by one with each run."""
    return max(logdir.glob("*_*"), key=lambda path: int(path.name.rsplit("_", 1)[1]))


def pretrained_checkpoint(loss: str, seed: int) -> str | None:
    """Checkpoint of a pre-training already completed on sondra-runs, None otherwise."""
    result = RUNS_DIR / f"pretrain-{loss}" / f"seed{seed}" / "result.json"
    return json.loads(result.read_text())["checkpoint"] if result.exists() else None


# ------------------------------- THE THREE KINDS OF RUNS -------------------------------
# each returns at once a handle on the remote run, whose .get() waits for its result


def launch_baseline(seed: int, commit: str, prefix: str = "", **overrides):
    return train.spawn(BASELINE_CONFIG, f"{prefix}baseline", seed, overrides, commit)


def launch_pretraining(loss: str, seed: int, commit: str, prefix: str = "", **overrides):
    overrides = {"loss.name": loss, **overrides}
    return train.spawn(PRETRAINING_CONFIG, f"{prefix}pretrain-{loss}", seed, overrides, commit)


def launch_finetuning(
    loss: str, seed: int, checkpoint: str, commit: str, prefix: str = "", **overrides
):
    overrides = {"model.pretrained_weights": checkpoint, **overrides}
    return train.spawn(BASELINE_CONFIG, f"{prefix}finetune-{loss}", seed, overrides, commit)


# ------------------------------- COMMANDS, RUN LOCALLY -------------------------------


def git_commit() -> str:
    """HEAD, flagged when tracked files differ from it: the containers get the working tree."""
    head = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"])
    if dirty:
        print(f"WARNING: uncommitted changes on top of {head}, results won't map to a commit")
        return f"{head}-dirty"
    return head


@app.local_entrypoint()
def smoke():
    """One epoch of each kind of run, end to end, logged under /runs/smoke/."""
    commit = git_commit()
    fetch_data.remote()  # returns at once when the data is already there

    loss = "NTXentLearnableTemp"
    baseline = launch_baseline(0, commit, prefix="smoke/", nepochs=1)
    pretraining = launch_pretraining(loss, 0, commit, prefix="smoke/", nepochs=1).get()
    finetuning = launch_finetuning(
        loss, 0, pretraining["checkpoint"], commit, prefix="smoke/", nepochs=1
    ).get()

    for result in (baseline.get(), pretraining, finetuning):
        print(json.dumps(result))


@app.local_entrypoint()
def pretrain(losses: str = "NTXentLearnableTemp,NTXentKLUnif", seeds: str = "0,1,2"):
    """The pre-trainings alone, in parallel: campaign reuses them once they are done."""
    commit = git_commit()
    fetch_data.remote()

    runs = [
        launch_pretraining(loss, int(seed), commit)
        for loss in losses.split(",")
        for seed in seeds.split(",")
    ]
    for run in runs:
        print(json.dumps(run.get()))


@app.local_entrypoint()
def campaign(losses: str = "NTXentLearnableTemp,NTXentKLUnif", seeds: str = "0,1,2"):
    """Baselines and fine-tunings for every loss and seed, with the pre-trainings they need."""
    commit = git_commit()
    fetch_data.remote()

    results = run_campaign.remote(losses.split(","), [int(s) for s in seeds.split(",")], commit)
    for result in results:
        print(json.dumps(result))
    print(summarize.remote())


@app.local_entrypoint()
def summary():
    print(summarize.remote())
