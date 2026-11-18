"""Packs a job's accepted model into a zip that a person can download and try on their own machine.

The zip holds the trained weights exactly as the verifiers accepted them (the checkpoint's own file, checked against
its manifest hash), the job's workload, the manifest, and for the news-topic template a script that loads it.
"""

from __future__ import annotations

import io
import json
import zipfile

LABELS = ["World", "Sports", "Business", "Sci/Tech"]

TRY_SCRIPT = '''"""Try the model Ekido trained for you.

    pip install torch transformers peft safetensors
    python try_model.py "Stocks rally as the central bank cuts rates" "Striker scores twice in the derby"

The first run downloads the public base model (distilbert-base-uncased, about 270 MB) from Hugging Face.
Your trained part is the small file next to this script: trainable.safetensors.
"""

import json
import sys
import warnings
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model
from safetensors.torch import load_file
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.utils import logging as hf_logging

# Loading the public base model reports its untrained head as "missing" before your weights replace it. That's expected.
hf_logging.set_verbosity_error()
hf_logging.disable_progress_bar()
warnings.filterwarnings("ignore")

LABELS = ["World", "Sports", "Business", "Sci/Tech"]
HERE = Path(__file__).resolve().parent
spec = json.loads((HERE / "workload.json").read_text(encoding="utf-8"))
base_id, revision, lora = spec["model"]["id"], spec["model"]["revision"], spec["lora"]

tokenizer = AutoTokenizer.from_pretrained(base_id, revision=revision)
base = AutoModelForSequenceClassification.from_pretrained(base_id, revision=revision, num_labels=4)
model = get_peft_model(
    base,
    LoraConfig(task_type=TaskType.SEQ_CLS, r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"], target_modules=lora["target_modules"]),
)
trained = load_file(str(HERE / "trainable.safetensors"))
params = {name: p for name, p in model.named_parameters() if p.requires_grad}
if set(params) != set(trained):
    sys.exit(f"These weights don't match the model: {len(set(params) ^ set(trained))} names differ.")
with torch.no_grad():
    for name, p in params.items():
        p.copy_(trained[name].to(p.dtype))
model.eval()

texts = sys.argv[1:] or [
    "India beat Australia in the final over",
    "Stocks rally as the central bank cuts interest rates",
    "NASA launches a new telescope to study distant galaxies",
    "UN Security Council meets on the Gaza ceasefire",
]
with torch.no_grad():
    for text in texts:
        encoded = tokenizer(text, truncation=True, max_length=spec.get("max_seq_length", 128), return_tensors="pt")
        probs = model(**encoded).logits.softmax(-1)[0]
        best = int(probs.argmax())
        print(f"{LABELS[best]:<9} {float(probs[best]):>4.0%}  {text}")
'''


def readme(job_id: str, step: int, manifest_hash: str, has_script: bool) -> str:
    lines = [
        "Your model, trained on Ekido",
        "",
        f"Job:            {job_id}",
        f"Final step:     {step}",
        f"Checkpoint:     {manifest_hash}",
        "",
        "trainable.safetensors  the trained weights, exactly as the verifiers accepted them",
        "manifest.json          the checkpoint manifest: file hashes, lineage and signer",
        "workload.json          what was trained: base model, data, settings",
    ]
    if has_script:
        lines += ["try_model.py           load it and classify news headlines (World, Sports, Business, Sci/Tech)", "", "    pip install torch transformers peft safetensors", '    python try_model.py "Your headline here"']
    else:
        lines += ["", "This job ran the synthetic test workload, so these weights are a test model, not a useful one."]
    return "\n".join(lines) + "\n"


def bundle(job_id: str, spec: dict, manifest: dict, manifest_hash: str, step: int, trainable: bytes) -> bytes:
    has_script = spec.get("template") == "distilbert-lora-agnews/v1"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("trainable.safetensors", trainable)
        archive.writestr("manifest.json", json.dumps(manifest, indent=2, sort_keys=True))
        archive.writestr("workload.json", json.dumps(spec, indent=2, sort_keys=True))
        archive.writestr("README.txt", readme(job_id, step, manifest_hash, has_script))
        if has_script:
            archive.writestr("try_model.py", TRY_SCRIPT)
    return buffer.getvalue()
