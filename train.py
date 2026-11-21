#!/usr/bin/env python3
"""
train.py - Fine-tune a transformer for Named Entity Recognition on CoNLL-2003.

Labels predicted by the fine-tuned model: PER, ORG, LOC, MISC (via the BIO tags
B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC, B-MISC, I-MISC, and O).
Note: CoNLL-2003 has NO DATE label. DATE entities are added at inference time
with spaCy and merged with the model output (see ner_engine.py).

Usage:
    python train.py                 # full training (3 epochs, ~14k train samples)
    python train.py --smoke         # smoke test (500 train, 200 val samples, 1 epoch)
    python train.py --smoke --model-name bert-base-cased

Before running (once, in this shell):
    export PYTORCH_ENABLE_MPS_FALLBACK=1
"""

import argparse
import json
import logging
import os
import sys
from collections import Counter

# Matplotlib must use the non-interactive "Agg" backend (no display needed).
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

# seqeval computes NER-aware precision / recall / F1 / accuracy.
from seqeval.metrics import (
    accuracy_score as seq_accuracy_score,
    f1_score as seq_f1_score,
    precision_score as seq_precision_score,
    recall_score as seq_recall_score,
    classification_report as seq_classification_report,
)

# ---------------------------------------------------------------------------
# CONFIGURATION (edit MODEL_NAME to switch architecture, e.g. "bert-base-cased")
# ---------------------------------------------------------------------------
MODEL_NAME = "distilbert-base-cased"  # default; overridable with --model-name
OUTPUT_DIR = "./model"                # where the fine-tuned model/tokenizer are saved
CHECKPOINT_DIR = "./checkpoints"      # intermediate training checkpoints
LABEL_DIST_PLOT = "label_distribution.png"
METRICS_FILE = "metrics.json"

# Training hyper-parameters (fixed for both smoke and full runs).
LEARNING_RATE = 2e-5
EPOCHS_FULL = 3
EPOCHS_SMOKE = 1
PER_DEVICE_BATCH_SIZE = 16
WEIGHT_DECAY = 0.01

# Smoke-test subset sizes.
SMOKE_TRAIN_SAMPLES = 500
SMOKE_EVAL_SAMPLES = 200
SMOKE_TEST_SAMPLES = 200

SEED = 42

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune a token-classification model on CoNLL-2003."
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke test: 500 train / 200 val samples, 1 epoch (fast, for CI/debug).",
    )
    parser.add_argument(
        "--model-name",
        default=MODEL_NAME,
        help='Hugging Face model ID (default: "%s")' % MODEL_NAME,
    )
    return parser.parse_args()


def load_conll2003():
    """
    Load CoNLL-2003, trying several Hugging Face dataset IDs in order.
    Returns (dataset, dataset_id).
    """
    candidates = ["eriktks/conll2003", "tomaarsen/conll2003", "conll2003"]
    last_error = None
    for ds_id in candidates:
        try:
            logger.info("Attempting to load dataset: %s", ds_id)
            dataset = load_dataset(ds_id)
            logger.info("Successfully loaded dataset: %s", ds_id)
            return dataset, ds_id
        except Exception as exc:  # noqa: BLE001 - we try the next candidate
            logger.warning("Failed to load %s: %s", ds_id, exc)
            last_error = exc
    raise RuntimeError(
        "Could not load any CoNLL-2003 dataset from Hugging Face Hub. "
        f"Last error: {last_error}"
    )


def get_label_list(dataset):
    """Read the NER label names from the dataset features (ner_tags)."""
    feature = dataset["train"].features["ner_tags"]
    # feature is a Sequence of ClassLabel; .feature.names gives the string labels.
    label_names = feature.feature.names
    logger.info("Label list from dataset features: %s", label_names)
    return label_names


def align_labels_with_tokens(labels, word_ids):
    """
    Align NER labels with tokenized sub-words.

    Rules:
      * special tokens (word_id is None)        -> -100 (ignored by the loss)
      * first sub-word of a word                -> the word's label
      * continuation sub-words of the same word -> -100 (ignored by the loss)

    Using -100 tells PyTorch's CrossEntropyLoss to skip those positions.
    """
    aligned_labels = []
    previous_word_idx = None
    for word_idx in word_ids:
        if word_idx is None:
            aligned_labels.append(-100)
        elif word_idx != previous_word_idx:
            aligned_labels.append(labels[word_idx])
        else:
            aligned_labels.append(-100)
        previous_word_idx = word_idx
    return aligned_labels


def tokenize_and_align(examples, tokenizer, label_list):
    """Tokenize a batch of examples and align the NER labels to sub-words."""
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,  # input is already a list of words
    )
    all_labels = examples["ner_tags"]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))
    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs


def compute_metrics(eval_pred, label_list):
    """
    Compute overall precision, recall, F1 and accuracy with seqeval.

    seqeval expects list-of-lists of string labels, so we convert the model's
    token-level predictions (and the -100-masked ground truth) back to strings,
    dropping every position whose label is -100.
    """
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)

    true_predictions = []
    true_labels = []
    for prediction, label in zip(predictions, labels):
        true_prediction = []
        true_label = []
        for p, l in zip(prediction, label):
            if l != -100:
                true_prediction.append(label_list[p])
                true_label.append(label_list[l])
        true_predictions.append(true_prediction)
        true_labels.append(true_label)

    return {
        "precision": seq_precision_score(true_labels, true_predictions),
        "recall": seq_recall_score(true_labels, true_predictions),
        "f1": seq_f1_score(true_labels, true_predictions),
        "accuracy": seq_accuracy_score(true_labels, true_predictions),
    }


def plot_label_distribution(dataset, label_list, output_path):
    """Count token-level labels across all splits and save a bar chart."""
    all_counts = Counter()
    for split in dataset:
        for label_ids in dataset[split]["ner_tags"]:
            for label_id in label_ids:
                all_counts[label_list[label_id]] += 1

    # Sort so "O" is last; entity labels first (alphabetical).
    ordered_labels = sorted(all_counts.keys(), key=lambda l: (l == "O", l))
    values = [all_counts[l] for l in ordered_labels]

    plt.figure(figsize=(10, 6))
    plt.bar(ordered_labels, values, color="steelblue")
    plt.title("CoNLL-2003 Label Distribution (token level, all splits)")
    plt.xlabel("Label")
    plt.ylabel("Token count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logger.info("Saved label distribution plot to %s", output_path)
    logger.info("Label counts: %s", dict(all_counts))


def convert_predictions_to_strings(predictions, labels, label_list):
    """Convert model predictions + labels to lists of string labels, dropping -100."""
    predictions = np.argmax(predictions, axis=2)
    true_predictions = []
    true_labels = []
    for prediction, label in zip(predictions, labels):
        true_prediction = []
        true_label = []
        for p, l in zip(prediction, label):
            if l != -100:
                true_prediction.append(label_list[p])
                true_label.append(label_list[l])
        true_predictions.append(true_prediction)
        true_labels.append(true_label)
    return true_labels, true_predictions


def main():
    args = parse_args()
    smoke_test = args.smoke
    epochs = EPOCHS_SMOKE if smoke_test else EPOCHS_FULL

    logger.info("=" * 60)
    logger.info("NER fine-tuning started")
    logger.info("Model: %s | Smoke test: %s | Epochs: %d", args.model_name, smoke_test, epochs)
    logger.info("MPS available: %s", torch.backends.mps.is_available())
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load dataset
    # ------------------------------------------------------------------
    dataset, ds_id = load_conll2003()
    label_list = get_label_list(dataset)
    num_labels = len(label_list)

    for split in dataset:
        logger.info("Split '%s': %d samples", split, len(dataset[split]))

    # ------------------------------------------------------------------
    # 2. Label distribution chart
    # ------------------------------------------------------------------
    plot_label_distribution(dataset, label_list, LABEL_DIST_PLOT)

    # ------------------------------------------------------------------
    # 3. Optional smoke-test subset
    # ------------------------------------------------------------------
    if smoke_test:
        logger.info(
            "SMOKE TEST: using %d train / %d val / %d test samples",
            SMOKE_TRAIN_SAMPLES,
            SMOKE_EVAL_SAMPLES,
            SMOKE_TEST_SAMPLES,
        )
        train_dataset = dataset["train"].select(
            range(min(SMOKE_TRAIN_SAMPLES, len(dataset["train"])))
        )
        eval_dataset = dataset["validation"].select(
            range(min(SMOKE_EVAL_SAMPLES, len(dataset["validation"])))
        )
        test_dataset = dataset["test"].select(
            range(min(SMOKE_TEST_SAMPLES, len(dataset["test"])))
        )
    else:
        train_dataset = dataset["train"]
        eval_dataset = dataset["validation"]
        test_dataset = dataset["test"]

    # ------------------------------------------------------------------
    # 4. Load tokenizer and model
    # ------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    id2label = {i: l for i, l in enumerate(label_list)}
    label2id = {l: i for i, l in enumerate(label_list)}
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )
    logger.info("Model %s loaded with %d labels", args.model_name, num_labels)

    # ------------------------------------------------------------------
    # 5. Tokenize and align labels
    # ------------------------------------------------------------------
    tokenized_train = train_dataset.map(
        lambda ex: tokenize_and_align(ex, tokenizer, label_list),
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing train",
    )
    tokenized_eval = eval_dataset.map(
        lambda ex: tokenize_and_align(ex, tokenizer, label_list),
        batched=True,
        remove_columns=eval_dataset.column_names,
        desc="Tokenizing validation",
    )
    tokenized_test = test_dataset.map(
        lambda ex: tokenize_and_align(ex, tokenizer, label_list),
        batched=True,
        remove_columns=test_dataset.column_names,
        desc="Tokenizing test",
    )

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    # ------------------------------------------------------------------
    # 6. Training arguments
    #    Handle the transformers version difference:
    #    older versions use `evaluation_strategy`, newer (>= 4.40 / 5.x)
    #    use `eval_strategy`. We inspect the signature and pick the right one.
    # ------------------------------------------------------------------
    import inspect

    ta_params = inspect.signature(TrainingArguments.__init__).parameters
    strategy_key = "evaluation_strategy" if "evaluation_strategy" in ta_params else "eval_strategy"

    trainer_params = inspect.signature(Trainer.__init__).parameters
    trainer_token_kwarg = "processing_class" if "processing_class" in trainer_params else "tokenizer"

    training_args = TrainingArguments(
        output_dir=CHECKPOINT_DIR,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_BATCH_SIZE,
        num_train_epochs=epochs,
        weight_decay=WEIGHT_DECAY,
        **{strategy_key: "epoch"},
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to="none",          # no telemetry / no external logging services
        logging_steps=10,
        save_total_limit=2,
        seed=SEED,
        # NOTE: deliberately NO fp16 / bf16 (MPS does not support them well).
    )
    logger.info("Training arguments configured (strategy key: %s)", strategy_key)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=data_collator,
        compute_metrics=lambda ep: compute_metrics(ep, label_list),
        **{trainer_token_kwarg: tokenizer},
    )

    # ------------------------------------------------------------------
    # 7. Train
    # ------------------------------------------------------------------
    logger.info("Starting training for %d epoch(s)...", epochs)
    train_result = trainer.train()
    logger.info("Training finished. Train metrics: %s", train_result.metrics)

    # ------------------------------------------------------------------
    # 8. Evaluate on the TEST split and produce the classification report
    # ------------------------------------------------------------------
    logger.info("Evaluating on the TEST split...")
    test_eval_metrics = trainer.evaluate(eval_dataset=tokenized_test)
    logger.info("Test eval metrics: %s", test_eval_metrics)

    predictions, labels, _ = trainer.predict(tokenized_test)
    true_labels, true_predictions = convert_predictions_to_strings(
        predictions, labels, label_list
    )

    report = seq_classification_report(
        true_labels, true_predictions, output_dict=True, zero_division=0
    )

    print("\n" + "=" * 60)
    print("CoNLL-2003 TEST classification report (seqeval)")
    print("=" * 60)
    print(seq_classification_report(true_labels, true_predictions, zero_division=0))

    # seqeval's output_dict uses entity keys plus "micro/macro/weighted avg".
    # Overall metrics are taken from "micro avg"; accuracy is computed separately.
    micro = report.get("micro avg", {})
    overall_metrics = {
        "precision": float(micro.get("precision", 0.0)),
        "recall": float(micro.get("recall", 0.0)),
        "f1": float(micro.get("f1-score", 0.0)),
        "accuracy": float(seq_accuracy_score(true_labels, true_predictions)),
    }

    per_entity = {}
    for entity in ("PER", "ORG", "LOC", "MISC"):
        if entity in report:
            per_entity[entity] = {
                "precision": float(report[entity].get("precision", 0.0)),
                "recall": float(report[entity].get("recall", 0.0)),
                "f1": float(report[entity].get("f1-score", 0.0)),
                "support": int(report[entity].get("support", 0)),
            }

    all_metrics = {
        "model_name": args.model_name,
        "dataset": ds_id,
        "smoke_test": smoke_test,
        "epochs": epochs,
        "overall": overall_metrics,
        "per_entity": per_entity,
        "test_samples": len(tokenized_test),
    }

    with open(METRICS_FILE, "w") as f:
        json.dump(all_metrics, f, indent=2)
    logger.info("Saved metrics to %s", METRICS_FILE)

    # ------------------------------------------------------------------
    # 9. Save model and tokenizer
    # ------------------------------------------------------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    logger.info("Saved model and tokenizer to %s", OUTPUT_DIR)
    logger.info("Done.")


if __name__ == "__main__":
    main()
