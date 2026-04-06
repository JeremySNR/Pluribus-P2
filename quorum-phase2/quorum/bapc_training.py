"""Behavioural Distillation Training for the BAPC Codec.

Instead of MSE reconstruction (which preserves the cone, not the semantics),
train the codec so the receiver behaves the same way when given latent prefix
embeddings as when given the sender's text message.

Loss components:
  L_distill:     KL(text-teacher logits || latent-student logits)
  L_separation:  -JS(matched latent || shuffled latent) — force receiver to use it
  L_align:       KL + cosine on logits to prevent degenerate behaviour
  L_orthogonality: ||W W^T - I||_F^2 on encoder weights to prevent cone collapse
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataclasses import dataclass
from typing import Optional

from .bapc_codec import BAPCCodec


@dataclass
class BAPCTrainingConfig:
    lr: float = 3e-4
    num_steps: int = 2000
    batch_size: int = 16
    teacher_window: int = 8
    temperature: float = 2.0
    lambda_sep: float = 0.1
    lambda_align: float = 0.1
    lambda_orth: float = 0.01
    log_every: int = 100


def js_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Jensen-Shannon divergence between two distributions (log-space safe)."""
    m = 0.5 * (p + q)
    return 0.5 * (F.kl_div(m.log(), p, reduction="batchmean") +
                  F.kl_div(m.log(), q, reduction="batchmean"))


def orthogonality_loss(weights: list[torch.Tensor]) -> torch.Tensor:
    """||W W^T - I||_F^2 summed over encoder weight matrices."""
    loss = torch.tensor(0.0)
    for W in weights:
        WWT = W @ W.T
        I = torch.eye(WWT.shape[0], device=W.device)
        loss = loss + (WWT - I).pow(2).mean()
    return loss


def collect_training_data(
    sender_model,
    sender_tokenizer,
    receiver_model,
    receiver_tokenizer,
    prompts: list[str],
    extraction_layer: int,
    max_gen_tokens: int = 40,
    teacher_window: int = 8,
) -> dict:
    """Collect paired (sender_hidden, receiver_teacher_logits) from text communication.

    For each prompt:
      1. Sender generates a text response
      2. Extract sender's hidden state from extraction_layer at last token
      3. Feed prompt + sender_text to receiver, record logits for first teacher_window tokens

    Returns dict with:
      sender_hiddens: [N, d_sender]
      receiver_teacher_logits: [N, teacher_window, vocab_size]
      sender_texts: list of generated text strings
      prompts: list of prompt strings
    """
    sender_hiddens = []
    teacher_logits = []
    sender_texts = []

    if sender_tokenizer.pad_token is None:
        sender_tokenizer.pad_token = sender_tokenizer.eos_token
    if receiver_tokenizer.pad_token is None:
        receiver_tokenizer.pad_token = receiver_tokenizer.eos_token

    for i, prompt in enumerate(prompts):
        inp = sender_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
        with torch.no_grad():
            gen_ids = sender_model.generate(
                **inp, max_new_tokens=max_gen_tokens,
                do_sample=True, temperature=0.8, top_p=0.95,
                pad_token_id=sender_tokenizer.eos_token_id,
            )
        gen_text = sender_tokenizer.decode(gen_ids[0][inp["input_ids"].shape[1]:],
                                           skip_special_tokens=True)
        sender_texts.append(gen_text)

        with torch.no_grad():
            sender_out = sender_model(**inp, output_hidden_states=True)
            h = sender_out.hidden_states[extraction_layer][0, -1, :]
        sender_hiddens.append(h)

        combined = prompt + " " + gen_text
        recv_inp = receiver_tokenizer(combined, return_tensors="pt",
                                       truncation=True, max_length=128)
        with torch.no_grad():
            recv_out = receiver_model(**recv_inp)
            logits = recv_out.logits[0]
            window = min(teacher_window, logits.shape[0])
            teacher_logits.append(logits[-window:])

        if (i + 1) % 50 == 0:
            print(f"    Collected {i+1}/{len(prompts)}", flush=True)

    max_window = min(teacher_window, min(t.shape[0] for t in teacher_logits))
    teacher_logits = torch.stack([t[-max_window:] for t in teacher_logits])

    return {
        "sender_hiddens": torch.stack(sender_hiddens),
        "receiver_teacher_logits": teacher_logits,
        "sender_texts": sender_texts,
        "prompts": prompts,
        "teacher_window": max_window,
    }


def train_bapc_codec(
    codec: BAPCCodec,
    receiver_model,
    receiver_tokenizer,
    training_data: dict,
    prompts: list[str],
    config: BAPCTrainingConfig | None = None,
) -> list[dict]:
    """Train a BAPC codec via behavioural distillation.

    The receiver model is frozen. Only codec parameters are trained.
    The codec learns to produce prefix embeddings that make the receiver
    behave the same as when it reads the sender's text.

    Args:
        codec: The BAPCCodec to train.
        receiver_model: Frozen CausalLM for computing student logits.
        receiver_tokenizer: Tokenizer for the receiver.
        training_data: Output of collect_training_data().
        prompts: Prompt strings for generating receiver inputs.
        config: Training configuration.

    Returns:
        List of loss component dicts.
    """
    if config is None:
        config = BAPCTrainingConfig()

    sender_hiddens = training_data["sender_hiddens"]
    teacher_logits = training_data["receiver_teacher_logits"]
    N = sender_hiddens.shape[0]
    T = config.temperature

    for p in receiver_model.parameters():
        p.requires_grad = False

    optimizer = optim.AdamW(codec.parameters(), lr=config.lr)
    history = []

    embed_layer = receiver_model.get_input_embeddings()

    for step in range(config.num_steps):
        idx = torch.randint(0, N, (min(config.batch_size, N),))
        h_batch = sender_hiddens[idx]
        teacher_batch = teacher_logits[idx]

        z_batch = codec.encode(h_batch)
        prefix_batch = codec.decode(z_batch)

        device = next(receiver_model.parameters()).device

        student_logits_list = []
        for b in range(h_batch.shape[0]):
            prompt = prompts[idx[b]]
            prompt_ids = receiver_tokenizer(prompt, return_tensors="pt",
                                            truncation=True, max_length=64)
            with torch.no_grad():
                prompt_embeds = embed_layer(prompt_ids["input_ids"][0].to(device))

            prefix_embeds = prefix_batch[b]  # [S, d_emb]
            combined_embeds = torch.cat([prefix_embeds, prompt_embeds], dim=0).unsqueeze(0)

            with torch.no_grad():
                out = receiver_model(inputs_embeds=combined_embeds)
            student_logits_list.append(out.logits[0])

        window = teacher_batch.shape[1]
        student_logits_all = torch.stack([sl[-window:] for sl in student_logits_list])

        p_teacher = F.softmax(teacher_batch / T, dim=-1).detach()
        log_p_student = F.log_softmax(student_logits_all / T, dim=-1)
        l_distill = F.kl_div(log_p_student, p_teacher, reduction="batchmean") * (T * T)

        perm = torch.randperm(h_batch.shape[0])
        z_shuffled = z_batch[perm]
        prefix_shuffled = codec.decode(z_shuffled)

        mismatch_logits_list = []
        for b in range(h_batch.shape[0]):
            prompt = prompts[idx[b]]
            prompt_ids = receiver_tokenizer(prompt, return_tensors="pt",
                                            truncation=True, max_length=64)
            with torch.no_grad():
                prompt_embeds = embed_layer(prompt_ids["input_ids"][0].to(device))

            prefix_embeds = prefix_shuffled[b]
            combined = torch.cat([prefix_embeds, prompt_embeds], dim=0).unsqueeze(0)
            with torch.no_grad():
                out = receiver_model(inputs_embeds=combined)
            mismatch_logits_list.append(out.logits[0])

        mismatch_logits_all = torch.stack([ml[-window:] for ml in mismatch_logits_list])

        p_matched = F.softmax(student_logits_all / T, dim=-1)
        p_mismatched = F.softmax(mismatch_logits_all / T, dim=-1)
        l_sep = -js_divergence(p_matched.detach(), p_mismatched.detach())

        l_align_cos = 1.0 - F.cosine_similarity(
            student_logits_all.reshape(-1, student_logits_all.shape[-1]),
            teacher_batch.reshape(-1, teacher_batch.shape[-1]),
            dim=-1
        ).mean()

        l_orth = orthogonality_loss(codec.get_encoder_weights())

        loss = (l_distill
                + config.lambda_sep * l_sep
                + config.lambda_align * l_align_cos
                + config.lambda_orth * l_orth)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(codec.parameters(), 1.0)
        optimizer.step()

        if step % config.log_every == 0:
            entry = {
                "step": step,
                "loss": loss.item(),
                "distill": l_distill.item(),
                "separation": l_sep.item(),
                "align_cos": l_align_cos.item(),
                "orthogonality": l_orth.item(),
            }
            history.append(entry)
            print(f"    Step {step}/{config.num_steps}: "
                  f"distill={l_distill.item():.4f} sep={l_sep.item():.4f} "
                  f"align={l_align_cos.item():.4f} orth={l_orth.item():.4f}",
                  flush=True)

    return history
