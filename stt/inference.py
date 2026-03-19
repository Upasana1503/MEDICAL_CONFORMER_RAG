import os

# -------------------------
# Thread safety (CRITICAL)
# -------------------------
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import string

from stt.model import ConformerCTC
from stt.audio import extract_log_mel, MAX_SECONDS, MAX_FRAMES

# -------------------------
# Config
# -------------------------
DEVICE = "cpu"
MODEL_PATH = "stt/conformer_ctc_best.pth"

# -------------------------
# Vocabulary (MUST MATCH TRAINING)
# -------------------------
BLANK_TOKEN = "_"
chars = (
    list(string.ascii_lowercase)
    + list(string.digits)
    + [" ", "'", ".", ","]
)

vocab = {BLANK_TOKEN: 0}
for i, ch in enumerate(chars, start=1):
    vocab[ch] = i

idx2char = {i: ch for ch, i in vocab.items()}

# -------------------------
# Greedy CTC decoding
# -------------------------
def greedy_decode(logits, input_lengths, blank=0):
    """
    logits: (B, T, vocab)
    input_lengths: (B,)
    """
    preds = logits.argmax(dim=-1)  # (B, T)
    results = []
    T_max = preds.size(1)

    for b in range(preds.size(0)):
        seq = []
        prev = blank
        length = min(int(input_lengths[b]), T_max)
        for t in range(length):
            p = preds[b, t].item()
            if p != blank and p != prev:
                seq.append(p)
            prev = p
        results.append(seq)

    return results


def ctc_beam_search_decode(logits, beam_width=10, blank=0):
    """
    Prefix beam search for CTC.
    logits: (B, T, V) - raw logits
    """
    probs = torch.softmax(logits, dim=-1)
    batch_size, T, V = probs.shape
    results = []

    for b in range(batch_size):
        beam = {(): (1.0, 0.0)}  # prefix -> (p_blank, p_non_blank)

        for t in range(T):
            new_beam = {}
            top_probs, top_indices = torch.topk(probs[b, t], min(20, V))

            for char_idx, char_prob in zip(top_indices, top_probs):
                char_idx = char_idx.item()
                char_prob = char_prob.item()

                for prefix, (p_b, p_nb) in beam.items():
                    if char_idx == blank:
                        n_p_b, n_p_nb = new_beam.get(prefix, (0.0, 0.0))
                        new_beam[prefix] = (n_p_b + char_prob * (p_b + p_nb), n_p_nb)
                    else:
                        end_char = prefix[-1] if prefix else None
                        n_prefix = prefix + (char_idx,)
                        n_p_b, n_p_nb = new_beam.get(n_prefix, (0.0, 0.0))

                        if char_idx == end_char:
                            new_beam[n_prefix] = (n_p_b, n_p_nb + char_prob * p_b)
                            c_p_b, c_p_nb = new_beam.get(prefix, (0.0, 0.0))
                            new_beam[prefix] = (c_p_b, c_p_nb + char_prob * p_nb)
                        else:
                            new_beam[n_prefix] = (n_p_b, n_p_nb + char_prob * (p_b + p_nb))

            beam = dict(
                sorted(new_beam.items(), key=lambda x: x[1][0] + x[1][1], reverse=True)[:beam_width]
            )

        best_path = list(max(beam.items(), key=lambda x: x[1][0] + x[1][1])[0])
        results.append(best_path)

    return results

# -------------------------
# Load model ONCE
# -------------------------
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
_model = ConformerCTC(
    vocab_size=checkpoint.get("vocab_size", len(vocab)),
    d_model=checkpoint.get("d_model", 256),
    n_layers=checkpoint.get("n_layers", 6),
    subsampling=checkpoint.get("subsampling", True),
)
_model.load_state_dict(checkpoint["model_state_dict"])
_model.to(DEVICE)
_model.eval()

# -------------------------
# Public API
# -------------------------
def transcribe_audio(
    audio_path,
    decode="beam",
    beam_width=10,
    max_seconds=MAX_SECONDS,
    max_frames=MAX_FRAMES,
):
    mel = extract_log_mel(audio_path, max_seconds=max_seconds, max_frames=max_frames)
    if mel.numel() == 0 or mel.shape[0] == 0:
        return "", 0.0

    x = mel.unsqueeze(0).to(DEVICE)
    input_lengths = torch.tensor([mel.shape[0]], device=DEVICE)

    with torch.no_grad():
        logits, _ = _model(x, input_lengths)

    if decode == "beam":
        pred_seqs = ctc_beam_search_decode(logits, beam_width=beam_width)
    else:
        pred_seqs = greedy_decode(logits, input_lengths)
    text = "".join(idx2char[i] for i in pred_seqs[0])

    probs = torch.softmax(logits, dim=-1)
    confidence = probs.max(dim=-1).values.mean().item()

    return text.strip(), confidence
