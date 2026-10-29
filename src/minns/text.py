"""Stage E: text embeddings of the TextGrid transcript, on the same instances as the audio.

Reads the tables minns.segments writes (outputs/features/words/<pid>.json) and embeds
every word with a multilingual text encoder, every layer, in context: each speaker's
segments are packed in time order into chunks of up to MAX_TOKENS tokens and encoded
together, so a word sees its neighbours, including those in the speaker's previous and
next segments. A word's vector is the mean of its subword tokens (tokenizer offsets);
a segment's or utterance's vector is the mean of the tokens of its words — the
SBERT mean pooling, for sbert-mpnet exactly its sentence embedding before
normalisation, but taken over the instance's tokens only.

Instances without words (an empty-text fallback, a stage-D utterance detected from
energy with no transcribed word in it) get a zero vector.

Feature sets (outputs/features/windows/<set>/<pid>.npz, float16, every layer):
  <encoder>-word   participant words (key word)
  <encoder>-seg    TextGrid segments (keys seg, dseg)
  <encoder>-utt    stage-D utterances (keys utt, dlg), when minns.utterances has run
                   before minns.segments

    python -m minns.text --encoder xlmr-large --limit 3
    python -m minns.text --encoder sbert-mpnet
    python -m minns.text --encoder mbert-base
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from minns.data import OUTPUTS, load_bucket
from minns.segments import WORD_DIR, seg_sets, word_spans
from minns.utterances import UTT_DIR, save

TEXT_ENCODERS = {
    "xlmr-large": "FacebookAI/xlm-roberta-large",          # 24 blocks, 1024-D
    "muril-large": "google/muril-large-cased",              # Indic + transliterated, 24 blocks, 1024-D
    "sbert-mpnet": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",   # the pipeline's SBERT
    "mbert-base": "google-bert/bert-base-multilingual-cased",   # multilingual BERT (kn/te/ta/en), 12 blocks, 768-D
}
MAX_TOKENS = 256


class TextEncoder:
    def __init__(self, name: str, device: str):
        from transformers import AutoModel, AutoTokenizer
        repo = TEXT_ENCODERS[name]
        local = OUTPUTS / "models" / repo.split("/")[-1]       # hand-fetched copy wins, as in extract.HFEncoder
        src = local if local.exists() else repo
        self.tok = AutoTokenizer.from_pretrained(src)
        self.model = AutoModel.from_pretrained(src).to(device).eval()
        self.device = device

    def n_tokens(self, text: str) -> int:
        return len(self.tok(text, add_special_tokens=False)["input_ids"])

    @torch.no_grad()
    def __call__(self, text: str) -> tuple[torch.Tensor, np.ndarray]:
        """(n_layers, n_tokens, dim) hidden states, (n_tokens, 2) character offsets; special tokens are (0, 0)."""
        enc = self.tok(text, return_offsets_mapping=True, truncation=True, max_length=512, return_tensors="pt")
        offsets = enc.pop("offset_mapping")[0].numpy()
        hs = self.model(**{k: v.to(self.device) for k, v in enc.items()}, output_hidden_states=True).hidden_states
        return torch.stack(hs)[:, 0], offsets


def chunks(table: dict, enc: TextEncoder) -> list[list[int]]:
    """Dialogue-segment indices packed per speaker, in time order, into chunks of <= MAX_TOKENS tokens."""
    out = []
    for role in sorted({s["role"] for s in table["segments"]}):
        cur, used = [], 0
        for i, s in enumerate(table["segments"]):
            if s["role"] != role or not s["text"]:
                continue
            n = enc.n_tokens(s["text"]) + 1
            if cur and used + n > MAX_TOKENS:
                out.append(cur)
                cur, used = [], 0
            cur.append(i)
            used += n
        if cur:
            out.append(cur)
    return out


def embed(table: dict, enc: TextEncoder) -> tuple[np.ndarray, np.ndarray, int]:
    """Token sums (n_words, L, D) and token counts (n_words,) of every word in the table, and how
    many words got no token (truncated chunk)."""
    w = table["words"]
    n_words = len(w["text"])
    by_seg: dict[int, list[int]] = {}
    for k, d in enumerate(w["dseg"]):
        if w["text"][k]:
            by_seg.setdefault(d, []).append(k)
    sums, counts = None, torch.zeros(n_words, device=enc.device)
    for chunk in chunks(table, enc):
        text, starts, ends, ids = "", [], [], []
        for d in chunk:
            off = len(text) + (1 if text else 0)
            text = f"{text} {table['segments'][d]['text']}" if text else table["segments"][d]["text"]
            for k in by_seg.get(d, []):
                starts.append(off + w["char"][k][0])
                ends.append(off + w["char"][k][1])
                ids.append(k)
        if not ids:
            continue
        hs, offsets = enc(text)
        if sums is None:
            sums = torch.zeros(n_words, hs.shape[0], hs.shape[2], device=enc.device)
        starts, ends, ids = np.asarray(starts), np.asarray(ends), np.asarray(ids)
        tok = np.flatnonzero(offsets[:, 1] > offsets[:, 0])
        # a token belongs to the word whose span holds its last character
        j = np.searchsorted(starts, offsets[tok, 1] - 1, side="right") - 1
        ok = (j >= 0) & (offsets[tok, 0] < ends[np.maximum(j, 0)])
        tok, j = tok[ok], ids[j[ok]]
        idx = torch.as_tensor(j, device=enc.device)
        sums.index_add_(0, idx, hs[:, torch.as_tensor(tok, device=enc.device)].float().transpose(0, 1))
        counts.index_add_(0, idx, torch.ones(len(j), device=enc.device))
    if sums is None:                     # no text at all (fallback bag)
        hs, _ = enc("a")
        sums = torch.zeros(n_words, hs.shape[0], hs.shape[2], device=enc.device)
    missing = int(sum(1 for k in range(n_words) if w["text"][k] and counts[k] == 0))
    return sums, counts, missing


def pool(sums: torch.Tensor, counts: torch.Tensor, group: np.ndarray, n_groups: int) -> np.ndarray:
    """Mean of the tokens of the words in each group (group = -1: none) -> (n_groups, L, D) float16."""
    keep = np.flatnonzero(group >= 0)
    g = torch.as_tensor(group[keep], device=sums.device)
    s = torch.zeros(n_groups, *sums.shape[1:], device=sums.device).index_add_(0, g, sums[keep])
    c = torch.zeros(n_groups, device=sums.device).index_add_(0, g, counts[keep])
    return (s / c.clamp(min=1)[:, None, None]).to(torch.float16).cpu().numpy()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoder", choices=list(TEXT_ENCODERS), required=True)
    ap.add_argument("--bucket", default="finetune", choices=["finetune", "pretrain"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    from minns.extract import pick_device
    df = load_bucket(args.bucket)
    if args.limit:
        df = df.head(args.limit)
    pids = df["participant_id"].tolist()
    missing = [p for p in pids if not (WORD_DIR / f"{p}.json").exists()]
    if missing:
        raise SystemExit(f"{len(missing)} participants have no tables yet (python -m minns.segments first)")
    out = OUTPUTS / "features/windows" / f"{args.encoder}-seg"
    todo = [p for p in pids if args.overwrite or not (out / f"{p}.npz").exists()]
    device = pick_device()
    print(f"{args.encoder}: {len(todo)}/{len(pids)} participants to embed, device={device}", flush=True)
    if not todo:
        return
    enc = TextEncoder(args.encoder, device)

    t0 = time.time()
    for i, pid in enumerate(todo, 1):
        table = json.loads((WORD_DIR / f"{pid}.json").read_text())
        w = table["words"]
        sums, counts, lost = embed(table, enc)
        info = {k: table[k] for k in ("rule", "fallback", "n_seg", "n_dseg", "n_word")}
        info.update(encoder=TEXT_ENCODERS[args.encoder], words_without_tokens=lost)

        p_word = np.asarray(table["p_word"])
        words = pool(sums[torch.as_tensor(p_word, device=device)], counts[torch.as_tensor(p_word, device=device)],
                     np.arange(len(p_word)), len(p_word))
        save(f"{args.encoder}-word", pid, {"word": (words, word_spans(table))}, info, dtype=np.float16,
             full_key="word")
        segs = pool(sums, counts, np.asarray(w["dseg"]), len(table["segments"]))
        save(f"{args.encoder}-seg", pid, seg_sets(segs, table), info, dtype=np.float16, full_key="seg")
        n_utt = 0
        if table["has_utterances"]:
            meta = json.loads((UTT_DIR / f"{pid}.json").read_text())
            spans = np.asarray(meta["dlg_spans"], dtype=np.float32).reshape(-1, 2)
            is_p = np.asarray(meta["dlg_roles"]) == 0
            utts = pool(sums, counts, np.asarray(w["dlg_utt"]), len(spans))
            n_utt = int((np.abs(utts.astype(np.float32)).sum((1, 2)) > 0).sum())
            save(f"{args.encoder}-utt", pid, {"utt": (utts[is_p], spans[is_p]), "dlg": (utts, spans)},
                 {**info, "utts_with_text": n_utt}, dtype=np.float16)
        if device == "mps":
            torch.mps.empty_cache()
        print(f"[{i}/{len(todo)}] {pid}  {len(p_word)} words, {table['n_dseg']} segments"
              + (f", {n_utt} utterances with text" if table["has_utterances"] else "")
              + (f"  ({lost} words lost to truncation)" if lost else ""), flush=True)
    print(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
