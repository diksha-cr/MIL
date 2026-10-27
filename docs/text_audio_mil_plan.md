# Stage E: audio + text multiple-instance models

Adds the transcript to the speech MIL models. Each instance has an audio vector and a
text vector for **the same time window**. Two designs without cross-attention (§3, §4)
are the baseline that the cross-attention model (§9) has to beat. The participant is
the bag throughout. Folds, thresholds and metrics are the same as in stages B–D, so
every row here can be compared with an existing row.

Code: `src/minns/segments.py`, `text.py`, `fusion.py`, `compare.py`,
`scripts/cluster_text.sh`. A test run on 60 participants is in §8.

---

## 0. What the data gives us (checked 2026-10-27)

| Fact | Consequence |
|---|---|
| The TextGrids (`outputs/data/cleaned/<pid>/output.TextGrid`, mirrored in `participant_jsons/*.json → transcript.segments`) have **one IntervalTier per speaker (S0, S1, S2, S?)**. Each interval is a **segment of text** (median 2 s, 5 words). There are **no word-level timings.** A few intervals are whole monologues (up to 97 s, 194 words). | Segments are the natural aligned instance. Word times have to be estimated (§1). Long intervals are cut into pieces. |
| Languages: kn 39 %, te 29 %, ta 19 %, en 10 %, te_kn 3 %, in native script with some code-mixed English. | The text encoder must be multilingual. Because the script gives away the language, text is **more** site- and language-confounded than audio (§6). |
| Roles come from `data.speaker_roles(rule="role")`. The stage-D utterances (`minns.utterances`) are cut from energy, not from the TextGrid. | Design 2 can use either segments or utterances as instances. For utterances, words are mapped onto them (§1). |
| The TextGrids hold speech only. The pipeline's SBERT step (`src/text/embeddings.py`), with its default `include_clinical=True`, also adds sentences built from personal-history answers. | We embed TextGrid speech only, which matches the pipeline's `--no-clinical-context`. |

---

## 1. Instances and word timings: `src/minns/segments.py`

Chosen order: **segment instances first, then syllable-split words. Build forced
alignment only if the word-level models beat the segment-level ones.**

- **Segments (`seg`, `dseg`).** Each TextGrid interval with text is one instance.
  The audio and text cover the same window by construction, and no alignment is
  needed. `seg` holds the participant's segments; `dseg` holds every speaker's
  segments in time order, with `role-seg`. Intervals longer than `MAX_SEC` (15 s,
  the same cap as the utterances) are cut at word boundaries into near-equal
  pieces.
- **Words (`word`).** Each segment's duration is split over its words in
  proportion to their syllable counts (`utterances.syllables`: aksharas for Indic
  scripts, vowel groups for Latin). This is approximate inside a segment and exact
  at its edges. Segments are short, so the error is bounded at about ±0.2–0.3 s.
  Whisper L23 states are also very contextual, so a slightly shifted window still
  mostly captures that word's representation.
- **Words → utterances.** A word belongs to the stage-D utterance of the same
  speaker that contains its midpoint. If none does, it goes to the nearest one
  within `UTT_SNAP_SEC` = 1 s. On the test set a median 82 % of participant words
  land in an utterance; containment alone gave 66 %.
- **No empty bags.** A recording without participant text gets one empty-text
  instance over the whole recording; a participant without words gets one
  empty-text word. This is flagged in `info.fallback`. Text encoders give such
  instances a zero vector.
- **No word tier exists (checked 2026-10-27).** The 1709 `cleaned/*/output.TextGrid`
  files are byte-identical to `raw_pool/*/transcripts/spontaneous.TextGrid`, and every
  tier is a speaker IntervalTier (S0–S5, S?) of segment text (median 2.5 s / 5 words,
  up to 53 words). The syllable split stays; single-word intervals are exact.
- **Upgrade path, only if word-level models beat segment-level ones:** CTC forced
  alignment inside each segment, using torchaudio `MMS_FA` + `uroman`, or
  AI4Bharat IndicWav2Vec on native script.

Hand-made features written per instance:

| Set | Keys | Content |
|---|---|---|
| `prosody-seg` | seg, dseg | `utterances.PROSODY_NAMES` (rate, pauses, response latency, F0, loudness) |
| `role-seg` | seg, dseg | one-hot participant / interviewer / other |
| `egemaps-seg` | seg, dseg | eGeMAPSv02 functionals (`--egemaps`) |
| `wordpros` | word | duration, syllables, segment syllable rate, position in segment, F0 mean/sd, loudness, voiced share |
| `<speech enc>-seg` / `-word` | seg, dseg / word | encoder states (every layer) from one pass over the recording, mean-pooled over the span (at least one 20 ms frame) |

Tables: `outputs/features/words/<pid>.json` holds the segments and words, plus the
maps from words to segments and to utterances.

---

## 2. Text features: `src/minns/text.py`

| Encoder | Repo | Notes |
|---|---|---|
| **`xlmr-large`** (default) | FacebookAI/xlm-roberta-large | token-level, kn/te/ta/en, 25 layers × 1024 |
| `muril-large` | google/muril-large-cased | Indic and transliterated text, suits code-mixing |
| `sbert-mpnet` | paraphrase-multilingual-mpnet-base-v2 | the pipeline's SBERT, 13 layers × 768 |

- **Context.** Each speaker's segments are packed, in time order, into chunks of up
  to 256 tokens and encoded together. A word therefore sees the speaker's
  neighbouring segments too.
- **Pooling.** A word's vector is the mean of its subword tokens, found from the
  tokenizer's offsets. A segment's or utterance's vector is the mean over the
  tokens of its words, which is SBERT's mean pooling.
- **Output.** `<enc>-word` (word), `<enc>-seg` (seg, dseg) and `<enc>-utt`
  (utt, dlg; needs stage D run first). All are float16, with every layer.
- **Choosing the text layer.** Probe it once (`minns.probe --encoder xlmr-large-seg --windows seg`),
  then fix it with `@L` (e.g. `xlmr-large-seg@16`). `--layers` then sweeps only the
  audio layer.

---

## 3. Design 2: aligned instances, no cross-attention (the baseline)

The instance is a segment (`seg` / `dseg`) or a stage-D utterance (`utt` / `dlg`),
and it carries `(audio_i, text_i)` for the same window.

**2a. Concatenation:** `x_i = [audio_i ; text_i ; prosody_i (; role_i)]`. There are
two ways to run it:

- `mil.py` unchanged (its MLP + attention / MI / mi pooling):
  `--encoder whisper-medium-seg+xlmr-large-seg@16+prosody-seg --windows seg --layers 23 24`.
  `load_packed` now accepts `set@L`, which pairs every audio layer with one text layer.
- `fusion.py concat-attn`: each modality is first projected to 128-D and
  layer-normalised, so a 1024-D audio vector does not swamp a 768-D text vector.
  This is the fair counterpart of the gate model.

**2b. Learned gate (`fusion.py gate-attn`):**

```
a_i = LN(W_a audio_i),  t_i = LN(W_t text_i)                  (128 each)
g_i = σ(W_g [a_i ; t_i])                                      (128, per dimension)
h_i = g_i ⊙ a_i + (1 − g_i) ⊙ t_i ;  then [h_i ; extra_i] → MLP (32, 64, 128) → gated attention → head
```

Where an instance has no text (`has_text = 0`), `t_i` is set to 0 and `g_i` is forced
to 1. The model prints the mean audio gate, overall and by language, and saves it
per participant in the OOF file.

---

## 4. Design 1: word instances

`w_j = [LN(W_a audio_word_j) ; LN(W_t text_word_j) ; wordpros_j]`, with syllable-split
word spans (§1).

| Model | Pooling |
|---|---|
| **1a `concat-attn` / `gate-attn` on `word`** | gated attention over all of the participant's words (median 204, max ~700) |
| **1b `concat-hier` / `gate-hier`** | attention over the words of each segment gives a segment vector; attention over the segments gives the bag vector |

The word bags are ~8× larger than the segment bags, and there are only ~63 training
positives. So nothing changes except the instances: same 128-D projections, the
`HIDDEN` sizes, dropout 0.25 and weight decay. Nothing is tuned on the folds.

**Decision rule for forced alignment:** build it only if word-level models (E8/E9)
beat segment-level ones (E4/E5) on `site_auc` with a paired CI that excludes 0.

---

## 5. Experiments: all on the same 5-fold CV of the 550 train participants

| # | Instances | Audio | Text | Fusion | Run with |
|---|---|---|---|---|---|
| E0 | — | stage-0 confounds (site+language, demographics) | | | `cluster_mil.sh stage0` |
| E1 | participant | — | SBERT mean of all participant segments | — | `minns.probe --encoder sbert-mpnet-seg --windows full` |
| E2 | seg / utt | whisper-medium | — | — | `fusion.py audio-attn` |
| E3 | seg / utt | — | xlmr-large | — | `fusion.py text-attn` (and the probe, for the layer) |
| **E4** | seg / utt | ✓ | ✓ | concat (2a) | `fusion.py concat-attn`, `mil.py` with `+` |
| **E5** | seg / utt | ✓ | ✓ | gate (2b) | `fusion.py gate-attn` |
| E6/E7 | word | ✓ / — | — / ✓ | — | `fusion.py audio-attn` / `text-attn` |
| **E8** | word | ✓ | ✓ | concat / gate | `fusion.py concat-attn gate-attn --windows word` |
| **E9** | word | ✓ | ✓ | hierarchical | `fusion.py concat-hier gate-hier --windows word` |
| **E10** | word | ✓ | ✓ | cross-attention, flat pooling | `fusion.py xattn-attn --windows word` |
| **E11** | word | ✓ | ✓ | cross-attention, words → segment → bag | `fusion.py xattn-hier --windows word` |

Every row adds `--extra prosody-seg` (segment) / `prosody` (utterance) / `wordpros`
(word), plus `role-seg` / `role` for `dseg` / `dlg`. Random CV comes first; site
CV only for the best 2–3 configs.

---

## 6. Metrics and reporting: the same as stages B–D

Every config reports what `evaluation.run_cv` produces:

- `oof_auc`, fold `auc ± sd`, `site_auc`, `ap`
- `f2`, `f1`, `ppv`, `recall`, `specificity`, `npv` at the F2 threshold taken from
  the other folds
- TP/FN/FP/TN
- `_oof.npz`

New:

1. **`lang_auc`**: within-language AUC. It is computed in `run_cv` and now
   reported by `mil.py` and `fusion.py`.
2. **Paired ΔAUC**: `python -m minns.compare delta --ref <oof.npz>:<audio-only config> --against <oof.npz>`.
   This is a paired bootstrap over participants (2000 resamples) for ΔAUC and
   Δwithin-site AUC, giving a 95 % CI and a one-sided p.
3. **Text confound**: `python -m minns.compare confound --set xlmr-large-seg@16 --windows seg`.
   It reports how well the participant-mean text vector predicts language and
   site, against the majority class.
4. **Gate values** by language, from `fusion.py`.

Results go to `outputs/results/stage_e/fusion_<audio>_<text>_<windows>_<cv>.csv`
(+ `_oof.npz`, with gates).

**A fused model only counts as better if** its `site_auc` beats the best audio-only
`site_auc` and E0, **and** the paired ΔAUC CI excludes 0. A higher pooled AUC alone is
not enough.

---

## 7. Cluster runs

```bash
sbatch scripts/cluster_utterances.sh segment                                   # stage D, if not done (for -utt text sets)
sbatch scripts/cluster_text.sh tables                                          # CPU, minutes
sbatch --gres=gpu:1 --mem=32G scripts/cluster_text.sh audio whisper-medium     # GPU
sbatch --gres=gpu:1 --mem=32G scripts/cluster_utterances.sh whisper-medium     # GPU, if whisper-medium-utt missing
sbatch --gres=gpu:1 --mem=32G scripts/cluster_text.sh text xlmr-large          # GPU; needs the hub (or outputs/models/xlm-roberta-large)
PROBE_ARGS="--windows seg" sbatch scripts/cluster_probe.sh xlmr-large-seg random   # pick the text layer
FUSION_ARGS="--audio whisper-medium-seg --text xlmr-large-seg@<L> --extra prosody-seg --windows seg --layers 23 24 \
  --models audio-attn text-attn concat-attn gate-attn" sbatch --gres=gpu:1 scripts/cluster_text.sh fusion
FUSION_ARGS="--audio whisper-medium-word --text xlmr-large-word@<L> --extra wordpros --windows word --layers 23 \
  --models audio-attn text-attn concat-attn gate-attn concat-hier gate-hier" sbatch --gres=gpu:1 scripts/cluster_text.sh fusion
```

Order: tables → audio → text → text-layer probe → segment fusion (E2–E5) → utterance
fusion → word fusion (E6–E9) → site CV on the best configs → `compare delta`.

---

## 8. Test run (local, 2026-10-27)

`scripts/smoke_text.sh` runs every step above on a subset of the train split. It
builds a scratch pipeline root whose bucket is the subset, and writes everything
under `$WORK`, never into `outputs/`:

```bash
bash scripts/smoke_text.sh                                                 # 60 participants, 15 positive
BUCKET=~/mh-ML-pipeline/outputs/data/buckets/1723_2026-10-13/finetune_bucket.csv \
  PY=.venv/bin/python N_POS=5 N_NEG=15 bash scripts/smoke_text.sh         # what was run here, ~5 min on M-series MPS
```

It passed on 60 participants (and on a separate 20). What it showed:

| Check | Result |
|---|---|
| Segments per bag (participant) | p10/50/90 = 13 / 26 / 42, min 4, after cutting intervals > 15 s (before: max 97 s, min 1 per bag) |
| Segment duration | p10/50/90 = 0.5 / 2.2 / 10.9 s |
| Words per bag / per segment | 116 / 204 / 333 (max 671); 1 / 5 / 21 per segment |
| Syllable-split word duration | 0.21 / 0.42 / 0.83 s |
| Participant words inside a stage-D utterance | median 82 % with the 1 s snap (66 % by containment alone) |
| Stage-D utterances with no text | 1 of 1178 |
| Words lost to tokenizer truncation | 0 (XLM-R and SBERT) |
| Empty bags / fallbacks | none |
| In-sample fit (train = score on all 60 bags, 30 epochs) | AUC 1.0 for all 6 fusion models, so gradients reach every path, including the hierarchy |
| Cross-attention (20 participants, 30 epochs, in-sample) | `xattn-attn` / `xattn-hier` AUC 1.0, all outputs finite; own-word text→audio weight 0.15 vs chance 0.14 (untrained on real folds, so no conclusion yet) |
| Confound: participant-mean vector → language / site | XLM-R L16 0.95 / 0.98, **Whisper L23 0.96 / 0.93** (majority 0.43 / 0.45) |
| Run time (60 participants) | tables 20 s, Whisper seg+word 3.6 min, XLM-R 1.2 min, segment fusion (4 models, 2 seeds × 10 epochs) 1 min |

At ~3 positives per fold, the CV AUCs from this run are noise; they only show that
every step runs. The confound result is the one early finding that matters: both
modalities identify language and site almost perfectly, so `site_auc` / `lang_auc`
decide whether a model is better.

---

## 9. Cross-attention: `fusion.py xattn-attn` / `xattn-hier` (E10, E11)

Instances are the participant's words, as in Design 1. Inside each TextGrid segment:

- every word's text vector (query) attends to the audio vectors of all words in the
  segment (keys/values), and every word's audio attends to the segment's text;
- 4 heads, residual + LayerNorm per modality, then `[a' ; t']` goes through the same
  body as the other models;
- `xattn-attn` pools all words with one attention; `xattn-hier` pools words into
  segments, then segments into the bag (as E9).

Attention never crosses a segment boundary, and the segment boundaries are the one
timing the TextGrid gives exactly. Inside a segment, each word attends to all of
the segment's audio rather than to its own syllable-split window. So the model
learns the alignment and does not depend on the estimated word times.

The diagnostic printed per language is the mean text→audio weight on the word's
own audio window, next to chance (1 / words in the segment). Well above chance
means the model leans on the estimated alignment; near chance means it uses the
segment as a whole.

It counts as better only under the §6 rule: compare it with E8/E9 (same instances,
no cross-attention) and E4/E5 (segments) by paired ΔAUC and `site_auc`.

```bash
FUSION_ARGS="--audio whisper-medium-word --text xlmr-large-word@<L> --extra wordpros --windows word --layers 23 \
  --models audio-attn concat-attn concat-hier gate-hier xattn-attn xattn-hier" \
  sbatch --gres=gpu:1 --mem=32G --job-name=fusion-xattn scripts/cluster_text.sh fusion
python -m minns.compare delta --ref outputs/results/stage_e/fusion_whisper-medium-word_xlmr-large-wordL<L>_word_random_oof.npz:concat-hier \
  --against outputs/results/stage_e/fusion_whisper-medium-word_xlmr-large-wordL<L>_word_random_oof.npz
```

Memory: attention is (words × words) per bag, with up to ~670 words, 4 heads and
batch 16. That is about 0.1 GB per tensor, fine on one GPU.

## Next, out of scope here

**Frame/token-level cross-attention.** XLM-R subword tokens attend to the Whisper
frames of their segment. This needs frame states stored per segment, which is far
larger than the pooled word vectors. Try it only if E10/E11 beat E8/E9.
