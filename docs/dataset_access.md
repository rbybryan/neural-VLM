# Dataset setup and portable paths

The repository contains code, not benchmark datasets or trained model weights.
Obtain each dataset from its provider under its current access conditions.
The synthetic CPU suite needs none of these datasets.

## Data root

Set `NEURAL_VLM_DATA_ROOT` to a directory you control. If unset, paths are relative to
`datasets/` under the working directory. Individual overrides take precedence:

```bash
export NEURAL_VLM_DATA_ROOT=/path/to/datasets
export NSD_ROOT="$NEURAL_VLM_DATA_ROOT/natural-scenes-dataset"
export THINGS_EEG2_ROOT="$NEURAL_VLM_DATA_ROOT/things_eeg_2"
export COCO_VAL2017="$NEURAL_VLM_DATA_ROOT/ms_coco/val2017"
```

See `analysis/paths.py` for the exact layouts. `COCO_TRAIN2017`, `THINGS_DB_ROOT`, and
`ARO_ROOT` are also configurable. Local dataset directories are gitignored.

## Benchmark inputs

| Dataset | Provider / instructions | Entry point |
|---|---|---|
| Winoground | [Hugging Face](https://huggingface.co/datasets/facebook/winoground); complete any access requirements | `scripts/prepare_eval_data.py winoground` |
| SugarCrepe | [Official repository](https://github.com/RAIVNLab/sugar-crepe); requires matching COCO images | `scripts/prepare_eval_data.py sugarcrepe` |
| NSD | [Natural Scenes Dataset](https://naturalscenesdataset.org/) | `analysis/nsd_loader.py` |
| THINGS-EEG2 | [Dataset](https://osf.io/3jk45/) | `analysis/things_eeg2_loader.py` |
| ARO | [Official repository](https://github.com/mertyg/vision-language-models-are-bows) | External preparation; no dedicated loader supplied |

For Winoground, authenticate locally if required, then run:

```bash
python scripts/prepare_eval_data.py winoground --output-dir data/eval/winoground
python scripts/prepare_eval_data.py sugarcrepe \
    --coco-val-dir "$COCO_VAL2017" --output-dir data/eval/sugarcrepe
```

Keep access tokens in your local credential store or environment. Never paste them into
code, logs, documentation, or committed configuration.

## Neural analyses

NSD expects `nsddata/`, `nsddata_betas/`, and `nsddata_stimuli/` under `NSD_ROOT`.
THINGS-EEG2 expects `preprocessed_data/` and `image_set/` under `THINGS_EEG2_ROOT`.
The original H3 plan named THINGS-MEG; the implementation follows the documented
[THINGS-EEG2 amendment](preregistration_amendment_01.md).

The historical [access checklist](data_access_checklist.md) records the broader research
plan. Its storage estimates and access assumptions are not guarantees of current availability.

## Training inputs

Prepare an image/conversation corpus in the schema accepted by `data/conversation.py`.
Pass its manifest and image directory via `--data-json` and `--image-root`.
Follow the original provider's preparation instructions; see
[LLaVA data preparation](https://github.com/haotian-liu/LLaVA/blob/main/docs/Data.md).
Use `data/decontaminate.py` to filter training images against evaluation inputs before
reporting benchmark results. See [training usage](USAGE.md).

No script assumes the author's cluster, storage quota, or login name.
