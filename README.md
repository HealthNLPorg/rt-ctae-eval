# rt-ctae-eval

 Evaluation and annotation adjudication tool for the ACS-CTAE Label Studio project, using [lseval](https://github.com/HealthNLPorg/lseval) as a backend.

## Installation

Clone this repository and install via `uv sync`.  For development activate the virtual environment via `source .venv/bin/activate`.  To run evaluation on an exported Label Studio project with adjudication output and correcting overlapping spans as correct:
```
uv run -m rt_ctae_eval.evaluate \
    --corpus_json ...json \
    --overlap \
    --output_dir adjudication_output \
    --filter_agreements \
    --adjudicate
```
