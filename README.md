# rt-ctae-eval

 Evaluation and annotation adjudication tool for the ACS-CTAE Label Studio project, using [lseval](https://github.com/HealthNLPorg/lseval) as a backend.

## Installation

Clone this repository and install via `uv sync`.  For development activate the virtual environment via `source .venv/bin/activate`.

## Overview
The ACS-CTAE project annotations identify mentions of radiation therapy with relevant signatures, adverse events, and causal relations between them.  In Label Studio these are modeled with span based entities with attributes and relations between them. Label Studio does not provide evaluation or adjudication functionality, which is the goal of this module.  The [lseval](https://github.com/HealthNLPorg/lseval) dependency provides most of the core functionality, but this module provides an entry point and project specific organization of data types and results.

## Functionality

Given an exported Label Studio annotation project, the module can compute scores at the corpus and document level between every pair of annotators, as well as generate JSON for an adjudication project. To run evaluation on an exported Label Studio project with adjudication output and correcting overlapping spans as correct:
```
uv run -m rt_ctae_eval.evaluate \
    --corpus_json ...json \
    --overlap \
    --output_dir adjudication_output \
    --filter_agreements \
    --adjudicate
```
