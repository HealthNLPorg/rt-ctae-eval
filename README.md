# rt-ctae-eval
## Evaluate
```
uv run python -m rt_ctae_eval.evaluate \
    --corpus_json ~/Downloads/latest_annotations_12_16_2025.json \
    --annotator_ids_tsv annotator_ids.tsv \
    --overlap \
    --annotator_ids_to_ignore 1 \
    --per_document
```
## Adjudicate (enough code re-used where we can probably collapse things down)
```
uv run python -m rt_ctae_eval.adjudicate \
    --corpus_json ~/Downloads/latest_annotations_12_16_2025.json \
    --annotator_ids_tsv annotator_ids.tsv \
    --overlap \
    --annotator_ids_to_ignore 1 \
    --output_dir ./adjudication_output
```
