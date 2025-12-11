import argparse
import polars as pl
from typing import Mapping
import json
import logging
from lseval.utils import organize_corpus_annotations_by_annotator
from .utils import score_corpus
from itertools import permutations

parser = argparse.ArgumentParser(description="")
parser.add_argument(
    "--corpus_json",
    required=True,
    help="Exported Label Studio JSON.",
)

parser.add_argument(
    "--annotator_ids_tsv",
    help="TSV with rows of the form <annotator name><tab><ID 1>,...,<ID N>",
)
parser.add_argument(
    "--overlap",
    action="store_true",
    help="Count predicted annotation spans as correct if they overlap by one character or more "
    + "with a reference annotation span. Not intended as a real evaluation method (since what "
    + "to do with multiple matches is not well defined) but useful for debugging purposes.",
)
parser.add_argument(
    "--per_document",
    action="store_true",
    help="Print document level stores",
)

logger = logging.getLogger(__name__)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


def get_id_to_annotator_mappping(annotator_ids_tsv: str) -> Mapping[int, str]:
    annotator_with_ids_df = pl.read_csv(annotator_ids_tsv, separator="\t")
    id_to_unique_annotator = {}
    for annnotator_name, clustered_ids in zip(
        annotator_with_ids_df["annotator_name"], annotator_with_ids_df["annotator_ids"]
    ):
        for annotator_id in map(int, clustered_ids.split(",")):
            id_to_unique_annotator[annotator_id] = annnotator_name
    return id_to_unique_annotator


def score_corpus_all_annnotators(
    corpus_json: str, annotator_ids_tsv: str, overlap: bool, per_document: bool
) -> None:
    with open(corpus_json, mode="rt") as f:
        raw_json_corpus = json.load(f)
    id_to_unique_annotator = get_id_to_annotator_mappping(annotator_ids_tsv)
    annotator_to_single_annotator_corpus = organize_corpus_annotations_by_annotator(
        raw_json_corpus=raw_json_corpus, id_to_unique_annotator=id_to_unique_annotator
    )
    for prediction_annotator, reference_annotator in permutations(
        annotator_to_single_annotator_corpus.keys(), r=2
    ):
        prediction_corpus = annotator_to_single_annotator_corpus[prediction_annotator]
        reference_corpus = annotator_to_single_annotator_corpus[reference_annotator]
        print(
            f"Prediction annotator {prediction_annotator} reference annotator {reference_annotator}"
        )
        score_corpus(
            prediction_corpus=prediction_corpus,
            reference_corpus=reference_corpus,
            overlap=overlap,
            per_document=per_document,
        )


def main() -> None:
    args = parser.parse_args()
    score_corpus_all_annnotators(
        corpus_json=args.corpus_json,
        annotator_ids_tsv=args.annotator_ids_tsv,
        overlap=args.overlap,
        per_document=args.per_document,
    )


if __name__ == "__main__":
    main()
