import argparse
import json
import logging
from lseval.utils import organize_corpus_annotations_by_annotator
from lseval.datatypes import SingleAnnotatorCorpus
from itertools import combinations
from .annotator import get_id_to_annotator_mappping, get_annotator_to_file_ids_mappping

logger = logging.getLogger(__name__)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
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
    "--annotator_to_file_ids_tsv",
    help="TSV with rows of the form <annotator name><tab><ID 1>,...,<ID N>",
    default=None,
)
parser.add_argument(
    "--annotator_ids_to_ignore",
    help="Specious fix to avoid wrangling JSON - probably should remove for 'primetime'",
    nargs="+",
    type=int,
)
parser.add_argument(
    "--overlap",
    action="store_true",
    help="Count predicted annotation spans as correct if they overlap by one character or more "
    + "with a reference annotation span. Not intended as a real evaluation method (since what "
    + "to do with multiple matches is not well defined) but useful for debugging purposes.",
)


def adjudicate_corpus(
    prediction_corpus: SingleAnnotatorCorpus,
    reference_corpus: SingleAnnotatorCorpus,
    overlap: bool,
) -> None:
    file_id_to_prediction_files = {
        annotated_file.file_id: annotated_file
        for annotated_file in prediction_corpus.annotated_files
    }
    file_id_to_reference_files = {
        annotated_file.file_id: annotated_file
        for annotated_file in reference_corpus.annotated_files
    }
    for file_id in sorted(
        file_id_to_prediction_files.keys() & file_id_to_reference_files.keys()
    ):
        reference_file = file_id_to_reference_files.get(
            file_id,
            None,
        )
        prediction_file = file_id_to_prediction_files.get(
            file_id,
            None,
        )
        if reference_file is None or prediction_file is None:
            logger.error(f"Missing annotations for {file_id}")
            logger.error(
                f"Reference file {'present' if reference_file is not None else 'absent'}, Prediction file {'present' if prediction_file is not None else 'absent'}"
            )
            continue


def adjudicate_corpus_all_annnotators(
    corpus_json: str,
    annotator_ids_tsv: str,
    annotator_to_file_ids_tsv: str | None,
    overlap: bool,
    annotator_ids_to_ignore: list[int],
) -> None:
    with open(corpus_json, mode="rt") as f:
        raw_json_corpus = json.load(f)
    id_to_unique_annotator = get_id_to_annotator_mappping(
        annotator_ids_tsv, annotator_ids_to_ignore
    )
    annotator_to_single_annotator_corpus = organize_corpus_annotations_by_annotator(
        raw_json_corpus=raw_json_corpus,
        id_to_unique_annotator=id_to_unique_annotator,
        annotator_ids_to_ignore=annotator_ids_to_ignore,
    )
    if annotator_to_file_ids_tsv is not None:
        annotator_to_file_ids_mapping = get_annotator_to_file_ids_mappping(
            annotator_to_file_ids_tsv
        )
    else:
        annotator_to_file_ids_mapping = {}
    for prediction_annotator, reference_annotator in combinations(
        annotator_to_single_annotator_corpus.keys(), r=2
    ):
        # Then it's open season
        if len(annotator_to_file_ids_mapping) == 0:
            prediction_corpus = annotator_to_single_annotator_corpus[
                prediction_annotator
            ]
            reference_corpus = annotator_to_single_annotator_corpus[reference_annotator]
        else:
            NotImplementedError(
                "Need to figure out mapping logic for file IDs for IAA etc"
            )
        logger.info(
            f"Prediction annotator {prediction_annotator} reference annotator {reference_annotator}"
        )
        adjudicate_corpus(
            prediction_corpus=prediction_corpus,
            reference_corpus=reference_corpus,
            overlap=overlap,
        )


def main() -> None:
    args = parser.parse_args()
    adjudicate_corpus_all_annnotators(
        corpus_json=args.corpus_json,
        annotator_ids_tsv=args.annotator_ids_tsv,
        annotator_to_file_ids_tsv=args.annotator_to_file_ids_tsv,
        overlap=args.overlap,
        annotator_ids_to_ignore=args.annotator_ids_to_ignore,
    )


if __name__ == "__main__":
    main()
