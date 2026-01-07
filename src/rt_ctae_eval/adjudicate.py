from collections.abc import Iterable
import argparse
import os
from typing import cast
import json
import logging
from lseval.utils import organize_corpus_annotations_by_annotator
from lseval.adjudication import build_adjudication_file
from lseval.datatypes import SingleAnnotatorCorpus, AnnotatedFile, Relation, Entity
from lseval.correctness_matrix import CorrectnessMatrix
from itertools import combinations
from .utils import score_file
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
    type=str,
    help="Exported Label Studio JSON.",
)

parser.add_argument(
    "--output_dir",
    required=True,
    type=str,
    help="Exported Label Studio JSON.",
)
parser.add_argument(
    "--annotator_ids_tsv",
    default=None,
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
    prediction_annotator: str,
    reference_annotator: str,
    prediction_corpus: SingleAnnotatorCorpus,
    reference_corpus: SingleAnnotatorCorpus,
    overlap: bool,
    output_dir: str,
) -> None:
    adjudication_json_path = os.path.join(
        output_dir, f"Adjudication_{prediction_annotator}_{reference_annotator}.json"
    )
    with open(adjudication_json_path, mode="w") as f:
        f.write("[")
    file_id_to_prediction_files = {
        annotated_file.file_id: annotated_file
        for annotated_file in prediction_corpus.annotated_files
    }
    file_id_to_reference_files = {
        annotated_file.file_id: annotated_file
        for annotated_file in reference_corpus.annotated_files
    }
    file_ids_not_shared_across_annotators = (
        file_id_to_prediction_files.keys() ^ file_id_to_reference_files.keys()
    )
    if len(file_ids_not_shared_across_annotators) > 5:
        logger.warning(
            "No shared annotations for %d files",
            len(file_ids_not_shared_across_annotators),
        )
    elif len(file_ids_not_shared_across_annotators) > 1:
        logger.warning(
            "No shared annotations for file IDs: %s",
            ", ".join(map(str, sorted(file_ids_not_shared_across_annotators))),
        )
    file_ids = sorted(
        file_id_to_prediction_files.keys() & file_id_to_reference_files.keys()
    )
    total_files = len(file_ids)
    for idx, file_id in enumerate(file_ids):
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

        assert isinstance(reference_file, AnnotatedFile) and isinstance(
            prediction_file, AnnotatedFile
        )
        reference_file_text = getattr(reference_file, "file_text", None)
        prediction_file_text = getattr(reference_file, "file_text", None)
        assert (
            reference_file_text is not None
            and reference_file_text == prediction_file_text
        )
        annotated_file_scores = score_file(
            file_id=file_id,
            prediction_file=prediction_file,
            reference_file=reference_file,
            overlap=overlap,
        )
        entity_correctness_matrices = cast(
            Iterable[CorrectnessMatrix[Entity]],
            [
                annotated_file_scores.rt_entity_correctness_matrix,
                annotated_file_scores.adverse_event_entity_correctness_matrix,
            ],
        )
        relation_correctness_matrices = cast(
            Iterable[CorrectnessMatrix[Relation]],
            [annotated_file_scores.causal_relation_correctness_matrix],
        )
        adjudication_file = build_adjudication_file(
            file_id=file_id,
            file_text=reference_file_text,
            total_files=total_files,
            reference_annotator=reference_annotator,
            prediction_annotator=prediction_annotator,
            # See if the types in lseval will work out just with Iterable etc
            entity_correctness_matrices=entity_correctness_matrices,
            relation_correctness_matrices=relation_correctness_matrices,
        )

        with open(adjudication_json_path, mode="a") as f:
            f.write(json.dumps(adjudication_file))
            if idx < total_files - 1:
                f.write(",")

    with open(adjudication_json_path, mode="a") as f:
        f.write("]")


def adjudicate_corpus_all_annnotators(
    corpus_json: str,
    output_dir: str,
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
            prediction_annotator=prediction_annotator,
            reference_annotator=reference_annotator,
            prediction_corpus=prediction_corpus,
            reference_corpus=reference_corpus,
            overlap=overlap,
            output_dir=output_dir,
        )


def main() -> None:
    args = parser.parse_args()
    adjudicate_corpus_all_annnotators(
        corpus_json=args.corpus_json,
        output_dir=args.output_dir,
        annotator_ids_tsv=args.annotator_ids_tsv,
        annotator_to_file_ids_tsv=args.annotator_to_file_ids_tsv,
        overlap=args.overlap,
        annotator_ids_to_ignore=args.annotator_ids_to_ignore,
    )


if __name__ == "__main__":
    main()
