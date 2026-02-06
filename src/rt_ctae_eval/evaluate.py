from collections.abc import Sequence, Iterable, Mapping
import argparse
from typing import cast
import json
import os
import logging
from lseval.adjudication import build_adjudication_file
from lseval.utils import organize_corpus_annotations_by_annotator
from .utils import (
    update_corpus_scores,
    score_file,
    update_correctness_matrix,
    merge_correctness_totals,
)
from lseval.datatypes import (
    SingleAnnotatorCorpus,
    DocTimeRel,
    Entity,
    Relation,
)
from lseval.correctness_matrix import CorrectnessMatrix, score_totals, Correctness
from .rt_ctae import AnnotatedCorpusScores, AnnotatedFileScores
from itertools import combinations
from functools import reduce
from tabulate import tabulate
from operator import itemgetter
from .annotator import get_id_to_annotator_mappping

logger = logging.getLogger(__name__)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    # level=logging.INFO,
    level=logging.ERROR,
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
parser.add_argument(
    "--adjudicate", action="store_true", help="Whether to do adjudication"
)
parser.add_argument(
    "--filter_agreements",
    action="store_true",
    help="Don't include agreements, files with agreements aren't counted",
)
parser.add_argument(
    "--per_document",
    action="store_true",
    help="Print document level stores",
)

parser.add_argument(
    "--both_ways",
    action="store_true",
    help="If you want to evaluate a particular annotator pair in both orders, only difference will be order of precision and recall and supports",
)

parser.add_argument(
    "--output_dir",
    required=True,
    type=str,
    help="Exported Label Studio JSON.",
)
parser.add_argument("--exclude_ids", type=str, default=None)


def exclusion_ids(exlude_ids_str: str) -> Sequence[int]:
    if "," in exlude_ids_str:
        return tuple(sorted(map(int, exlude_ids_str.split(","))))
    elif "-" in exlude_ids_str:
        first, second = exlude_ids_str.split("-")
        return range(int(first), int(second))
    else:
        logger.warning("Ill-formed exclusion string: %s", exlude_ids_str)
        return []


def score_corpus(
    prediction_annotator: str,
    reference_annotator: str,
    prediction_corpus: SingleAnnotatorCorpus,
    reference_corpus: SingleAnnotatorCorpus,
    exclusion_ids: Sequence[int],
    overlap: bool,
    adjudicate: bool,
    per_document: bool,
    filter_agreements: bool,
    output_dir: str,
) -> None:
    if adjudicate:
        total_instances = 0
        tmp_adjudication_json_path = os.path.join(
            output_dir,
            f".tmp_Adjudication_{prediction_annotator}_Prediction_{reference_annotator}_Reference.jsonl",
        )
        adjudication_json_path = os.path.join(
            output_dir,
            f"Adjudication_{prediction_annotator}_Prediction_{reference_annotator}_Reference.json",
        )

    def is_valid_file_id(file_id: int) -> bool:
        return len(exclusion_ids) == 0 or file_id not in exclusion_ids

    annotated_corpus_scores = AnnotatedCorpusScores(
        rt_entity_correctness_matrix=CorrectnessMatrix(),
        adverse_event_entity_correctness_matrix=CorrectnessMatrix(),
        causal_relation_correctness_matrix=CorrectnessMatrix(),
    )
    file_id_to_prediction_files = {
        annotated_file.file_id: annotated_file
        for annotated_file in prediction_corpus.annotated_files
        if is_valid_file_id(annotated_file.file_id)
    }
    file_id_to_reference_files = {
        annotated_file.file_id: annotated_file
        for annotated_file in reference_corpus.annotated_files
        if is_valid_file_id(annotated_file.file_id)
    }
    file_ids = sorted(
        file_id_to_prediction_files.keys() | file_id_to_reference_files.keys()
    )
    total_files = len(file_ids)
    for file_id in file_ids:
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
                f"Reference/{reference_annotator} file {'present' if reference_file is not None else 'absent'}, Prediction/{prediction_annotator}/file {'present' if prediction_file is not None else 'absent'}"
            )
            continue
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
                *annotated_file_scores.dtr_correctness_matrices.values(),
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
            filter_agreements=filter_agreements,
        )
        if adjudicate and not (filter_agreements and adjudication_file is None):
            total_instances += 1
            with open(tmp_adjudication_json_path, mode="a", encoding="utf-8") as f:
                f.write(json.dumps(adjudication_file) + "\n")
        annotated_corpus_scores = update_corpus_scores(
            annotated_corpus_scores, annotated_file_scores
        )
        if per_document:
            print(f"File {file_id} scores:")
            print_metrics(annotated_file_scores)

    print("Corpus scores:")
    print_metrics(annotated_corpus_scores)
    print_dtr_by_category(annotated_corpus_scores)
    if adjudicate:
        with (
            open(adjudication_json_path, mode="w", encoding="UTF-8") as wf,
            open(tmp_adjudication_json_path, mode="r", encoding="UTF-8") as rf,
        ):
            wf.write("[")
            for idx, line in enumerate(rf):
                wf.write(line.strip())
                if idx < total_instances - 1:
                    wf.write(",")
            wf.write("]")
        if os.path.exists(tmp_adjudication_json_path):
            os.remove(tmp_adjudication_json_path)


def print_cui_by_category(
    annotated_collection_scores: AnnotatedFileScores | AnnotatedCorpusScores,
) -> None:
    def build_row(
        cui: str, correctness_totals: Mapping[Correctness, int]
    ) -> tuple[list[str], int]:
        cui_f1, cui_prec, cui_recall, cui_support = score_totals(correctness_totals)
        return [
            cui,
            f"{cui_f1:.3f}",
            f"{cui_prec:.3f}",
            f"{cui_recall:.3f}",
            f"{cui_support}",
        ], cui_support

    rows_and_supports = (
        build_row(cui, correctness_totals)
        for (
            cui,
            correctness_totals,
        ) in annotated_collection_scores.cui_correctness_totals.items()
    )
    sorted_rows = list(
        map(itemgetter(0), sorted(rows_and_supports, key=itemgetter(1), reverse=True))
    )

    print(
        tabulate(sorted_rows, headers=["CUI", "F1", "Precision", "Recall", "Support"])
    )


def print_dtr_by_category(
    annotated_collection_scores: AnnotatedFileScores | AnnotatedCorpusScores,
) -> None:
    def build_row(
        dtr: DocTimeRel, correctness_matrix: CorrectnessMatrix[Entity]
    ) -> tuple[list[str], int]:
        return [
            dtr.value,
            f"{correctness_matrix.get_f1():.3f}",
            f"{correctness_matrix.get_precision():.3f}",
            f"{correctness_matrix.get_recall():.3f}",
            f"{correctness_matrix.get_support()}",
        ], correctness_matrix.get_support()

    rows_and_supports = (
        build_row(dtr, correctness_matrix)
        for (
            dtr,
            correctness_matrix,
        ) in annotated_collection_scores.dtr_correctness_matrices.items()
        if dtr is not None
    )
    sorted_rows = list(
        map(itemgetter(0), sorted(rows_and_supports, key=itemgetter(1), reverse=True))
    )
    print(
        tabulate(
            sorted_rows,
            headers=["DTR Category", "F1", "Precision", "Recall", "Support"],
        )
    )


def print_metrics(
    annotated_collection_scores: AnnotatedFileScores | AnnotatedCorpusScores,
) -> None:
    cumulative_dtr_matrix = reduce(
        update_correctness_matrix,
        annotated_collection_scores.dtr_correctness_matrices.values(),
    )
    cumulative_cui_totals = reduce(
        merge_correctness_totals,
        annotated_collection_scores.cui_correctness_totals.values(),
    )
    rt_row = [
        "Radiotherapy Treatments",
        f"{annotated_collection_scores.rt_entity_correctness_matrix.get_f1():.3f}",
        f"{annotated_collection_scores.rt_entity_correctness_matrix.get_precision():.3f}",
        f"{annotated_collection_scores.rt_entity_correctness_matrix.get_recall():.3f}",
        f"{annotated_collection_scores.rt_entity_correctness_matrix.get_support()}",
    ]
    adverse_event_row = [
        "Adverse Events",
        f"{annotated_collection_scores.adverse_event_entity_correctness_matrix.get_f1():.3f}",
        f"{annotated_collection_scores.adverse_event_entity_correctness_matrix.get_precision():.3f}",
        f"{annotated_collection_scores.adverse_event_entity_correctness_matrix.get_recall():.3f}",
        f"{annotated_collection_scores.adverse_event_entity_correctness_matrix.get_support()}",
    ]
    dtr_row = [
        "DocTimeRel",
        f"{cumulative_dtr_matrix.get_f1():.3f}",
        f"{cumulative_dtr_matrix.get_precision():.3f}",
        f"{cumulative_dtr_matrix.get_recall():.3f}",
        f"{cumulative_dtr_matrix.get_support()}",
    ]
    cui_f1, cui_prec, cui_recall, cui_support = score_totals(cumulative_cui_totals)
    cui_row = [
        "CUIs",
        f"{cui_f1:.3f}",
        f"{cui_prec:.3f}",
        f"{cui_recall:.3f}",
        f"{cui_support}",
    ]
    causal_relation_row = [
        "Causal Relations",
        f"{annotated_collection_scores.causal_relation_correctness_matrix.get_f1():.3f}",
        f"{annotated_collection_scores.causal_relation_correctness_matrix.get_precision():.3f}",
        f"{annotated_collection_scores.causal_relation_correctness_matrix.get_recall():.3f}",
        f"{annotated_collection_scores.causal_relation_correctness_matrix.get_support()}",
    ]
    rows = [
        rt_row,
        adverse_event_row,
        dtr_row,
        cui_row,
        causal_relation_row,
    ]
    print(
        tabulate(rows, headers=["Annotation", "F1", "Precision", "Recall", "Support"])
    )


def score_corpus_all_annnotators(
    corpus_json: str,
    annotator_ids_tsv: str,
    annotator_ids_to_ignore: list[int],
    exclude_ids: str | None,
    overlap: bool,
    adjudicate: bool,
    both_ways: bool,
    filter_agreements: bool,
    per_document: bool,
    output_dir: str,
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
    if exclude_ids is not None:
        file_exclusion_ids = exclusion_ids(exclude_ids)
    else:
        file_exclusion_ids = []
    for prediction_annotator, reference_annotator in combinations(
        annotator_to_single_annotator_corpus.keys(), r=2
    ):
        score_corpus_annotator_pair(
            prediction_annotator=prediction_annotator,
            reference_annotator=reference_annotator,
            annotator_to_single_annotator_corpus=annotator_to_single_annotator_corpus,
            exclusion_ids=file_exclusion_ids,
            overlap=overlap,
            adjudicate=adjudicate,
            filter_agreements=filter_agreements,
            both_ways=both_ways,
            per_document=per_document,
            output_dir=output_dir,
        )


def score_corpus_annotator_pair(
    prediction_annotator: str,
    reference_annotator: str,
    annotator_to_single_annotator_corpus: Mapping[str, SingleAnnotatorCorpus],
    exclusion_ids: Sequence[int],
    overlap: bool,
    adjudicate: bool,
    filter_agreements: bool,
    both_ways: bool,
    per_document: bool,
    output_dir: str,
) -> None:
    prediction_corpus = annotator_to_single_annotator_corpus[prediction_annotator]
    reference_corpus = annotator_to_single_annotator_corpus[reference_annotator]
    logger.info(
        f"Prediction annotator {prediction_annotator} reference annotator {reference_annotator}"
    )
    score_corpus(
        prediction_annotator=prediction_annotator,
        reference_annotator=reference_annotator,
        prediction_corpus=prediction_corpus,
        reference_corpus=reference_corpus,
        exclusion_ids=exclusion_ids,
        overlap=overlap,
        adjudicate=adjudicate,
        per_document=per_document,
        filter_agreements=filter_agreements,
        output_dir=output_dir,
    )
    if both_ways:
        logger.info(
            f"Prediction annotator {reference_annotator} reference annotator {prediction_annotator}"
        )
        score_corpus(
            prediction_annotator=reference_annotator,
            reference_annotator=prediction_annotator,
            prediction_corpus=reference_corpus,
            reference_corpus=prediction_corpus,
            exclusion_ids=exclusion_ids,
            overlap=overlap,
            adjudicate=False,
            per_document=per_document,
            filter_agreements=filter_agreements,
            output_dir=output_dir,
        )


def main() -> None:
    args = parser.parse_args()
    score_corpus_all_annnotators(
        corpus_json=args.corpus_json,
        annotator_ids_tsv=args.annotator_ids_tsv,
        annotator_ids_to_ignore=args.annotator_ids_to_ignore,
        exclude_ids=args.exclude_ids,
        overlap=args.overlap,
        adjudicate=args.adjudicate,
        filter_agreements=args.filter_agreements,
        both_ways=args.both_ways,
        per_document=args.per_document,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
