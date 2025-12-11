from lseval.datatypes import Entity, Relation, SingleAnnotatorCorpus, AnnotatedFile
from more_itertools import partition
from lseval.correctness_matrix import CorrectnessMatrix
from lseval.score import (
    build_entity_correctness_matrix,
    build_relation_correctness_matrix,
)
import logging
from .rt_ctae import (
    AnnotatedFileScores,
    RTEntity,
    AdverseEventEntity,
    CausalRelation,
    cuis_are_radiation_treatment,
    cuis_are_adverse_event,
)


logger = logging.getLogger(__name__)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


def is_rt_entity(entity: Entity) -> bool:
    return cuis_are_radiation_treatment(set(entity.cuis))


def is_adverse_event_entity(entity: Entity) -> bool:
    return cuis_are_adverse_event(set(entity.cuis))


def get_rt_entity_correctness_matrix(
    prediction_file: AnnotatedFile,
    reference_file: AnnotatedFile,
    overlap: bool,
) -> CorrectnessMatrix[RTEntity]:
    if prediction_file.file_id != reference_file.file_id:
        ValueError(
            f"Mismatched file IDs, predicted {prediction_file.file_id} - reference {reference_file.file_id}"
        )
    prediction_rt_entities = {
        RTEntity(
            file_id=prediction_file.file_id,
            span=entity.span,
            dtr=entity.dtr,
            cuis=entity.cuis,
            text=entity.text,
        )
        for entity in prediction_file.entities
        if is_rt_entity(entity)
    }
    reference_rt_entities = {
        RTEntity(
            file_id=reference_file.file_id,
            span=entity.span,
            dtr=entity.dtr,
            cuis=entity.cuis,
            text=entity.text,
        )
        for entity in reference_file.entities
        if is_rt_entity(entity)
    }
    return build_entity_correctness_matrix(
        predicted_entities=prediction_rt_entities,
        reference_entities=reference_rt_entities,
        overlap=overlap,
    )


def get_adverse_event_entity_correctness_matrix(
    prediction_file: AnnotatedFile,
    reference_file: AnnotatedFile,
    overlap: bool,
) -> CorrectnessMatrix[AdverseEventEntity]:
    if prediction_file.file_id != reference_file.file_id:
        ValueError(
            f"Mismatched file IDs, predicted {prediction_file.file_id} - reference {reference_file.file_id}"
        )
    prediction_adverse_event_entities = {
        AdverseEventEntity(
            file_id=prediction_file.file_id,
            span=entity.span,
            dtr=entity.dtr,
            cuis=entity.cuis,
            text=entity.text,
        )
        for entity in prediction_file.entities
        if is_adverse_event_entity(entity)
    }
    reference_adverse_event_entities = {
        AdverseEventEntity(
            file_id=reference_file.file_id,
            span=entity.span,
            dtr=entity.dtr,
            cuis=entity.cuis,
            text=entity.text,
        )
        for entity in reference_file.entities
        if is_adverse_event_entity(entity)
    }
    return build_entity_correctness_matrix(
        predicted_entities=prediction_adverse_event_entities,
        reference_entities=reference_adverse_event_entities,
        overlap=overlap,
    )


def get_causal_relation_correctness_matrix(
    prediction_file: AnnotatedFile,
    reference_file: AnnotatedFile,
    # rt_entity_correctness_matrix: CorrectnessMatrix[RTEntity],
    # adverse_event_entity_correctness_matrix: CorrectnessMatrix[AdverseEventEntity],
    overlap: bool,
    directed: bool = False,
) -> CorrectnessMatrix[CausalRelation]:
    def is_valid_relation(relation: Relation) -> bool:
        first_adverse_second_rt = is_adverse_event_entity(
            relation.arg1
        ) and is_rt_entity(relation.arg2)
        first_rt_second_adverse = is_rt_entity(
            relation.arg1
        ) and is_adverse_event_entity(relation.arg2)
        return first_adverse_second_rt or first_rt_second_adverse

    def get_valid_relations(annotated_file: AnnotatedFile) -> list[Relation]:
        invalid_relation_iter, valid_relation_iter = partition(
            is_valid_relation, annotated_file.relations
        )
        invalid_relations = list(invalid_relation_iter)
        if len(invalid_relations) > 0:
            logger.info(
                f"File with ID {annotated_file.file_id} has {len(invalid_relations)} invalid relations."
            )
        return list(valid_relation_iter)

    if prediction_file.file_id != reference_file.file_id:
        ValueError(
            f"Mismatched file IDs, predicted {prediction_file.file_id} - reference {reference_file.file_id}"
        )
    valid_prediction_relations = [
        CausalRelation(
            file_id=reference_file.file_id,
            arg1=relation.arg1,
            arg2=relation.arg2,
            label=relation.label,
            directed=directed,
        )
        for relation in get_valid_relations(prediction_file)
    ]
    valid_reference_relations = [
        CausalRelation(
            file_id=prediction_file.file_id,
            arg1=relation.arg1,
            arg2=relation.arg2,
            label=relation.label,
            directed=directed,
        )
        for relation in get_valid_relations(reference_file)
    ]
    return build_relation_correctness_matrix(
        predicted_relations=valid_prediction_relations,
        reference_relations=valid_reference_relations,
        overlap=overlap,
    )


def score_file(
    prediction_file: AnnotatedFile,
    reference_file: AnnotatedFile,
    overlap: bool,
) -> AnnotatedFileScores:
    return AnnotatedFileScores(
        rt_entity_correctness_matrix=get_rt_entity_correctness_matrix(
            prediction_file=prediction_file,
            reference_file=reference_file,
            overlap=overlap,
        ),
        adverse_event_entity_correctness_matrix=get_adverse_event_entity_correctness_matrix(
            prediction_file=prediction_file,
            reference_file=reference_file,
            overlap=overlap,
        ),
        causal_relation_correctness_matrix=get_causal_relation_correctness_matrix(
            prediction_file=prediction_file,
            reference_file=reference_file,
            overlap=overlap,
        ),
    )


def score_corpus(
    prediction_corpus: SingleAnnotatorCorpus,
    reference_corpus: SingleAnnotatorCorpus,
    overlap: bool,
    per_document: bool,
) -> None:
    rt_total_tp = 0
    rt_total_fp = 0
    rt_total_tn = 0
    rt_total_fn = 0
    adverse_total_tp = 0
    adverse_total_fp = 0
    adverse_total_tn = 0
    adverse_total_fn = 0
    relation_total_tp = 0
    relation_total_fp = 0
    relation_total_tn = 0
    relation_total_fn = 0
    file_id_to_prediction_files = {
        annotated_file.file_id: annotated_file
        for annotated_file in prediction_corpus.annotated_files
    }
    file_id_to_reference_files = {
        annotated_file.file_id: annotated_file
        for annotated_file in reference_corpus.annotated_files
    }
    for file_id, prediction_file in file_id_to_prediction_files.items():
        reference_file = file_id_to_reference_files.get(
            file_id, AnnotatedFile(file_id=file_id, entities=frozenset(), relations=frozenset())
        )
        annotated_file_scores = score_file(
            prediction_file=prediction_file,
            reference_file=reference_file,
            overlap=overlap,
        )

        rt_total_tp += len(
            annotated_file_scores.rt_entity_correctness_matrix.true_positives
        )
        rt_total_fp += len(
            annotated_file_scores.rt_entity_correctness_matrix.false_positives
        )
        rt_total_tn += 0
        rt_total_fn += len(
            annotated_file_scores.rt_entity_correctness_matrix.false_negatives
        )
        adverse_total_tp += len(
            annotated_file_scores.adverse_event_entity_correctness_matrix.true_positives
        )
        adverse_total_fp += len(
            annotated_file_scores.adverse_event_entity_correctness_matrix.false_positives
        )
        adverse_total_tn += 0
        adverse_total_fn += len(
            annotated_file_scores.adverse_event_entity_correctness_matrix.false_negatives
        )
        relation_total_tp += len(
            annotated_file_scores.causal_relation_correctness_matrix.true_positives
        )
        relation_total_fp += len(
            annotated_file_scores.causal_relation_correctness_matrix.false_positives
        )
        relation_total_tn += 0
        relation_total_fn += len(
            annotated_file_scores.causal_relation_correctness_matrix.false_negatives
        )
        if per_document:
            print(f"File {file_id} scores:")
            print_metrics(annotated_file_scores)


def print_metrics(annotated_files_cores: AnnotatedFileScores) -> None:
    print(
        f"RT Entities Precision:     \t{annotated_files_cores.rt_entity_correctness_matrix.get_precision()}"
    )
    print(
        f"RT Entities Recall:        \t{annotated_files_cores.rt_entity_correctness_matrix.get_recall()}"
    )
    print(
        f"RT Entities F1:            \t{annotated_files_cores.rt_entity_correctness_matrix.get_f1()}"
    )
    print(
        f"RT Entities Precision:     \t{annotated_files_cores.rt_entity_correctness_matrix.get_precision()}"
    )
    print(
        f"RT Entities Recall:        \t{annotated_files_cores.rt_entity_correctness_matrix.get_recall()}"
    )
    print(
        f"RT Entities F1:            \t{annotated_files_cores.rt_entity_correctness_matrix.get_f1()}"
    )
    print(
        f"Causal Relations Precision:\t{annotated_files_cores.causal_relation_correctness_matrix.get_precision()}"
    )
    print(
        f"Causal Relations Recall:   \t{annotated_files_cores.causal_relation_correctness_matrix.get_recall()}"
    )
    print(
        f"Causal Relations F1:       \t{annotated_files_cores.causal_relation_correctness_matrix.get_f1()}"
    )
