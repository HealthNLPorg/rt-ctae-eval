from operator import attrgetter
from lseval.datatypes import Entity, Relation, SingleAnnotatorCorpus, AnnotatedFile
from more_itertools import partition
from lseval.correctness_matrix import CorrectnessMatrix
from lseval.score import (
    build_entity_correctness_matrix,
    build_relation_correctness_matrix,
)
import logging
from typing import Mapping
from itertools import groupby
from .rt_ctae import (
    AnnotatedFileScores,
    EventType,
    RTEntity,
    AdverseEventEntity,
    CausalRelation,
)


logger = logging.getLogger(__name__)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


def get_causal_relation_correctness_matrix(
    prediction_file: AnnotatedFile,
    reference_file: AnnotatedFile,
    overlap: bool,
    directed: bool = False,
) -> CorrectnessMatrix[CausalRelation]:
    def is_valid_relation(relation: Relation) -> bool:
        first_adverse_second_rt = isinstance(
            relation.arg1, AdverseEventEntity
        ) and isinstance(relation.arg2, RTEntity)
        first_rt_second_adverse = isinstance(relation.arg1, RTEntity) and isinstance(
            relation.arg2, AdverseEventEntity
        )
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
            logger.error(f"{invalid_relations}")
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
            source_annotations=relation.source_annotations,
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
            source_annotations=relation.source_annotations,
        )
        for relation in get_valid_relations(reference_file)
    ]
    return build_relation_correctness_matrix(
        predicted_relations=valid_prediction_relations,
        reference_relations=valid_reference_relations,
        overlap=overlap,
    )


def to_rt_entity(entity: Entity, file_id: int) -> RTEntity:
    return RTEntity(
        file_id=file_id,
        span=entity.span,
        text=entity.text,
        dtr=entity.dtr,
        label=entity.label,
        cuis=entity.cuis,
        source_annotations=entity.source_annotations,
    )


def to_adverse_event_entity(entity: Entity, file_id: int) -> AdverseEventEntity:
    return AdverseEventEntity(
        file_id=file_id,
        span=entity.span,
        text=entity.text,
        dtr=entity.dtr,
        label=entity.label,
        cuis=entity.cuis,
        source_annotations=entity.source_annotations,
    )


def parse_entities(annotated_file: AnnotatedFile) -> Mapping[EventType, set[Entity]]:
    get_label = attrgetter("label")
    event_type_to_instances = {}
    for label, entities in groupby(
        sorted(annotated_file.entities, key=get_label), key=get_label
    ):
        event_type = EventType(label)
        match event_type:
            case EventType.RTEntity:
                event_type_to_instances[EventType.RTEntity] = {
                    to_rt_entity(entity, annotated_file.file_id) for entity in entities
                }
            case EventType.AdverseEventEntity:
                event_type_to_instances[EventType.AdverseEventEntity] = {
                    to_adverse_event_entity(entity, annotated_file.file_id)
                    for entity in entities
                }
            case _:
                ValueError(f"Unsupported event type {event_type} for label {label}")
    return event_type_to_instances


def score_file(
    prediction_file: AnnotatedFile,
    reference_file: AnnotatedFile,
    overlap: bool,
) -> AnnotatedFileScores:
    if prediction_file.file_id != reference_file.file_id:
        ValueError(
            f"Mismatched file IDs, predicted {prediction_file.file_id} - reference {reference_file.file_id}"
        )

    event_type_to_prediction_entities = parse_entities(prediction_file)
    event_type_to_reference_entities = parse_entities(reference_file)
    return AnnotatedFileScores(
        rt_entity_correctness_matrix=build_entity_correctness_matrix(
            predicted_entities=event_type_to_prediction_entities[EventType.RTEntity],
            reference_entities=event_type_to_reference_entities[EventType.RTEntity],
            overlap=overlap,
        ),
        adverse_event_entity_correctness_matrix=build_entity_correctness_matrix(
            predicted_entities=event_type_to_prediction_entities[
                EventType.AdverseEventEntity
            ],
            reference_entities=event_type_to_reference_entities[
                EventType.AdverseEventEntity
            ],
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
            # AnnotatedFile(file_id=file_id, entities=frozenset(), relations=frozenset()),
            None,
        )
        prediction_file = file_id_to_prediction_files.get(
            file_id,
            # AnnotatedFile(file_id=file_id, entities=frozenset(), relations=frozenset()),
            None,
        )
        if reference_file is None or prediction_file is None:
            logger.error(f"Missing annotations for {file_id}")
            logger.error(
                f"Reference file {'present' if reference_file is not None else 'absent'}, Prediction file {'present' if prediction_file is not None else 'absent'}"
            )
            continue
        annotated_file_scores = score_file(
            prediction_file=prediction_file,
            reference_file=reference_file,
            overlap=overlap,
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
