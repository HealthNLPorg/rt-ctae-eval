from operator import attrgetter
from lseval.datatypes import Entity, Relation, AnnotatedFile, DocTimeRel
from functools import partial
from lseval.correctness_matrix import CorrectnessMatrix
from lseval.score import (
    build_entity_correctness_matrix,
    build_relation_correctness_matrix,
)
import logging
from typing import Mapping
from itertools import groupby, chain
from collections.abc import Iterable
from .rt_ctae import (
    AnnotatedFileScores,
    AnnotatedCorpusScores,
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


def is_valid_relation(relation: Relation) -> bool:
    first_adverse_second_rt = isinstance(
        relation.arg1, AdverseEventEntity
    ) and isinstance(relation.arg2, RTEntity)
    first_rt_second_adverse = isinstance(relation.arg1, RTEntity) and isinstance(
        relation.arg2, AdverseEventEntity
    )
    return first_adverse_second_rt or first_rt_second_adverse


def to_rt_entity(entity: Entity, file_id: int) -> RTEntity:
    return RTEntity(
        file_id=file_id,
        label_studio_id=entity.label_studio_id,
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
        label_studio_id=entity.label_studio_id,
        span=entity.span,
        text=entity.text,
        dtr=entity.dtr,
        label=entity.label,
        cuis=entity.cuis,
        source_annotations=entity.source_annotations,
    )


def parse_dtr_entities(
    annotated_file: AnnotatedFile,
) -> Mapping[DocTimeRel, set[Entity]]:
    get_dtr = attrgetter("dtr")

    return {
        dtr: set(entities)
        for dtr, entities in groupby(
            sorted(annotated_file.entities, key=get_dtr), key=get_dtr
        )
    }


def build_category_correctness_matrices[T](
    predicted_category_entities: Mapping[T, set[Entity]],
    reference_category_entities: Mapping[T, set[Entity]],
    overlap: bool,
) -> Mapping[T, CorrectnessMatrix[Entity]]:
    category_correctness_matrices = {}
    for category in (
        predicted_category_entities.keys() | reference_category_entities.keys()
    ):
        category_correctness_matrices[category] = build_entity_correctness_matrix(
            predicted_entities=predicted_category_entities.get(category, set()),
            reference_entities=reference_category_entities.get(category, set()),
            overlap=overlap,
        )
    return category_correctness_matrices


def parse_cui_entities(
    annotated_file: AnnotatedFile,
) -> Mapping[str, set[Entity]]:
    cui_to_entity = {}
    for entity in annotated_file.entities:
        for cui in entity.cuis:
            if cui not in cui_to_entity:
                cui_to_entity[cui] = set()
            else:
                cui_to_entity[cui].add(entity)
    return cui_to_entity


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


def recoordinate_causal_relation(
    label_studio_id_to_entity: Mapping[str, Entity],
    relation: Relation,
) -> CausalRelation | None:
    updated_arg1 = label_studio_id_to_entity.get(relation.arg1.label_studio_id)
    updated_arg2 = label_studio_id_to_entity.get(relation.arg2.label_studio_id)
    if updated_arg1 is None or updated_arg2 is None:
        ValueError(
            f"Missing ID mapping information in {label_studio_id_to_entity.keys()} for one of {relation.arg1.label_studio_id} {relation.arg2.label_studio_id}"
        )
        return None
    updated_relation = CausalRelation(
        file_id=relation.file_id,
        arg1=updated_arg1,
        arg2=updated_arg2,
        label=relation.label,
        source_annotations=relation.source_annotations,
    )
    assert is_valid_relation(updated_relation)
    return updated_relation


def recoordinate_causal_relations(
    annotated_file: AnnotatedFile, updated_entities: Iterable[Entity]
) -> list[Relation]:
    id_to_entity = {}
    for entity in updated_entities:
        label_studio_id = getattr(entity, "label_studio_id")
        if label_studio_id is None:
            ValueError(f"Missing Label Studio ID for entity {entity}")
        stored = id_to_entity.get(label_studio_id)
        if stored is not None:
            ValueError(f"Duplicate Label Studio IDs for entities {entity} and {stored}")
        id_to_entity[str(label_studio_id)] = entity
    local_recoordinate = partial(recoordinate_causal_relation, id_to_entity)
    return list(filter(None, map(local_recoordinate, annotated_file.relations)))


def score_file(
    file_id: int,
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
        file_id=file_id,
        rt_entity_correctness_matrix=build_entity_correctness_matrix(
            predicted_entities=event_type_to_prediction_entities.get(
                EventType.RTEntity, set()
            ),
            reference_entities=event_type_to_reference_entities.get(
                EventType.RTEntity, set()
            ),
            overlap=overlap,
        ),
        adverse_event_entity_correctness_matrix=build_entity_correctness_matrix(
            predicted_entities=event_type_to_prediction_entities.get(
                EventType.AdverseEventEntity, set()
            ),
            reference_entities=event_type_to_reference_entities.get(
                EventType.AdverseEventEntity, set()
            ),
            overlap=overlap,
        ),
        causal_relation_correctness_matrix=build_relation_correctness_matrix(
            predicted_relations=recoordinate_causal_relations(
                prediction_file,
                chain.from_iterable(event_type_to_prediction_entities.values()),
            ),
            reference_relations=recoordinate_causal_relations(
                reference_file,
                chain.from_iterable(event_type_to_reference_entities.values()),
            ),
            overlap=overlap,
        ),
        dtr_correctness_matrices=build_category_correctness_matrices(
            predicted_category_entities=parse_dtr_entities(prediction_file),
            reference_category_entities=parse_dtr_entities(reference_file),
            overlap=overlap,
        ),
        cui_correctness_totals={
            cui: correctness_matrix.to_correctness_totals()
            for cui, correctness_matrix in build_category_correctness_matrices(
                predicted_category_entities=parse_cui_entities(prediction_file),
                reference_category_entities=parse_cui_entities(reference_file),
                overlap=overlap,
            ).items()
        },
    )


def warned_set_update[T](needs_updates: set[T], has_updates: set[T]) -> set[T]:
    initial_total = len(needs_updates) + len(has_updates)
    needs_updates.update(has_updates)
    new_total = len(needs_updates)
    difference = initial_total - new_total
    if difference > 0:
        ValueError(f"Set has {difference} non-unique entries.")
        return set()
    return needs_updates


def update_correctness_matrix[T](
    needs_updates: CorrectnessMatrix[T], has_updates: CorrectnessMatrix[T]
) -> CorrectnessMatrix[T]:
    needs_updates.true_negatives = warned_set_update(
        needs_updates.true_negatives, has_updates.true_negatives
    )
    needs_updates.true_positives = warned_set_update(
        needs_updates.true_positives, has_updates.true_positives
    )
    needs_updates.false_negatives = warned_set_update(
        needs_updates.false_negatives, has_updates.false_negatives
    )
    needs_updates.false_positives = warned_set_update(
        needs_updates.false_positives, has_updates.false_positives
    )
    needs_updates.support += has_updates.support
    return needs_updates


def update_corpus_scores(
    corpus_scores: AnnotatedCorpusScores, file_scores: AnnotatedFileScores
) -> AnnotatedCorpusScores:
    corpus_scores.rt_entity_correctness_matrix = update_correctness_matrix(
        corpus_scores.rt_entity_correctness_matrix,
        file_scores.rt_entity_correctness_matrix,
    )
    corpus_scores.adverse_event_entity_correctness_matrix = update_correctness_matrix(
        corpus_scores.adverse_event_entity_correctness_matrix,
        file_scores.adverse_event_entity_correctness_matrix,
    )
    corpus_scores.causal_relation_correctness_matrix = update_correctness_matrix(
        corpus_scores.causal_relation_correctness_matrix,
        file_scores.causal_relation_correctness_matrix,
    )
    return corpus_scores


def warned_merge[T](arg1: set[T], arg2: set[T]) -> set[T]:
    original_total = len(arg1) + len(arg2)
    merged = arg1 & arg2
    if original_total > len(merged):
        ValueError(
            f"arg1 and arg2 have {original_total - len(merged)} non-unique entries"
        )
        return set()
    return merged


# different from the update based version
def merge_correctness_matrices[T](
    arg1: CorrectnessMatrix[T], arg2: CorrectnessMatrix[T]
) -> CorrectnessMatrix[T]:
    return CorrectnessMatrix(
        true_negatives=warned_merge(arg1.true_negatives, arg2.true_negatives),
        true_positives=warned_merge(arg1.true_positives, arg2.true_positives),
        false_negatives=warned_merge(arg1.false_negatives, arg2.false_negatives),
        false_positives=warned_merge(arg1.false_positives, arg2.false_positives),
    )
