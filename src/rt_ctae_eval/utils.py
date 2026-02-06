from collections import defaultdict
import operator
from lseval.datatypes import Entity, Relation, AnnotatedFile, DocTimeRel, overlap_match
from functools import partial
from lseval.correctness_matrix import CorrectnessMatrix, Correctness
from lseval.score import (
    build_entity_correctness_matrix,
    build_relation_correctness_matrix,
)
import logging
from typing import Mapping, cast
from itertools import chain
from collections.abc import Iterable, Set, Collection
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
) -> Mapping[DocTimeRel, Collection[Entity]]:
    updated_mapping = {}
    for entity in annotated_file.entities:
        dtr = entity.dtr
        if dtr not in updated_mapping.keys():
            updated_mapping[dtr] = set()
        else:
            updated_mapping[dtr].add(entity)
    return updated_mapping


def build_category_correctness_matrices[T](
    predicted_category_entities: Mapping[T, Collection[Entity]],
    reference_category_entities: Mapping[T, Collection[Entity]],
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
) -> Mapping[str, Collection[Entity]]:
    cui_to_entity = {}
    for entity in annotated_file.entities:
        for cui in entity.cuis:
            if cui not in cui_to_entity.keys():
                cui_to_entity[cui] = set()
            else:
                cui_to_entity[cui].add(entity)
    return cui_to_entity


def parse_entities(
    annotated_file: AnnotatedFile,
) -> Mapping[EventType, Collection[Entity]]:
    event_type_to_instances = defaultdict(set)
    for entity in annotated_file.entities:
        label = getattr(entity, "label")
        event_type = EventType(label)
        match event_type:
            case EventType.RTEntity:
                event_type_to_instances[EventType.RTEntity].add(
                    to_rt_entity(entity, annotated_file.file_id)
                )
            case EventType.AdverseEventEntity:
                event_type_to_instances[EventType.AdverseEventEntity].add(
                    to_adverse_event_entity(entity, annotated_file.file_id)
                )
            case _:
                raise ValueError(
                    f"Unsupported event type {event_type} for label {label}"
                )
    return event_type_to_instances


def entity_match(overlap: bool, entity1: Entity, entity2: Entity) -> bool:
    # naive bc type sensitivity should be handled elsewhere
    return (
        overlap_match(entity1.span, entity2.span)
        if overlap
        else entity1.span == entity2.span
    )


def recoordinate_relation_argument(
    entity: Entity, correctness_matrix: CorrectnessMatrix[Entity], overlap: bool
) -> Entity:
    local_entity_match = partial(entity_match, overlap, entity)
    true_positive_possibility = next(
        filter(local_entity_match, correctness_matrix.true_positives), None
    )
    if true_positive_possibility is not None:
        return true_positive_possibility
    false_positive_possibility = next(
        filter(local_entity_match, correctness_matrix.false_positives), None
    )
    if false_positive_possibility is not None:
        return false_positive_possibility
    raise ValueError(
        "No options for either relation argument represented in correctness matrix."
    )


def recoordinate_false_negative_relations(
    false_negative_relations: Iterable[Relation],
    entity_to_correctness_matrix: Mapping[Entity, CorrectnessMatrix[Entity]],
    overlap: bool,
) -> Iterable[Relation]:
    for relation in false_negative_relations:
        arg1_correctness_matrix = entity_to_correctness_matrix[relation.arg1]
        arg1_correctness = arg1_correctness_matrix.get_correctness(relation.arg1)
        arg2_correctness_matrix = entity_to_correctness_matrix[relation.arg2]
        arg2_correctness = arg2_correctness_matrix.get_correctness(relation.arg2)
        match arg1_correctness, arg2_correctness:
            case Correctness.FALSE_NEGATIVE, Correctness.FALSE_NEGATIVE:
                # Everything is all set
                yield relation
            case Correctness.FALSE_NEGATIVE, Correctness.NA:
                yield CausalRelation(
                    file_id=relation.file_id,
                    arg1=relation.arg1,
                    arg2=recoordinate_relation_argument(
                        relation.arg2, arg2_correctness_matrix, overlap
                    ),
                    label=relation.label,
                    source_annotations=relation.source_annotations,
                )
            case Correctness.NA, Correctness.FALSE_NEGATIVE:
                yield CausalRelation(
                    file_id=relation.file_id,
                    arg1=recoordinate_relation_argument(
                        relation.arg1, arg1_correctness_matrix, overlap
                    ),
                    arg2=relation.arg2,
                    label=relation.label,
                    source_annotations=relation.source_annotations,
                )
            case Correctness.NA, Correctness.NA:
                yield CausalRelation(
                    file_id=relation.file_id,
                    arg1=recoordinate_relation_argument(
                        relation.arg1, arg1_correctness_matrix, overlap
                    ),
                    arg2=recoordinate_relation_argument(
                        relation.arg2, arg2_correctness_matrix, overlap
                    ),
                    label=relation.label,
                    source_annotations=relation.source_annotations,
                )
            case _:
                raise ValueError(
                    "Arguments to a false negative (reference) relations are definitionally either false negatives (reference) themselves, or are true/false positive but are represented in the correctness matrix by a predicted entity"
                )


def recoordinate_causal_relation(
    label_studio_id_to_entity: Mapping[str, Entity],
    relation: Relation,
) -> CausalRelation | None:
    updated_arg1 = label_studio_id_to_entity.get(relation.arg1.label_studio_id)
    updated_arg2 = label_studio_id_to_entity.get(relation.arg2.label_studio_id)
    if updated_arg1 is None or updated_arg2 is None:
        raise ValueError(
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
    if not is_valid_relation(updated_relation):
        logger.warning(f"Relation is out of order {updated_relation}")
    return updated_relation


def recoordinate_causal_relations(
    annotated_file: AnnotatedFile, updated_entities: Iterable[Entity]
) -> Set[Relation]:
    id_to_entity = {}
    for entity in updated_entities:
        label_studio_id = getattr(entity, "label_studio_id")
        if label_studio_id is None:
            raise ValueError(f"Missing Label Studio ID for entity {entity}")
        stored = id_to_entity.get(label_studio_id)
        if stored is not None:
            raise ValueError(
                f"Duplicate Label Studio IDs for entities {entity} and {stored}"
            )
        id_to_entity[str(label_studio_id)] = entity
    local_recoordinate = partial(recoordinate_causal_relation, id_to_entity)
    return set(
        filter(operator.is_not_none, map(local_recoordinate, annotated_file.relations))
    )


def score_file(
    file_id: int,
    prediction_file: AnnotatedFile,
    reference_file: AnnotatedFile,
    overlap: bool,
) -> AnnotatedFileScores:
    if prediction_file.file_id != reference_file.file_id:
        raise ValueError(
            f"Mismatched file IDs, predicted {prediction_file.file_id} - reference {reference_file.file_id}"
        )

    event_type_to_prediction_entities = parse_entities(prediction_file)
    event_type_to_reference_entities = parse_entities(reference_file)
    rt_entity_correctness_matrix = build_entity_correctness_matrix(
        predicted_entities=event_type_to_prediction_entities.get(
            EventType.RTEntity, set()
        ),
        reference_entities=event_type_to_reference_entities.get(
            EventType.RTEntity, set()
        ),
        overlap=overlap,
    )
    adverse_event_entity_correctness_matrix = build_entity_correctness_matrix(
        predicted_entities=event_type_to_prediction_entities.get(
            EventType.AdverseEventEntity, set()
        ),
        reference_entities=event_type_to_reference_entities.get(
            EventType.AdverseEventEntity, set()
        ),
        overlap=overlap,
    )
    entity_to_correctness_matrix = {}
    for event_type, entities in chain(
        event_type_to_prediction_entities.items(),
        event_type_to_reference_entities.items(),
    ):
        match event_type:
            case EventType.RTEntity:
                for entity in entities:
                    entity_to_correctness_matrix[entity] = rt_entity_correctness_matrix
            case EventType.AdverseEventEntity:
                for entity in entities:
                    entity_to_correctness_matrix[entity] = (
                        adverse_event_entity_correctness_matrix
                    )
    raw_relation_correctness_matrix = build_relation_correctness_matrix(
        predicted_relations=recoordinate_causal_relations(
            prediction_file,
            chain.from_iterable(event_type_to_prediction_entities.values()),
        ),
        reference_relations=recoordinate_causal_relations(
            reference_file,
            chain.from_iterable(event_type_to_reference_entities.values()),
        ),
        overlap=overlap,
    )
    causal_relation_correctness_matrix = CorrectnessMatrix(
        true_positives=cast(
            set[CausalRelation], raw_relation_correctness_matrix.true_positives
        ),
        false_positives=cast(
            set[CausalRelation], raw_relation_correctness_matrix.false_positives
        ),
        false_negatives=cast(
            set[CausalRelation],
            set(
                recoordinate_false_negative_relations(
                    false_negative_relations=raw_relation_correctness_matrix.false_negatives,
                    entity_to_correctness_matrix=entity_to_correctness_matrix,
                    overlap=overlap,
                )
            ),
        ),
    )
    return AnnotatedFileScores(
        file_id=file_id,
        rt_entity_correctness_matrix=rt_entity_correctness_matrix,
        adverse_event_entity_correctness_matrix=adverse_event_entity_correctness_matrix,
        causal_relation_correctness_matrix=causal_relation_correctness_matrix,
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
        raise ValueError(f"Set has {difference} non-unique entries.")
    return needs_updates


def update_correctness_matrix[T](
    needs_updates: CorrectnessMatrix[T], has_updates: CorrectnessMatrix[T]
) -> CorrectnessMatrix[T]:
    needs_updates.true_negatives = warned_set_update(
        set(needs_updates.true_negatives), set(has_updates.true_negatives)
    )
    needs_updates.true_positives = warned_set_update(
        set(needs_updates.true_positives), set(has_updates.true_positives)
    )
    needs_updates.false_negatives = warned_set_update(
        set(needs_updates.false_negatives), set(has_updates.false_negatives)
    )
    needs_updates.false_positives = warned_set_update(
        set(needs_updates.false_positives), set(has_updates.false_positives)
    )
    return needs_updates


def update_category_correctness_matrices[S, T](
    needs_updates: Mapping[S, CorrectnessMatrix[T]],
    has_updates: Mapping[S, CorrectnessMatrix[T]],
) -> Mapping[S, CorrectnessMatrix[T]]:
    updated_mapping = {}
    for category in needs_updates.keys() | has_updates.keys():
        updated_mapping[category] = update_correctness_matrix(
            needs_updates.get(category, CorrectnessMatrix()),
            has_updates.get(category, CorrectnessMatrix()),
        )
    return updated_mapping


def merge_correctness_totals[T](
    arg1: Mapping[T, int],
    arg2: Mapping[T, int],
) -> Mapping[T, int]:
    updated_mapping = {}
    for correctness in arg1.keys() | arg2.keys():
        updated_mapping[correctness] = arg1.get(correctness, 0) + arg2.get(
            correctness, 0
        )
    return updated_mapping


def update_category_to_correctness_totals[S, T](
    needs_updates: Mapping[S, Mapping[T, int]],
    has_updates: Mapping[S, Mapping[T, int]],
) -> Mapping[S, Mapping[T, int]]:
    updated_mapping = {}
    for category in needs_updates.keys() | has_updates.keys():
        updated_mapping[category] = {}
        needs_updates_correctness_counts = needs_updates.get(category, {})
        has_updates_correctness_counts = has_updates.get(category, {})
        updated_mapping[category] = merge_correctness_totals(
            needs_updates_correctness_counts, has_updates_correctness_counts
        )
    return updated_mapping


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

    corpus_scores.dtr_correctness_matrices = update_category_correctness_matrices(
        corpus_scores.dtr_correctness_matrices,
        file_scores.dtr_correctness_matrices,
    )

    corpus_scores.cui_correctness_totals = update_category_to_correctness_totals(
        corpus_scores.cui_correctness_totals,
        file_scores.cui_correctness_totals,
    )
    return corpus_scores
