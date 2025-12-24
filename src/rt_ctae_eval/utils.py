from operator import attrgetter
from lseval.datatypes import Entity, Relation, SingleAnnotatorCorpus, AnnotatedFile
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


def update_correctness_matrix(
    needs_updates: CorrectnessMatrix, has_updates: CorrectnessMatrix
) -> CorrectnessMatrix:
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


def score_corpus(
    prediction_corpus: SingleAnnotatorCorpus,
    reference_corpus: SingleAnnotatorCorpus,
    overlap: bool,
    per_document: bool,
) -> None:
    annotated_corpus_scores = AnnotatedCorpusScores(
        rt_entity_correctness_matrix=CorrectnessMatrix(),
        adverse_event_entity_correctness_matrix=CorrectnessMatrix(),
        causal_relation_correctness_matrix=CorrectnessMatrix(),
    )
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
            file_id=file_id,
            prediction_file=prediction_file,
            reference_file=reference_file,
            overlap=overlap,
        )
        annotated_corpus_scores = update_corpus_scores(
            annotated_corpus_scores, annotated_file_scores
        )
        if per_document:
            print(f"File {file_id} scores:")
            print_metrics(annotated_file_scores)

    print("Corpus scores:")
    print_metrics(annotated_corpus_scores)


def print_metrics(
    annotated_collection_scores: AnnotatedFileScores | AnnotatedCorpusScores,
) -> None:
    print(
        f"RT Entities Precision:     \t{annotated_collection_scores.rt_entity_correctness_matrix.get_precision()}"
    )
    print(
        f"RT Entities Recall:        \t{annotated_collection_scores.rt_entity_correctness_matrix.get_recall()}"
    )
    print(
        f"RT Entities F1:            \t{annotated_collection_scores.rt_entity_correctness_matrix.get_f1()}"
    )
    print(
        f"Adverse Event Entities Precision:     \t{annotated_collection_scores.adverse_event_entity_correctness_matrix.get_precision()}"
    )
    print(
        f"Adverse Event Entities Recall:        \t{annotated_collection_scores.adverse_event_entity_correctness_matrix.get_recall()}"
    )
    print(
        f"Adverse Event Entities F1:            \t{annotated_collection_scores.adverse_event_entity_correctness_matrix.get_f1()}"
    )
    print(
        f"Causal Relations Precision:\t{annotated_collection_scores.causal_relation_correctness_matrix.get_precision()}"
    )
    print(
        f"Causal Relations Recall:   \t{annotated_collection_scores.causal_relation_correctness_matrix.get_recall()}"
    )
    print(
        f"Causal Relations F1:       \t{annotated_collection_scores.causal_relation_correctness_matrix.get_f1()}"
    )
