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
    return cuis_are_radiation_treatment(entity.cuis)


def is_adverse_event_entity(entity: Entity) -> bool:
    return cuis_are_adverse_event(entity.cuis)


def get_rt_entity_correctness_matrix(
    prediction_file: AnnotatedFile,
    reference_file: AnnotatedFile,
    overlap: bool,
) -> CorrectnessMatrix[RTEntity]:
    prediction_rt_entities = {
        RTEntity(span=entity.span, dtr=entity.dtr, cuis=entity.cuis, text=entity.text)
        for entity in prediction_file.entities
        if is_rt_entity(entity)
    }
    reference_rt_entities = {
        RTEntity(span=entity.span, dtr=entity.dtr, cuis=entity.cuis, text=entity.text)
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
    prediction_adverse_event_entities = {
        AdverseEventEntity(
            span=entity.span, dtr=entity.dtr, cuis=entity.cuis, text=entity.text
        )
        for entity in prediction_file.entities
        if is_adverse_event_entity(entity)
    }
    reference_adverse_event_entities = {
        AdverseEventEntity(
            span=entity.span, dtr=entity.dtr, cuis=entity.cuis, text=entity.text
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

    valid_prediction_relations = [
        CausalRelation(
            arg1=relation.arg1,
            arg2=relation.arg2,
            label=relation.label,
            directed=directed,
        )
        for relation in get_valid_relations(prediction_file)
    ]
    valid_reference_relations = [
        CausalRelation(
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
    pass
