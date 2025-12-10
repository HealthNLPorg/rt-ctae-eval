from lseval.datatypes import Entity, SingleAnnotatorCorpus, AnnotatedFile
from lseval.correctness_matrix import CorrectnessMatrix
from lseval.score import (
    build_entity_correctness_matrix,
)
from .rt_ctae import (
    AnnotatedFileScores,
    RTEntity,
    AdverseEventEntity,
    CausalRelation,
    cuis_are_radiation_treatment,
    cuis_are_adverse_event,
)


def get_rt_entity_correctness_matrix(
    prediction_file: AnnotatedFile,
    reference_file: AnnotatedFile,
    overlap: bool,
) -> CorrectnessMatrix[RTEntity]:
    def is_rt_entity(entity: Entity) -> bool:
        return cuis_are_radiation_treatment(entity.cuis)

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
    per_document: bool,
) -> CorrectnessMatrix[AdverseEventEntity]:
    def is_adverse_event_entity(entity: Entity) -> bool:
        return cuis_are_adverse_event(entity.cuis)

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
    rt_entity_correctness_matrix: CorrectnessMatrix[RTEntity],
    adverse_event_entity_correctness_matrix: CorrectnessMatrix[AdverseEventEntity],
    overlap: bool,
    per_document: bool,
) -> CorrectnessMatrix[CausalRelation]:
    pass


def score_file(
    prediction_file: AnnotatedFile,
    reference_file: AnnotatedFile,
    overlap: bool,
    per_document: bool,
) -> AnnotatedFileScores:
    pass


def score_corpus(
    prediction_corpus: SingleAnnotatorCorpus,
    reference_corpus: SingleAnnotatorCorpus,
    overlap: bool,
    per_document: bool,
) -> None:
    pass
