from collections.abc import Mapping
from dataclasses import dataclass


from enum import Enum
from lseval.correctness_matrix import CorrectnessMatrix, Correctness
from typing import Any

from lseval.datatypes import Entity, Relation, DocTimeRel

RT_CUI = "C1522449"


def cuis_are_radiation_treatment(cuis: set[str]) -> bool:
    return RT_CUI in cuis and len(cuis) == 1


def cuis_are_adverse_event(cuis: set[str]) -> bool:
    return RT_CUI not in cuis


class NaranjoScale(Enum):
    DOUBTFUL = "Doubtful"
    POSSIBLE = "Possible"
    PROBABLE = "Probable"
    CERTAIN = "Certain"
    NA = "N/A"

    @classmethod
    def _missing_(cls, value):
        return NaranjoScale.NA


class EventType(Enum):
    RTEntity = "Radiotherapy Treatment"
    AdverseEventEntity = "Adverse Event"
    NA = "N/A"

    @classmethod
    def _missing_(cls, value):
        return EventType.NA


@dataclass(eq=True, frozen=True)
class RTEntity(Entity):
    def __post_init__(self):
        if not cuis_are_radiation_treatment(set(self.cuis)):
            ValueError(f"{self} is not a proper RT entity")


@dataclass(eq=True, frozen=True)
class AdverseEventEntity(Entity):
    def __post_init__(self):
        if not cuis_are_adverse_event(set(self.cuis)):
            ValueError(f"{self} is not a proper RT entity")


@dataclass(eq=True, frozen=True)
class CausalRelation(Relation):
    def __post_init__(self):
        if not isinstance(self.arg1, RTEntity):
            ValueError(
                f"{self.arg1} is not a radiation treatment - convention is radiation treatment is the anchor"
            )
        if not isinstance(self.arg2, AdverseEventEntity):
            ValueError(
                f"{self.arg2} is not a adverse event - convention is adverse event is the anchor"
            )
        if not CausalRelation.validate_naranjo_label(self.label):
            ValueError(f"Invalid causality label {self.label}")

    @staticmethod
    def validate_naranjo_label(label: Any) -> bool:
        return label in NaranjoScale


@dataclass
class AnnotatedFileScores:
    file_id: int
    rt_entity_correctness_matrix: CorrectnessMatrix[RTEntity]
    adverse_event_entity_correctness_matrix: CorrectnessMatrix[AdverseEventEntity]
    causal_relation_correctness_matrix: CorrectnessMatrix[CausalRelation]
    dtr_correctness_matrices: Mapping[DocTimeRel, CorrectnessMatrix[Entity]] = {}
    cui_correctness_totals: Mapping[str, Mapping[Correctness, int]] = {}


@dataclass
class AnnotatedCorpusScores:
    rt_entity_correctness_matrix: CorrectnessMatrix[RTEntity]
    adverse_event_entity_correctness_matrix: CorrectnessMatrix[AdverseEventEntity]
    causal_relation_correctness_matrix: CorrectnessMatrix[CausalRelation]
    dtr_correctness_matrices: Mapping[DocTimeRel, CorrectnessMatrix[Entity]] = {}
    cui_correctness_totals: Mapping[str, Mapping[Correctness, int]] = {}
