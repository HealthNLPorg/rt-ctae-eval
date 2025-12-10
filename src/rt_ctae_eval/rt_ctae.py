from dataclasses import dataclass

from lseval.correctness_matrix import CorrectnessMatrix
from typing import Any

from lseval.datatypes import Entity, Relation
from .meta import RT_CUI, NaranjoScale


def cuis_are_radiation_treatment(cuis: set[str]) -> bool:
    return RT_CUI in cuis and len(cuis) == 1


def cuis_are_adverse_event(cuis: set[str]) -> bool:
    return RT_CUI not in cuis


@dataclass
class RTEntity(Entity):
    def __post_init__(self):
        if not cuis_are_radiation_treatment(self.cuis):
            ValueError(f"{self} is not a proper RT entity")


@dataclass
class AdverseEventEntity(Entity):
    def __post_init__(self):
        if not cuis_are_adverse_event(self.cuis):
            ValueError(f"{self} is not a proper RT entity")


@dataclass
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
    rt_entity_correctness_matrix: CorrectnessMatrix[RTEntity]
    adverse_event_entity_correctness_matrix: CorrectnessMatrix[AdverseEventEntity]
    causal_relation_correctness_matrix: CorrectnessMatrix[CausalRelation]
