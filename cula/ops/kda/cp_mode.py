# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Intracard context-parallel dispatch vocabulary shared by the KDA backends."""

from __future__ import annotations

from enum import Enum


class NotSplittableError(ValueError):
    """Intracard CP cannot split the given shape. Subclasses ValueError so
    existing ``except ValueError`` callers keep working."""


class CPMode(Enum):
    """OFF = serial path; AUTO = engage when profitable; FORCE = engage
    whenever splittable, else NotSplittableError."""

    OFF = "off"
    AUTO = "auto"
    FORCE = "force"

    @classmethod
    def parse(cls, use_intracard_cp, use_cp=None) -> CPMode | None:
        """Public use_intracard_cp value ("auto"/True/False; use_cp is a
        deprecated alias) -> mode. None stays None (backend default)."""
        if use_intracard_cp is not None and use_cp is not None:
            raise TypeError("Pass only one of use_intracard_cp or use_cp.")
        value = use_intracard_cp if use_intracard_cp is not None else use_cp
        if value is None or isinstance(value, cls):
            return value
        # Identity checks (not ==): `1 == True` would match stray ints.
        if value is True:
            return cls.FORCE
        if value is False:
            return cls.OFF
        if value == "auto":
            return cls.AUTO
        raise ValueError(f'use_intracard_cp must be "auto", True, or False, got {value!r}')
