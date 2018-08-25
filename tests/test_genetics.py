#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `prometheus_ml` package."""

import pytest
from prometheus_ml.genetics import Individual


def test_individual_creation():
    """Test the CLI."""
    ind = Individual()
    assert ind.scoring is None
