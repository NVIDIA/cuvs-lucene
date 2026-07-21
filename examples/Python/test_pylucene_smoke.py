# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from pylucene_smoke import SELECTED_CASES, initialize_pylucene_context, run_case


@pytest.fixture(scope="session")
def pylucene_context(request):
    context = initialize_pylucene_context()
    yield context

    if request.session.testsfailed == 0:
        print(
            "PyLucene smoke suite passed: "
            f"loaded {context.cuvs_lucene_jar.name}, used {context.cuvs_java_jar.name}, "
            f"ran {len(SELECTED_CASES)} case(s)"
        )


@pytest.mark.parametrize("case", SELECTED_CASES, ids=lambda case: case.name)
def test_pylucene_smoke_case(pylucene_context, case):
    run_case(
        case,
        pylucene_context.codec_class,
        pylucene_context.codec_cache,
        pylucene_context.jarray,
    )
