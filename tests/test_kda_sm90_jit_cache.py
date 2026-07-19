# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from cula.ops.kda.sm90 import k1


def test_k1_compile_cache_is_bounded(monkeypatch):
    cache = {}
    compile_calls = []

    def fake_compile(*key):
        compile_calls.append(key)
        return key

    monkeypatch.setattr(k1, "_k1_kernel_cache", cache)
    monkeypatch.setattr(k1, "_compile_k1", fake_compile)

    for head_count in range(k1._K1_KERNEL_CACHE_MAXSIZE + 1):
        k1._get_compiled_k1(head_count, 1.0, -5.0, False)

    assert len(cache) == k1._K1_KERNEL_CACHE_MAXSIZE
    assert (0, 1.0, -5.0, False) not in cache
    assert len(compile_calls) == k1._K1_KERNEL_CACHE_MAXSIZE + 1

    k1._get_compiled_k1(k1._K1_KERNEL_CACHE_MAXSIZE, 1.0, -5.0, False)
    assert len(compile_calls) == k1._K1_KERNEL_CACHE_MAXSIZE + 1
