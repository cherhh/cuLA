# KDA Backend Dispatch

## Context

SM90 KDA prefill has two in-tree implementations. FlashKDA provides the preferred CuTeDSL K1+K2 path but does not support every call shape or option. The fully-fused CUDA C++ path supports GVA and remains the fallback. Neither backend implements backward.

## API

`cula.kda.kda_prefill` is the runtime-selected entry point. It runs outside `torch.compile` graphs and requires grad-disabled execution. Existing explicit exports remain available for callers that need a fixed implementation.

Backend priority is:

1. `flashkda`
2. `fully_fused`

`CULA_BACKEND_FLASHKDA=0` and `CULA_BACKEND_FULLY_FUSED=0` disable their respective backends.

## Selection Contract

Each backend first probes its runtime dependency, then verifies the complete call contract. Verification covers architecture, shapes, dtypes, execution mode, gate settings, optional buffers, CP controls, and unsupported keyword arguments.

A rejection records its reason and continues to the next backend. If every backend rejects, dispatch raises one error containing all reasons. Exceptions raised by verifier code or by an accepted implementation are not converted into rejections because they indicate implementation defects rather than unsupported calls.

Availability probes and implementation imports remain lazy. The registry contains no numerical fallback because both implementations live in this repository and unsupported calls should fail explicitly.

## Validation

CPU tests cover registry ordering, environment controls, verifier acceptance and rejection, exception propagation, and FlashKDA-to-fully-fused fallback. SM90 tests compare each selected backend with its explicit entry point. The fat-wheel build verifies that SM90 and SM100/SM103 bindings use distinct object files.
