# Security Policy

`legate-boost` is a gradient-boosted-machine library implemented on top of
[Legate](https://github.com/nv-legate/legate) and
[cuPyNumeric](https://github.com/nv-legate/cupynumeric) — NVIDIA's
distributed Python runtime. It is a Python library with a C++/CUDA task
implementation (`src/`) that the Legate runtime schedules across CPUs and
GPUs in a single process or a distributed job. It is invoked in-process
and inherits the caller's privilege.

Its security posture is shaped by the inputs it ingests — training and
inference arrays, categorical encoders, persisted model objects — and by
the Legate task-graph boundary the C++/CUDA kernels execute beneath.

## Reporting a Vulnerability

Please report security vulnerabilities privately through one of the channels
below. **Do not open a public GitHub issue, PR, or discussion** for a
suspected vulnerability.

1. **NVIDIA Vulnerability Disclosure Program (preferred)**
   <https://www.nvidia.com/en-us/security/>
   Submit through the NVIDIA PSIRT web form. This is the fastest path to
   triage and tracking.

2. **Email NVIDIA PSIRT**
   psirt@nvidia.com — encrypt sensitive reports with the
   [NVIDIA PSIRT PGP key](https://www.nvidia.com/en-us/security/pgp-key).

3. **GitHub Private Vulnerability Reporting**
   Use the **Security** tab on this repository → *Report a vulnerability*.

Please include, where possible:

- Affected component (e.g. a specific model under `src/models/`, the
  target encoder, the Legate mapper, the build's `RAPIDS.cmake` bootstrap)
- legate-boost version, Legate version, cuPyNumeric version, CUDA
  version, and OS
- Reproduction steps and a minimal proof-of-concept input or model
- Impact assessment (memory corruption, code execution, DoS, info
  disclosure)
- Any relevant CWE / CVE identifiers

NVIDIA PSIRT will acknowledge receipt and coordinate triage, fix
development, and coordinated disclosure. More on NVIDIA's response
process: <https://www.nvidia.com/en-us/security/psirt-policies/>.

## Security Architecture & Context

**Classification:** Library (Python frontend with C++/CUDA task
implementations executed by the Legate runtime).

**Primary security responsibility:** Safely ingest tabular / numeric
training and inference inputs, fit boosted ensembles and related models
via Legate tasks, and round-trip model state through pickle without
crashing the host process or corrupting memory.

**Components and trust boundaries:**

- **`legateboost/`** — Python package. scikit-learn-compatible
  estimators (`legateboost.py`), input validation (`input_validation.py`),
  encoders (`encoder.py`), objectives / metrics (`objectives.py`,
  `metrics.py`), Shapley value computation (`shapley.py`), and pickle
  support helpers (`utils.py` — see `__getstate__` / state restoration).
- **`src/`** — C++/CUDA Legate task implementations:
  - `src/models/tree/`, `src/models/krr/`, `src/models/nn/` — tree,
    kernel-ridge-regression, and neural-network model kernels.
  - `src/encoder/target_encoder.{cc,cu,h}` — target encoder.
  - `src/utils/`, `src/special/`, `src/cpp_utils/` — math primitives
    and helpers.
  - `src/legate_library.{cc,h}`, `src/mapper.{cc,h}` — Legate
    library registration and task-to-processor mapping.
- **Build glue** (`CMakeLists.txt`, `cmake/legateboost/`) — bootstraps
  `RAPIDS.cmake` from the rapids-cmake project at configure time.
- **Distributed execution** — the Legate runtime schedules the tasks
  in `src/` across the available CPU / GPU processors, in a single
  process or across a multi-rank job. Inter-rank data transfer is
  Legate's responsibility, not legate-boost's.

**Out of scope for this policy:** vulnerabilities in Legate, cuPyNumeric,
CUDA, the NVIDIA driver, NumPy, scikit-learn, or the `rapids-cmake`
project itself. Vulnerabilities in *how* legate-boost integrates with
those projects — task input handling, pickle state, the cmake bootstrap
fetch — are in scope.

## Threat Model

The threats below trace to specific components and patterns in this
repository. Several have already been observed and remediated through
the [RAPIDS Security Audit](https://github.com/orgs/rapidsai/projects/207);
they are listed so callers and integrators understand the classes of bugs
the library defends against.

1. **Unchecked array indices in GPU / CPU kernels.**
   Task kernels under `src/models/` and `src/encoder/` accept array
   regions from the Legate runtime and index into them based on
   caller-supplied feature counts, bin counts, encoder cardinalities,
   and similar metadata. Pathological combinations of these values
   have been shown to drive out-of-bounds reads or writes in both the
   CPU and CUDA code paths. A hostile caller — or input data with
   surprising cardinality — is the canonical trigger.

2. **Pickle / joblib deserialization of fitted models.**
   `legateboost/utils.py` implements `__getstate__` / state restoration
   so that fitted estimators round-trip through `pickle.dump` /
   `joblib.dump`. Loading a pickled estimator from an untrusted source
   is equivalent to arbitrary code execution by design of the
   pickle protocol — not specific to legate-boost, but legate-boost
   inherits it.

3. **Build-time fetch of `RAPIDS.cmake` without integrity check.**
   `cmake/legateboost/get_rapids_cmake.cmake` performs
   `file(DOWNLOAD https://raw.githubusercontent.com/rapidsai/rapids-cmake/branch-${rapids-cmake-version}/RAPIDS.cmake ...)`
   with no `EXPECTED_HASH`. The URL references a *branch* (mutable),
   and no checksum is asserted on the downloaded content. A
   compromised `rapids-cmake` branch, or in-path tampering on the
   build host, can substitute the CMake bootstrap script that
   subsequently controls the entire build (CPM-fetched dependencies,
   compiler flags).

4. **Input shape / dtype assumptions at the Python ↔ C++ boundary.**
   `legateboost/input_validation.py` is the load-bearing validation
   layer between Python and the C++/CUDA tasks. Any path that bypasses
   it — direct task invocation, callers that monkey-patch the
   estimators, future estimator types added without routing through
   the validator — re-exposes the kernels to the OOB-index class of
   bug above.

5. **Input scale causing DoS or OOM.**
   Tree-boosting fits grow proportionally to `n_samples × n_features ×
   max_depth × n_estimators`, and Shapley-value computation has
   super-linear cost in feature count. A caller able to influence those
   parameters can exhaust GPU/CPU memory or wall-clock time without
   producing a memory-safety bug.

6. **NaN / Inf / extreme numeric inputs.**
   Some kernels assume finite, well-formed `float`/`double` inputs.
   NaN propagation through reductions, divide-by-zero in normalization
   primitives, or extreme values feeding exp/log transforms can yield
   incorrect results or numerical hangs.

7. **Distributed-execution trust boundary.**
   When run as a multi-rank Legate job, legate-boost participates in
   Legate's cross-rank data movement and task scheduling. The trust
   model is Legate's: all ranks are assumed to be mutually
   authenticated peers on a private network.

8. **CI/CD workflow weaknesses.**
   The repository's GitHub Actions workflows have historically had
   `${{ }}` expression-injection points, reusable workflows referenced
   by mutable tag, and missing top-level `permissions:` blocks. These
   are not runtime threats to the library but are part of the
   repository's supply-chain surface.

## Critical Security Assumptions

legate-boost is a library and inherits the caller's privilege; the
following are assumed of the caller / deployer.

- **Inputs are well-formed and within documented ranges.**
  legate-boost expects caller-supplied arrays to have valid dtypes,
  consistent shapes, finite values (or behavior documented for
  NaN/Inf), and feature/bin cardinalities that do not overflow when
  multiplied. Callers ingesting data from external sources should
  validate before fitting.

- **Pickled model artifacts are trusted.**
  Estimators round-trip through pickle; loading a pickled estimator
  from an untrusted source is arbitrary code execution. Production
  pipelines should treat pickled legate-boost objects as code — sign,
  scope, and provenance-verify them.

- **Resource limits are imposed externally.**
  legate-boost does not cap memory or time per call. Fit and Shapley
  computations can be made arbitrarily expensive by input shape.
  Callers operating on untrusted inputs should run legate-boost in a
  process with cgroup / ulimit / container memory and CPU limits.

- **Distributed cluster peers are mutually trusted.**
  Multi-rank Legate execution assumes authenticated peers on a private
  network — this is Legate's trust model and legate-boost inherits it.

- **The build host's outbound network is trusted, or `RAPIDS.cmake` is
  pinned.**
  Until the build's `RAPIDS.cmake` fetch enforces an `EXPECTED_HASH`
  and references an immutable ref, the build host's view of
  `raw.githubusercontent.com` is load-bearing. Operators concerned
  with build-time supply-chain integrity should pre-stage a pinned
  `RAPIDS.cmake` and configure the build to consume it instead.

- **The Legate runtime and cuPyNumeric are themselves trusted.**
  legate-boost relies on Legate's task isolation, region permissions,
  and inter-rank transport. Bugs in those layers are out of scope for
  this policy and should be reported to the Legate / cuPyNumeric
  projects.

- **GPU memory is not a confidentiality boundary.**
  Multiple processes sharing a GPU may observe each other's GPU memory
  through driver-level side channels. legate-boost assumes the caller
  has provisioned the GPU appropriately (MIG, exclusive process,
  container isolation) when confidentiality matters.

## Supported Versions

Security fixes are issued against the current release line. Older
releases are generally not back-ported; upgrade to the latest supported
version to receive fixes.

## Dependency Security

legate-boost tracks CVEs in its direct dependencies — notably `legate`,
`cupynumeric`, `numpy`, and `scikit-learn`. High-severity upstream
advisories may trigger out-of-band releases. The build-time dependency
on `rapids-cmake` is fetched dynamically; see threat #3 above.
