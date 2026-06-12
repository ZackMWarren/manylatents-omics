# Data Kinds

**Typed internal data representations for omics workflows.**

This defines the schema seam between data loaders and ops/algorithms. Each kind
carries its own structural semantics (named dimensions, coordinates, required
fields), so ops can read and validate structure instead of guessing axes by
convention.

## Directory Structure

```
manylatents/singlecell/data/
├── kinds/kinds.py                  # Typed kinds: LabeledArray, SparseGraph, Trajectory
├── adapters/formats/adapters.py    # Generic: AnnData → LabeledArray  (from_anndata)
├── adapters/sources/tenx.py        # Specific: 10x .h5 loader          (make_data)
├── manifests/datasets_10x.csv      # Dataset registry (the spreadsheet's home)
├── manifests/registry.py           # Registry loader — single source of truth
├── anndata.py                      # existing datamodule
├── anndata_dataset.py
└── cellxgene_census.py
```

## The Problem We're Solving

- ❌ **AnnData as internal type**: Can't cleanly represent trajectories, time-series, or graphs.
- ❌ **Bare numpy arrays + positional convention**: Dims get reordered, and code has to *guess* what axis 0 means.
- ✅ **Typed kinds with named dims**: Structure is self-describing. Ops read dims, never guess them.

## The Kinds

Every kind subclasses `Kind` (`kinds.py`) and implements four things:
**constructor**, `validate()` (runs on `load`), `serialize(path)`, and `load(path)`.

| Kind | Wraps | Required structure | Status |
|------|-------|--------------------|--------|
| `LabeledArray` | `xarray.DataArray` | dims `cell`, `gene` (+ optional `time`) | ✅ implemented |
| `SparseGraph` | `torch_geometric.data.Data` | a `Data` graph; attrs declared per-op | 🚧 minimal (validate + require) |
| `TrajectoryXXX` | — | TBD (MIOFlow-style paths) | ⛔ placeholder stub, not implemented |

> **Note:** the trajectory kind currently exists only as the placeholder class
> `TrajectoryXXX` whose methods are no-ops. Don't depend on it yet — see
> *Adding a New Kind* below for the shape it should take when fleshed out.

### LabeledArray

**xarray DataArray with named dimensions.** The primary kind for cell×gene
matrices and other labeled array data.

- **Required dims:** `cell`, `gene`  **Optional:** `time`
- **Metadata:** domain attrs (e.g. `genome`) live in `.attrs`; labels (e.g.
  `gene_ids`) live in `.coords`.
- **Accessor:** `.da` returns the underlying `DataArray`.

```python
import numpy as np
import xarray as xr
from manylatents.singlecell.data.kinds.kinds import LabeledArray

da = xr.DataArray(
    np.random.rand(1000, 2000),                 # 1000 cells × 2000 genes
    dims=["cell", "gene"],
    coords={"cell": cell_ids, "gene": gene_names},
    attrs={"genome": "GRCh38"},
)
kind = LabeledArray(da)
kind.validate()        # wraps a non-empty DataArray? returns self, else raises
```

**`validate()`** is a structural check, not a contract check: it confirms the
kind wraps a non-empty `DataArray` and returns `self` (so it chains). It does
**not** enforce specific dims — that's `require`'s job.

**`require(*dims, coords=())`** is how an op declares the structure it consumes.
It raises a clear `ValueError` if a named dim or coord is missing, and returns
`self`:

```python
def my_op(kind: LabeledArray) -> LabeledArray:
    kind.require("cell", "gene", "time")    # raises "requires dims [...]" if absent
    result = kind.da.mean(dim="time")
    return LabeledArray(result)
```

**Serialization** is zarr. `load` reads the array back with `open_dataarray`, so
the round-trip is name-agnostic (works for the unnamed arrays the adapter
produces) and runs `validate()` on read:

```python
kind.serialize("data.zarr")
loaded = LabeledArray.load("data.zarr")      # validates on read
```

### SparseGraph (🚧 minimal)

Wraps a `torch_geometric.data.Data` graph. `validate()` checks it is a `Data`;
`require(*attrs)` asserts named attributes exist; `serialize`/`load` use
`torch.save`/`torch.load`. `torch_geometric` is a heavy dependency, so it is
imported lazily inside the methods that use it.

```python
from manylatents.singlecell.data.kinds.kinds import SparseGraph
graph = SparseGraph(data)        # data is a torch_geometric Data
graph.validate()
graph.require("edge_index")
```

### Trajectory (⛔ stub)

`TrajectoryXXX` is a placeholder for MIOFlow-style cell-fate trajectories. Its
methods are no-ops today; it is intentionally unfinished.

## AnnData Adapter

The edge where AnnData is converted to a typed kind — **the only place AnnData
is used internally.** Located in `adapters/formats/adapters.py`:

```python
import scanpy as sc
from manylatents.singlecell.data.adapters.formats.adapters import from_anndata

adata = sc.read_h5ad("data.h5ad")
kind = from_anndata(adata, coords={"cell": adata.obs_names, "gene": adata.var_names})
# returns a validated LabeledArray; downstream code never touches AnnData
```

For 10x `.h5` files, `adapters/sources/tenx.py` wraps this with the right field
checks (rejects non-scRNA-seq modalities, requires `gene_ids`/`genome`):

```python
from manylatents.singlecell.data.adapters.sources.tenx import make_data
kind = make_data("filtered_feature_bc_matrix.h5")   # -> validated LabeledArray
```

## Example Ops

Minimal demonstration ops live in `tests/singlecell/test_op/example_ops.py`.
They show an op declaring its contract via `require` up front:

```python
from tests.singlecell.test_op.example_ops import temporal_analysis, basic_filter

temporal_analysis(kind)                  # raises unless kind has a 'time' dim
basic_filter(kind, min_expression=0.1)   # needs only cell/gene
```

These are exercised by `TestExampleOps` in `tests/singlecell/test_kinds.py` so
they stay in sync with the kind API.

## Dataset Manifest (the spreadsheet's home)

The Geomancer "Datasets for Geomancer - 10x Genomics" spreadsheet lives in-repo
at **`manylatents/singlecell/data/manifests/datasets_10x.csv`** and is read
through the registry **`manylatents.singlecell.data.manifests`**. This is the
single source of truth for *what data exists* — nothing hardcodes dataset paths.

**The spreadsheet is expected to change.** To update it, drop in a fresh export
with the same columns (only `Dataset_Name` and `wget_commands` — which must hold
a `.h5` URL — are required; extra columns are ignored). To use a private or newer
copy without editing the repo, set `GEOMANCER_DATASETS_CSV` to its path; it
overrides the in-repo default.

```python
from manylatents.singlecell.data.manifests import (
    manifest_path,    # active CSV path (env override, else in-repo default)
    load_manifest,    # -> list[DatasetEntry(name, url)] with a usable .h5 link
    select_random,    # -> n random entries (seed= for reproducibility)
)

for entry in load_manifest():
    print(entry.name, entry.url)
```

### Adding a dataset

1. Add a row to `datasets_10x.csv` (or your `GEOMANCER_DATASETS_CSV` copy) with a
   `Dataset_Name` and a `wget_commands` cell containing the `.h5` download URL.
2. It is now discoverable via `load_manifest()` / `select_random()` — no code
   change needed.
3. Load it through the standard path: download the `.h5`, then `make_data(path)`.

The real-data test `test_random_10x_dataset_loads_validates_and_enforces_dims`
samples from this manifest and runs each dataset through `make_data`, validating
structure on read. It is marked `network`/`slow` and skips when the manifest is
absent or the host is offline.

## Adding a New Kind

1. **Define the class** in `kinds/kinds.py`:

```python
class MyKind(Kind):
    def __init__(self, data):
        self._data = data

    def validate(self):
        if not _is_valid(self._data):
            raise ValueError("MyKind: <what's wrong>")
        return self

    def serialize(self, path: str) -> None:
        ...

    @classmethod
    def load(cls, path):
        data = ...                    # read from disk
        return cls(data).validate()   # validation runs on read
```

2. **Write tests** in `tests/singlecell/test_kinds.py`:
   - a **round-trip** test (construct → serialize → load → identical), and
   - a **rejection** test (malformed input fails on `validate`/`load`).

3. **Update this README** with the kind, its required structure, and usage.

## Invariants Every Kind Must Hold

```
construct → validate()
          → serialize(path)
          → load(path) → validate()      ← validation runs on read
          → identical to the original
```

- Malformed data is rejected on read, not silently accepted.
- Named dims survive slicing/transposing.
- Coords/attrs are preserved across the round-trip.

## Coordination with Downstreams

- **Model ops (manylatents#269):** consume `LabeledArray` with dims `cell`,
  `gene` (optional `time`). That contract is enforced per-op via `require`.
- **MIOFlow / RITINI:** will flesh out the `Trajectory` and `SparseGraph` stubs;
  op shapes finalize when those land.

---

*Last updated: 2026-06-11*
