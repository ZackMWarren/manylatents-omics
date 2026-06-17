# Data Kinds

**Typed internal data representations for omics workflows.**

This defines the schema seam between data loaders and ops/algorithms. Each kind
carries its own structural semantics (named dimensions, coordinates, required
fields), so ops can read and validate structure instead of guessing axes by
convention.

## Directory Structure

```
manylatents/singlecell/data/
├── kinds/kinds.py                  # Typed kinds: LabeledArray, SparseGraph
├── adapters/formats/adapters.py    # Generic: AnnData → LabeledArray  (from_anndata)
├── adapters/sources/tenx.py        # Specific: 10x .h5 loader          (make_data)
├── manifests/datasets_10x.csv      # Dataset registry (the spreadsheet's home)
├── manifests/tenx_registry.py      # 10x registry loader — single source of truth
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
**constructor** (calls `validate()`), `validate()`, `serialize(path)`, and
`load(path)` (constructs, so it validates too).

| Kind | Wraps | Required structure | Status |
|------|-------|--------------------|--------|
| `LabeledArray` | `xarray.DataArray` | a non-empty `DataArray`; dims declared per-op | ✅ implemented |
| `SparseGraph` | two `numpy` arrays | `edges` (E×2 integer) + `node_ids` (1-D) | ✅ implemented |

### LabeledArray

**xarray DataArray with named dimensions.** The primary kind for cell×gene
matrices and other labeled array data.

- **Required structure:** a non-empty `DataArray`. The kind itself enforces no
  specific dims — `cell`/`gene` are established by `from_anndata` and required at
  the op via `require("cell", "gene")` (`time` optional). See `validate` vs
  `require` below.
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

### SparseGraph

**A graph as two plain `numpy` arrays:** an `edges` edge list and a `node_ids`
vector. No heavy graph dependency — just numpy.

- **Required structure:** `edges` is `E×2` with an integer dtype (each row a
  `(source, target)` pair); `node_ids` is 1-D.
- **Accessor:** `.data` returns the `(edges, node_ids)` tuple.

`validate()` enforces the shapes and the integer edge dtype, returning `self`.
**Serialization** is `np.savez_compressed`; `load` reads the arrays back and
runs `validate()`.

```python
import numpy as np
from manylatents.singlecell.data.kinds.kinds import SparseGraph

edges = np.array([[0, 1], [1, 2], [2, 0]])   # E×2 integer edge list
node_ids = np.array([0, 1, 2])               # 1-D node ids
graph = SparseGraph(edges, node_ids)
graph.validate()                             # checks shapes + integer dtype

graph.serialize("graph.npz")
loaded = SparseGraph.load("graph.npz")       # validates on read
edges, node_ids = loaded.data
```


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

These are exercised by `TestExampleOps` in
`tests/singlecell/test_kinds/test_example_ops.py` so they stay in sync with the
kind API.

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
    tenx_manifest_path,    # active CSV path (env override, else in-repo default)
    load_tenx_manifest,    # -> list[TenxDatasetEntry(name, url)] with a usable .h5 link
    select_random_tenx,    # -> n random entries (seed= for reproducibility)
)

for entry in load_tenx_manifest():
    print(entry.name, entry.url)
```

### Adding a dataset

1. Add to an exisiting manifests website's file (i.e tenx_registry.py), or create a new file for
   a new source of datasets. Ensure that there is a `Dataset_Name` and a `wget_commands` cell containing the `.h5` download URL. Add additional metadata in additional columns.
2. It is now discoverable via `load_tenx_manifest()` / `select_random_tenx()` — no
   code change needed.
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
        self.validate()               # validation runs on construction

    def validate(self):
        if not _is_valid(self._data):
            raise ValueError("MyKind: <what's wrong>")
        return self

    def serialize(self, path: str) -> None:
        ...

    @classmethod
    def load(cls, path):
        data = ...                    # read from disk
        return cls(data)              # __init__ validates, so load validates too
```

2. **Write tests** in `tests/singlecell/test_kinds`:
   - a **round-trip** test (construct → serialize → load → identical), and
   - a **rejection** test (malformed input fails on `validate`/`load`).
   - additional tests specific to the kind

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
  `gene`. That contract is enforced per-op via `require`.
- **MIOFlow / RITINI:** consume `LabeledArray` with `time` coord.

---

*Last updated: 2026-06-16*
