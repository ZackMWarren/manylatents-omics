# Data Kinds

**Typed internal data representations for omics workflows.**

This defines the schema seam between data loaders and ops/algorithms. Each kind
carries its own structural semantics (named dimensions, coordinates, required
fields), so ops can read and validate structure instead of guessing axes by
convention.

## Directory Structure

```
manylatents/
├── kinds/base.py                       # Kind: the abstract base class (the contract)
├── kinds/labeled_array.py              # LabeledArray kind
├── kinds/sparse_graph.py               # SparseGraph kind
├── kinds/__init__.py                   # Re-exports Kind, LabeledArray, SparseGraph
└── singlecell/data/
    ├── adapters/formats/adapters.py    # Generic: AnnData → LabeledArray  (from_anndata)
    ├── adapters/sources/tenx.py        # Specific: 10x .h5 loader          (read_tenx)
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

Every kind is a **frozen dataclass** that subclasses `Kind` (`base.py`) and lives
in its own module (`labeled_array.py`, `sparse_graph.py`). `frozen=True` makes
instances immutable after construction; `eq=False` keeps identity-based equality
(the auto-generated `__eq__` would compare the wrapped arrays element-wise and
raise on the ambiguous truth value). `base.py` declares five abstract members
every kind implements: `validate()`, `serialize(path)`, `load(path)` (constructs,
so it validates too), plus `require(*dims, coords=())` and `tagged(op_name)` — the
two halves of geomancy's op protocol (see [Op protocol](#op-protocol-require-tagged--provenance)
below). Each kind also runs `validate()` from its `__post_init__`, so validation
fires on construction. All three kinds are re-exported from the `kinds` package,
so import them from `manylatents.kinds`.

| Kind | Wraps | Required structure | Status |
|------|-------|--------------------|--------|
| `LabeledArray` | `xarray.DataArray` | a non-empty `DataArray`; dims declared per-op | ✅ implemented |
| `SparseGraph` | two `numpy` arrays | `edges` (E×2 integer) + `node_ids` (1-D) | ✅ implemented |

### Op protocol: `require`, `tagged` & provenance

Beyond validation and persistence, every kind participates in geomancy's op
protocol through two methods and a provenance trail. This is what lets geomancy's
op registry declare contracts and record what it ran, without this repo importing
geomancy.

**`require(*dims, coords=())`** is how an op declares the structure it consumes.
It raises a clear `ValueError` if a named dim/component or coord is missing, and
returns `self` for chaining. Each kind interprets the names against its own
structure — `LabeledArray` checks its `DataArray` dims/coords; `SparseGraph`
checks its named components (`edges`, `node_ids`) and carries no coords.

**`tagged(op_name)`** returns a *copy* with `op_name` appended to the kind's
`provenance` trail. Because kinds are frozen, the original is untouched — the op
registry calls `tagged` as it runs each op to build up an ordered record of the
pipeline:

```python
kind = LabeledArray(da)                       # provenance == ()
kind = kind.tagged("normalize").tagged("log1p")
kind.provenance                               # ("normalize", "log1p")
```

**`provenance`** is a `tuple[str, ...]` field on every kind, defaulting to `()`
and normalized to a tuple on construction. It **survives serialization** — the
zarr (`LabeledArray`) and `.npz` (`SparseGraph`) round-trips both carry the trail
through, so `load` returns a kind with the same provenance it was saved with.

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
from manylatents.kinds import LabeledArray

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

**Serialization** is zarr and runs `validate()` on read. Dense arrays are written
natively; the round-trip is name-agnostic (works for the unnamed arrays the
adapter produces).

```python
kind.serialize("data.zarr")
loaded = LabeledArray.load("data.zarr")      # validates on read
```

> **Sparse data must be a pydata `sparse.COO`, not a scipy sparse matrix.** zarr
> can't serialize a sparse-backed `DataArray` directly, so `serialize` stores the
> bare COO components (`coords`, `data`, `fill_value`) plus dims/coords/attrs and
> rebuilds the array on `load`. This path is keyed on the backing array being a
> `sparse.COO` — a scipy CSR/CSC/COO matrix is not recognized by xarray as an
> array backing and won't round-trip. Convert first:
>
> ```python
> import sparse
> coo = sparse.COO.from_scipy_sparse(scipy_csr)   # scipy -> pydata sparse.COO
> da = xr.DataArray(coo, dims=["cell", "gene"], coords={...})
> LabeledArray(da).serialize("data.zarr")
> ```

### SparseGraph

**A graph as two plain `numpy` arrays:** an `edges` edge list and a `node_ids`
vector. No heavy graph dependency — just numpy.

- **Required structure:** `edges` is `E×2` with an integer dtype (each row a
  `(source, target)` pair); `node_ids` is 1-D.
- **Accessor:** `.data` returns the `(edges, node_ids)` tuple.

`validate()` enforces the shapes and the integer edge dtype, returning `self`.
`require` accepts the graph's named components (`edges`, `node_ids`) and rejects
any coords, since a bare edge list carries none. **Serialization** is
`np.savez_compressed`; the `provenance` trail rides along as a string array, and
`load` reads the arrays back (restoring provenance) and runs `validate()`.

```python
import numpy as np
from manylatents.kinds import SparseGraph

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
from manylatents.singlecell.data.adapters.sources.tenx import read_tenx
kind = read_tenx("filtered_feature_bc_matrix.h5")   # -> validated LabeledArray
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
`tests/kinds/test_example_ops.py` so they stay in sync with the
kind API.

## Dataset Manifest (the spreadsheet's home)

The Geomancer "Datasets for Geomancer - 10x Genomics" spreadsheet lives in-repo
at **`manylatents/singlecell/data/manifests/datasets_10x.csv`** and is read
through the registry **`manylatents.singlecell.data.manifests`**. This is the
single source of truth for *what data exists* — nothing hardcodes dataset paths.

**The spreadsheet is expected to change.** To update it, drop in a fresh export
with the same columns (only `Dataset_Name` and `wget_commands` — which must hold
a `.h5` URL — are required; extra columns are addtional medtadata). To use a private or newer
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

## Adding a New Kind

1. **Define the class** in its own module, `kinds/my_kind.py`, as a frozen
   dataclass subclassing `Kind` from `base.py`:

```python
from dataclasses import dataclass

from .base import Kind


@dataclass(frozen=True, eq=False)
class MyKind(Kind):
    data: SomeType
    provenance: tuple[str, ...] = ()  # op protocol: the trail tagged() appends to

    def __post_init__(self):
        object.__setattr__(self, "provenance", tuple(self.provenance))
        self.validate()               # validation runs on construction
        # To normalize a field on a frozen instance, bypass the guard:
        # object.__setattr__(self, "data", _normalize(self.data))

    def validate(self):
        if not _is_valid(self.data):
            raise ValueError("MyKind: <what's wrong>")
        return self

    def require(self, *dims: str, coords: tuple[str, ...] = ()) -> "MyKind":
        # raise ValueError("requires ...") if a named part/coord is absent
        return self

    def tagged(self, op_name: str) -> "MyKind":
        return MyKind(self.data, self.provenance + (op_name,))

    def serialize(self, path: str) -> None:
        ...                           # persist provenance too, so it round-trips

    @classmethod
    def load(cls, path):
        data = ...                    # read from disk (incl. provenance)
        return cls(data)              # __post_init__ validates, so load validates too
```

2. **Re-export it** from `kinds/__init__.py` so it can be imported from the
   `kinds` package alongside the others.

3. **Write tests** in `tests/kinds`:
   - a **round-trip** test (construct → serialize → load → identical), and
   - a **rejection** test (malformed input fails on `validate`/`load`).
   - additional tests specific to the kind

4. **Update this README** with the kind, its required structure, and usage.

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
- The `provenance` trail is preserved across the round-trip.

## Coordination with Downstreams

- **Model ops (manylatents#269):** consume `LabeledArray` with dims `cell`,
  `gene`. That contract is enforced per-op via `require`.
- **MIOFlow / RITINI:** consume `LabeledArray` with `time` coord.

---

*Last updated: 2026-07-06*
