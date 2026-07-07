"""Adapters: convert ecosystem formats/sources to typed kinds at the ingestion edge.

``formats/`` holds generic converters (e.g. AnnData -> LabeledArray); ``sources/``
holds source-specific loaders (e.g. 10x ``.h5``). Imports are kept lazy at the
submodule level so importing this package does not pull in scanpy/scipy.
"""
