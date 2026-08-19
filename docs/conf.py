"""Sphinx configuration — fleet standard via py-canon."""

from py_canon.sphinx import configure

configure(globals())

# Repo-specific: the API docs are dense with numpy/scipy/pandas types, so
# resolve those references rather than rendering them as plain text.
intersphinx_mapping.update(  # noqa: F821
    {
        "numpy": ("https://numpy.org/doc/stable/", None),
        "scipy": ("https://docs.scipy.org/doc/scipy/", None),
        "pandas": ("https://pandas.pydata.org/docs/", None),
    }
)
