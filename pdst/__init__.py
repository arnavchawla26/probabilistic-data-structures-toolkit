"""probabilistic-data-structures-toolkit: from-scratch sketches for big streams.

Three classic space/accuracy-tradeoff data structures, implemented from first
principles (pure standard library, no numpy/mmh3/etc.):

- ``BloomFilter`` -- approximate set membership, no false negatives.
- ``CountMinSketch`` -- approximate per-item frequency counting over a stream.
- ``HyperLogLog`` -- approximate distinct-count (cardinality) estimation.

See the package README for the theory behind each structure and the
``pdst`` command-line tool for a runnable accuracy-vs-exact comparison
against Python's own ``set``/``Counter``.
"""

from .bloom_filter import BloomFilter, optimal_parameters as bloom_optimal_parameters
from .count_min_sketch import CountMinSketch
from .hyperloglog import HyperLogLog

__all__ = [
    "BloomFilter",
    "bloom_optimal_parameters",
    "CountMinSketch",
    "HyperLogLog",
]

__version__ = "0.1.0"
