"""Logging utilities."""

import logging
import sys


# Create logger
LOG = logging.getLogger("icl_retrieval")
LOG.setLevel(logging.INFO)

# Create console handler
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
handler.setFormatter(formatter)

# Add handler to logger
if not LOG.handlers:
    LOG.addHandler(handler)
