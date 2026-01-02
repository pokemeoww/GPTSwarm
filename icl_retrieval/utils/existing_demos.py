"""Pre-defined demo lists for fixed demo mode (optional).

This file provides fixed demo lists for each prompt type.
These are used when demo_method='fixed' in the environment variables.
"""

# Propose demos - for proposing word candidates
list_propose_demos = []

# If-correct demos - for validating word-meaning pairs
list_if_correct_demos = []

# Suggest demos - for generating improvement suggestions
list_suggest_demos = []

# Value demos - for evaluating word feasibility
list_value_demos = []

# Note: These lists are empty by default.
# You can populate them with fixed examples if you want to use demo_method='fixed'
# Otherwise, use demo_method='retrieved' or 'reranked' for dynamic retrieval
