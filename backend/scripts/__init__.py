"""Operational one-off scripts for the Football Predictor backend.

Scripts here are run by hand (or through their /predict/admin/* counterpart)
rather than imported by the request path. They only import the database layer
lazily so ``python scripts/<name>.py --help`` works without a database.
"""
