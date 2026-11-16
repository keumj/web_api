"""Isolated KRW-based global asset research signals.

The schema belongs in the separate ``global_asset_signals`` PostgreSQL database.
Importing this package does not connect to a database or collect any data.
Run ``python -m global_signals.refresh`` explicitly; runtime reads use PostgreSQL.
"""
