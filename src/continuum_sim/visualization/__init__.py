"""Visualization modules are imported explicitly by their consumers.

The package facade deliberately performs no eager imports. Several historical
interactive viewers still depend on the retired motor/backend interfaces and
must not become transitive dependencies of Scenario artifact export.
"""

__all__: list[str] = []
