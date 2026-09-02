# Contracts

`contracts/` contains shared implementation-agnostic primitives used across multiple capabilities and applications.

Initial groups:

```text
contracts/
├── spatial/
├── temporal/
├── observations/
└── maps/
```

Repository-wide contracts may define data shape, invariants, provenance, confidence, coordinate frames, timestamps, identifiers, artifact references, and serialization boundaries.

Capability-specific contracts remain owned by the module that defines the capability. For example, learned point embedding contracts belong to `point-representation`; they should not be moved here merely because another module consumes them.

The root contract package should stay small and stable enough to support independently evolving modules and applications.
