# Temporal Contracts

Repository-wide temporal primitives such as timestamps, time ranges, synchronization metadata, and clock/source identity belong here when shared across multiple capabilities.

`contextual_mapping_contracts.Timestamp` stores non-negative integer nanoseconds with an
explicit clock identifier; cross-clock conversion is never implicit.
