# Spatial Contracts

Repository-wide spatial primitives such as coordinate-frame identifiers, poses, transforms, bounds, and geometry references belong here when shared by multiple capabilities.

`contextual_mapping_contracts.FrameId` provides stable frame identity. Transform and pose values
remain deferred until a concrete cross-capability boundary requires them.
