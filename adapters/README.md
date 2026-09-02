# Adapters

Repository-level adapters integrate external systems that are shared by more than one capability or application.

Initial adapter groups:

```text
adapters/
├── datasets/
├── ros2/
└── map-storage/
```

An adapter used exclusively by one module should remain inside that module's infrastructure layer. Shared adapters belong here.

Examples include dataset normalization, ROS 2 boundary translation, persistent map storage, artifact stores, and repository-wide external transport integration.

Adapters must translate external representations into project contracts instead of exposing third-party schemas directly to modules.
