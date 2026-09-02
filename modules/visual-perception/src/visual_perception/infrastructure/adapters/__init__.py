"""Real backend adapters selected by the module benchmark (#174).

Issues #186-#189: these adapters require GPU hardware and downloaded model
checkpoints that are not available in every development environment. Until
implemented for real, each module here exposes a class that satisfies its
port's shape but raises
:class:`~visual_perception.domain.errors.BackendUnavailableError` when used,
so the canonical pipeline fails explicitly instead of silently falling back
to a fake. See #190 for real-hardware validation.
"""
