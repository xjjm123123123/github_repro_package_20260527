"""Goal-buffer helper for learned-policy evaluation."""

from src.legacy.run_metaworld_mem import PartialObsWrapper


def wrap_with_goal_buffer(env, **kwargs):
    """Enable the explicit goal cache used in the GoalBuffer evaluation."""
    kwargs = dict(kwargs)
    kwargs["memory_mode"] = "goal_buffer"
    return PartialObsWrapper(env, **kwargs)
