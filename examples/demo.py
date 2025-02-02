from expression.engine import Engine
from expression.expressions import Happy, Sad

# Initialize Engine
engine = Engine()

# Queue animations with different transition and animation durations & interpolations
engine.queue_animation(
    Happy(),
    transition_duration=0.2,
    animation_duration=2.0,
    interpolation="ease-in-out",
)
engine.queue_animation(
    Sad(),
    transition_duration=0.2,
    animation_duration=1.5,
    interpolation="linear",
)

# Start the rendering engine
engine.run()
