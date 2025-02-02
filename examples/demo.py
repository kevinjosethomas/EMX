from expression.engine import Engine
from expression.expressions import Happy, Sad

# Initialize Engine
engine = Engine()

# Queue animations with different durations & interpolations
engine.queue_animation(Happy(), duration=2.0, interpolation="ease-in-out")
engine.queue_animation(Sad(), duration=1.5, interpolation="linear")

# Start the rendering engine
engine.run()
