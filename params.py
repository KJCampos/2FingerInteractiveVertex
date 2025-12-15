class Params:
    """
    All tunable knobs live here so you don't hunt through code.
    """
    def __init__(self):
        # Particle count (keep modest for now)
        self.num_particles = 10000

        # Particle visuals
        self.particle_radius_px = 1

        # Physics
        self.gravity = 0.15         # downward pull (sim units / s^2)
        self.damping = 0.985        # velocity multiplier per step
        self.max_speed = 1.5        # clamp speed (sim units / s)

        # "Fluid-ish" cohesion (soft neighbor attraction)
        self.neighbor_radius = 0.04
        self.cohesion = 0.35

        # Hand interaction
        self.hand_strength = 3.0
        self.hand_radius = 0.14     # influence radius (sim units)

        # If auto_mode=True:
        #   open hand => push, pinch => pull
        # If auto_mode=False:
        #   uses force_mode_push toggle
        self.auto_mode = True
        self.force_mode_push = True
