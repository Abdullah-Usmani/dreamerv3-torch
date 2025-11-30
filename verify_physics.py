from envs.dmc import ContinualCartPole
import numpy as np

def verify():
    print("Initializing Environment...")
    env = ContinualCartPole()
    
    # --- TEST TASK 2 (MOON GRAVITY) ---
    print("\n--- Testing Task 2 (Moon Gravity) ---")
    env.set_task(2)
    env.reset()
    
    # Check internal MuJoCo gravity vector
    g_z = env._env.physics.model.opt.gravity[2]
    print(f"Physics Engine Gravity Z: {g_z}")
    
    if g_z == -1.62:
        print("✅ SUCCESS: Gravity modified correctly.")
    else:
        print(f"❌ FAILURE: Expected -1.62, got {g_z}")

    # --- TEST TASK 1 (WIND) ---
    print("\n--- Testing Task 1 (Wind) ---")
    env.set_task(1)
    env.reset()
    
    # Step the environment to apply wind
    env.step(np.array([0.0]))
    
    # Check force applied to the pole body ("pole_1")
    # Index 0 is X-axis force
    force_x = env._env.physics.named.data.xfrc_applied["pole_1", 0]
    print(f"Physics Engine Applied Force X: {force_x}")
    
    if force_x == 1.0:
        print("✅ SUCCESS: Wind force applied correctly.")
    else:
        print(f"❌ FAILURE: Expected 1.0, got {force_x}")

if __name__ == "__main__":
    verify()