import pickle
import numpy as np

# 1. Update this path to your actual checkpoint file
checkpoint_path = "/home/hice1/kagrawal74/scratch/DRL/ray_results/PPO_selfplay_rec/PPO_Soccer_edcea_00000_0_2026-03-29_16-36-14/checkpoint_001000/checkpoint-1000"

with open(checkpoint_path, "rb") as f:
    data = pickle.load(f)

# The 'worker' key contains the state of the rollout workers
worker_state = pickle.loads(data["worker"])

print("--- Checkpoint Structure ---")
print(f"Available keys in worker_state: {list(worker_state.keys())}")

# In many versions, weights are inside 'policy_states'
# Let's try to find them dynamically
policy_data = None
if "policy_states" in worker_state:
    policy_data = worker_state["policy_states"]
    print("Found weights in 'policy_states'")
elif "state" in worker_state:
    policy_data = worker_state["state"]
    print("Found weights in 'state'")

if policy_data:
    # Check if our policies exist in this folder
    if "default" in policy_data and "opponent_1" in policy_data:
        # In some versions, weights are further nested under 'weights' or 'worker_weights'
        def_p = policy_data["default"]
        opp_p = policy_data["opponent_1"]
        
        # If these are dicts, let's look for the weight tensors
        def_w = def_p.get("weights", def_p)
        opp_w = opp_p.get("weights", opp_p)
        
        first_key = next(iter(def_w))
        is_identical = np.array_equal(def_w[first_key], opp_w[first_key])
        
        print(f"\nComparing layer: {first_key}")
        if is_identical:
            print("✅ SUCCESS: Weights match! Self-play is active.")
        else:
            print("❌ Weights differ. The callback hasn't triggered for this checkpoint yet.")
    else:
        print(f"Policies found: {list(policy_data.keys())}")
else:
    print("Could not automatically locate policy weights. Please check the 'Available keys' printed above.")