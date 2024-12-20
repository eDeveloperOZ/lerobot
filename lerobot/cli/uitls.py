import numpy as np
import argparse

def none_or_int(value):
    if value.lower() == "none":
        return None
    try:
        return int(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Invalid int or None: {value}") from e
    

# Simiulation utils 

def init_sim_calibration(robot, cfg):
    """Initialize calibration for simulation"""
    start_pos = np.array(robot.leader_arms.main.calibration["start_pos"])
    axis_directions = np.array(cfg.get("axis_directions", [1])) 
    offsets = np.array(cfg.get("offsets", [0])) * np.pi
    
    return {
        "start_pos": start_pos,
        "axis_directions": axis_directions, 
        "offsets": offsets
    }

def real_positions_to_sim(real_positions, axis_directions, start_pos, offsets):
    """Convert real robot positions to simulation positions"""
    return axis_directions * (real_positions - start_pos) * 2.0 * np.pi / 4096 + offsets