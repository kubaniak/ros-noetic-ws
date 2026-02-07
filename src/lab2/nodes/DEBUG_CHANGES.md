# Debug Changes and Fixes

## Changes Made

### 1. RRT* Planning Fixes (l2_planning.py)

**Problem:** 
- Only 0.3% success rate (79 nodes in 10,000 iterations)
- 72% rejection rate (4408 collisions, 1461 small moves)
- Path stopped at 48.46m from 60.86m goal

**Root Cause:**
- Timestep was 1.0 seconds, causing trajectories to jump 0.5m at each step
- With robot radius of 0.22m, these large jumps caused frequent collisions
- Robot controller would rotate in place when angle error is large (cos(angle)~0)

**Fix:**
- **Reduced timestep from 1.0s to 0.5s** (line 62)
  - Smaller jumps = fewer collisions
  - More granular collision checking

**Added Debug Output:**
- Start position collision check
- Timestep display
- Sample trajectory analysis at iteration 100
- Collision pixel statistics

**Expected Improvements:**
- Success rate should increase to 30-50%
- Path should reach closer to goal
- More nodes in final tree

---

### 2. Trajectory Rollout Fixes (l2_follow_path.py)

**Problem:**
- Robot selecting controls with huge rotation errors
- Example: theta = 4.75 rad when goal is -0.600 rad
- Rotation cost not being considered until within 0.1m

**Root Cause:**
- `MIN_TRANS_DIST_TO_USE_ROT` was 0.1m (TRANS_GOAL_TOL)
- Robot ignored rotation until extremely close
- Would overshoot and spin wildly

**Fix:**
- **Increased MIN_TRANS_DIST_TO_USE_ROT from 0.1m to 1.0m** (line 30)
  - Robot now cares about orientation when within 1m
  - Prevents wild spinning near waypoints

**Added Debug Output:**
- Cost breakdown for top 3 options showing:
  - End position (x, y, theta)
  - Distance to goal
  - Rotation error in radians
  - Individual cost components (rot_cost, obs_cost)
  - Total cost
- Warning message when rotation cost is/isn't being considered

**Expected Improvements:**
- Robot should approach waypoints more smoothly
- Should maintain better heading alignment
- Less overshooting and spinning

---

## How to Test

### Test RRT* Planning
```bash
cd /workspaces/ros-noetic-ws/src/lab2/nodes
./l2_planning.py
```

**Look for:**
- Timestep shows 0.5s (not 1.0s)
- Start position: all pixels should be free
- Success rate at iteration 100 should be >5%
- At iteration 100, should see sample trajectory analysis
- Collision check should show high percentage of free pixels
- Should reach closer to goal (within 5-10m if not all the way)

### Test Trajectory Rollout
```bash
# Terminal 1
roslaunch rob521_lab2 willowgarage_world.launch

# Terminal 2
roslaunch rob521_lab2 map_view.launch

# Terminal 3
rosrun rob521_lab2 l2_follow_path.py
```

**Look for:**
- "PATH LOADING DEBUG" section showing correct waypoints
- "MIN_TRANS_DIST_TO_USE_ROT=1.0m" in output
- Cost breakdown showing rotation errors
- Message showing if rotation cost IS/NOT being considered
- When within 1m of waypoint: "Rotation cost IS being considered"
- Rotation errors (rot_err) should be considered in cost
- Loop time should be < 0.200s

---

## Debug Output Examples

### Good RRT* Output
```
=== RRT* PLANNING DEBUG ===
Start: [0. 0. 0.]
Goal: [ 42.05 -44.  ]
Distance to goal: 60.86m
Timestep: 0.5s                    <-- Should be 0.5
Start position check: 154/154 pixels are free  <-- All free!

RRT* iter 100: nodes=35, ball_r=2.500, success_rate=30.0%  <-- >5%!
  Stats: bounds=0, collision=45, small_move=20, duplicate=5
  Closest node to goal: 34 at distance 55.12m  <-- Making progress

  DEBUG: Sample trajectory analysis:
    Collision check: 1423/1447 pixels are free  <-- ~98% free!
```

### Good Trajectory Rollout Output
```
Current pose: x=1.500, y=-0.600, theta=-0.500
Current goal (waypoint 2): x=1.659, y=-0.600, theta=-0.600
Distance to goal: 0.159m           <-- Within 1m

Top 3 control options (MIN_TRANS_DIST_TO_USE_ROT=1.0m):
  1. v=0.130, w=0.182 -> pos=(1.65,-0.61), theta=-0.32rad, 
     dist=0.011m, rot_err=0.280rad, rot_cost=0.0280, obs_cost=0.0020, total=0.041
                                      ^^^^^^^^^^^^ Rotation IS considered!
  -> Rotation cost IS being considered (dist 0.159m < 1.0m)
```

### Bad Output to Watch For
```
# RRT* - Bad
Start position check: 98/154 pixels are free  <-- OBSTACLES AT START!
RRT* iter 100: success_rate=0.5%               <-- TOO LOW
Collision check: 342/1447 pixels are free      <-- ONLY 24% FREE

# Trajectory Rollout - Bad
Distance to goal: 0.357m
  -> Rotation cost NOT considered              <-- Should be considered!
  1. v=0.260, w=1.092 -> theta=4.75rad         <-- Huge rotation error!
     rot_cost=0.0000                           <-- Not being penalized!
```

---

## If Issues Persist

### RRT* Still Has Low Success Rate
1. Check start position - may need to adjust starting point
2. Increase timestep reduction (try 0.3s or 0.25s)
3. Reduce max velocities (self.vel_max, self.rot_vel_max)
4. Check goal position is reachable (not in obstacle)

### Robot Still Overshooting
1. Increase MIN_TRANS_DIST_TO_USE_ROT to 2.0m
2. Increase ROT_DIST_MULT from 0.1 to 0.5 or 1.0
3. Reduce TRANS_VEL_OPTS (make robot slower)
4. Increase ROT_GOAL_TOL (currently 0.3 rad = 17°)

### Loop Time Still Too Slow
- Collision check optimizations already in place
- If still >0.2s, reduce CONTROL_HORIZON from 5 to 3 seconds
- Or reduce horizon_timesteps by increasing INTEGRATION_DT

---

## Summary

**Key Fixes:**
1. RRT* timestep: 1.0s → 0.5s
2. Trajectory rollout rotation consideration: 0.1m → 1.0m

**Expected Results:**
- RRT* should generate paths that reach goal or get much closer
- Robot should follow waypoints smoothly with good heading alignment
- Loop time should be comfortable <0.2s

Run the tests and share the debug output!
