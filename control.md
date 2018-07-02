# Control Module

  - DBW Node
  - Waypoint follower

## DBW Node

The Drive By Wire node is in charge of moving the vehicle using the steering, throttle and break actuators. The node subscribes to the following nodes:

- /current_velocity
- /twist_cmd
- /vehicle/dbw_enabled

## Waypoint Follower

This node subscribes to `/final_waypoints` data and publishes `/twist_cmd` for the dbw node to use.

For validation the project includes the [dbw_test.py](ros/src/twist_controller/dbw_test.py) generating 3 files:

  - steers.csv
  - throttles.csv
  - brakes.csv

execute it as follows

```bash
roslaunch ros/src/twist_controller/launch/dbw_test.launch
```

Then execute the script to plot the result.

```bash
python process_csv.py
```

Results are:

### Steering

![steering](ros/src/twist_controller/steers.png)

### Throttle

![throttle](ros/src/twist_controller/throttles.png)

### Brakes

![Brakes](ros/src/twist_controller/brakes.png)
