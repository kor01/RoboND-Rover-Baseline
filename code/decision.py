import numpy as np
from rover_state import RoverState
from geometry import rad_to_degree


# This is where you can build a decision tree for determining throttle, brake and steer 
# commands based on the output of the perception_step() function
def decision_step(rover: RoverState):

    # Implement conditionals to decide what to do given perception data
    # Here you're all set up with some basic functionality but you'll need to
    # improve on this decision tree to do a good job of navigating autonomously!

    # Example:
    # Check if we have vision data to make decisions with

    rover.step += 1
    if rover.nav_angles is not None:
        # Check for Rover.mode status
        if rover.mode == 'forward':
            # Check the extent of navigable terrain
            if len(rover.nav_angles) >= rover.stop_forward:
                # If mode is forward, navigable terrain looks good 
                # and velocity is below max, then throttle 
                if rover.vel < rover.max_vel:
                    # Set throttle value to throttle setting
                    rover.throttle = rover.throttle_set
                else: # Else coast
                    rover.throttle = 0
                rover.brake = 0
                # Set steering to average angle clipped to the range +/- 15

                # randomized policy pick to get out of deterministic policy
                
                if rover.directions is None or len(rover.directions) == 0:
                    direction = np.mean(rover.nav_angles)
                    if rover.step - rover.last_roll > 500:
                        rover.direction_pick = None
                else:
                    if rover.direction_pick is None \
                        or rover.direction_pick >= len(rover.directions):
                        rover.direction_pick = np.random.randint(0, len(rover.directions))
                        rover.last_roll = rover.step
                    
                    direction = rover.directions[rover.direction_pick]
                    print('picked direction:', direction, rover.step)

                rover.steer = np.clip(rad_to_degree(direction), -15, 15)

            # If there's a lack of navigable terrain pixels then go to 'stop' mode
            else:
                    # Set mode to "stop" and hit the brakes!
                    rover.throttle = 0
                    # Set brake to stored brake value
                    rover.brake = rover.brake_set
                    rover.steer = 0
                    rover.mode = 'stop'

        # If we're already in "stop" mode then make different decisions
        elif rover.mode == 'stop':
            print('in stop mode')
            # If we're in stop mode but still moving keep braking
            if rover.vel > 0.2:
                rover.throttle = 0
                rover.brake = rover.brake_set
                rover.steer = 0
            # If we're not moving (vel < 0.2) then do something else
            elif rover.vel <= 0.2:
                
                mean_angle = np.pi / 2

                if len(rover.nav_angles) > 0:
                    mean_angle = abs(rover.nav_angles.mean())
                
                print('current mean_angle', mean_angle)
                    
                # Now we're stopped and we have vision data to see if there's a path forward
                if len(rover.nav_angles) < rover.go_forward or mean_angle > np.pi / 8:
                    rover.throttle = 0
                    # Release the brake to allow turning
                    rover.brake = 0
                    # Turn range is +/- 15 degrees, when stopped the next line will induce 4-wheel turning
                    rover.steer = -15 # Could be more clever here about which way to turn
                # If we're stopped but see sufficient navigable terrain in front then go!
                else:
                    # Set throttle back to stored value
                    rover.throttle = rover.throttle_set
                    # Release the brake
                    rover.brake = 0
                    # Set steer to mean angle
                    direction = np.mean(rover.nav_angles)
                    rover.steer = np.clip(
                        rad_to_degree(direction), -15, 15)
                    rover.mode = 'forward'
    # Just to make the rover do something 
    # even if no modifications have been made to the code
    else:
        rover.throttle = rover.throttle_set
        rover.steer = 0
        rover.brake = 0
        
    # If in a state where want to pickup a rock send pickup command
    if rover.near_sample and rover.vel == 0 and not rover.picking_up:
        rover.send_pickup = True
    
    return rover

