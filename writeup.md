## Project: Search and Sample Return

---


**The goals / steps of this project are the following:**  

**Training / Calibration**  

* Download the simulator and take data in "Training Mode"
* Test out the functions in the Jupyter Notebook provided
* Add functions to detect obstacles and samples of interest (golden rocks)
* Fill in the `process_image()` function with the appropriate image processing steps (perspective transform, color threshold etc.) to get from raw images to a map.  The `output_image` you create in this step should demonstrate that your mapping pipeline works.
* Use `moviepy` to process the images in your saved dataset with the `process_image()` function.  Include the video you produce as part of your submission.

**Autonomous Navigation / Mapping**

* Fill in the `perception_step()` function within the `perception.py` script with the appropriate image processing functions to create a map and update `Rover()` data (similar to what you did with `process_image()` in the notebook).
* Fill in the `decision_step()` function within the `decision.py` script with conditional statements that take into consideration the outputs of the `perception_step()` in deciding how to issue throttle, brake and steering commands.
* Iterate on your perception and decision function until your rover does a reasonable (need to define metric) job of navigating and mapping.  

[//]: # (Image References)

[image1]: ./misc/my_rover.png


## Project Approach

In this section, I will introduce the high level approach I used in this project

### Perception

1. **full camera model in perspective transform:**

    - perspective Transform:

      - perspective transform is used in this project to convert image segmentation of navigable terrain to coordinates described in rover body fixed frame and world frame.

      - these transformations allows the cruise directions detection and map construction

    - Problems with CV2 based method:

      - the **original cv2 base** method is not robust in the context of mobile robot because the transformation parameter proposed by cv2 is coupled with the orientation of the robot. When the pitch and roll of the robot are changed, this approach introduces errors.

      - One possible remedy is to use the transformation only in the context where the approximation is relatively accurate, i.e. drop the frames when the pitch or roll is high.

      - This remedy is **not ideal** since it limits the dexterity of the robot and hurts the performance of mapping, resulting in low map coverage.

      - It also hurts the performance of decision making step, since the high frame drop rate can results in inaccurate direction estimation and introduces delay.

    - Fully Calibrated Camera Model:

      - in this project a fully calibrated camera model is introduced to replace the cv2 algorithm.

      - the camera model explicitly takes roll and pitch of the imu reading, and used these parameters to generate appropriate perspective transform.

      - the initial camera internal parameter can be regressed by numerical optimization.

      - the result shows **significant performance** gain in the situation where roll or pitch is highly deviate from zero, which in turn improves the map fidelity by 30% and better map coverage. Since it does not apply constraints on motion, the flexibility in decision making implementation is also improved.


2. **geometric representation and interpolation:**

      - one problem associated with perspective transform is the pixels near the sight singularity is sparsely distributed. The pixels near sight singularity can be very helpful in direction detection.

      - In this project I introduced a ray based geometric representation to the perspective transformed particles, which served as a interpolation method and results in improvement in detection range and better direction detection.


3. **direction detection, sample detection and obstacles detection:**

      - the ray based representation obtained through the above steps is used for direction detection (to determine where to go)

      - the detection rule is simple heuristic clustering of rays based on its length and angles (using mesh clustering method).

      - the directions are generated as the mean angle of each cluster ( as opposed to the baseline implementation which uses the mean angle of all detected particles)


### Decision:

the implementation of decision making step used in this project adhere the baseline implementation (simple reactive approach, without additional planning), and improves in following perspectives:

1. **better direction selection:**

    - the direction choice of the baseline version used the mean angle of all particles. Here I used a random selection of the directions detected in the above perception pipeline.

    - in reactive approach the decision does not rely on the memory (current map) of the rover, the random direction selection is helpful in mapping coverage and prevent the rover from getting stuck into a fixed path route.

    - since the action of taking one option when facing many direction choices takes an interval of time to accomplish, one additional trick is introduced to make the rover direction selection consistent over a short period of steps, that is the choice of direction represented by random int will only changed every 100 steps, or when the number of choices is changed.

2. **better mode switch predicate:**

 - the switch predicate of ```forward``` and ```stop``` mode is improved, instead of using the number of navi-angle, I used length of navigable rays in the front sight (-3 to 3 degree) of the rover, as indicator of stop and resume, when results in much better collision avoidance (to map edge and obstacles)



## Technical Details:

1. Camera Model

    the camera model used here is a standard mathematical projection model with parameters:

     * view point position (described in robot body fixed frame)

     * projection plane origin coordinate (also described in robot body fixed frame)

    when these parameters are known, we can compute the projection of every ground pixel (z = 0), if we know the rotational from world frame to robot body fixed frame.

    inversely, we can also recover the planar coordinate (described in robot frame) of each ground pixel.

    the cv2 perspective transform is a special case with the assumption that the roll and pitch is zero.

    the detail of mathematical symbolic induction using this geometric model can be found in Mathematica notebook: ```code/perspective_full_equation.nb```

    the symbolic expression obtained by this geometric model replicated as ```numpy``` code  in ```perspective.py```

    the require camera parameters (```view_point and project_plane_origin```) is obtained by numerical optimization, using the calibration correspondence provided by this project (the grid corner pixel position and the position expressed in body fixed frame)

    this numerical optimization is implemented in notebook ```camera_calibration.ipynb```

    in ```perception.ipynb``` video clip, you can see the effectiveness of this approach.

2. Mesh Clustering

    in our problem, we need to cluster the particles from the perspective transform to ray based representation.

    This requires to cluster the particles into rays.

    Since we have a relatively fixed geometric dimension, a generic clustering algorithm (K-means or Euclidean clustering) is not appropriate and the performance is not acceptable.

    Here I used a simple mesh based clustering method, implemented in ```code/mesh_cluster.py```


3. Ray Based Representation

    Ray here is defined as the collection of linear interval on the half line passing through origin

    This geometric entity is particularly helpful in perspective transform interpolation and direction detection

    Rays are geometric entity with two properties:

      - angle, the angle of the ray w.r.t ```x-axis```
      - segments, represented by a list of (start, end) tuple

    the class is define in ```code/ray_detect.py```, and an associated function ```particles_to_rays()``` cluster rays from a set of particles

    the demonstration of the usefulness in this project is in ```perception.ipynb```



## [Rubric](https://review.udacity.com/#!/rubrics/916/view) Points
### All Rubric Points are well addressed as explanation below

---
#### Writeup / README

 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf:

    - You're reading it!

#### Notebook Analysis
 1. Run the functions provided in the notebook on test images (first with the test data provided, next on data you have recorded). Add/modify functions to allow for color selection of obstacles and rock samples:

      - the process to obtain camera parameter is illustrated in ```calibration.ipynb```
      - The Analysis and experiment of perception algorithm is illustrated in notebook ```perception_analysis.ipynb```


2. Populate the `process_image()` function with the appropriate analysis steps to map pixels identifying navigable terrain, obstacles and rock samples into a worldmap.  Run `process_image()` on your test data using the `moviepy` functions provided to create video output of your result:

      - This is also implemented in ```perception_analysis.ipynb```


#### Autonomous Navigation and Mapping

1. Fill in the `perception_step()` (at the bottom of the `perception.py` script) and `decision_step()` (in `decision.py`) functions in the autonomous mapping scripts and an explanation is provided in the writeup of how and why these functions were modified as they were:

    * this code are already implemented, the algorithm and approach is explanation in the above sections: ```Project Approach``` and ```Technical Details```


2. Launching in autonomous mode your rover can navigate and map autonomously.  Explain your results and how you might improve them in your writeup.  

    * the trail on my machine achieves ```78.2% fidelity``` and ```93.8% map coverage``` and recovers the positions of ```6 sample rocks```.


3. The robot direction pick is random hence in some trail it takes some time (about 10 mins) for the rover to finish the mapping (> 95% map rate), below is a snapshot running on my machine (```intel i7 4 phisical cores```, ```anaconda numpy with intel-mkl```, ```cpu utlization:570% - 800%```)

  ![alt text][image1]


#### Parameter to Reproduce the Experiment:


1. frame rate: 8

2. quality: ```Fantastic```

3. resolution: ```1024x768```

#### Future Improvement:

The quality of the mapping and sample retrieval can be further improved by:

1. Apply probability and Bayes filtering method to perception state, to obtain better state and map estimation.

2. Use better computer vision method for image segmentation, to segment objects of interests more accurately.

3. Among the objects of interests, only navigable terrain lives in x-y plane, thus satisfies the assumption of perspective transform. To accurately track all the objects of interests more advanced depth estimation and stereo camera can be used.

4. The reactive control used in this project is simple to implement, better map traverse speed can be achieved by planning-control architecture: a trajectory plan is generated from current map, and a control algorithm is implement to follow the trajectory.
