# Auto Align


A blender add-on that automatically aligns objects parallel to the world axis.

![inform](https://user-images.githubusercontent.com/49553394/167562850-3d33a7bc-5613-444d-a69c-d27a0e06b169.gif)

## Usage

The figure below briefly shows how this add-on works. The axis represents the local axis of the model (Dual-clutch gearbox HD by [Artec 3D](https://www.artec3d.com/3d-models/dual-clutch-gearbox-hd)).

<img src="https://user-images.githubusercontent.com/49553394/167296309-6ee7b458-dd90-46a1-b57d-1fd521bc70f5.png" width="420" height="420">

There are three options available in the `3D Viewport > Sidebar > Item > Auto Align`.

* `Rotate` : Rotate the selected object to match the world axis.
* `Rotate & Bake` : Same as `Rotate`, but rotation is applied.
* `Keep Position & Bake` : Keep the object fixed and only change the local axis.

## Algorithm

The proper orientation of the object is calculated as follows:

1. RANSAC (Random sample consensus)
   * Randomly select two orthogonal faces to create a candidate orientation. 
   * Find the faces that corresponding to the axis within a specific threshold (<5 deg).
   * Criterion is to maximize the sum of the areas of the face.

2. Correction
   * Fine-tune the candidate orientation.
   * Find the weighted median of the points on each axis and rotate the orientation.
   * Repeat until convergence.

3. XYZ adjustment
   * To avoid continuous flips and maintain consistency while repeating the same operation.
   * Determine XYZ axis to have the least difference from the initial mesh orientation.

For each object, only up to 10000 faces are considered in the process. This is due to performance reason. If there are more than 10000 faces, larger faces would be considered first.
