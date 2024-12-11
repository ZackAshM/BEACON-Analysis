File `rfhpol.npz` contains 'pos' and 'cables' corresponding to ENU positions relative to antenna 1 and cable delays in ns
- This file was made by Andrew Zeolla by calibrating pulsed (ground/drone) data in HPOL to obtain precise positions and delays.
- VPOL positions can be approximated to be the same

File `scint.npz` contains 'pos_corners', 'pos_centers', 'cables', 'tilts', 'dimensions', and 'face_normals' corresponding to the ENU positions of the corners and centers of the scint panels, the cable delays in ns referenced to ch 0 (scint 1), the N-S and E-W tilts, the length, width, and depth of the detectors, and the ENU normal vectors of the faces