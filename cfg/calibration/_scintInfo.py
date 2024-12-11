import numpy as np
from scipy.constants import c

scintInfo = [
    [ # pos_corners
        [ # scint 1
            [-33.72417075,  0.6330881765 , 8.200910978],
            [-33.79664659,  -1.465871533 , 8.409910456],
            [-34.54870527,  -1.410340439 , 8.616906447],
            [-34.40287908,  0.6997246309 , 8.375907352],
        ],
        [ # scint 2
            [-48.72475541,  -7.3506648  , 13.17800999],
            [-48.01778605,  -10.27033603,  13.4621113],
            [-48.73978881,  -9.004291085,  13.20240775],
            [-50.08747321,  -6.546610366,  13.50320033],
        ],
        [ # scint 3
            [-35.76569522,  -17.7244834  , 11.77007522],
            [-35.8514241,   -19.53469739 , 11.52566944],
            [-36.62116271,  -19.64908378 , 12.14996472],
            [-35.80193052,  -17.71448926 , 12.15387505],
        ],
        [ # scint 4
            [-34.90941878,-33.76765383,11.15621504],
            [-34.96156604,-36.02209014,10.6491024],
            [-36.02205192,-35.64338593,10.95149864],
            [-35.74013245,-33.88536925,11.05510982],
        ],
    ],
    [ # pos_centers
        [-34.11832134,  -0.3852945007, 8.400908904], # scint 1
        [-48.89178817,  -8.292420328,  13.33640755], # scint 2
        [-36.00961123,  -18.6551332, 11.89987119], # scint 3
        [-35.40873416, -34.82934718, 10.95300657], # scint 4
    ],
    [ # cable delays (referenced to ch 0); Comes from stacked scint runs 21-27
        0,# 1e9*70/c,
        -30.63,# 1e9*60/c,
        -0.74,# 1e9*70/c,
        0.06,# 1e9*70/c,
    ],
    [ # tilts [N-S, E-W] (+ = southern/western edge higher than northern/eastern edge)
        [5.3, 14.2],
        [3.2, 18.0],
        [-2.3, 17.3],
        [0.7, 15.0],
    ],
    [ # dimensions [length, width, depth] in m (https://pos.sissa.it/301/401/pdf)
        [1.875, 0.8, 0.01],
        [1.875, 0.8, 0.01],
        [1.875, 0.8, 0.01],
        [1.875, 0.8, 0.01],
    ],
    [ # face normal vectors in ENU; calculated first order using independent rotations by their NS and EW tilts
        [ # scint 1
            [ 0.24530739,  0.08954824,  0.96530068], # length * width face (UPish)
            [ 0.96944535, -0.02265919, -0.24425862], # length * depth face (EASTish)
            [ 0.        ,  0.9957247 , -0.09237059], # width * depth face (NORTHish)
        ],
        [ # scint 2
            [ 0.30901699,  0.05308941,  0.94957359],
            [ 0.95105652, -0.01724979, -0.30853516],
            [ 0.        ,  0.99844076, -0.0558215 ],
        ],
        [ # scint 3
            [ 0.29737487, -0.03831626,  0.95399164],
            [ 0.9547608 ,  0.01193419, -0.29713531],
            [ 0.        ,  0.9991944 ,  0.04013179],
        ],
        [ # scint 4
            [ 0.25881905,  0.01180072,  0.96585374],
            [ 0.96592583, -0.00316199, -0.25879973],
            [ 0.        ,  0.99992537, -0.012217  ],
        ],
    ],
]

np.savez('scint.npz', pos_corners=scintInfo[0], pos_centers=scintInfo[1], cables=scintInfo[2], tilts=scintInfo[3], 
         dimensions=scintInfo[4], face_normals=scintInfo[5])