import numpy as np
import matplotlib.pyplot as plt

    
def create_2D_points_on_arc(x0, x1, R, ds):
    # Calculate midpoint of the chord
    xm = (x0 + x1) / 2
    # Calculate the distance between x0 and x1
    chord_length = abs(x1 - x0)
    # Calculate the distance from the midpoint to the center of the circle
    d = np.sqrt(R**2 - (chord_length / 2)**2)
    
    ym = -d
    
    # Calculate the angle corresponding to the arc length ds
    delta_theta = np.atan(0.5 * chord_length / d) * 2
    diameter = R * 2
    circumference = diameter * np.pi
    
    arc_length = circumference * delta_theta / (2 * np.pi)
    n_segments = int(arc_length / ds) 
    d_theta = delta_theta / n_segments


    vec_m_to_1 = np.array(
        [
        - chord_length / 2,
        d
    ]
    )

    l_vectors = []
    for i in range(n_segments+1):
        d_theta_i = d_theta * i

        A = np.array([
            [np.cos(d_theta_i),  np.sin(d_theta_i)],
            [-np.sin(d_theta_i), np.cos(d_theta_i)]
        ])

        vec_rotated = A @ vec_m_to_1
        l_vectors.append(vec_rotated)
    
    ar_vectors_relative = np.array(l_vectors)

    ar_coords = ar_vectors_relative 
    return ar_coords - vec_m_to_1


def extend_arc_3D(ar, R):
    ar_x = ar[:,0]
    x0 = ar_x.min()
    x1 = ar_x.max()
    xm = (x0 + x1) / 2
    chord_length = abs(x1 - x0)
    d = np.sqrt(R**2 - (chord_length / 2)**2)
    ym = -d
    

    ar_y = ar[:,1]
    ar_z = -1 * (np.sqrt(R**2 - (ar_x - xm)**2) + ym)
    return np.column_stack([ar_x, ar_y, ar_z])
        


def create_double_arc_geometry(x0, z0, x1, z1, R_y, R_z, ds):
    ar_arc_y = create_2D_points_on_arc(x0, x1, R_y, ds)
    ar_arc_yz = extend_arc_3D(ar_arc_y, R_z)
    # print(f'''
    # ar_arc_y.shape : {ar_arc_y.shape}
    # ar_arc_yz.shape : {ar_arc_yz.shape}
    # ''')

    # print('ar_arc_y')
    # print(ar_arc_y)
    # print('======='*5)
    # print('ar_arc_yz')
    # print(ar_arc_yz)
    # print('======='*5)
    n_points = ar_arc_y.shape[0]
    ar_z0 = np.linspace(z0, z1, n_points)
    # print(ar_z0)
    ar_arc_yz[:,2] += ar_z0
    return ar_arc_yz

if __name__ == '__main__':
    ar = create_double_arc_geometry(x0=0,z0=-30,x1=25000, z1=-30, R_y=250000, R_z=250000, ds=1)
    print(ar.shape)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(ar[:,0], ar[:,1], ar[:,2])
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label hoi')
    fig.savefig('points.png')