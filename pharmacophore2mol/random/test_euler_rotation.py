
import numpy as np
import matplotlib.pyplot as plt

def get_euler_rotation_matrix(angles):
    """
    Naive Euler angle rotation (Z-Y-X convention).
    angles: (ax, ay, az) in radians
    """
    ax, ay, az = angles
    
    # Rotation matrices around X, Y, Z axes
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(ax), -np.sin(ax)],
                   [0, np.sin(ax), np.cos(ax)]])
                   
    Ry = np.array([[np.cos(ay), 0, np.sin(ay)],
                   [0, 1, 0],
                   [-np.sin(ay), 0, np.cos(ay)]])
                   
    Rz = np.array([[np.cos(az), -np.sin(az), 0],
                   [np.sin(az), np.cos(az), 0],
                   [0, 0, 1]])
                   
    return Rz @ Ry @ Rx


def get_euler_v2(angles):
    ax, ay, az = angles
    
    # Rotation matrices around X, Y, Z axes
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(ax), -np.sin(ax)],
                   [0, np.sin(ax), np.cos(ax)]])
                   
    Ry = np.array([[np.cos(ay), 0, np.sin(ay)],
                   [0, 1, 0],
                   [-np.sin(ay), 0, np.cos(ay)]])
                   
    Rz = np.array([[np.cos(az), -np.sin(az), 0],
                   [np.sin(az), np.cos(az), 0],
                   [0, 0, 1]])
                   
    return Rx @ Ry @ Rx @ Ry @ Rz @ Rx @ Ry @ Rx

def get_uniform_rotation_matrix_qr(rng):
    """
    Correct uniform sampling on SO(3) using QR decomposition of Gaussian matrix.
    Implements Mezzadri's algorithm to fix QR sign ambiguity bias.
    """
    M = rng.standard_normal((3, 3))
    Q, R = np.linalg.qr(M)
    
    # Fix bias from QR canonical form
    # The diagonal of R gives the signs correction factor
    d = np.diag(R)
    ph = np.sign(d)
    
    # Apply signs to Q to make the distribution uniform
    Q = Q * ph
    
    # Now check determinant to force it into SO(3) (Rotation instead of Reflection)
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    
    
    return Q

def get_uniform_rotation_matrix_euler(rng):
    """
    Generates a uniform random rotation matrix using Euler angles.
    Uses the "Arvo" method (1992) or standard sampling from the Haar measure.
    We use Z-Y-Z (Intrinsic) convention as it is standard for this derivation.
    
    alpha ~ U(0, 2pi)
    gamma ~ U(0, 2pi)
    cos(beta) ~ U(-1, 1)  -> beta = arccos(2*u - 1)
    """
    # Sample uniform variables
    u1 = rng.random()
    u2 = rng.random()
    u3 = rng.random()
    
    # Map to angles
    alpha = 2 * np.pi * u1             # 1st Z rotation
    beta = np.arccos(2 * u2 - 1)       # 2nd axis rotation (from pole to pole)
    gamma = 2 * np.pi * u3             # 3rd axis (Z again) rotation
    
    # Build rotation matrices (Z-X-Z convention)
    # Rz(alpha)
    ca, sa = np.cos(alpha), np.sin(alpha)
    Rz_alpha = np.array([
        [ca, -sa, 0],
        [sa,  ca, 0],
        [0,   0,  1]
    ])
    
    # Rx(beta) - use X axis for the tipping
    cb, sb = np.cos(beta), np.sin(beta)
    Rx_beta = np.array([
        [1,   0,  0],
        [0,  cb, -sb],
        [0,  sb,  cb]
    ])
    
    # Rz(gamma) - Z axis again
    cg, sg = np.cos(gamma), np.sin(gamma)
    Rz_gamma = np.array([
        [cg, -sg, 0],
        [sg,  cg, 0],
        [0,   0,  1]
    ])
    
    # Combine: R = Rz(alpha) * Rx(beta) * Rz(gamma)
    return Rz_alpha @ Rx_beta @ Rz_gamma


def get_perfect_zyx_rotation(rng, max_angles):
    """
    Generates a rotation using Z-Y-X (Yaw-Pitch-Roll) convention that is
    BOTH uniform on the sphere (if max_angles=PI) AND has independent axes.
    
    The trick is sampling the Pitch (Y) angle using arcsin() instead of uniform.
    
    max_angles: (max_yaw, max_pitch, max_roll)
    """
    # 1. Yaw (Z) - Uniform
    # range: [-max_z, max_z]
    yaw = rng.uniform(-max_angles[2], max_angles[2])
    
    # 2. Pitch (Y) - Area-Preserving Sampling (The Fix)
    # The 'bunching' happens because uniform angle != uniform area.
    # We sample sin(beta) uniformly instead of beta.
    # range for sine: [-sin(max_y), sin(max_y)]
    # This is valid for max_y up to 90 degrees (PI/2).
    
    limit_y = max_angles[1]
    # Clamp to PI/2 because ZYX singularity happens at 90 degrees anyway
    if limit_y > np.pi/2: limit_y = np.pi/2
        
    sin_limit = np.sin(limit_y)
    u = rng.uniform(-sin_limit, sin_limit)
    pitch = np.arcsin(u)
    
    # 3. Roll (X) - Uniform
    roll = rng.uniform(-max_angles[0], max_angles[0])
    
    # Build Matrix Z-Y-X
    cz, sz = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    
    cy, sy = np.cos(pitch), np.sin(pitch)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    
    cx, sx = np.cos(roll), np.sin(roll)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    
    return Rz @ Ry @ Rx


def R_axis(axis_idx, angle):
        c, s = np.cos(angle), np.sin(angle)
        if axis_idx == 0: # X
            return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        elif axis_idx == 1: # Y
            return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        elif axis_idx == 2: # Z
            return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        
def get_perfect_axis_tilt_rotation(rng, azimuth_axis, max_azimuth, max_tilt):
    """
    Generates a rotation defined by a primary axis (0=X, 1=Y, 2=Z) acting as the pole.
    1. Azimuth: Uniform rotation AROUND the pole [-max_azimuth, max_azimuth].
    2. Tilt: Uniform deviation OF the pole [0, max_tilt].
    
    This creates a perfect circular cone of probability density around the chosen axis.
    """
    # 1. Sample Azimuth (Spin around pole)
    azimuth = rng.uniform(-max_azimuth, max_azimuth)
    
    # 2. Sample Tilt (Tip away from pole) - Area preserving
    z_min = np.cos(max_tilt)
    u = rng.uniform(z_min, 1.0)
    tilt = np.arccos(u)
    
    # 3. Sample Tilt Direction (Which way to tip?)
    tilt_direction = rng.uniform(0, 2 * np.pi)
    
    # Construct Matrix based on Axis using "Sandwich" logic:
    # R = R_axis(azimuth + phi) @ R_perp(tilt) @ R_axis(-phi)
    
    # Helper for simple axis rotations
    
            
    # We need a perpendicular axis to tip AROUND.
    # If Pole=Z(2), tipping axis can be Y(1).
    # If Pole=Y(1), tipping axis can be X(0).
    # If Pole=X(0), tipping axis can be Z(2).
    perp_axis = (azimuth_axis - 1) % 3 
    
    phi = tilt_direction
    
    # 1. Rotate to align tipping plane
    R_align = R_axis(azimuth_axis, -phi)
    
    # 2. Perform the tilt (using perp axis)
    R_tilt = R_axis(perp_axis, tilt)
    
    # 3. Rotate back + Azimuth
    R_final = R_axis(azimuth_axis, azimuth + phi)
    
    # Composition: Final @ Tilt @ Align
    return R_final @ R_tilt @ R_align, azimuth


def get_rotation_matrix_hybrid(rng, max_angles):
    """
    Hybrid Logic for Nodes.py:
    1. If max_angles are all PI (180 deg), use Arvo/Uniform Sphere method (Z-X-Z with arccos).
    2. If max_angles are restricted (e.g. planar rotation), use independent Naive Euler (which may introduce non-uniformity in cases where more than one axis is allowed to rotate).
    """
    # 1. Check if we want full uniform rotation (180 degrees on all axes)
    if np.all(np.isclose(max_angles, np.pi)):
        # Arvo's method (Fast Random Rotation Matrices, James Arvo 1992)
        # Uses Z-X-Z convention with non-uniform sampling of the middle angle.
        # This is faster than QR and equally correct for SO(3) uniformity.
        
        u1, u2, u3 = rng.random(size=3)
        
        # 1. First Z rotation (alpha)
        alpha = 2 * np.pi * u1
        
        # 2. X rotation (beta) - The "tipping" angle
        # Sample cos(beta) uniformly in [-1, 1], so beta in [0, pi]
        # This distributes the pole correctly.
        beta = np.arccos(2 * u2 - 1) 

        # 3. Second Z rotation (gamma)
        gamma = 2 * np.pi * u3
        
        # Construct matrices for Z-X-Z
        ca, sa = np.cos(alpha), np.sin(alpha)
        Rz_alpha = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])
        
        cb, sb = np.cos(beta), np.sin(beta)
        Rx_beta = np.array([[1, 0, 0], [0, cb, -sb], [0, sb, cb]])
        
        cg, sg = np.cos(gamma), np.sin(gamma)
        Rz_gamma = np.array([[cg, -sg, 0], [sg, cg, 0], [0, 0, 1]])

        return Rz_alpha @ Rx_beta @ Rz_gamma

    else:
        # 2. Limited Rotation: Use naive Euler angles in the defined box.
        # For small angles, the "pole bunching" distortion is negligible.
        # We sample uniformly in the user's defined box [-max, max].
        ax = rng.uniform(-max_angles[0], max_angles[0])
        ay = rng.uniform(-max_angles[1], max_angles[1])
        az = rng.uniform(-max_angles[2], max_angles[2])
        
        # Rotation matrices
        cx, sx = np.cos(ax), np.sin(ax)
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        
        cy, sy = np.cos(ay), np.sin(ay)
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        
        cz, sz = np.cos(az), np.sin(az)
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        
        return Rz @ Ry @ Rx

def visualize_rotations(n_samples=5000):
    start_vec = np.array([1.0, 0.0, 0.0]) # Point on x-axis
    
    # 0=X, 1=Y, 2=Z
    azimuth_axis = 2  # <--- Change this manually (0=X, 1=Y, 2=Z)
    max_azimuth = np.pi  # Full 360 spin
    max_tilt = np.deg2rad(30) # 15 degree cone
    
    euler_points = []
    qr_points = []
    
    azimuth_colors = []

    rng = np.random.default_rng(42)
    
    # Generate points
    print(f"Generating {n_samples} random rotations and applying to vector {start_vec}...")
    for _ in range(n_samples):
        # 1. Perfect Axis-Tilt Rotation
        # This will test the cone/azimuth logic
        R_euler_mat, az = get_perfect_axis_tilt_rotation(rng, azimuth_axis, max_azimuth, max_tilt)
        euler_points.append(R_euler_mat @ start_vec)
        azimuth_colors.append(az) # Store for coloring
        
        # 2. Correct QR (Haar Uniform) - as control
        Q = get_uniform_rotation_matrix_qr(rng)
        qr_points.append(Q @ start_vec)
    
    exit()
    print("Done generating points. Now plotting...")
    euler_points = np.array(euler_points)
    qr_points = np.array(qr_points)
    
    # Normalize azimuth for colormap [-pi, pi] -> [0, 1]
    norm = plt.Normalize(-max_azimuth, max_azimuth)
    colors = plt.cm.hsv(norm(azimuth_colors))

    # Plot
    fig = plt.figure(figsize=(14, 7))
    
    # Wireframe sphere for reference
    u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
    x = np.cos(u)*np.sin(v)
    y = np.sin(u)*np.sin(v)
    z = np.cos(v)

    # Euler Plot
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_wireframe(x, y, z, color="gray", alpha=0.1)
    
    # Use computed colors
    sc = ax1.scatter(euler_points[:,0], euler_points[:,1], euler_points[:,2], s=2, alpha=0.6, c=colors)
    
    ax1.set_title(f"Perfect Axis-Tilt Sampling\nAxis={azimuth_axis}, Tilt={np.rad2deg(max_tilt):.1f}deg")
    ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
    
    # Add colorbar for Azimuth
    sm = plt.cm.ScalarMappable(cmap=plt.cm.hsv, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax1, label='Azimuth Angle (rad)')

    # QR Plot
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_wireframe(x, y, z, color="gray", alpha=0.1)
    ax2.scatter(qr_points[:,0], qr_points[:,1], qr_points[:,2], s=2, alpha=0.6, c='green')
    ax2.set_title(f"QR Decomposition (Control)\nComplete Uniform Sphere")
    ax2.set_xlabel('X'); ax2.set_ylabel('Y'); ax2.set_zlabel('Z')
    # ax2.set_aspect('equal')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    try:
        visualize_rotations()
    except Exception as e:
        print(e)
