import numpy as np

def random_orientation_deviation(num_samples=100000):
    rand_quats1 = np.random.randn(num_samples, 4)
    rand_quats2 = np.random.randn(num_samples, 4)
    rand_quats1 /= np.linalg.norm(rand_quats1, axis=1, keepdims=True)
    rand_quats2 /= np.linalg.norm(rand_quats2, axis=1, keepdims=True)
    
    dot_products = np.abs(np.sum(rand_quats1 * rand_quats2, axis=1))
    dot_products = np.clip(dot_products, -1.0, 1.0)
    
    angles = 2 * np.arccos(dot_products)  # radians
    
    mean_angle_deg = np.degrees(np.mean(angles))
    return mean_angle_deg

print(f"Expected random orientation deviation: {random_orientation_deviation():.2f} degrees")

