import struct
import numpy as np

def inspect_coordinates(file_path):
    with open(file_path, 'rb') as f:
        data = f.read()
    
    n_points = len(data) // 20
    print(f"Number of points: {n_points}")
    
    positions = []
    for i in range(n_points):
        offset = i * 20
        x, y, z = struct.unpack('<fff', data[offset:offset+12])
        positions.append([x, y, z])
    
    positions = np.array(positions)
    print(f"X range: {positions[:, 0].min()} to {positions[:, 0].max()}")
    print(f"Y range: {positions[:, 1].min()} to {positions[:, 1].max()}")
    print(f"Z range: {positions[:, 2].min()} to {positions[:, 2].max()}")
    print(f"Mean: {positions.mean(axis=0)}")

if __name__ == "__main__":
    inspect_coordinates('sciconnav-gemini/data/concept_coordinates_128.bin')