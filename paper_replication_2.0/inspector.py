import pickle

# Load the pickle file
with open('data_murty_185/all_data.pickle', 'rb') as f:
    fmri_data = pickle.load(f)

# Inspect the type and structure of the data
# print(f"Type of the data: {type(fmri_data)}")

# If it's a numpy array or list, print the shape
#if isinstance(fmri_data, (list, np.ndarray)):
    #print(f"Shape of the data: {len(fmri_data)}")

# If it's a dictionary, rrint the keys of the dictionary
# print(f"Keys in the dictionary: {list(fmri_data.keys())}")

# output: Keys in the dictionary: ['dp', 'p1', 'p2', 'p3', 'p4']

# If the dictionary contains nested dictionaries, recursively explore the structure
# for key, value in fmri_data.items():
#     print(f"Key: {key}, Type of value: {type(value)}")
    
#     if isinstance(value, dict):
#         # Explore the nested dictionary
#         for sub_key, sub_value in value.items():
#             print(f"  Sub-key: {sub_key}, Sub-value type: {type(sub_value)}")

# output can be:
# Type of the data: <class 'dict'>
# Keys in the dictionary: ['dp', 'p1', 'p2', 'p3', 'p4']
# Key: dp, Type of value: <class 'dict'>
#   Sub-key: rffa, Sub-value type: <class 'numpy.ndarray'>
#   Sub-key: lffa, Sub-value type: <class 'numpy.ndarray'>
#   Sub-key: reba, Sub-value type: <class 'numpy.ndarray'>
#   Sub-key: leba, Sub-value type: <class 'numpy.ndarray'>
# Key: p1, Type of value: <class 'dict'>
#   Sub-key: lffa, Sub-value type: <class 'numpy.ndarray'>
#   Sub-key: rffa, Sub-value type: <class 'numpy.ndarray'>
#   Sub-key: leba, Sub-value type: <class 'numpy.ndarray'>
#   Sub-key: reba, Sub-value type: <class 'numpy.ndarray'>
#   Sub-key: lppa, Sub-value type: <class 'numpy.ndarray'>
#   Sub-key: rppa, Sub-value type: <class 'numpy.ndarray'>
# Key: p2, Type of value: <class 'dict'>
#   Sub-key: lffa, Sub-value type: <class 'numpy.ndarray'>
#   Sub-key: rffa, Sub-value type: <class 'numpy.ndarray'>
#   Sub-key: leba, Sub-value type: <class 'numpy.ndarray'>
#   Sub-key: reba, Sub-value type: <class 'numpy.ndarray'>
#   Sub-key: lppa, Sub-value type: <class 'numpy.ndarray'>
#   Sub-key: rppa, Sub-value type: <class 'numpy.ndarray'>
# Key: p3, Type of value: <class 'dict'>
#   Sub-key: lffa, Sub-value type: <class 'numpy.ndarray'>
#   Sub-key: rffa, Sub-value type: <class 'numpy.ndarray'>
#   Sub-key: leba, Sub-value type: <class 'numpy.ndarray'>
#   Sub-key: reba, Sub-value type: <class 'numpy.ndarray'>
#   Sub-key: lppa, Sub-value type: <class 'numpy.ndarray'>
#   Sub-key: rppa, Sub-value type: <class 'numpy.ndarray'>
# Key: p4, Type of value: <class 'dict'>
#   Sub-key: lffa, Sub-value type: <class 'numpy.ndarray'>
#   Sub-key: rffa, Sub-value type: <class 'numpy.ndarray'>
#   Sub-key: leba, Sub-value type: <class 'numpy.ndarray'>
#   Sub-key: reba, Sub-value type: <class 'numpy.ndarray'>
#   Sub-key: lppa, Sub-value type: <class 'numpy.ndarray'>
#   Sub-key: rppa, Sub-value type: <class 'numpy.ndarray'>

# The value for each brain region key is a NumPy array containing the fMRI data for that region. 
# These arrays represent the brain activation or signal intensity in the specified region.

# Check the shape of data
print(f"Shape of lffa data in p1: {fmri_data['p1']['lffa'].shape}")

# output: Shape of lffa data in p1: (185, 325)
# 185: stimuli images number, 325: number of voxels in one area, each voxel represents a small 3D volume of brain tissue
# Each row corresponds to the fMRI response to a different stimulus, and the 325 columns represent the voxel-level activation in response to that stimulus in the Left FFA.

p1_data = fmri_data['p1']

total_voxels = 0

for region, voxel_data in p1_data.items():
    print(f"Region: {region}, Number of voxels: {voxel_data.shape[1]}")

    total_voxels += voxel_data.shape[1]
    
print(f"Total number of voxels in P1: {total_voxels}")

# Initialize a variable to track voxel index positions
voxel_start_idx = 0

# Print the ordering of the voxels
print("Voxel ordering for P1:")

for region, voxel_data in p1_data.items():
    num_voxels = voxel_data.shape[1]  # Assuming second dimension is number of voxels
    voxel_end_idx = voxel_start_idx + num_voxels
    print(f"Region: {region}, Voxel range: {voxel_start_idx} to {voxel_end_idx - 1}")
    voxel_start_idx = voxel_end_idx  # Move the start index to the next region

# Shape of lffa data in p1: (185, 325)
# Region: lffa, Number of voxels: 325
# Region: rffa, Number of voxels: 463
# Region: leba, Number of voxels: 535
# Region: reba, Number of voxels: 1121
# Region: lppa, Number of voxels: 715
# Region: rppa, Number of voxels: 840
# Total number of voxels in P1: 3999
# Voxel ordering for P1:
# Region: lffa, Voxel range: 0 to 324
# Region: rffa, Voxel range: 325 to 787
# Region: leba, Voxel range: 788 to 1322
# Region: reba, Voxel range: 1323 to 2443
# Region: lppa, Voxel range: 2444 to 3158
# Region: rppa, Voxel range: 3159 to 3998

