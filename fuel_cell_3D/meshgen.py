import tifffile
import numpy as np
from scipy.ndimage import label, generate_binary_structure

refinement_factor = 2*1.26 # problem size = refinement_factor^3 * 2k

num_vexels_x = int(10*refinement_factor)
num_vexels_y = int(20*refinement_factor)
num_vexels_z = int(10*refinement_factor)

total_num_vexels = num_vexels_x*num_vexels_y*num_vexels_z

M_3d = np.zeros((num_vexels_x,num_vexels_y,num_vexels_z)).astype(np.uint8)
M_3d[:,:int(num_vexels_y/2),:] = 2
M_3d[:,int(num_vexels_y/2):,:int(num_vexels_z/2)] = 1
tifffile.imwrite('M_3d_3phases_num_cell_'+str(total_num_vexels)+'.tif', M_3d, compression='lzw')

