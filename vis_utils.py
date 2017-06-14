""" net visualization utilities
A few functions to visualize units 
This set of functions operate on numpy arrays.

\author Luis Gonzalo Sanchez Giraldo (University of Miami) 
"""
import numpy as np
import math

def tilePatches(patch_set, ncol=None, nspace=1):
    """ Tile patches on a image for visualization
    Input: 
        -- patch_set: 3 or 4-dimensional numpy array. Last dimension is the patch index, 
        or in other words the number of tiles.
        -- ncol: is the number of patches to be tiled horizontally.
        -- nspace: is the number of pixels between tiles.
    Output:
        -- tiled_patches: An image containng the tiled patches
    """
    n_patches = patch_set.shape[-1]
    if ncol is None:
        ncol = int(math.ceil(math.sqrt(n_patches)))
    nrow = n_patches/ncol + int(n_patches%ncol > 0)
        
    tiled_nrows = (patch_set.shape[0]+nspace)*(nrow) - nspace
    tiled_ncols = (patch_set.shape[1]+nspace)*(ncol) - nspace 
    if len(patch_set.shape) == 4:
        tiled_depth = patch_set.shape[2]
    elif len(patch_set.shape) == 3:
        tiled_depth = 1
    tiled_image = np.ones([tiled_nrows, tiled_ncols, tiled_depth])*np.max(patch_set);
    for iRow in range(nrow):
        for iCol in range(ncol):
            t_row_str = iRow*(patch_set.shape[0] + nspace);
            t_col_str = iCol*(patch_set.shape[1] + nspace);
            t_row_end = (iRow + 1)*(patch_set.shape[0] + nspace) - nspace;
            t_col_end = (iCol + 1)*(patch_set.shape[1] + nspace) - nspace;
            f_idx = (iRow)*ncol + iCol;
            if f_idx >= n_patches:
                break
            if len(patch_set.shape) == 4:
                tiled_image[t_row_str:t_row_end, t_col_str:t_col_end, :] = patch_set[:, :, : , f_idx]
            else:
                tiled_image[t_row_str:t_row_end, t_col_str:t_col_end,0] = patch_set[:, :, f_idx]
    return tiled_image
