import sys
import os
import logging
import cv2
import numpy as np
import open3d as o3d
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.sparse as ss
import scipy.sparse.linalg

# Configure logging
logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
image_row = 0
image_col = 0

# Use TkAgg backend for matplotlib
mpl.use("TkAgg")


def mask_visualization(M, save_path=None):
    """Visualizing the mask (size : "image width" * "image height")"""
    global image_row, image_col
    mask = np.copy(np.reshape(M, (image_row, image_col)))
    plt.figure()
    plt.imshow(mask, cmap='gray')
    plt.title('Mask')
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Mask visualization saved to: {save_path}")


def normal_visualization(N, save_path=None):
    """Visualizing the unit normal vector in RGB color space"""
    global image_row, image_col
    N_map = np.copy(np.reshape(N, (image_row, image_col, 3)))
    # Rescale to [0,1] float number
    N_map = (N_map + 1.0) / 2.0
    plt.figure()
    plt.imshow(N_map)
    plt.title('Normal map')
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Normal map visualization saved to: {save_path}")


def depth_visualization(D, save_path=None):
    """Visualizing the depth on 2D image"""
    global image_row, image_col
    D_map = np.copy(np.reshape(D, (image_row, image_col)))
    plt.figure()
    plt.imshow(D_map)
    plt.colorbar(label='Distance to Camera')
    plt.title('Depth map')
    plt.xlabel('X Pixel')
    plt.ylabel('Y Pixel')
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Depth map visualization saved to: {save_path}")


def save_ply(Z, filepath):
    """Convert depth map to point cloud and save it to ply file"""
    global image_row, image_col

    Z_map = np.reshape(Z, (image_row, image_col)).copy()
    data = np.zeros((image_row * image_col, 3), dtype=np.float32)

    # Let all point float on a base plane
    baseline_val = np.min(Z_map)
    Z_map[np.where(Z_map == 0)] = baseline_val

    for i in range(image_row):
        for j in range(image_col):
            idx = i * image_col + j
            data[idx][0] = j
            data[idx][1] = i
            data[idx][2] = Z_map[image_row - 1 - i][j]
    # Output to ply file
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data)
    o3d.io.write_point_cloud(filepath, pcd, write_ascii=True)
    logger.info(f"Point cloud saved to: {filepath}")


def show_ply(filepath):
    """Show the result of saved ply file"""
    pcd = o3d.io.read_point_cloud(filepath)
    o3d.visualization.draw_geometries([pcd], point_show_normal=True)


def read_bmp(filepath):
    """Read the .bmp file"""
    global image_row, image_col
    if not os.path.exists(filepath):
        logger.error(f"Image file not found: {filepath}")
        return None

    image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if image is None:
        logger.error(f"Failed to read image: {filepath}")
        return None

    image_row, image_col = image.shape
    return image


def read_lightsrc(filepath):
    """Read the light source"""
    if not os.path.exists(filepath):
        logger.error(f"Light source file not found: {filepath}")
        return None

    l = []
    try:
        with open(filepath) as f:
            for i in range(0, 6):
                x = f.read(6)
                x = f.readline().replace(")\n", "").replace("(", "")
                x = x.split(",")
                l.append(x)

        l = np.array(l).astype(float)
        for i in range(l.shape[0]):
            t = np.linalg.norm(l[i])
            l[i] = l[i] / t

        return l
    except Exception as e:
        logger.error(f"Error reading light source file: {e}")
        return None


def normal_estimation(image_lst, L):
    """Calculate normal map"""
    global image_row, image_col

    normal_map = np.zeros((image_row, image_col, 3))

    for x in range(image_row):
        for y in range(image_col):
            I_lst = []
            for num in range(len(image_lst)):
                I_lst.append(image_lst[num][x][y])
            I_vec = np.array(I_lst)
            KdN = np.dot(np.dot(np.linalg.inv(np.dot(L.T, L)), L.T), I_vec)
            if KdN.any() == 0:
                N = KdN
            else:
                N = KdN / np.linalg.norm(KdN)
            normal_map[x][y] = N

    return normal_map


def create_mask(N):
    """Create a mask"""
    global image_row, image_col

    mask = np.zeros((image_row, image_col))
    for x in range(image_row):
        for y in range(image_col):
            if N[x][y].any() != 0:
                mask[x][y] = 1

    valid_pixels = np.sum(mask)

    return mask, valid_pixels


def depth_estimation(N, mask):
    """Calculate depth map"""
    global image_row, image_col

    row_idx, col_idx = np.where(mask != 0)
    pix = row_idx.size
    pix_order = np.copy(mask)

    for i in range(pix):
        pix_order[row_idx[i]][col_idx[i]] = i

    # Generate M & V
    row = np.empty(0)
    col = np.empty(0)
    val = np.empty(0)
    V = np.zeros((2 * pix, 1))

    for i in range(pix):
        # Image coordinate
        img_x = row_idx[i]
        img_y = col_idx[i]
        # Normal vector
        nx = N[img_x, img_y, 0]
        ny = N[img_x, img_y, 1]
        nz = N[img_x, img_y, 2]

        # Real y direction
        row_y = 2 * i
        if mask[img_x + 1, img_y]:
            row = np.append(row, [row_y, row_y])
            col = np.append(col, [i, pix_order[img_x + 1][img_y]])
            val = np.append(val, [1, -1])
            if nz == 0:
                V[row_y][0] = 0
            else:
                V[row_y][0] = (-ny / nz)

        # Real x direction
        row_x = 2 * i + 1
        if mask[img_x, img_y + 1]:
            row = np.append(row, [row_x, row_x])
            col = np.append(col, [i, pix_order[img_x][img_y + 1]])
            val = np.append(val, [-1, 1])
            if nz == 0:
                V[row_x][0] = 0
            else:
                V[row_x][0] = (-nx / nz)

    M = ss.csr_matrix((val, (row, col)), shape=(2 * pix, pix))

    logger.info("Solving linear system for depth estimation...")
    MtM = M.T @ M
    MtV = M.T @ V
    z = scipy.sparse.linalg.spsolve(MtM, MtV)

    # Transfer z to z_map
    z_map = np.copy(mask)
    index = 0
    for x in range(image_row):
        for y in range(image_col):
            if z_map[x][y] != 0:
                z_map[x][y] = z[index]
                index += 1

    z_map = np.clip(z_map, -40, 20)

    return z_map


def process_object(obj_name):
    """Process a single object"""
    logger.info(f"Processing object: {obj_name}")

    obj_dir = os.path.join("testcase", obj_name)
    if not os.path.exists(obj_dir):
        logger.error(f"Object directory not found: {obj_dir}")
        return False

    image_paths = [os.path.join(obj_dir, f"pic{i}.bmp") for i in range(1, 7)]
    all_images = []

    for img_path in image_paths:
        image = read_bmp(img_path)
        if image is None:
            logger.error(f"Failed to load image: {img_path}")
            return False
        all_images.append(image)
    logger.info(f"Loaded 6 images: {obj_dir} pic1~6.bmp {all_images[0].shape}")

    light_path = os.path.join(obj_dir, "LightSource.txt")
    logger.info(f"Reading light source from {light_path}...")
    Light_src = read_lightsrc(light_path)
    if Light_src is None:
        return False
    logger.info(f"Light source loaded successfully (6 light directions)")

    logger.info("Starting normal estimation...")
    N = normal_estimation(all_images, Light_src)
    normal_visualization(N)
    plt.savefig(os.path.join(obj_dir, "normal_map.png"))
    logger.info("Normal estimation completed")

    logger.info("Creating mask...")
    Mask, valid_pixels = create_mask(N)
    mask_visualization(Mask)
    plt.savefig(os.path.join(obj_dir, "mask.png"))
    logger.info(f"Mask created with {valid_pixels} valid pixels")

    logger.info("Starting depth estimation...")
    Z = depth_estimation(N, Mask)
    depth_visualization(Z)
    plt.savefig(os.path.join(obj_dir, "depth_map.png"))
    logger.info("Depth estimation completed")

    ply_path = os.path.join(obj_dir, f"{obj_name}.ply")
    save_ply(Z, ply_path)
    show_ply(ply_path)

    logger.info(f"Completed: {obj_name}\n")
    return True


def main():
    if len(sys.argv) < 2:
        print("Usage: python exe.py <model1> [<model2> ...]")
        sys.exit(1)

    model_list = sys.argv[1:]
    for model_name in model_list:
        process_object(model_name)
    print("All tasks completed.")


if __name__ == '__main__':
    main()
