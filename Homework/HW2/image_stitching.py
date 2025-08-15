import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
from typing import List, Tuple, Optional
import logging
from tabulate import tabulate

# Set up logging
logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


class Stitcher:
    """
    Image stitching class using SIFT features and homography estimation.
    """

    def __init__(self,
                 ransac_threshold: float = 1.0,
                 ransac_iterations: int = 1000):
        """
        Initialize the stitcher with RANSAC parameters.
        
        Args:
            ransac_threshold: Distance threshold for RANSAC inliers
            ransac_iterations: Number of RANSAC iterations
        """
        self.ransac_threshold = ransac_threshold
        self.ransac_iterations = ransac_iterations

    def stitch(self,
               paths: List[str],
               blending_mode: str = "linearBlending",
               matching_threshold: float = 0.75) -> np.ndarray:
        """
        Main stitching function.
        
        Args:
            paths: List of image paths [left_path, right_path]
            blending_mode: Blending method ('noBlending', 'linearBlending', 'linearBlendingWithConstant')
            matching_threshold: Lowe's ratio test threshold
            
        Returns:
            Stitched image as numpy array
        """
        if len(paths) != 2:
            raise ValueError("Exactly two image paths are required")

        left_path, right_path = paths

        # Load images
        left_gray, left_color = self._read_image(left_path)
        right_gray, right_color = self._read_image(right_path)

        # Extract features
        kp_left, des_left = self._extract_sift_features(left_gray)
        kp_right, des_right = self._extract_sift_features(right_gray)

        logger.info(
            f"Left keypoints: {len(kp_left)}, Right keypoints: {len(kp_right)}"
        )

        # Match features
        matches = self._match_features(kp_left, kp_right, des_left, des_right,
                                       matching_threshold)
        logger.info(f"Matches found: {len(matches)}")

        if len(matches) < 4:
            raise ValueError(
                "Not enough matches found for homography estimation")

        # Visualize matches
        self._draw_matches([left_color, right_color], matches)

        # Estimate homography using RANSAC
        logger.info("Estimating homography using RANSAC...")
        homography = self._estimate_homography_ransac(matches)
        logger.info("Homography estimation completed")

        # Warp and blend images
        logger.info("Starting Warping...")
        result = self._warp_and_blend([left_color, right_color], homography,
                                      blending_mode)

        return result

    def _read_image(self, path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Read image and convert to grayscale.
        
        Args:
            path: Image file path
            
        Returns:
            Tuple of (grayscale_image, color_image)
        """
        try:
            color_img = cv2.imread(path)
            if color_img is None:
                raise ValueError(f"Could not load image: {path}")
            gray_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
            return gray_img, color_img
        except Exception as e:
            logger.error(f"Error reading image {path}: {e}")
            raise

    def _extract_sift_features(self,
                               image: np.ndarray) -> Tuple[List, np.ndarray]:
        """
        Extract SIFT features from image.
        
        Args:
            image: Grayscale image
            
        Returns:
            Tuple of (keypoints, descriptors)
        """
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(image, None)
        return keypoints, descriptors

    def _match_features(
            self, kp_left: List, kp_right: List, des_left: np.ndarray,
            des_right: np.ndarray,
            threshold: float) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Match features using Lowe's ratio test.
        
        Args:
            kp_left, kp_right: Keypoints from left and right images
            des_left, des_right: Descriptors from left and right images
            threshold: Lowe's ratio test threshold
            
        Returns:
            List of matched point pairs
        """
        if des_left is None or des_right is None:
            return []

        matches = []

        for i, desc_left in enumerate(des_left):
            # Find two nearest neighbors
            distances = np.linalg.norm(des_right - desc_left, axis=1)
            sorted_indices = np.argsort(distances)

            if len(sorted_indices) < 2:
                continue

            nearest_dist = distances[sorted_indices[0]]
            second_nearest_dist = distances[sorted_indices[1]]

            # Lowe's ratio test
            if nearest_dist < threshold * second_nearest_dist:
                pt_left = (int(kp_left[i].pt[0]), int(kp_left[i].pt[1]))
                pt_right = (int(kp_right[sorted_indices[0]].pt[0]),
                            int(kp_right[sorted_indices[0]].pt[1]))
                matches.append((pt_left, pt_right))

        return matches

    def _draw_matches(self, images: List[np.ndarray], matches: List) -> None:
        """
        Visualize feature matches between two images.
        
        Args:
            images: List of [left_image, right_image]
            matches: List of matched point pairs
        """
        img_left, img_right = images
        h_left, w_left = img_left.shape[:2]
        h_right, w_right = img_right.shape[:2]

        # Create visualization canvas
        canvas = np.zeros((max(h_left, h_right), w_left + w_right, 3),
                          dtype=np.uint8)
        canvas[:h_left, :w_left] = img_left
        canvas[:h_right, w_left:] = img_right

        # Draw matches
        for pt_left, pt_right in matches:
            pt_right_shifted = (pt_right[0] + w_left, pt_right[1])

            # Draw keypoints
            cv2.circle(canvas, pt_left, 3, (0, 0, 255), 2)
            cv2.circle(canvas, pt_right_shifted, 3, (0, 255, 0), 2)

            # Draw connecting line
            cv2.line(canvas, pt_left, pt_right_shifted, (255, 0, 0), 1)

        # Display and save
        cv2.imshow("Feature Matches", canvas)
        cv2.waitKey(0)
        cv2.imwrite("feature_matches.jpg", canvas)
        cv2.destroyAllWindows()

    def _estimate_homography_ransac(self, matches: List) -> np.ndarray:
        """
        Estimate homography matrix using RANSAC.
        
        Args:
            matches: List of matched point pairs
            
        Returns:
            3x3 homography matrix
        """
        if len(matches) < 4:
            raise ValueError(
                "At least 4 matches required for homography estimation")

        # Convert matches to numpy arrays
        src_pts = np.array([match[1] for match in matches],
                           dtype=np.float32)  # right image
        dst_pts = np.array([match[0] for match in matches],
                           dtype=np.float32)  # left image

        best_homography = None
        max_inliers = 0

        for _ in range(self.ransac_iterations):
            # Randomly select 4 points
            indices = random.sample(range(len(matches)), 4)
            sample_src = src_pts[indices]
            sample_dst = dst_pts[indices]

            # Estimate homography
            try:
                homography = self._solve_homography(sample_src, sample_dst)
                if homography is None:
                    continue

                # Count inliers
                inliers = self._count_inliers(src_pts, dst_pts, homography)

                if inliers > max_inliers:
                    max_inliers = inliers
                    best_homography = homography

            except Exception as e:
                logger.debug(f"Error in homography estimation: {e}")
                continue

        logger.info(f"Maximum inliers: {max_inliers}")
        return best_homography

    def _solve_homography(self, src_pts: np.ndarray,
                          dst_pts: np.ndarray) -> Optional[np.ndarray]:
        """
        Solve homography using DLT (Direct Linear Transform).
        
        Args:
            src_pts: Source points (4x2)
            dst_pts: Destination points (4x2)
            
        Returns:
            3x3 homography matrix or None if failed
        """
        try:
            # Construct matrix A for Ah = 0
            A = []
            for i in range(4):
                x, y = src_pts[i]
                u, v = dst_pts[i]

                A.append([-x, -y, -1, 0, 0, 0, x * u, y * u, u])
                A.append([0, 0, 0, -x, -y, -1, x * v, y * v, v])

            A = np.array(A)

            # Solve using SVD
            _, _, vt = np.linalg.svd(A)
            h = vt[-1]

            # Reshape to 3x3 matrix
            H = h.reshape(3, 3)

            # Normalize
            if abs(H[2, 2]) > 1e-8:
                H = H / H[2, 2]
                return H
            else:
                return None

        except Exception as e:
            logger.debug(f"Error solving homography: {e}")
            return None

    def _count_inliers(self, src_pts: np.ndarray, dst_pts: np.ndarray,
                       homography: np.ndarray) -> int:
        """
        Count inliers for given homography.
        
        Args:
            src_pts: Source points
            dst_pts: Destination points
            homography: 3x3 homography matrix
            
        Returns:
            Number of inliers
        """
        inliers = 0

        for i in range(len(src_pts)):
            # Transform source point
            src_homogeneous = np.append(src_pts[i], 1)
            transformed = homography @ src_homogeneous

            if abs(transformed[2]) > 1e-8:
                transformed = transformed / transformed[2]
                error = np.linalg.norm(transformed[:2] - dst_pts[i])

                if error < self.ransac_threshold:
                    inliers += 1

        return inliers

    def _warp_and_blend(self, images: List[np.ndarray], homography: np.ndarray,
                        blending_mode: str) -> np.ndarray:
        """
        Warp right image and blend with left image.
        
        Args:
            images: List of [left_image, right_image]
            homography: 3x3 homography matrix
            blending_mode: Blending method
            
        Returns:
            Stitched image
        """
        img_left, img_right = images
        h_left, w_left = img_left.shape[:2]
        h_right, w_right = img_right.shape[:2]

        # Create output canvas
        output_height = max(h_left, h_right)
        output_width = w_left + w_right
        warped_img = np.zeros((output_height, output_width, 3), dtype=np.uint8)

        # Place left image
        if blending_mode == "noBlending":
            warped_img[:h_left, :w_left] = img_left

        print(" ==== Original Matrix ==== ")
        print(
            f"{tabulate(homography, floatfmt='.3f', numalign='right', tablefmt='grid')}"
        )

        # Warp right image
        inv_homography = np.linalg.inv(homography)
        print(" ======= INV Matrix ======= ")
        print(
            f"{tabulate(inv_homography, floatfmt='.3f', numalign='right', tablefmt='grid')}"
        )

        for i in range(output_height):
            for j in range(output_width):
                # Map output coordinates to right image coordinates
                output_coords = np.array([j, i, 1])
                right_coords = inv_homography @ output_coords

                if abs(right_coords[2]) > 1e-8:
                    right_coords = right_coords / right_coords[2]
                    x, y = int(round(right_coords[0])), int(
                        round(right_coords[1]))

                    # Check bounds
                    if 0 <= x < w_right and 0 <= y < h_right:
                        warped_img[i, j] = img_right[y, x]
        logger.info("Warping Completed")

        # Apply blending
        logger.info("Starting Blending...")
        blender = Blender()
        if blending_mode == "linearBlending":
            warped_img = blender.linear_blending([img_left, warped_img])
        elif blending_mode == "linearBlendingWithConstant":
            warped_img = blender.linear_blending_with_constant(
                [img_left, warped_img])
        logger.info("Blending Completed")

        # Remove black borders
        warped_img = self._remove_black_borders(warped_img)

        return warped_img

    def _remove_black_borders(self, image: np.ndarray) -> np.ndarray:
        """
        Remove black borders from stitched image.
        
        Args:
            image: Input image
            
        Returns:
            Cropped image
        """
        h, w = image.shape[:2]

        # Find valid region
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Get bounding box of largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            return image[y:y + h, x:x + w]

        return image


class Blender:
    """
    Image blending utilities.
    """

    def linear_blending(self, images: List[np.ndarray]) -> np.ndarray:
        """
        Linear blending of overlapping regions.
        
        Args:
            images: List of [left_image, right_image]
            
        Returns:
            Blended image
        """
        img_left, img_right = images
        h, w = img_right.shape[:2]
        h_left, w_left = img_left.shape[:2]

        # Create masks
        left_mask = self._create_mask(img_left, h, w)
        right_mask = self._create_mask(img_right, h, w)

        # Find overlap
        overlap_mask = left_mask & right_mask

        # Create alpha mask
        alpha_mask = self._create_alpha_mask(overlap_mask)

        # Blend
        result = np.copy(img_right)
        result[:h_left, :w_left] = img_left

        for i in range(h):
            for j in range(w):
                if overlap_mask[i, j]:
                    alpha = alpha_mask[i, j]
                    result[i, j] = alpha * img_left[i, j] + (
                        1 - alpha) * img_right[i, j]

        return result

    def linear_blending_with_constant(self,
                                      images: List[np.ndarray],
                                      constant_width: int = 10) -> np.ndarray:
        """
        Linear blending with constant width around seam.
        
        Args:
            images: List of [left_image, right_image]
            constant_width: Width of blending region
            
        Returns:
            Blended image
        """
        img_left, img_right = images
        h, w = img_right.shape[:2]
        h_left, w_left = img_left.shape[:2]

        # Create masks
        left_mask = self._create_mask(img_left, h, w)
        right_mask = self._create_mask(img_right, h, w)

        # Find overlap
        overlap_mask = left_mask & right_mask

        # Create alpha mask with constant width
        alpha_mask = self._create_alpha_mask_constant_width(
            overlap_mask, constant_width)

        # Blend
        result = np.copy(img_right)
        result[:h_left, :w_left] = img_left

        for i in range(h):
            for j in range(w):
                if overlap_mask[i, j]:
                    alpha = alpha_mask[i, j]
                    result[i, j] = alpha * img_left[i, j] + (
                        1 - alpha) * img_right[i, j]

        return result

    def _create_mask(self, image: np.ndarray, target_h: int,
                     target_w: int) -> np.ndarray:
        """Create binary mask for non-zero pixels."""
        h, w = image.shape[:2]
        mask = np.zeros((target_h, target_w), dtype=bool)

        for i in range(min(h, target_h)):
            for j in range(min(w, target_w)):
                if np.any(image[i, j] > 0):
                    mask[i, j] = True

        return mask

    def _create_alpha_mask(self, overlap_mask: np.ndarray) -> np.ndarray:
        """Create alpha mask for linear blending."""
        h, w = overlap_mask.shape
        alpha_mask = np.zeros((h, w), dtype=np.float32)

        for i in range(h):
            # Find overlap region in this row
            overlap_indices = np.where(overlap_mask[i])[0]
            if len(overlap_indices) > 0:
                min_idx, max_idx = overlap_indices[0], overlap_indices[-1]
                if min_idx < max_idx:
                    # Linear interpolation
                    for j in range(min_idx, max_idx + 1):
                        alpha_mask[
                            i, j] = 1.0 - (j - min_idx) / (max_idx - min_idx)

        return alpha_mask

    def _create_alpha_mask_constant_width(self, overlap_mask: np.ndarray,
                                          constant_width: int) -> np.ndarray:
        """Create alpha mask with constant width blending."""
        h, w = overlap_mask.shape
        alpha_mask = np.zeros((h, w), dtype=np.float32)

        for i in range(h):
            overlap_indices = np.where(overlap_mask[i])[0]
            if len(overlap_indices) > 0:
                min_idx, max_idx = overlap_indices[0], overlap_indices[-1]
                if min_idx < max_idx:
                    middle_idx = (min_idx + max_idx) // 2

                    # Left region
                    for j in range(min_idx, middle_idx + 1):
                        if j >= middle_idx - constant_width:
                            alpha_mask[
                                i,
                                j] = 1.0 - (j - min_idx) / (max_idx - min_idx)
                        else:
                            alpha_mask[i, j] = 1.0

                    # Right region
                    for j in range(middle_idx + 1, max_idx + 1):
                        if j <= middle_idx + constant_width:
                            alpha_mask[
                                i,
                                j] = 1.0 - (j - min_idx) / (max_idx - min_idx)
                        else:
                            alpha_mask[i, j] = 0.0

        return alpha_mask


def main():
    """Main function to run image stitching with command line arguments."""
    import sys

    # Check command line arguments
    if len(sys.argv) != 6:
        logger.error(
            "Usage: python stitcher.py <left_image> <right_image> <output_image> --blend <blending_mode>"
        )
        logger.error(
            "Blending modes: noBlending, linearBlending, linearBlendingWithConstant"
        )
        sys.exit(1)

    # Parse command line arguments
    left_path = sys.argv[1]
    right_path = sys.argv[2]
    output_path = sys.argv[3]

    # Check for --blend flag
    if sys.argv[4] != "--blend":
        logger.error("Expected '--blend' flag")
        logger.error(
            "Usage: python exe.py <left_image> <right_image> <output_image> --blend <blending_mode>"
        )
        sys.exit(1)

    blending_mode = sys.argv[5]

    # Validate blending mode
    valid_modes = [
        "noBlending", "linearBlending", "linearBlendingWithConstant"
    ]
    if blending_mode not in valid_modes:
        logger.error(f"Invalid blending mode '{blending_mode}'")
        logger.error(f"Valid modes: {', '.join(valid_modes)}")
        sys.exit(1)

    try:
        # Display input images
        img_left = cv2.imread(left_path)
        img_right = cv2.imread(right_path)

        if img_left is None:
            logger.error(f"Could not load left image: {left_path}")
            sys.exit(1)
        if img_right is None:
            logger.error(f"Could not load right image: {right_path}")
            sys.exit(1)

        logger.info(
            f"Loaded images: {left_path} ({img_left.shape}) and {right_path} ({img_right.shape})"
        )

        # Show input images (optional - comment out if running headless)
        combined = np.hstack([img_left, img_right])
        cv2.imshow("Input Images", combined)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Create stitcher and stitch images
        logger.info(
            f"Startinging image stitching with {blending_mode} mode...")
        stitcher = Stitcher(ransac_threshold=1.0, ransac_iterations=1000)
        result = stitcher.stitch([left_path, right_path], blending_mode, 0.75)

        # Save result
        success = cv2.imwrite(output_path, result)
        if not success:
            logger.error(f"Could not save result to {output_path}")
            sys.exit(1)

        logger.info(f"Result saved to: {output_path}")

        # Display result (optional - comment out if running headless)
        cv2.imshow("Stitched Result", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        logger.info("Image stitching completed successfully")

    except Exception as e:
        logger.error(f"Error during stitching: {e}")
        logger.error(f"Stitching failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
