import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

class PatchMatch:
    def __init__(self, source_image, mask, patch_size):
        self.source_image = source_image
        self.mask = mask
        self.patch_size = patch_size
        self.half_size = patch_size // 2
        self.h, self.w = source_image.shape[:2]
        
        self.masked_coords = np.array(np.where(mask == True)).T
        
        self.is_valid_coords = np.logical_not(mask)
        # self.is_valid_coords[:self.half_size] = False
        # self.is_valid_coords[-self.half_size:] = False
        # self.is_valid_coords[:, :self.half_size] = False
        # self.is_valid_coords[:, -self.half_size:] = False
        
        # for y, x in self.masked_coords:
        #     self.is_valid_coords[y - self.half_size: y + self.half_size + 1, x - self.half_size: x + self.half_size + 1] = False
        
        self.valid_coords = np.array(np.where(self.is_valid_coords)).T
        
        self.nnf = None
        
    def coord_in_image(self, x, y):
        return x >= 0 and y >= 0 and x < self.w and y < self.h
        
    def patch_in_image(self, x, y):
        return x - self.half_size >= 0 and y - self.half_size >= 0 and x + self.half_size < self.w and y + self.half_size < self.h

    # def patch_in_image(self, coord):
    #     return self.patch_in_image(coord[0], coord[1])
    
    # def patch_has_masked(self, x, y):
    #     return np.any(self.mask[y - self.half_size: y + self.half_size + 1, x - self.half_size: x + self.half_size + 1])
    
    # def patch_has_masked(self, coord):
    #     return self.patch_has_masked(coord[0], coord[1])
    
    # def compute_patch_distance(self, patch_image1, patch_coords1, patch_image2, patch_coords2):
    #     return np.sum((patch_image1[patch_coords1[1] - self.half_size: patch_coords1[1] + self.half_size + 1, patch_coords1[0] - self.half_size: patch_coords1[0] + self.half_size + 1]
    #                    - patch_image2[patch_coords2[1] - self.half_size: patch_coords2[1] + self.half_size + 1, patch_coords2[0] - self.half_size: patch_coords2[0] + self.half_size + 1]) ** 2)
    
    def compute_patch_distance(self, patch_image1, patch_coords1, patch_image2, patch_coords2):
        def get_patch(image, coords):
            x, y = coords
            half_size = self.half_size
            patch = image[max(0, y - half_size): y + half_size + 1, max(0, x - half_size): x + half_size + 1]
            if patch.shape[0] != 2 * half_size + 1 or patch.shape[1] != 2 * half_size + 1:
                patch = np.pad(patch, ((max(0, half_size - y), max(0, y + half_size + 1 - image.shape[0])),
                                    (max(0, half_size - x), max(0, x + half_size + 1 - image.shape[1])),
                                    (0, 0)), 'constant')
            return patch

        patch1 = get_patch(patch_image1, patch_coords1)
        patch2 = get_patch(patch_image2, patch_coords2)
        return np.sum((patch1 - patch2) ** 2)

    def random_initialize_nnf(self):
        self.nnf = np.zeros((self.h, self.w, 2), dtype=np.int32)
        for y in range(self.h):
            for x in range(self.w):
                if self.mask[y, x]:
                    random_coord = self.valid_coords[np.random.choice(len(self.valid_coords))]
                    self.nnf[y, x] = random_coord[::-1]  # Store as (x, y)
                else:
                    self.nnf[y, x] = [x, y]
        return

    def paint_matched(self, target):
        for y, x in self.masked_coords:
            match = self.nnf[y, x]
            target[y, x] = self.source_image[match[1], match[0]]
        return

    def run(self, iterations=5, random_nnf=True, nnf=None):
        """
        Perform inpainting using PatchMatch.
        """
        inpainted_image = self.source_image.copy()
        
        if random_nnf:
            self.random_initialize_nnf()
        else:
            self.nnf = nnf
        self.paint_matched(inpainted_image)
        
        plt.subplot(1, 2, 1)
        plt.title("Original Image with Hole")
        plt.imshow(cv2.cvtColor(self.source_image.astype(np.uint8), cv2.COLOR_BGR2RGB))
        plt.subplot(1, 2, 2)
        plt.title(f"Inpainted Image - Iteration 0")
        plt.imshow(cv2.cvtColor(inpainted_image.astype(np.uint8), cv2.COLOR_BGR2RGB))
        plt.pause(0.01)  # Pause to update the plot
        plt.clf()  # Clear the figure for the next iteration

        for it in range(iterations):
            if it % 2 == 0:
                for_list = self.masked_coords
            else:
                for_list = reversed(self.masked_coords)
            
            for y, x in for_list:
                current_patch = [x, y]
                
                # Propagation step
                best_match = self.nnf[y, x]
                best_distance = self.compute_patch_distance(inpainted_image, current_patch, self.source_image, best_match)
                
                if it % 2 == 0:
                    neighbor_list = [(-1, 0), (0, -1)]
                else:
                    neighbor_list = [(1, 0), (0, 1)]
                for dy, dx in neighbor_list:
                    neighbor_x = x + dx
                    neighbor_y = y + dy
                    
                    candidate = self.nnf[neighbor_y, neighbor_x] - np.array([dx, dy])
                    if self.coord_in_image(candidate[0], candidate[1]) and self.is_valid_coords[candidate[1], candidate[0]]:
                        distance = self.compute_patch_distance(inpainted_image, current_patch, self.source_image, candidate)
                        if distance < best_distance:
                            best_match = candidate
                            best_distance = distance
                
                # Random search
                search_radius = max(self.h, self.w)
                while search_radius > 1:
                    rx = best_match[0] + np.random.randint(-search_radius, search_radius + 1)
                    ry = best_match[1] + np.random.randint(-search_radius, search_radius + 1)
                    candidate = [rx, ry]
                    if self.coord_in_image(rx, ry) and self.is_valid_coords[ry, rx]:
                        distance = self.compute_patch_distance(inpainted_image, current_patch, self.source_image, candidate)
                        if distance < best_distance:
                            best_match = candidate
                            best_distance = distance
                    search_radius //= 2
                
                self.nnf[y, x] = best_match
            
            # Reconstruct inpainted image
            self.paint_matched(inpainted_image)
            
            # Update plot for each iteration
            plt.subplot(1, 2, 1)
            plt.title("Original Image with Hole")
            plt.imshow(cv2.cvtColor(self.source_image.astype(np.uint8), cv2.COLOR_BGR2RGB))
            plt.subplot(1, 2, 2)
            plt.title(f"Inpainted Image - Iteration {it + 1}")
            plt.imshow(cv2.cvtColor(inpainted_image.astype(np.uint8), cv2.COLOR_BGR2RGB))
            plt.pause(0.01)  # Pause to update the plot
            plt.clf()  # Clear the figure for the next iteration

        return inpainted_image


def donwsample(image, new_height, new_width):
    h, w = image.shape[:2]
    image = image[:new_height * (h // new_height), :new_width * (w // new_width)]
    image = image.reshape(new_height, h // new_height, new_width, w // new_width, -1)
    return image.mean(axis=(1, 3))

def upsample(nnf, new_height, new_width):
    h, w = nnf.shape[:2]
    nnf = nnf.astype(np.float32)
    nnf[..., 0] = (nnf[..., 0] + 0.5) * new_width / w - 0.5
    nnf[..., 1] = (nnf[..., 1] + 0.5) * new_height / h - 0.5
    nnf = cv2.resize(nnf, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    nnf = np.round(nnf).astype(np.int32)
    return nnf

# Example Usage
if __name__ == "__main__":
    # Load image and mask
    # image = cv2.imread("test_data/simple.png")
    # mask = cv2.imread("test_data/simple.mask.png")
    # mask_color = [255, 255, 255]
    # image = cv2.imread("test_data/magicRoom.png").astype(np.float32)
    # mask = cv2.imread("test_data/magicRoom.mask.png")
    # mask_color = [0, 255, 0]
    image = cv2.imread("test_data/yexinjia.png").astype(np.float32)
    mask = cv2.imread("test_data/yexinjia.mask.png")
    mask_color = [255, 0, 0]
    
    mask = np.all(mask == mask_color, axis=-1)
    origin_image = image.copy()
    image[mask] = 0

    patch_size = 5
    h, w = image.shape[:2]
    total_level = 0
    h_list = [h]
    w_list = [w]
    while h // 2 >= patch_size and w // 2 >= patch_size:
        h //= 2
        w //= 2
        h_list.append(h)
        w_list.append(w)
        total_level += 1
    
    plt.figure(figsize=(12, 6))
    
    tmp_nnf = None
    for level in range(total_level):
        tmp_image = donwsample(image, h, w)
        tmp_mask = donwsample(mask, h, w).squeeze(-1).astype(np.bool)
        solver = PatchMatch(tmp_image, tmp_mask, patch_size)
        iterations = 2 * (level + 2) + 1
        if level == 0:
            inpainted_image = solver.run(iterations, random_nnf=True, nnf=None)
        else:
            inpainted_image = solver.run(iterations, random_nnf=False, nnf=tmp_nnf)
        h = h_list[- level - 2]
        w = w_list[- level - 2]
        tmp_nnf = upsample(solver.nnf, h, w)
        
    
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(origin_image.astype(np.uint8), cv2.COLOR_BGR2RGB))
    plt.subplot(1, 2, 2)
    plt.title(f"Inpainted Image - Final")
    plt.imshow(cv2.cvtColor(inpainted_image.astype(np.uint8), cv2.COLOR_BGR2RGB))
    plt.show()