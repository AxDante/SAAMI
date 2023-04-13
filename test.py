from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor
import cv2

image = cv2.imread('data/test_img.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

sam_checkpoint = "models/sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

mask_generator = SamAutomaticMaskGenerator(sam)



# sam = sam_model_registry["vit_h"](checkpoint="<path/to/checkpoint>")
# mask_generator = SamAutomaticMaskGenerator(sam)
# masks = mask_generator.generate(<your_image>)