import os
import glob
import tifffile as tiff
from cellpose import denoise, models, io
from skimage import measure 
import pandas as pd

class CellposeSegmenter:
    def __init__(self, gpu=True, model_type='nuclei', diam_mean=19.0):
        """Initialize Cellpose models"""
        super().__init__()  # Add this line
        self.gpu = gpu
        self.model_type = model_type
        self.diam_mean = diam_mean
        
        # Initialize models
        try:
            self.denoise_model = denoise.DenoiseModel(
                gpu=self.gpu, 
                model_type='denoise_nuclei', 
                diam_mean=self.diam_mean
            )
            print("Denoise model initialized")
            
            self.cellpose_model = models.CellposeModel(
                gpu=self.gpu, 
                model_type=self.model_type, 
                diam_mean=self.diam_mean
            )
            print("Cellpose model initialized")
            
        except Exception as e:
            print(f"Error initializing models: {str(e)}")
            raise

    def check_files_in_folder(self, input_folder, output_folder):
        """
        Process all TIF images in a folder: denoise, segment, and save the output.
        """
        try:
            print(f"Checking input folder: {input_folder}")
            
            # Print all contents of the input folder
            print(f"Contents of input folder {input_folder}:")
            for item in os.listdir(input_folder):
                print(f"- {item}")

            
        except Exception as e:
            print(f"Error in check_files_in_folder: {str(e)}")
            raise

    def denoise_and_segment_image(self, img, img_name, output_path):
        """
        Perform denoising and segmentation on a 3D image using Cellpose.
        Returns the masks and centroids DataFrame
        """
        print(f"Processing file: {img_name}, shape: {img.shape}")

        # Perform denoising
        imgs_dn = self.denoise_model.eval(
            img,
            batch_size=4,
            channel_axis=None,
            normalize=True,
            rescale=None,
            diameter=19,
            tile=True,
            do_3D=True,
            tile_overlap=0.1,
            bsize=224
        )
        print("Denoise Down!")

        # Perform segmentation
        masks, flows, styles = self.cellpose_model.eval(
            x=imgs_dn,
            batch_size=8,
            do_3D=True,
            diameter=self.cellpose_model.diam_mean,
            channel_axis=None
        )
        print("Mask Segmentation Down!")

        # Extract centroids
        props = measure.regionprops_table(masks, properties=['centroid', 'area'])
        centroids_df = pd.DataFrame(props)
        
        # Save centroids
        centroids_path = f"{output_path}_centroids.csv"
        centroids_df.to_csv(centroids_path)
        print("Get Centroids Down!")

        # Save masks
        io.save_masks(
            images=img,
            masks=masks,
            flows=flows,
            file_names=output_path,
            png=False,
            tif=True,
            channels=[0, 0]
        )
        print("Save Mask Down!")

        return {
            'masks': masks,
            'centroids_df': centroids_df,
            'centroids_path': centroids_path,
            'mask_path': output_path
        }

    def process_images_in_folder(self, input_folder, output_folder):
        """Process all TIF images in a folder"""
        results = {}
        
        os.makedirs(output_folder, exist_ok=True)
        file_paths = glob.glob(os.path.join(input_folder, "*.tif"))
        print(f"Found {len(file_paths)} TIF files in input folder")

        for file_path in file_paths:
            try:
                img = tiff.imread(file_path)
                file_name = os.path.basename(file_path)
                file_base = "cp_masks_" + os.path.splitext(file_name)[0]
                output_path = os.path.join(output_folder, file_base)

                result = self.denoise_and_segment_image(
                    img=img,
                    img_name=file_name,
                    output_path=output_path
                )
                
                results[file_name] = result
                print(f"\nSuccessfully saved masks and centroids at [{output_folder}]\n")

            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
                results[file_name] = {'error': str(e)}

        return results
