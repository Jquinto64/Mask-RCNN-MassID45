## Installation

#### OPTION 1: If on the Killarney (Compute Canada) or similar cluster with prebuilt wheels 
```bash
module load StdEnv/2020 gcc/9.3.0
module load python/3.10.2
module load cuda/11.8.0
module load opencv/4.8.0

cd projects/your_aip_project/your_folder
ENV_NAME=mask_rcnn
virtualenv --no-download virtualenvs/$ENV_NAME

source /home/your_name/projects/your_aip_project/your_name/virtualenvs/$ENV_NAME/bin/activate

python -m pip show pip
pip install --no-index --upgrade pip

pip install --no-index detectron2
pip install torchvision==0.14.1 torchaudio==0.13.1
pip install --no-index opencv

pip install numpy==1.23.0
pip install sahi==0.11.18

pip install --no-index  cython
pip install --no-index  scipy
pip install --no-index  shapely
pip install --no-index  timm
pip install --no-index  h5py
pip install --no-index  submitit
pip install --no-index  scikit-image
pip install --no-index typing-extensions

# Implements patch for lazyconfig -- see PR: https://github.com/facebookresearch/detectron2/pull/3755/files#diff-882061b431ed3670f5b4a045ae0f4e1f140cc785db77fe77585b820bdec6f73d
cp detectron2_modifications/defaults.py /home/your_name/projects/your_aip_project/your_name/virtualenvs/$ENV_NAME/lib/python3.10/site-packages/detectron2/engine/defaults.py

cp detectron2_modifications/augmentation_impl.py /home/your_name/projects/your_aip_project/your_name/virtualenvs/ENV_NAME/lib/python3.10/site-packages/detectron2/data/transforms/augmentation_impl.py

# Fix ensures images without annotations do not throw an error during training
cp detectron2_modifications/dataset_mapper.py /home/your_name/projects/your_aip_project/your_name/virtualenvs/$ENV_NAME/lib/python3.10/site-packages/detectron2/data/dataset_mapper.py


# Move the following files to miniconda/envs/maskdino/lib/python3.8/site-packages/sahi/:
cp sahi_modifications/detectron2.py /home/your_name/projects/your_aip_project/your_name/virtualenvs/$ENV_NAME/lib/python3.10/site-packages/sahi/models/detectron2.py

cp sahi_modifications/annotation.py /home/your_name/projects/your_aip_project/your_name/virtualenvs/$ENV_NAME/lib/python3.10/site-packages/sahi/annotation.py

```
##### OPTION 2: If your cluster supports arbitrary package versions
```bash
conda create --name mask_rcnn python=3.8 -y
conda activate mask_rcnn
# optional: module load cuda-11.3 
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install opencv-python==4.9.0.80

# Navigate to Mask2Former directory
cd Mask-RCNN

# Install prebuilt detectron2 - see https://detectron2.readthedocs.io/en/latest/tutorials/install.html
python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html

# Move the following files to the location of your conda environments (.e.g, miniconda/envs/maskdino/lib/python3.8/site-packages/detectron2/)

# Implements patch for lazyconfig -- see PR: https://github.com/facebookresearch/detectron2/pull/3755/files#diff-882061b431ed3670f5b4a045ae0f4e1f140cc785db77fe77585b820bdec6f73d
cp detectron2_modifications/defaults.py /miniconda/envs/mask_rcnn/lib/python3.8/site-packages/detectron2/engine/defaults.py

cp detectron2_modifications/augmentation_impl.py /miniconda/envs/mask_rcnn/lib/python3.8/site-packages/detectron2/data/transforms/augmentation_impl.py

# Fix ensures images without annotations do not throw an error during training
cp detectron2_modifications/dataset_mapper.py /miniconda/envs/mask_rcnn/lib/python3.8/site-packages/detectron2/data/dataset_mapper.py

# Install certain versions of packages to avoid errors later on
pip install numpy==1.23.1
pip install pillow==9.5.0
pip install sahi==0.11.18
pip install pycocotools

# Move the following files to miniconda/envs/maskdino/lib/python3.8/site-packages/sahi/:
cp sahi_modifications/detectron2.py /miniconda/envs/mask_rcnn/lib/python3.8/site-packages/sahi/models/detectron2.py

cp sahi_modifications/annotation.py /miniconda/envs/mask_rcnn/lib/python3.8/site-packages/sahi/annotation.py
```
## Instructions for Inference on your Images
1.  First download the MS-COCO pretrained R50 checkpoint and place it in this folder:
```bash
wget https://dl.fbaipublicfiles.com/detectron2/new_baselines/mask_rcnn_R_50_FPN_400ep_LSJ/42019571/model_final_14d201.pkl
``` 

2. Download the model checkpoints from [Zenodo](https://zenodo.org/records/15479862/files/model_checkpoints.zip?download=1). This submodule is for Mask R-CNN so use `model_final_mask_rcnn.pth`.
3. Run `standalone_inference.py` as follows:
```bash
python standalone_inference_mask_rcnn.py --model_path path/to/model_final_mask_rcnn.pth
--imgs_directory path/to/your_images --output_dir path/to/your_output_folder --config configs/new_baselines/mask_rcnn_R_50_FPN_100ep_LSJ.py
``` 

## MassID45 Training and Inference Instructions
1. First download the MS-COCO pretrained R50 checkpoint using
```bash
wget https://dl.fbaipublicfiles.com/detectron2/new_baselines/mask_rcnn_R_50_FPN_400ep_LSJ/42019571/model_final_14d201.pkl
```
2. Ensure your dataset is structured according to the COCO dataset format:
```
your_dataset/
  annotations/
    instances_{train,val}2017.json
  {train,val}2017/
    # image files that are mentioned in the corresponding json
```
3. Make any desired modifications to the data augmentations and training hyperparameters using `configs/new_baselines/mask_rcnn_R_50_FPN_100ep_LSJ.py`.

4. Modify the  `--dataset_path` argument in `train.sh` to reflect the location of your training data. Then run the training script with 
```bash
sbatch train.sh
```
Outputs will be saved in the folder specified by `train.output_dir`.

5.  Replace `--dataset_img_path` and `--dataset_json_path` in `sahi_inference.sh` with the locations of your validation or testing data, respectively, then run inference with: 
```bash
sbatch sahi_inference.sh
```
Results will appear in the `runs/predict` folder under the name specified by the `--exp_name` argument.

