<h2>TensorflowU2Net-Image-Segmentation-Augmented-GastrointestinalPolyp (2024/03/25)</h2>

This is the sixth experimental Image Segmentation project for GastrointestinalPolyp  based on
the <a href="https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-API">Tensorflow-Image-Segmentation-API</a>, and
<a href="https://drive.google.com/file/d/1f9Bwjt2rKstDRgv5zztfrBl-DLZNo--c/view?usp=sharing">
GastrointestinalPolyp-ImageMask-Dataset.zip</a> 
<br>
<br>
Segmentation samples by TensorflowU2Net Model.<br>
<img src="./projects/TensorflowU2Net/GastrointestinalPolyp/asset/segmentation_samples.png" width="720" height="auto">
<br>
<br>

We will use an online dataset augmentation strategy based on Python script <a href="./src/ImageMaskAugmentor.py">
ImageMaskAugmentor.py</a> to train GastrointestinalPolyp Segmentation Model.<br><br>

Please see our experiments.<br> 
<li>
<a href="https://github.com/atlan-antillia/Image-Segmentation-Gastrointestinal-Polyp">
Image-Segmentation-Gastrointestinal-Polyp</a>
</li>

<li>
<a href="https://github.com/sarah-antillia/TensorflowMultiResUNet-Segmentation-Gastrointestinal-Polyp">
TensorflowMultiResUNet-Segmentation-Gastrointestinal-Polyp 
</a>
</li>
<li>
<a href="https://github.com/sarah-antillia/TensorflowSwinUNet-Image-Segmentation-Augmented-GastrointestinalPolyp">
TensorflowSwinUNet-Image-Segmentation-Augmented-GastrointestinalPolyp 
</a>
</li>
<li>
<a href="https://github.com/sarah-antillia/TensorflowEfficientUNet-Image-Segmentation-Augmented-GastrointestinalPolyp">
TensorflowEfficientUNet-Image-Segmentation-Augmented-GastrointestinalPolyp 
</a>
</li>
<li>
<a href="https://github.com/sarah-antillia/TensorflowSharpUNet-Image-Segmentation-Augmented-GastrointestinalPolyp">
TensorflowSharpUNet-Image-Segmentation-Augmented-GastrointestinalPolyp 
</a>
</li>

<br>
We use TensorflowU2Net Model
<a href="./src/TensorflowU2Net.py">TensorflowU2Net.py</a> for this GastrointestinalPolyp Segmentation.<br>

Our TensorflowU2Net class is based on the following implementation.<br>
<a href="https://github.com/hasibzunair/sharp-unets/blob/master/demo.ipynb">
Hasibzunair: sharp-unets</a><br>
<br>
 
<h3>1. Dataset Citation</h3>

The image dataset used here has been taken from the following kaggle web site.<br>
<a href="https://www.kaggle.com/datasets/debeshjha1/kvasirseg">Kvasir-SEG Data (Polyp segmentation & detection)</a>
<br><br>
<b>About Dataset</b>
<pre>
Kvasir-SEG information:
The Kvasir-SEG dataset (size 46.2 MB) contains 1000 polyp images and their corresponding ground truth 
from the Kvasir Dataset v2. The images' resolution in Kvasir-SEG varies from 332x487 to 1920x1072 pixels. 
The images and its corresponding masks are stored in two separate folders with the same filename. 
The image files are encoded using JPEG compression, facilitating online browsing. 
The open-access dataset can be easily downloaded for research and educational purposes.
</pre>

<b>Applications of the Dataset</b><br>
<pre>
The Kvasir-SEG dataset is intended to be used for researching and developing new and improved methods 
for segmentation, detection, localization, and classification of polyps. 
Multiple datasets are prerequisites for comparing computer vision-based algorithms, and this dataset 
is useful both as a training dataset or as a validation dataset. These datasets can assist the 
development of state-of-the-art solutions for images captured by colonoscopes from different manufacturers. 
Further research in this field has the potential to help reduce the polyp miss rate and thus improve 
examination quality. The Kvasir-SEG dataset is also suitable for general segmentation and bounding box 
detection research. In this context, the datasets can accompany several other datasets from a wide 
range of fields, both medical and otherwise.
</pre>
<!--
<b>Ground Truth Extraction</b><br>
<pre>
We uploaded the entire Kvasir polyp class to Labelbox and created all the segmentations using this application. 
The Labelbox is a tool used for labeling the region of interest (ROI) in image frames, i.e., the polyp regions 
for our case. We manually annotated and labeled all of the 1000 images with the help of medical experts. 
After annotation, we exported the files to generate masks for each annotation. 
The exported JSON file contained all the information about the image and the coordinate points for generating 
the mask. To create a mask, we used ROI coordinates to draw contours on an empty black image and fill the 
contours with white color. The generated masks are a 1-bit color depth images. The pixels depicting polyp tissue, 
the region of interest, are represented by the foreground (white mask), while the background (in black) does not 
contain positive pixels. Some of the original images contain the image of the endoscope position marking probe, 
ScopeGuide TM, Olympus Tokyo Japan, located in one of the bottom corners, seen as a small green box. 
As this information is superfluous for the segmentation task, we have replaced these with black boxes in the 
Kvasir-SEG dataset.
</pre>
-->
See also:
<pre>
Kvasir-SEG
https://paperswithcode.com/dataset/kvasir-seg
</pre>


<h3>
<a id="2">
2 GastrointestinalPolyp-ImageMask Dataset
</a>
</h3>
 If you would like to train this GastrointestinalPolyp Segmentation model by yourself,
please download the dataset from the google drive 
<a href="https://drive.google.com/file/d/1f9Bwjt2rKstDRgv5zztfrBl-DLZNo--c/view?usp=sharing">
GastrointestinalPolyp-ImageMask-Dataset.zip</a>.
<br>


Please see also the <a href="https://github.com/sarah-antillia/ImageMask-Dataset-GastrointestinalPolyp">ImageMask-Dataset-GastrointestinalPolyp</a>.<br>
Please expand the downloaded ImageMaskDataset and place them under <b>./dataset</b> folder to be

<pre>
./dataset
└─GastrointestinalPolyp
    ├─test
    │  ├─images
    │  └─masks
    ├─train
    │  ├─images
    │  └─masks
    └─valid
        ├─images
        └─masks
</pre>
 
 
<b>GastrointestinalPolyp Dataset Statistics</b><br>
<img src ="./projects/TensorflowU2Net/GastrointestinalPolyp/GastrointestinalPolyp_Statistics.png" width="512" height="auto"><br>
As shown above, the number of images of train and valid dataset is not necessarily large. 
<br>

<h3>
<a id="3">
3 TensorflowU2Net
</a>
</h3>
This <a href="./src/TensorflowU2Net.py">TensorflowU2Net</a> model is slightly flexibly customizable by a configuration file.<br>
For example, <b>TensorflowU2Net/GastrointestinalPolyp</b> model can be customizable
by using <a href="./projects/TensorflowU2Net/GastrointestinalPolyp/train_eval_infer.config">train_eval_infer.config</a>
<pre>
; train_eval_infer.config
; 2024/03/25 (C) antillia.com

[model]
model          = "TensorflowU2Net"
generator      = True
image_width    = 512
image_height   = 512
image_channels = 3
num_classes    = 1

num_layers     = 6
base_filters  = 16

activation     = "mish"
optimizer      = "Adam"
;dropout_rate   = 0.02
learning_rate  = 0.0001
clipvalue      = 0.5
loss           = "bce_dice_loss"
metrics        = ["binary_accuracy"]
show_summary   = False

[dataset]
datasetclass  = "ImageMaskDataset"
resize_interpolation = "cv2.INTER_CUBIC"

[train]
save_model_file = "best_model.h5"

dataset_splitter = True
learning_rate_reducer = True
reducer_patience      = 5
steps_per_epoch       = 200
validation_steps      = 100

epochs        = 100
batch_size    = 4
patience      = 10
metrics       = ["binary_accuracy", "val_binary_accuracy"]
model_dir     = "./models"
save_weights_only = True

eval_dir      = "./eval"
image_datapath = "../../../dataset/GastrointestinalPolyp/train/images/"
mask_datapath  = "../../../dataset/GastrointestinalPolyp/train/masks/"
create_backup  = False

[eval]
image_datapath = "../../../dataset/GastrointestinalPolyp/valid/images/"
mask_datapath  = "../../../dataset/GastrointestinalPolyp/valid/masks/"

[test]
image_datapath = "../../../dataset/GastrointestinalPolyp/test/images/"
mask_datapath  = "../../../dataset/GastrointestinalPolyp/test/masks/"

[infer] 
images_dir = "../../../dataset/GastrointestinalPolyp/test/images/"
output_dir    = "./test_output"
merged_dir    = "./test_output_merged"

[mask]
blur      = True
binarize  = True
threshold = 80

[generator]
debug        = True
augmentation = True

[augmentor]
vflip    = True
hflip    = True
rotation = True
angles   = [0, 90, 180, 270,]
shrinks  = [0.8]
shears   = [0.2]
transformer = True
alpah       = 1300
sigmoid     = 8
</pre>
Depending on these parameters in [augmentor] section, it will generate vflipped, hflipped, rotated, shrinked,
sheared, elastic-transformed images and masks
from the original images and masks in the folders specified by image_datapath and mask_datapath in 
[train] and [eval] sections.<br>
<pre>
[train]
image_datapath = "../../../dataset/GastrointestinalPolyp/train/images/"
mask_datapath  = "../../../dataset/GastrointestinalPolyp/train/masks/"
[eval]
image_datapath = "../../../dataset/GastrointestinalPolyp/valid/images/"
mask_datapath  = "../../../dataset/GastrointestinalPolyp/valid/masks/"
</pre>

For more detail on ImageMaskAugmentor.py, please refer to
<a href="https://github.com/sarah-antillia/Image-Segmentation-ImageMaskDataGenerator">
Image-Segmentation-ImageMaskDataGenerator.</a>.
    
<br>

<h3>
3.1 Training
</h3>
Please move to the <b>./projects/TensorflowU2Net/GastrointestinalPolyp</b> folder,<br>
and run the following bat file to train TensorflowUNet model for GastrointestinalPolyp.<br>
<pre>
./1.train_generator.bat
</pre>
, which simply runs <a href="./src/TensorflowUNetGeneratorTrainer.py">TensorflowUNetGeneratorTrainer.py </a>
in the following way.

<pre>
python ../../../src/TensorflowUNetGeneratorTrainer.py ./train_eval_infer.config
</pre>
Train console output:<br>
<img src="./projects/TensorflowU2Net/GastrointestinalPolyp/asset/train_console_output_at_epoch_67.png" width="720" height="auto"><br>
Train metrics:<br>
<img src="./projects/TensorflowU2Net/GastrointestinalPolyp/asset/train_metrics.png" width="720" height="auto"><br>
Train losses:<br>
<img src="./projects/TensorflowU2Net/GastrointestinalPolyp/asset/train_losses.png" width="720" height="auto"><br>
<br>
The following debug setting is helpful whether your parameters in [augmentor] section are good or not good.
<pre>
[generator]
debug     = True
</pre>
You can check the yielded images and mask files used in the actual train-eval process in the following folders under
<b>./projects/TensorflowU2Net/GastrointestinalPolyp/</b>.<br> 
<pre>
generated_images_dir
generated_masks_dir
</pre>

Sample images in generated_images_dir<br>
<img src="./projects/TensorflowU2Net/GastrointestinalPolyp/asset/sample_images_in_generated_images_dir.png"
 width="1024" height="auto"><br>
Sample masks in generated_masks_dir<br>
<img src="./projects/TensorflowU2Net/GastrointestinalPolyp/asset/sample_masks_in_generated_masks_dir.png"
 width="1024" height="auto"><br>

<h3>
3.2 Evaluation
</h3>
Please move to the <b>./projects/TensorflowU2Net/GastrointestinalPolyp</b> folder,<br>
and run the following bat file to evaluate TensorflowUNet model for GastrointestinalPolyp.<br>
<pre>
./2.evaluate.bat
</pre>
<pre>
python ../../../src/TensorflowUNetEvaluator.py ./train_eval_infer.config
</pre>
Evaluation console output:<br>
<img src="./projects/TensorflowU2Net/GastrointestinalPolyp/asset/evaluate_console_output_at_epoch_67.png" width="720" height="auto">
<pre>
Test loss    :0.2469
Test accuracy:0.9401000142097473
</pre>

As shown above, the loss and accuracy for the test dataset is slightly worse than that of the fourth experiment based on TensorflowEfficientUNet Model 
<a href="https://github.com/sarah-antillia/TensorflowEfficientUNet-Image-Segmentation-Augmented-GastrointestinalPolyp">TensorflowEfficientUNet-Image-Segmentation-Augmented-GastrointestinalPolyp</a>
<br>
<img src=
"https://github.com/sarah-antillia/TensorflowEfficientUNet-Image-Segmentation-Augmented-GastrointestinalPolyp/blob/main/projects/TensorflowEfficientUNet/GastrointestinalPolyp/asset/evaluate_console_output_at_epoch_39.png" 
width="720" height="auto"><br>

<pre>
Test loss    :0.1439
Test accuracy:0.9678999781608582
</pre>

<br>
<h2>
3.3 Inference
</h2>
Please move to a <b>./projects/TensorflowU2Net/GastrointestinalPolyp</b> folder<br>
,and run the following bat file to infer segmentation regions for images by the Trained-TensorflowUNet model for GastrointestinalPolyp.<br>
<pre>
./3.infer.bat
</pre>
<pre>
python ../../../src/TensorflowUNetInferencer.py ./train_eval_infer.config
</pre>
Sample test images<br>
<img src="./projects/TensorflowU2Net/GastrointestinalPolyp/asset/test_images.png" width="1024" height="auto"><br>
Sample test mask (ground_truth)<br>
<img src="./projects/TensorflowU2Net/GastrointestinalPolyp/asset/test_masks.png" width="1024" height="auto"><br>

<br>
Inferred test masks<br>
<img src="./projects/TensorflowU2Net/GastrointestinalPolyp/asset/test_output.png" width="1024" height="auto"><br>
<br>
Merged test images and inferred masks<br> 
<img src="./projects/TensorflowU2Net/GastrointestinalPolyp/asset/test_output_merged.png" width="1024" height="auto"><br> 


Enlarged segementation samples<br>
<table>
<tr>
<td>
test/images/cju0qx73cjw570799j4n5cjze.jpg<br>
<img src="./dataset/GastrointestinalPolyp/test/images/cju0qx73cjw570799j4n5cjze.jpg" width="512" height="auto">

</td>
<td>
Inferred merged/cju0qx73cjw570799j4n5cjze.jpg<br>
<img src="./projects/TensorflowU2Net/GastrointestinalPolyp/test_output_merged/cju0qx73cjw570799j4n5cjze.jpg" width="512" height="auto">
</td> 
</tr>

<tr>
<td>
test/images/cju0roawvklrq0799vmjorwfv.jpg<br>
<img src="./dataset/GastrointestinalPolyp/test/images/cju0roawvklrq0799vmjorwfv.jpg" width="512" height="auto">

</td>
<td>
Inferred merged/cju0roawvklrq0799vmjorwfv.jpg<br>
<img src="./projects/TensorflowU2Net/GastrointestinalPolyp/test_output_merged/cju0roawvklrq0799vmjorwfv.jpg" width="512" height="auto">
</td> 
</tr>


<tr>
<td>
test/images/cju0t4oil7vzk099370nun5h9.jpg<br>
<img src="./dataset/GastrointestinalPolyp/test/images/cju0t4oil7vzk099370nun5h9.jpg" width="512" height="auto">

</td>
<td>
Inferred merged/cju0t4oil7vzk099370nun5h9.jpg<br>
<img src="./projects/TensorflowU2Net/GastrointestinalPolyp/test_output_merged/cju0t4oil7vzk099370nun5h9.jpg" width="512" height="auto">
</td> 
</tr>


<tr>
<td>
test/images/cju2wtwj87kys0855kx6mddzw.jpg<br>
<img src="./dataset/GastrointestinalPolyp/test/images/cju2wtwj87kys0855kx6mddzw.jpg" width="512" height="auto">

</td>
<td>
Inferred merged/cju2wtwj87kys0855kx6mddzw.jpg<br>
<img src="./projects/TensorflowU2Net/GastrointestinalPolyp/test_output_merged/cju2wtwj87kys0855kx6mddzw.jpg" width="512" height="auto">
</td> 
</tr>


<tr>
<td>
test/images/cju7d9seq29zd0871nzl2uu5m.jpg<br>
<img src="./dataset/GastrointestinalPolyp/test/images/cju7d9seq29zd0871nzl2uu5m.jpg" width="512" height="auto">

</td>
<td>
Inferred merged/cju7d9seq29zd0871nzl2uu5m.jpg<br>
<img src="./projects/TensorflowU2Net/GastrointestinalPolyp/test_output_merged/cju7d9seq29zd0871nzl2uu5m.jpg" width="512" height="auto">
</td> 
</tr>

</table>


<h3>
References
</h3>

<b>1. Kvasir-SEG Data (Polyp segmentation & detection)</b><br>
<pre>
https://www.kaggle.com/datasets/debeshjha1/kvasirseg
</pre>

<b>2. Kvasir-SEG: A Segmented Polyp Dataset</b><br>
Debesh Jha, Pia H. Smedsrud, Michael A. Riegler, P˚al Halvorsen,<br>
Thomas de Lange, Dag Johansen, and H˚avard D. Johansen<br>
<pre>
https://arxiv.org/pdf/1911.07069v1.pdf
</pre>

<b>3. keras-unet-collection: _model_u2net_2d</b><br>
Yingkai (Kyle) Sha<br>
<pre>
 https://github.com/yingkaisha/keras-unet-collection/blob/main/keras_unet_collection/_model_u2net_2d.py
</pre>

<b>4. TensorflowSwinUNet-Image-Segmentation-Augmented-GastrointestinalPolyp</b><br>
Toshiyuki Arai @antillia.com<br>
<pre>
https://github.com/sarah-antillia/TensorflowSwinUNet-Image-Segmentation-Augmented-GastrointestinalPolyp
</pre>

<b>5. TensorflowMultiResUNet-Segmentation-Gastrointestinal-Polyp</b><br>
Toshiyuki Arai @antillia.com<br>
<pre>
https://github.com/sarah-antillia/TensorflowMultiResUNet-Segmentation-Gastrointestinal-Polyp
</pre>

<b>6. TensorflowUNet3Plus-Segmentation-Gastrointestinal-Polyp</b><br>
Toshiyuki Arai @antillia.com<br>
<pre>
https://github.com/sarah-antillia/TensorflowUNet3Plus-Segmentation-Gastrointestinal-Polyp
</pre>

<b>7. TensorflowEfficientUNet-Segmentation-Gastrointestinal-Polyp</b><br>
Toshiyuki Arai @antillia.com<br>
<pre>
https://github.com/sarah-antillia/TensorflowEfficientUNet-Image-Segmentation-Augmented-GastrointestinalPolyp
</pre>

<b>8. TensorflowSharpUNet-Segmentation-Gastrointestinal-Polyp</b><br>
Toshiyuki Arai @antillia.com<br>
<pre>
https://github.com/sarah-antillia/TensorflowSharpUNet-Image-Segmentation-Augmented-GastrointestinalPolyp
</pre>

