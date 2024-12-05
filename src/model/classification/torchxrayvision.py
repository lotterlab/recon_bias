import torch.nn as nn
from collections import OrderedDict
import torchvision
import torch
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")
import skimage

class XRayCenterCrop(object):
    def crop_center(self, img):
        _, y, x = img.shape
        crop_size = np.min([y, x])
        startx = x // 2 - (crop_size // 2)
        starty = y // 2 - (crop_size // 2)
        return img[:, starty:starty + crop_size, startx:startx + crop_size]

    def __call__(self, img):
        return self.crop_center(img)
    
class XRayResizer(object):
    def __init__(self, size, engine="skimage"):
        self.size = size
        self.engine = engine
        # if 'cv2' in sys.modules:
        #     print("Setting XRayResizer engine to cv2 could increase performance.")

    def __call__(self, img):
        if self.engine == "skimage":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return skimage.transform.resize(img, (1, self.size, self.size), mode='constant', preserve_range=True).astype(np.float32)
        elif self.engine == "cv2":
            import cv2  # pip install opencv-python
            return cv2.resize(img[0, :, :],
                              (self.size, self.size),
                              interpolation=cv2.INTER_AREA
                              ).reshape(1, self.size, self.size).astype(np.float32)
        else:
            raise Exception("Unknown engine, Must be skimage (default) or cv2.")


chexpert_pathologies = ["Enlarged Cardiomediastinum",
                            "Cardiomegaly",
                            "Lung Opacity",
                            "Lung Lesion",
                            "Edema",
                            "Consolidation",
                            "Pneumonia",
                            "Atelectasis",
                            "Pneumothorax",
                            "Pleural Effusion",
                            "Pleural Other",
                            "Fracture",
                            "Support Devices"]


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))

warning_log = {}

def fix_resolution(x, resolution: int, model: nn.Module):
    """Check resolution of input and resize to match requested."""

    # just skip it if upsample was removed somehow
    if not hasattr(model, 'upsample') or (model.upsample == None):
        return x

    if (x.shape[2] != resolution) | (x.shape[3] != resolution):
        if not hash(model) in warning_log:
            print("Warning: Input size ({}x{}) is not the native resolution ({}x{}) for this model. A resize will be performed but this could impact performance.".format(x.shape[2], x.shape[3], resolution, resolution))
            warning_log[hash(model)] = True
        return model.upsample(x)
    return x


def warn_normalization(x):
    """Check normalization of input and warn if possibly wrong. When 
    processing an image that may likely not have the correct 
    normalization we can issue a warning. But running min and max on 
    every image/batch is costly so we only do it on the first image/batch.
    """

    # Only run this check on the first image so we don't hurt performance.
    if not "norm_check" in warning_log:
        x_min = x.min()
        x_max = x.max()
        if torch.logical_or(-255 < x_min, x_max < 255) or torch.logical_or(x_min < -1024, 1024 < x_max):
            print(f'Warning: Input image does not appear to be normalized correctly. The input image has the range [{x_min:.2f},{x_max:.2f}] which doesn\'t seem to be in the [-1024,1024] range. This warning may be wrong though. Only the first image is tested and we are only using a heuristic in an attempt to save a user from using the wrong normalization.')
            warning_log["norm_correct"] = False
        else:
            warning_log["norm_correct"] = True

        warning_log["norm_check"] = True

def op_norm(outputs, op_threshs):
    """Normalize outputs according to operating points for a given model.
    Args: 
        outputs: outputs of self.classifier(). torch.Size(batch_size, num_tasks) 
        op_threshs_arr: torch.Size(batch_size, num_tasks) with self.op_threshs expanded.
    Returns:
        outputs_new: normalized outputs, torch.Size(batch_size, num_tasks)
    """
    # expand to batch size so we can do parallel comp
    op_threshs = op_threshs.expand(outputs.shape[0], -1)

    # initial values will be 0.5
    outputs_new = torch.zeros(outputs.shape, device=outputs.device) + 0.5

    # only select non-nan elements otherwise the gradient breaks
    mask_leq = (outputs < op_threshs) & ~torch.isnan(op_threshs)
    mask_gt = ~(outputs < op_threshs) & ~torch.isnan(op_threshs)

    # scale outputs less than thresh
    outputs_new[mask_leq] = outputs[mask_leq] / (op_threshs[mask_leq] * 2)
    # scale outputs greater than thresh
    outputs_new[mask_gt] = 1.0 - ((1.0 - outputs[mask_gt]) / ((1 - op_threshs[mask_gt]) * 2))

    return outputs_new


class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Modified from torchvision to have a variable number of input channels

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self,
                 growth_rate=32,
                 block_config=(6, 12, 24, 16),
                 num_init_features=64,
                 bn_size=4,
                 drop_rate=0,
                 num_classes=len(chexpert_pathologies),
                 in_channels=1,
                 weights=None,
                 op_threshs=None,
                 apply_sigmoid=False
                 ):

        super(DenseNet, self).__init__()

        self.apply_sigmoid = apply_sigmoid
        self.weights = weights

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(in_channels, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

        # needs to be register_buffer here so it will go to cuda/cpu easily
        self.register_buffer('op_threshs', op_threshs)

    def __repr__(self):
        if self.weights is not None:
            return "XRV-DenseNet121-{}".format(self.weights)
        else:
            return "XRV-DenseNet"

    def features2(self, x):
        x = fix_resolution(x, 224, self)
        warn_normalization(x)

        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
        return out

    def forward(self, x):
        x = fix_resolution(x, 224, self)
        print('Afer fix resolution')
        print(f"Shape: {x.shape}")
        print(f"Dtype: {x.dtype}")
        print(f"Values: {x}")

        features = self.features2(x)
        out = self.classifier(features)

        if hasattr(self, 'apply_sigmoid') and self.apply_sigmoid:
            out = torch.sigmoid(out)

        if hasattr(self, "op_threshs") and (self.op_threshs != None):
            out = torch.sigmoid(out)
            out = op_norm(out, self.op_threshs)
        return out

import os 
import numpy as np
import pprint
import pandas as pd
import collections
from skimage.io import imread
import random
from typing import Dict


def apply_window(arr, center, width, y_min=0, y_max=255):
    y_range = y_max - y_min
    arr = arr.astype('float64')
    width = float(width)

    below = arr <= (center - width / 2)
    above = arr > (center + width / 2)
    between = np.logical_and(~below, ~above)

    arr[below] = y_min
    arr[above] = y_max
    if between.any():
        arr[between] = (
                ((arr[between] - center) / width + 0.5) * y_range + y_min
        )

    return arr

def normalize(img, maxval, reshape=False):
    """Scales images to be roughly [-1024 1024]."""

    if img.max() > maxval:
        raise Exception("max image value ({}) higher than expected bound ({}).".format(img.max(), maxval))

    img = (2 * (img.astype(np.float32) / maxval) - 1.) * 1024

    if reshape:
        # Check that images are 2D arrays
        if len(img.shape) > 2:
            img = img[:, :, 0]
        if len(img.shape) < 2:
            print("error, dimension lower than 2 for image")

        # add color channel
        img = img[None, :, :]

    return img


def apply_random_window_width(img, min_width, max_width=256):
    width = np.random.randint(min_width, max_width + 1)
    img = apply_window(img, 256. / 2, width, y_min=0, y_max=255)
    return img

def apply_transforms(sample, transform, seed=None) -> Dict:
    """Applies transforms to the image and masks.
    The seeds are set so that the transforms that are applied
    to the image are the same that are applied to each mask.
    This way data augmentation will work for segmentation or 
    other tasks which use masks information.
    """

    if seed is None:
        MAX_RAND_VAL = 2147483647
        seed = np.random.randint(MAX_RAND_VAL)

    if transform is not None:
        random.seed(seed)
        torch.random.manual_seed(seed)
        sample["img"] = transform(sample["img"])

        if "pathology_masks" in sample:
            for i in sample["pathology_masks"].keys():
                random.seed(seed)
                torch.random.manual_seed(seed)
                sample["pathology_masks"][i] = transform(sample["pathology_masks"][i])

        if "semantic_masks" in sample:
            for i in sample["semantic_masks"].keys():
                random.seed(seed)
                torch.random.manual_seed(seed)
                sample["semantic_masks"][i] = transform(sample["semantic_masks"][i])

    return sample

thispath = os.path.dirname(os.path.realpath(__file__))
datapath = os.path.join(thispath, "data")

class Dataset():
    def __init__(self):
        pass

    def totals(self):
        counts = [dict(collections.Counter(items[~np.isnan(items)]).most_common()) for items in self.labels.T]
        return dict(zip(self.pathologies, counts))

    def __repr__(self):
        pprint.pprint(self.totals())
        return self.string()

    def check_paths_exist(self):
        if not os.path.isdir(self.imgpath):
            raise Exception("imgpath must be a directory")
        if not os.path.isfile(self.csvpath):
            raise Exception("csvpath must be a file")

    def limit_to_selected_views(self, views):
        """This function is called by subclasses to filter the
        images by view based on the values in .csv['view']
        """
        if type(views) is not list:
            views = [views]
        if '*' in views:
            # if you have the wildcard, the rest are irrelevant
            views = ["*"]
        self.views = views

        # missing data is unknown
        self.csv.view.fillna("UNKNOWN", inplace=True)

        if "*" not in views:
            self.csv = self.csv[self.csv["view"].isin(self.views)]  # Select the view

# Dataset
class CheX_Dataset(Dataset):
    """CheXpert Dataset

    CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison.
    Jeremy Irvin *, Pranav Rajpurkar *, Michael Ko, Yifan Yu, Silviana Ciurea-Ilcus, Chris Chute,
    Henrik Marklund, Behzad Haghgoo, Robyn Ball, Katie Shpanskaya, Jayne Seekins, David A. Mong,
    Safwan S. Halabi, Jesse K. Sandberg, Ricky Jones, David B. Larson, Curtis P. Langlotz,
    Bhavik N. Patel, Matthew P. Lungren, Andrew Y. Ng. https://arxiv.org/abs/1901.07031

    Dataset website here:
    https://stanfordmlgroup.github.io/competitions/chexpert/

    A small validation set is provided with the data as well, but is so tiny, it not included
    here.
    """

    def __init__(self,
                 imgpath,
                 csvpath=os.path.join(datapath, "chexpert_train.csv.gz"),
                 views=["PA"],
                 transform=None,
                 data_aug=None,
                 flat_dir=True,
                 seed=0,
                 unique_patients=True,
                 min_window_width=None,
                 labels_to_use=None,
                 use_class_balancing=False,
                 use_no_finding=False, 
                 split = 'train_class'
                 ):

        super(CheX_Dataset, self).__init__()
        np.random.seed(seed)  # Reset the seed so all runs are the same.

        self.pathologies = ["Enlarged Cardiomediastinum",
                            "Cardiomegaly",
                            "Lung Opacity",
                            "Lung Lesion",
                            "Edema",
                            "Consolidation",
                            "Pneumonia",
                            "Atelectasis",
                            "Pneumothorax",
                            "Pleural Effusion",
                            "Pleural Other",
                            "Fracture",
                            "Support Devices"]

        if use_no_finding:
            self.pathologies.append("No Finding")

        self.pathologies = sorted(self.pathologies)

        self.imgpath = imgpath
        self.transform = transform
        self.data_aug = data_aug
        self.csvpath = csvpath
        self.csv = pd.read_csv(self.csvpath)
        self.views = views
        self.min_window_width = min_window_width
        self.use_class_balancing = use_class_balancing
        self.split = split
        print('class balancing', use_class_balancing)

        self.csv["view"] = "Lateral" if "lateral" in self.csv["Path"] else "Frontal"
        self.csv.loc[(self.csv["view"] == "Frontal"), "view"] = self.csv["AP/PA"]  # If Frontal change with the corresponding value in the AP/PA column otherwise remains Lateral
        self.csv["view"] = self.csv["view"].replace({'Lateral': "L"})  # Rename Lateral with L
        
        if 'split' in self.csv.columns:
            self.csv = self.csv[self.csv['split'] == self.split].reset_index(drop=True)
        else:
            raise ValueError("The dataset CSV must include a 'split' column.")

        if labels_to_use:
            label_col = labels_to_use[0]
            label_classes = labels_to_use[1:]

            # only keep entries that are in the labels
            idx = self.csv[label_col].isin(label_classes)
            self.csv = self.csv[idx].copy()

            label_map = {}
            for label_i, label_name in enumerate(label_classes):
                label_map[label_name] = label_i
            self.labels = self.csv[label_col].map(label_map).values
            self.idxs_per_label = {}
            for label_i in range(len(label_classes)):
                self.idxs_per_label[label_i] = np.where(self.labels == label_i)[0]

            # if not self.use_class_balancing:
            #     print('Are you sure you dont want to use class balancing??')
        else:
            # Get our classes.
            self.csv.sort_values(by='Path', inplace=True)
            healthy = self.csv["No Finding"] == 1
            self.labels = []
            for pathology in self.pathologies:
                assert pathology in self.csv.columns
                if pathology == "No Finding":
                    for idx, row in self.csv.iterrows():
                        if row['No Finding'] != row['No Finding']: # only reassign if nan
                            if (row[6:18] == 1).sum():
                                self.csv.loc[idx, 'No Finding'] = 0
                elif pathology != "Support Devices":
                    self.csv.loc[healthy, pathology] = 0

                mask = self.csv[pathology]

                self.labels.append(mask.values)
            self.labels = np.asarray(self.labels).T
            self.labels = self.labels.astype(np.float32)

            # Make all the -1 values into nans to keep things simple
            self.labels[self.labels == -1] = np.nan

        # Rename pathologies
        #self.pathologies = list(np.char.replace(self.pathologies, "Pleural Effusion", "Effusion"))

        # patientid
        self.csv["patientid"] = self.csv["PatientID"]

        # age
        self.csv['age_years'] = self.csv['Age'] * 1.0
        self.csv['Age'][(self.csv['Age'] == 0)] = None

        # sex
        self.csv['sex_male'] = self.csv['Sex'] == 'Male'
        self.csv['sex_female'] = self.csv['Sex'] == 'Female'

        self.csv.sort_values(by='Path', inplace=True)
        self.csv = self.csv.reset_index(drop=True)
        
        print(f'Loaded {len(self)} samples for split {self.split}')

    def string(self):
        return self.__class__.__name__ + " num_samples={} views={} data_aug={}".format(len(self), self.views, self.data_aug)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.use_class_balancing: # hack to sample idx again
            idx = np.random.randint(len(self.idxs_per_label))
            idx = np.random.choice(self.idxs_per_label[idx])

        sample = {}
        sample["idx"] = idx
        sample["lab"] = self.labels[idx]
        sample["path"] = self.csv['Path'].iloc[idx]

        imgid = self.csv['Path'].iloc[idx]
        imgid = imgid.replace("CheXpert-v1.0-small/", "")
        img_path = os.path.join(self.imgpath, imgid)
        img = imread(img_path)

        # apply windowing
        if self.min_window_width:
            img = apply_random_window_width(img, self.min_window_width, max_width=256)

        sample["img"] = normalize(img, maxval=255, reshape=True)

        sample = apply_transforms(sample, self.transform)
        sample = apply_transforms(sample, self.data_aug)

        return sample