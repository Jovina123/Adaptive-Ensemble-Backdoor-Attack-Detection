import sys
import os

import torch
import torchvision
import numpy as np
import torch.nn.functional as F

from sklearn.utils import shuffle


# ============================================================
# PATH SETUP
# ============================================================

BEATRIX_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BEATRIX_DIR, "../.."))

# Beatrix's own config.py must come first.
sys.path.insert(0, BEATRIX_DIR)

# Project-level modules come second.
sys.path.insert(1, PROJECT_ROOT)

import config

from classifier_models import (
    PreActResNet18,
    ResNet18,
    PreActResNet34,
)

from dataloader import get_dataloader
from networks.models import Generator, NetC_MNIST
from utils import progress_bar


# ============================================================
# BACKDOOR CREATION
# ============================================================

def create_targets_bd(targets, opt):

    if opt.attack_mode == "all2one":

        bd_targets = torch.ones_like(targets) * opt.target_label

    elif opt.attack_mode == "all2all":

        bd_targets = torch.tensor(
            [(int(label) + 1) % opt.num_classes for label in targets],
            device=targets.device
        )

    else:
        raise Exception(
            "{} attack mode is not implemented".format(
                opt.attack_mode
            )
        )

    return bd_targets.to(opt.device)


def create_bd(inputs, targets, netG, netM, opt):

    bd_targets = create_targets_bd(
        targets,
        opt
    )

    patterns = netG(inputs)

    patterns = netG.normalize_pattern(
        patterns
    )

    masks_output = netM.threshold(
        netM(inputs)
    )

    bd_inputs = (
        inputs
        + (patterns - inputs) * masks_output
    )

    return (
        bd_inputs,
        bd_targets,
        patterns,
        masks_output
    )


def create_cross(inputs1, inputs2, netG, netM, opt):

    patterns2 = netG(inputs2)

    patterns2 = netG.normalize_pattern(
        patterns2
    )

    masks_output = netM.threshold(
        netM(inputs2)
    )

    inputs_cross = (
        inputs1
        + (patterns2 - inputs1) * masks_output
    )

    return (
        inputs_cross,
        patterns2,
        masks_output
    )


# ============================================================
# INTERMEDIATE FEATURE EXTRACTION
# ============================================================

class LayerActivations:

    def __init__(self, model, opt):

        self.opt = opt
        self.model = model
        self.model.eval()

        self.features = None
        self.hook = None
        self.hook_name = None

        self.build_hook()


    def build_hook(self):

        # ----------------------------------------------------
        # CIFAR-10 / GTSRB
        # ----------------------------------------------------

        if hasattr(self.model, "layer4"):

            layer4 = getattr(
                self.model,
                "layer4"
            )

            self.hook = layer4.register_forward_hook(
                self.hook_fn
            )

            self.hook_name = "layer4"

            print(
                "Feature hook: layer4"
            )

            return


        # ----------------------------------------------------
        # MNIST
        #
        # NetC_MNIST does not necessarily have layer4.
        # Find a suitable convolutional layer automatically.
        # ----------------------------------------------------

        candidate_layers = []

        for name, module in self.model.named_modules():

            if isinstance(
                module,
                (
                    torch.nn.Conv2d,
                    torch.nn.Sequential
                )
            ):

                candidate_layers.append(
                    (name, module)
                )


        if not candidate_layers:

            raise RuntimeError(
                "Could not find a suitable intermediate "
                "feature layer in the model."
            )


        # Prefer the last convolutional layer.
        conv_layers = [
            (name, module)
            for name, module in candidate_layers
            if isinstance(module, torch.nn.Conv2d)
        ]


        if conv_layers:

            name, module = conv_layers[-1]

        else:

            name, module = candidate_layers[-1]


        self.hook = module.register_forward_hook(
            self.hook_fn
        )

        self.hook_name = name

        print(
            "Feature hook:",
            self.hook_name
        )


    def hook_fn(self, module, input, output):

        # Use output because this is the actual
        # intermediate representation produced by
        # the selected layer.

        self.features = output.detach()


    def remove_hook(self):

        if self.hook is not None:

            self.hook.remove()

            self.hook = None


    def run_hook(self, x):

        self.features = None

        with torch.no_grad():

            self.model(x)

        if self.features is None:

            raise RuntimeError(
                "Feature hook did not capture any features."
            )

        return self.features


# ============================================================
# EVALUATION / FEATURE EXTRACTION
# ============================================================

def eval(
    netC,
    netG,
    netM,
    test_dl1,
    test_dl2,
    opt
):

    print(" Eval:")

    n_output_batches = 3
    n_output_images = 3

    total_sample = 0

    total_correct_clean = 0
    total_correct_bd = 0
    total_correct_cross = 0


    clean_feature = []
    bd_feature = []

    ori_label = []
    bd_label = []


    intermedia_feature = LayerActivations(
        netC.to(opt.device),
        opt
    )


    for batch_idx, (
        batch1,
        batch2
    ) in enumerate(
        zip(test_dl1, test_dl2)
    ):

        inputs, targets = batch1
        inputs2, targets2 = batch2


        inputs1 = inputs.to(
            opt.device
        )

        targets1 = targets.to(
            opt.device
        )

        inputs2 = inputs2.to(
            opt.device
        )

        targets2 = targets2.to(
            opt.device
        )


        bs = inputs1.shape[0]

        total_sample += bs


        # ====================================================
        # CLEAN
        # ====================================================

        with torch.no_grad():

            preds_clean = netC(
                inputs1
            )

        correct_clean = torch.sum(
            torch.argmax(
                preds_clean,
                1
            ) == targets1
        )

        total_correct_clean += (
            correct_clean.item()
        )

        acc_clean = (
            total_correct_clean
            * 100.0
            / total_sample
        )


        # ====================================================
        # BACKDOOR
        # ====================================================



        with torch.no_grad():

            (
                inputs_bd,
                targets_bd,
                _,
                _
            ) = create_bd(
                inputs1,
                targets1,
                netG,
                netM,
                opt
            )


            preds_bd = netC(
                inputs_bd
            )


        correct_bd = torch.sum(
            torch.argmax(
                preds_bd,
                1
            ) == targets_bd
        )

        total_correct_bd += (
            correct_bd.item()
        )

        acc_bd = (
            total_correct_bd
            * 100.0
            / total_sample
        )


        # ====================================================
        # CROSS
        # ====================================================

        with torch.no_grad():

            (
                inputs_cross,
                _,
                _
            ) = create_cross(
                inputs1,
                inputs2,
                netG,
                netM,
                opt
            )


            preds_cross = netC(
                inputs_cross
            )


        correct_cross = torch.sum(
            torch.argmax(
                preds_cross,
                1
            ) == targets1
        )

        total_correct_cross += (
            correct_cross.item()
        )

        acc_cross = (
            total_correct_cross
            * 100.0
            / total_sample
        )


        # ====================================================
        # FEATURE EXTRACTION
        # ====================================================

        clean_features_batch = (
            intermedia_feature
            .run_hook(inputs1)
            .cpu()
        )


        bd_features_batch = (
            intermedia_feature
            .run_hook(inputs_bd)
            .cpu()
        )


        clean_feature.append(
            clean_features_batch
        )

        bd_feature.append(
            bd_features_batch
        )


        ori_label.append(
            torch.argmax(
                preds_clean,
                1
            ).cpu()
        )

        bd_label.append(
            torch.argmax(
                preds_bd,
                1
            ).cpu()
        )


        # ====================================================
        # PROGRESS
        # ====================================================

        progress_bar(
            batch_idx,
            len(test_dl1),
            (
                "Acc Clean: {:.3f} | "
                "Acc Bd: {:.3f} | "
                "Acc Cross: {:.3f}"
            ).format(
                acc_clean,
                acc_bd,
                acc_cross
            )
        )


        # ====================================================
        # SAVE SAMPLE BACKDOOR IMAGES
        # ====================================================

        if batch_idx < n_output_batches:

            if hasattr(opt, "temps"):

                dir_temps = os.path.join(
                    opt.temps,
                    opt.dataset
                )

                os.makedirs(
                    dir_temps,
                    exist_ok=True
                )


                available_images = min(
                    n_output_images,
                    inputs_bd.shape[0]
                )


                if available_images > 0:

                    subs = []

                    for i in range(
                        available_images
                    ):

                        subs.append(
                            inputs_bd[
                                i:i + 1,
                                :,
                                :,
                                :
                            ]
                        )


                    try:

                        images = netG.denormalize_pattern(
                            torch.cat(
                                subs,
                                dim=3
                            )
                        )


                        file_name = (
                            "{}_{}_sample_{}.png"
                            .format(
                                opt.dataset,
                                opt.attack_mode,
                                batch_idx
                            )
                        )


                        file_path = os.path.join(
                            dir_temps,
                            file_name
                        )


                        torchvision.utils.save_image(
                            images,
                            file_path,
                            normalize=True,
                            pad_value=1
                        )

                    except Exception as e:

                        print(
                            "\nWarning: could not "
                            "save sample image:",
                            e
                        )


    # ========================================================
    # REMOVE HOOK
    # ========================================================

    intermedia_feature.remove_hook()


    # ========================================================
    # COMBINE FEATURES
    # ========================================================

    data = {

        "clean_feature":
            torch.cat(
                clean_feature,
                dim=0
            ),

        "bd_feature":
            torch.cat(
                bd_feature,
                dim=0
            ),

        "ori_label":
            torch.cat(
                ori_label,
                dim=0
            ),

        "bd_label":
            torch.cat(
                bd_label,
                dim=0
            )
    }


    print()
    print(
        "Feature extraction completed."
    )

    print(
        "Clean feature shape:",
        tuple(
            data["clean_feature"].shape
        )
    )

    print(
        "Backdoor feature shape:",
        tuple(
            data["bd_feature"].shape
        )
    )


    # ========================================================
    # SAVE FEATURE DATA
    # ========================================================

    if getattr(
        opt,
        "save_feature_data",
        False
    ):

        dir_data = os.path.join(
            PROJECT_ROOT,
            "feature_data"
        )


        ckpt_folder = os.path.join(
            dir_data,
            opt.dataset,
            opt.attack_mode,
            "target_" + str(
                opt.target_label
            )
        )


        os.makedirs(
            ckpt_folder,
            exist_ok=True
        )


        ckpt_path = os.path.join(
            ckpt_folder,
            "data.pt"
        )


        torch.save(
            data,
            ckpt_path
        )


        print(
            "Feature data saved:",
            ckpt_path
        )


    return data


# ============================================================
# LOAD MODEL / DATA
# ============================================================

def train(opt):

    # ========================================================
    # MODEL
    # ========================================================

    if opt.dataset == "cifar10":

        netC = PreActResNet18().to(
            opt.device
        )

    elif opt.dataset == "gtsrb":

        netC = PreActResNet18(
            num_classes=43
        ).to(
            opt.device
        )

    elif opt.dataset == "mnist":

        netC = NetC_MNIST().to(
            opt.device
        )

    else:

        raise Exception(
            "Invalid dataset"
        )


    # ========================================================
    # CHECKPOINT PATH
    # ========================================================

    ckpt_folder = os.path.join(
        PROJECT_ROOT,
        "checkpoints",
        opt.dataset,
        opt.attack_mode,
        "target_" + str(
            opt.target_label
        )
    )


    ckpt_path = os.path.join(
        ckpt_folder,
        "{}_{}_ckpt.pth.tar".format(
            opt.attack_mode,
            opt.dataset
        )
    )


    print()
    print(
        "Checkpoint:",
        ckpt_path
    )


    if not os.path.exists(
        ckpt_path
    ):

        raise FileNotFoundError(
            "\nCheckpoint not found:\n"
            + ckpt_path
            + "\n"
            "Make sure the corresponding "
            "backdoored model exists."
        )


    # ========================================================
    # LOAD CHECKPOINT
    #
    # weights_only=False is required for these older
    # Beatrix checkpoints because they contain argparse.Namespace.
    # Only use this with a trusted checkpoint.
    # ========================================================

    state_dict = torch.load(
        ckpt_path,
        map_location=opt.device,
        weights_only=False
    )


    # ========================================================
    # CLASSIFIER
    # ========================================================

    print("load C")

    netC.load_state_dict(
        state_dict["netC"]
    )

    netC.to(
        opt.device
    )

    netC.eval()

    netC.requires_grad_(
        False
    )


    # ========================================================
    # GENERATOR
    # ========================================================

    print("load G")

    netG = Generator(
        opt
    )

    netG.load_state_dict(
        state_dict["netG"]
    )

    netG.to(
        opt.device
    )

    netG.eval()

    netG.requires_grad_(
        False
    )


    # ========================================================
    # MASK GENERATOR
    # ========================================================

    print("load M")

    netM = Generator(
        opt,
        out_channels=1
    )

    netM.load_state_dict(
        state_dict["netM"]
    )

    netM.to(
        opt.device
    )

    netM.eval()

    netM.requires_grad_(
        False
    )


    # ========================================================
    # DATALOADER
    # ========================================================

    opt.n_iters = 1
    opt.batchsize = 256

    test_dl1 = get_dataloader(
        opt,
        train=False
    )

    test_dl2 = get_dataloader(
        opt,
        train=False
    )


    # ========================================================
    # FEATURE DATA CACHE
    # ========================================================

    dir_data = os.path.join(
        PROJECT_ROOT,
        "feature_data"
    )


    feature_folder = os.path.join(
        dir_data,
        opt.dataset,
        opt.attack_mode,
        "target_" + str(
            opt.target_label
        )
    )


    feature_path = os.path.join(
        feature_folder,
        "data.pt"
    )


    # IMPORTANT:
    # Delete/recreate feature data when testing
    # after changing the feature-hook implementation.
    #
    # We use it if it already exists.

    if os.path.exists(
        feature_path
    ):

        print()
        print(
            "Loading cached feature data:"
        )

        print(
            feature_path
        )


        data = torch.load(
            feature_path,
            map_location="cpu",
            weights_only=False
        )

    else:

        data = eval(
            netC,
            netG,
            netM,
            test_dl1,
            test_dl2,
            opt
        )


    return data


# ============================================================
# BACKDOOR DATASET SELECTION
# ============================================================

def bd_dataset(
    X_all,
    y_all,
    n_poison=100,
    num_class=10,
    target_class=[],
    source_class=[1],
    balance=True
):

    if len(y_all.shape) > 1:

        labels = np.argmax(
            y_all,
            axis=-1
        )

    else:

        labels = y_all


    train_classes = np.arange(
        num_class
    )


    clean_x = []
    clean_y = []
    poison_y = []


    if balance:

        for tc in train_classes:

            if tc in target_class:
                continue

            if tc not in source_class:
                continue


            index = np.where(
                labels == tc
            )


            clean_x.append(
                X_all[
                    index
                ][0:n_poison]
            )


            clean_y.append(
                y_all[
                    index
                ][0:n_poison]
            )


            poison_index = np.where(
                labels == target_class[0]
            )


            poison_y.append(
                y_all[
                    poison_index
                ][0:n_poison]
            )


        clean_x = torch.cat(
            clean_x,
            dim=0
        )

        clean_y = torch.cat(
            clean_y,
            dim=0
        )

        poison_y = torch.cat(
            poison_y,
            dim=0
        )


    else:

        index = np.where(
            labels != target_class[0]
        )


        clean_x.append(
            X_all[
                index
            ][0:n_poison]
        )


        clean_y.append(
            y_all[
                index
            ][0:n_poison]
        )


        poison_index = np.where(
            labels == target_class[0]
        )


        poison_y.append(
            y_all[
                poison_index
            ][0:n_poison]
        )


        clean_x = torch.cat(
            clean_x,
            dim=0
        )

        clean_y = torch.cat(
            clean_y,
            dim=0
        )

        poison_y = torch.cat(
            poison_y,
            dim=0
        )


    return (
        clean_x,
        poison_y
    )


# ============================================================
# GAUSSIAN KERNEL
# ============================================================

def gaussian_kernel(
    x1,
    x2,
    kernel_mul=2.0,
    kernel_num=5,
    fix_sigma=0,
    mean_sigma=0
):

    x1_sample_size = x1.shape[0]
    x2_sample_size = x2.shape[0]


    x1_tile_shape = []
    x2_tile_shape = []
    norm_shape = []


    for i in range(
        len(x1.shape) + 1
    ):

        if i == 1:

            x1_tile_shape.append(
                x2_sample_size
            )

        else:

            x1_tile_shape.append(
                1
            )


        if i == 0:

            x2_tile_shape.append(
                x1_sample_size
            )

        else:

            x2_tile_shape.append(
                1
            )


        if not (
            i == 0
            or i == 1
        ):

            norm_shape.append(
                i
            )


    tile_x1 = torch.unsqueeze(
        x1,
        1
    ).repeat(
        x1_tile_shape
    )


    tile_x2 = torch.unsqueeze(
        x2,
        0
    ).repeat(
        x2_tile_shape
    )


    L2_distance = torch.square(
        tile_x1 - tile_x2
    ).sum(
        dim=norm_shape
    )


    # ========================================================
    # BANDWIDTH
    # ========================================================

    if fix_sigma:

        bandwidth = fix_sigma

    elif mean_sigma:

        bandwidth = torch.mean(
            L2_distance
        )

    else:

        bandwidth = torch.median(
            L2_distance.reshape(
                L2_distance.shape[0],
                -1
            )
        )


    bandwidth = bandwidth / (
        kernel_mul ** (
            kernel_num // 2
        )
    )


    bandwidth_list = [
        bandwidth
        * (
            kernel_mul ** i
        )
        for i in range(
            kernel_num
        )
    ]


    kernel_val = [
        torch.exp(
            -L2_distance
            / bandwidth_temp
        )
        for bandwidth_temp
        in bandwidth_list
    ]


    return sum(
        kernel_val
    )


# ============================================================
# KMMD
# ============================================================

def kmmd_dist(
    x1,
    x2
):

    if x1.shape[0] == 0:
        return np.array(
            [0.0]
        )

    if x2.shape[0] == 0:
        return np.array(
            [0.0]
        )


    X_total = torch.cat(
        [
            x1,
            x2
        ],
        0
    )


    Gram_matrix = gaussian_kernel(
        X_total,
        X_total,
        kernel_mul=2.0,
        kernel_num=2,
        fix_sigma=0,
        mean_sigma=0
    )


    n = int(
        x1.shape[0]
    )

    m = int(
        x2.shape[0]
    )


    x1x1 = Gram_matrix[
        :n,
        :n
    ]

    x2x2 = Gram_matrix[
        n:,
        n:
    ]

    x1x2 = Gram_matrix[
        :n,
        n:
    ]


    diff = (
        torch.mean(x1x1)
        + torch.mean(x2x2)
        - 2 * torch.mean(x1x2)
    )


    diff = (
        (m * n)
        / (m + n)
        * diff
    )


    return diff.cpu().numpy()


# ============================================================
# BEATRIX FEATURE CORRELATION DETECTOR
# ============================================================

class Feature_Correlations:

    def __init__(
        self,
        POWER_list,
        mode="mad"
    ):

        self.power = POWER_list
        self.mode = mode


    def train(
        self,
        in_data
    ):

        self.in_data = in_data


        if "mad" in self.mode:

            (
                self.medians,
                self.mads
            ) = self.get_median_mad(
                self.in_data
            )


            (
                self.mins,
                self.maxs
            ) = self.minmax_mad()


    def minmax_mad(
        self
    ):

        mins = []
        maxs = []


        for L, mm in enumerate(
            zip(
                self.medians,
                self.mads
            )
        ):

            medians = mm[0]
            mads = mm[1]


            if L == len(mins):

                mins.append(
                    [None]
                    * len(self.power)
                )

                maxs.append(
                    [None]
                    * len(self.power)
                )


            for p, P in enumerate(
                self.power
            ):

                mins[L][p] = (
                    medians[p]
                    - mads[p] * 10
                )


                maxs[L][p] = (
                    medians[p]
                    + mads[p] * 10
                )


        return (
            mins,
            maxs
        )


    def G_p(
        self,
        ob,
        p
    ):

        temp = ob.detach()

        temp = temp ** p


        temp = temp.reshape(
            temp.shape[0],
            temp.shape[1],
            -1
        )


        temp = torch.matmul(
            temp,
            temp.transpose(
                dim0=2,
                dim1=1
            )
        )


        temp = temp.triu()


        temp = (
            temp.sign()
            * torch.abs(temp)
            ** (1 / p)
        )


        temp = temp.reshape(
            temp.shape[0],
            -1
        )


        self.num_feature = (
            temp.shape[-1] / 2
        )


        return temp


    def get_median_mad(
        self,
        feat_list
    ):

        medians = []
        mads = []


        for L, feat_L in enumerate(
            feat_list
        ):

            if L == len(medians):

                medians.append(
                    [None]
                    * len(self.power)
                )

                mads.append(
                    [None]
                    * len(self.power)
                )


            for p, P in enumerate(
                self.power
            ):

                g_p = self.G_p(
                    feat_L,
                    P
                )


                current_median = (
                    g_p.median(
                        dim=0,
                        keepdim=True
                    )[0]
                )


                current_mad = (
                    torch.abs(
                        g_p
                        - current_median
                    ).median(
                        dim=0,
                        keepdim=True
                    )[0]
                )


                medians[L][p] = (
                    current_median
                )

                mads[L][p] = (
                    current_mad
                )


        return (
            medians,
            mads
        )


    def get_deviations_(
        self,
        feat_list
    ):

        deviations = []
        batch_deviations = []


        for L, feat_L in enumerate(
            feat_list
        ):

            dev = 0


            for p, P in enumerate(
                self.power
            ):

                g_p = self.G_p(
                    feat_L,
                    P
                )


                dev += (
                    F.relu(
                        self.mins[L][p]
                        - g_p
                    )
                    / torch.abs(
                        self.mins[L][p]
                        + 1e-6
                    )
                ).sum(
                    dim=1,
                    keepdim=True
                )


                dev += (
                    F.relu(
                        g_p
                        - self.maxs[L][p]
                    )
                    / torch.abs(
                        self.maxs[L][p]
                        + 1e-6
                    )
                ).sum(
                    dim=1,
                    keepdim=True
                )


            batch_deviations.append(
                dev.cpu()
                .detach()
                .numpy()
            )


        batch_deviations = np.concatenate(
            batch_deviations,
            axis=1
        )


        deviations.append(
            batch_deviations
        )


        deviations = (
            np.concatenate(
                deviations,
                axis=0
            )
            / self.num_feature
            / len(self.power)
        )


        return deviations


    def get_deviations(
        self,
        feat_list
    ):

        deviations = []
        batch_deviations = []


        for L, feat_L in enumerate(
            feat_list
        ):

            dev = 0


            for p, P in enumerate(
                self.power
            ):

                g_p = self.G_p(
                    feat_L,
                    P
                )


                dev += torch.sum(
                    torch.abs(
                        g_p
                        - self.medians[L][p]
                    )
                    / (
                        self.mads[L][p]
                        + 1e-6
                    ),
                    dim=1,
                    keepdim=True
                )


            batch_deviations.append(
                dev.cpu()
                .detach()
                .numpy()
            )


        batch_deviations = np.concatenate(
            batch_deviations,
            axis=1
        )


        deviations.append(
            batch_deviations
        )


        deviations = (
            np.concatenate(
                deviations,
                axis=0
            )
            / self.num_feature
            / len(self.power)
        )


        return deviations


# ============================================================
# THRESHOLD DETERMINATION
# ============================================================

def threshold_determine(
    clean_feature_target,
    ood_detection
):

    test_deviations_list = []

    step = 5


    # Need enough samples for 5-fold splitting.
    if len(clean_feature_target) < step:

        raise RuntimeError(
            "Not enough clean samples to "
            "calculate Beatrix thresholds."
        )


    for i in range(step):

        index_mask = np.ones(
            (
                len(clean_feature_target),
            )
        )


        start = (
            i
            * int(
                len(clean_feature_target)
                // step
            )
        )


        end = (
            (i + 1)
            * int(
                len(clean_feature_target)
                // step
            )
        )


        index_mask[
            start:end
        ] = 0


        clean_feature_target_train = (
            clean_feature_target[
                np.where(
                    index_mask == 1
                )
            ]
        )


        clean_feature_target_test = (
            clean_feature_target[
                np.where(
                    index_mask == 0
                )
            ]
        )


        ood_detection.train(
            in_data=[
                clean_feature_target_train
            ]
        )


        test_deviations = (
            ood_detection.get_deviations_(
                [
                    clean_feature_target_test
                ]
            )
        )


        test_deviations_list.append(
            test_deviations
        )


    test_deviations = np.concatenate(
        test_deviations_list,
        0
    )


    test_deviations_sort = np.sort(
        test_deviations,
        0
    )


    percentile_95 = (
        test_deviations_sort[
            int(
                len(test_deviations_sort)
                * 0.95
            )
        ][0]
    )


    percentile_99 = (
        test_deviations_sort[
            min(
                int(
                    len(test_deviations_sort)
                    * 0.99
                ),
                len(test_deviations_sort) - 1
            )
        ][0]
    )


    print(
        f"percentile_95:{percentile_95}"
    )

    print(
        f"percentile_99:{percentile_99}"
    )


    return (
        percentile_95,
        percentile_99
    )


# ============================================================
# BEATRIX DETECTOR
# ============================================================

class BEAT_detector:

    def __init__(
        self,
        opt,
        clean_test=500,
        bd_test=500,
        order_list=np.arange(1, 9)
    ):

        self.opt = opt

        self.test_target_label = (
            opt.target_label
        )

        self.order_list = order_list


        if opt.dataset in [
            "cifar10",
            "gtsrb",
            "mnist"
        ]:

            self.clean_data_perclass = 30

        else:

            raise Exception(
                "Invalid dataset"
            )


        self.clean_test = clean_test
        self.bd_test = bd_test


    def _detecting(
        self,
        data
    ):

        opt = self.opt


        clean_feature = (
            data["clean_feature"]
            .to(opt.device)
        )


        bd_feature = (
            data["bd_feature"]
            .to(opt.device)
        )


        ori_label = (
            data["ori_label"]
            .cpu()
        )


        bd_label = (
            data["bd_label"]
            .cpu()
        )


        (
            clean_feature,
            bd_feature,
            ori_label,
            bd_label
        ) = shuffle(
            clean_feature,
            bd_feature,
            ori_label,
            bd_label
        )


        print()
        print(
            "BEATRIX GRAM-MATRIX OOD DETECTION"
        )
        print()


        ood_detection = Feature_Correlations(
            POWER_list=self.order_list,
            mode="mad"
        )


        J_t = []
        threshold_list = []


        for test_target_label in range(
            opt.num_classes
        ):

            print(
                f"*****class:{test_target_label}*****"
            )


            # =================================================
            # CLEAN SAMPLES FOR THIS CLASS
            # =================================================

            class_indices = np.where(
                ori_label.numpy()
                == test_target_label
            )[0]


            if len(class_indices) < (
                self.clean_data_perclass
                + 5
            ):

                print(
                    "Not enough clean "
                    "samples for class",
                    test_target_label
                )

                continue


            clean_feature_target = (
                clean_feature[
                    class_indices
                ]
            )


            clean_feature_defend = (
                clean_feature_target[
                    :self.clean_data_perclass
                ]
            )


            # =================================================
            # THRESHOLD
            # =================================================

            (
                threshold_95,
                threshold_99
            ) = threshold_determine(
                clean_feature_defend,
                ood_detection
            )


            threshold_list.append(
                [
                    test_target_label,
                    threshold_95,
                    threshold_99
                ]
            )


            ood_detection.train(
                in_data=[
                    clean_feature_defend
                ]
            )


            clean_feature_test = (
                clean_feature_target[
                    -self.clean_test:
                ]
            )


            clean_label_test = np.zeros(
                (
                    clean_feature_test.shape[0],
                )
            )


            # =================================================
            # TARGET CLASS
            # =================================================

            if (
                test_target_label
                == opt.target_label
            ):

                bd_feature_test, _ = (
                    bd_dataset(
                        bd_feature,
                        ori_label,
                        n_poison=self.bd_test,
                        num_class=opt.num_classes,
                        target_class=[
                            test_target_label
                        ],
                        source_class=list(
                            np.arange(
                                opt.num_classes
                            )
                        ),
                        balance=False
                    )
                )


                bd_label_test = np.ones(
                    (
                        bd_feature_test.shape[0],
                    )
                )


                feature_test = torch.cat(
                    [
                        clean_feature_test,
                        bd_feature_test
                    ],
                    0
                )


                label_test = np.concatenate(
                    [
                        clean_label_test,
                        bd_label_test
                    ],
                    0
                )


                # =============================================
                # TP95
                # =============================================

                clean_deviations_sort = (
                    np.sort(
                        ood_detection
                        .get_deviations_(
                            [
                                clean_feature_test
                            ]
                        ),
                        0
                    )
                )


                bd_deviations_sort = (
                    np.sort(
                        ood_detection
                        .get_deviations_(
                            [
                                bd_feature_test
                            ]
                        ),
                        0
                    )
                )


                if len(
                    clean_deviations_sort
                ) > 0:

                    idx95 = min(
                        int(
                            len(
                                clean_deviations_sort
                            ) * 0.95
                        ),
                        len(
                            clean_deviations_sort
                        ) - 1
                    )


                    percentile_95 = np.where(
                        bd_deviations_sort
                        > clean_deviations_sort[
                            idx95
                        ],
                        1,
                        0
                    )


                    print(
                        "percentile_95:{}"
                        ",TP95:{}".format(
                            clean_deviations_sort[
                                idx95
                            ],
                            percentile_95.sum()
                            / len(
                                bd_deviations_sort
                            )
                        )
                    )


                    idx99 = min(
                        int(
                            len(
                                clean_deviations_sort
                            ) * 0.99
                        ),
                        len(
                            clean_deviations_sort
                        ) - 1
                    )


                    percentile_99 = np.where(
                        bd_deviations_sort
                        > clean_deviations_sort[
                            idx99
                        ],
                        1,
                        0
                    )


                    print(
                        "percentile_99:{}"
                        ",TP99:{}".format(
                            clean_deviations_sort[
                                idx99
                            ],
                            percentile_99.sum()
                            / len(
                                bd_deviations_sort
                            )
                        )
                    )


            else:

                feature_test = (
                    clean_feature_test
                )

                label_test = (
                    clean_label_test
                )


            # =================================================
            # DETECTION
            # =================================================

            test_deviations = (
                ood_detection
                .get_deviations_(
                    [
                        feature_test
                    ]
                )
            )


            ood_label_95 = np.where(
                test_deviations
                > threshold_95,
                1,
                0
            ).squeeze()


            ood_label_99 = np.where(
                test_deviations
                > threshold_99,
                1,
                0
            ).squeeze()


            # Make sure scalar results are arrays.
            ood_label_95 = np.atleast_1d(
                ood_label_95
            )

            ood_label_99 = np.atleast_1d(
                ood_label_99
            )

            label_test = np.atleast_1d(
                label_test
            )


            # =================================================
            # FALSE NEGATIVE / POSITIVE
            # =================================================

            false_negative_95 = np.where(
                label_test
                - ood_label_95
                > 0,
                1,
                0
            )


            false_negative_99 = np.where(
                label_test
                - ood_label_99
                > 0,
                1,
                0
            )


            false_positive_95 = np.where(
                label_test
                - ood_label_95
                < 0,
                1,
                0
            )


            false_positive_99 = np.where(
                label_test
                - ood_label_99
                < 0,
                1,
                0
            )


            print(
                "false_negative_95:{},"
                "false_negative_99:{}".format(
                    false_negative_95.sum(),
                    false_negative_99.sum()
                )
            )


            print(
                "false_positive_95:{},"
                "false_positive_99:{}".format(
                    false_positive_95.sum(),
                    false_positive_99.sum()
                )
            )


            # =================================================
            # GROUP FEATURES
            # =================================================

            clean_mask = (
                ood_label_95 == 0
            )

            bd_mask = (
                ood_label_95 == 1
            )


            clean_feature_group = (
                feature_test[
                    clean_mask
                ]
            )


            bd_feature_group = (
                feature_test[
                    bd_mask
                ]
            )


            # =================================================
            # KMMD
            # =================================================

            if (
                clean_feature_group.shape[0]
                < 1
                or
                bd_feature_group.shape[0]
                < 1
            ):

                kmmd = np.array(
                    [0.0]
                )

            else:

                # Spatial average of feature maps.
                #
                # This works for:
                #   [N,C,H,W]
                #
                # and also protects MNIST if the selected
                # feature representation has an unexpected
                # shape.

                if (
                    clean_feature_group.dim()
                    >= 4
                ):

                    clean_feature_flat = (
                        torch.mean(
                            clean_feature_group,
                            dim=tuple(
                                range(
                                    2,
                                    clean_feature_group.dim()
                                )
                            )
                        )
                    )


                    bd_feature_flat = (
                        torch.mean(
                            bd_feature_group,
                            dim=tuple(
                                range(
                                    2,
                                    bd_feature_group.dim()
                                )
                            )
                        )
                    )

                else:

                    clean_feature_flat = (
                        clean_feature_group
                    )

                    bd_feature_flat = (
                        bd_feature_group
                    )


                kmmd = kmmd_dist(
                    clean_feature_flat,
                    bd_feature_flat
                )


            kmmd_value = float(
                np.asarray(kmmd).reshape(-1)[0]
            )


            print(
                f"KMMD:{kmmd_value}."
            )


            J_t.append(
                kmmd_value
            )


        # =====================================================
        # FINAL BEATRIX SCORE
        # =====================================================

        print()
        print(
            "J values:"
        )

        print(
            J_t
        )


        if len(J_t) == 0:

            raise RuntimeError(
                "No valid class results "
                "were produced."
            )


        J_t = np.asarray(
            J_t
        )


        J_t_median = np.median(
            J_t
        )


        J_MAD = np.median(
            np.abs(
                J_t
                - J_t_median
            )
        )


        J_star = (
            np.abs(
                J_t
                - J_t_median
            )
            / 1.4826
            / (
                J_MAD
                + 1e-6
            )
        )


        print()
        print(
            "BEATRIX J* SCORES"
        )


        for i, score in enumerate(
            J_star
        ):

            print(
                "class {}: {:.2f}".format(
                    i,
                    score
                )
            )


        self._save_result_to_dir(
            result=[
                J_star
            ]
        )


    # ========================================================
    # SAVE RESULT
    # ========================================================

    def _save_result_to_dir(
        self,
        result
    ):

        opt = self.opt


        result_dir = os.path.join(
            PROJECT_ROOT,
            "results",
            opt.dataset
        )


        result_path = os.path.join(
            result_dir,
            opt.attack_mode,
            "target_" + str(
                opt.target_label
            )
        )


        os.makedirs(
            result_path,
            exist_ok=True
        )


        output_path = os.path.join(
            result_path,
            "{}_{}_output.txt".format(
                opt.attack_mode,
                opt.dataset
            )
        )


        with open(
            output_path,
            "w+"
        ) as f:

            J_star_to_save = [
                str(value)
                for value in result[0]
            ]


            f.write(
                ", ".join(
                    J_star_to_save
                )
                + "\n"
            )


        print()
        print(
            "Result saved:"
        )

        print(
            output_path
        )


# ============================================================
# MAIN
# ============================================================

def main(k):

    opt = (
        config
        .get_arguments()
        .parse_args()
    )


    # ========================================================
    # DEVICE
    # ========================================================

    opt.device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


    print(
        opt.device
    )


    # ========================================================
    # TARGET CLASS
    # ========================================================

    opt.target_label = k


    print(
        "-" * 50
        + "opt.target_label:"
        + str(
            opt.target_label
        )
    )


    # ========================================================
    # NUMBER OF CLASSES
    # ========================================================

    if opt.dataset in [
        "mnist",
        "cifar10"
    ]:

        opt.num_classes = 10

    elif opt.dataset == "gtsrb":

        opt.num_classes = 43

    else:

        raise Exception(
            "Invalid Dataset"
        )


    # ========================================================
    # IMAGE SIZE
    # ========================================================

    if opt.dataset == "cifar10":

        opt.input_height = 32
        opt.input_width = 32
        opt.input_channel = 3


    elif opt.dataset == "gtsrb":

        opt.input_height = 32
        opt.input_width = 32
        opt.input_channel = 3


    elif opt.dataset == "mnist":

        opt.input_height = 28
        opt.input_width = 28
        opt.input_channel = 1


    else:

        raise Exception(
            "Invalid Dataset"
        )


    # ========================================================
    # RUN
    # ========================================================

    data = train(
        opt
    )


    beat_detector = BEAT_detector(
        opt
    )


    beat_detector._detecting(
        data
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    # Parse once so CUDA_VISIBLE_DEVICES
    # can be configured.

    opt = (
        config
        .get_arguments()
        .parse_args()
    )


    os.environ[
        "CUDA_VISIBLE_DEVICES"
    ] = opt.gpu

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Your current MNIST/CIFAR setup has only target_0
    # checkpoints available.
    #
    # Therefore we run target 0 only.
    #
    # Change this to a loop later if you have checkpoints
    # for target_0 through target_9.
    # --------------------------------------------------------

    main(0)