import argparse


def get_arguments():

    parser = argparse.ArgumentParser(
        description="Beatrix backdoor detection"
    )

    # ========================================================
    # DIRECTORIES
    # ========================================================

    parser.add_argument(
        "--checkpoints",
        type=str,
        default="../../checkpoints/"
    )

    parser.add_argument(
        "--data_root",
        type=str,
        default="../../data/"
    )

    parser.add_argument(
        "--temps",
        type=str,
        default="../../temps"
    )

    parser.add_argument(
        "--result",
        type=str,
        default="./results"
    )

    # ========================================================
    # DEVICE
    # ========================================================

    parser.add_argument(
        "--device",
        type=str,
        default="cpu"
    )

    parser.add_argument(
        "--gpu",
        type=str,
        default="0"
    )

    # ========================================================
    # DATASET
    # ========================================================

    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        choices=["mnist", "cifar10", "gtsrb", "vggface"]
    )

    parser.add_argument(
        "--input_height",
        type=int,
        default=None
    )

    parser.add_argument(
        "--input_width",
        type=int,
        default=None
    )

    parser.add_argument(
        "--input_channel",
        type=int,
        default=None
    )

    parser.add_argument(
        "--num_classes",
        type=int,
        default=None
    )

    # ========================================================
    # DATALOADER
    # ========================================================

    parser.add_argument(
        "--batchsize",
        type=int,
        default=64
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=4
    )

    parser.add_argument(
        "--random_crop",
        type=int,
        default=4
    )

    parser.add_argument(
        "--random_rotation",
        type=int,
        default=0
    )

    # ========================================================
    # ATTACK
    # ========================================================

    parser.add_argument(
        "--attack_mode",
        type=str,
        default="all2one",
        choices=["all2one", "all2all"]
    )

    parser.add_argument(
        "--target_label",
        type=int,
        default=0
    )

    parser.add_argument(
        "--p_attack",
        type=float,
        default=0.1
    )

    parser.add_argument(
        "--p_cross",
        type=float,
        default=0.1
    )

    parser.add_argument(
        "--mask_density",
        type=float,
        default=0.032
    )

    # ========================================================
    # TRAINING
    # ========================================================

    parser.add_argument(
        "--n_iters",
        type=int,
        default=2
    )

    parser.add_argument(
        "--lr_G",
        type=float,
        default=0.01
    )

    parser.add_argument(
        "--lr_C",
        type=float,
        default=0.01
    )

    parser.add_argument(
        "--lr_M",
        type=float,
        default=0.01
    )

    # ========================================================
    # LOSS
    # ========================================================

    parser.add_argument(
        "--lambda_div",
        type=float,
        default=1.0
    )

    parser.add_argument(
        "--lambda_norm",
        type=float,
        default=100.0
    )

    # ========================================================
    # SCHEDULERS
    # ========================================================

    parser.add_argument(
        "--schedulerG_milestones",
        type=int,
        nargs="+",
        default=[200, 300, 400, 500]
    )

    parser.add_argument(
        "--schedulerC_milestones",
        type=int,
        nargs="+",
        default=[100, 200, 300, 400]
    )

    parser.add_argument(
        "--schedulerM_milestones",
        type=int,
        nargs="+",
        default=[10, 20]
    )

    parser.add_argument(
        "--schedulerG_lambda",
        type=float,
        default=0.1
    )

    parser.add_argument(
        "--schedulerC_lambda",
        type=float,
        default=0.1
    )

    parser.add_argument(
        "--schedulerM_lambda",
        type=float,
        default=0.1
    )

    # ========================================================
    # BEATRIX / NEURAL CLEANSE SETTINGS
    # ========================================================

    parser.add_argument(
        "--init_cost",
        type=float,
        default=1e-3
    )

    parser.add_argument(
        "--atk_succ_threshold",
        type=float,
        default=99.0
    )

    parser.add_argument(
        "--early_stop",
        type=bool,
        default=True
    )

    parser.add_argument(
        "--early_stop_threshold",
        type=float,
        default=99.0
    )

    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=25
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=5
    )

    parser.add_argument(
        "--cost_multiplier",
        type=float,
        default=2.0
    )

    parser.add_argument(
        "--epoch",
        type=int,
        default=50
    )

    parser.add_argument(
        "--EPSILON",
        type=float,
        default=1e-7
    )

    parser.add_argument(
        "--to_file",
        type=bool,
        default=True
    )

    parser.add_argument(
        "--n_times_test",
        type=int,
        default=1
    )

    parser.add_argument(
        "--save_feature_data",
        type=bool,
        default=False
    )

    parser.add_argument(
        "--true_target_label",
        type=int,
        default=None
    )

    parser.add_argument(
        "--total_label",
        type=int,
        default=None
    )

    return parser
