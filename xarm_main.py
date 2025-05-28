import os
import argparse
import json
from pathlib import Path
import importlib
import copy
import functools
import datetime

from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
from lightning.pytorch.strategies import DDPStrategy
from lightning import seed_everything
import torch.distributed as dist

# Use BaseTrainer directly
from robovlms.train.base_trainer import BaseTrainer
# Import the new XArmDataModule
from robovlms.data.datamodule.xarm_datamodule import XArmDataModule # Adapted path
from robovlms.data.data_utils import preprocess_image
from robovlms.utils.setup_callback import SetupCallback


def get_date_str(): #
    return str(datetime.date.today()) #


def get_obj_from_str(string, reload=False): #
    module, cls = string.rsplit(".", 1) #
    if reload: #
        module_imp = importlib.import_module(module) #
        importlib.reload(module_imp) #
    return getattr(importlib.import_module(module, package=None), cls) #


def instantiate_from_config(config): #
    if not "target" in config: #
        raise KeyError("Expected key `target` to instantiate.") #
    return get_obj_from_str(config["target"])(**config.get("params", dict())) #


def init_lr_monitor_callback(): #
    return LearningRateMonitor(logging_interval="step") #


def init_setup_callback(config_dict): # Renamed arg to avoid conflict
    return SetupCallback( #
        now=str(datetime.datetime.now()).replace(" ", "_"), #
        logdir=config_dict["log_dir"], #
        ckptdir=config_dict["output_dir"], #
        cfgdir=config_dict["log_dir"], #
        config=config_dict, #
    )


def init_trainer_config_fn(configs_dict): # Renamed from init_trainer_config to avoid conflict, and arg
    trainer_config = copy.deepcopy(configs_dict["trainer"]) #
    trainer_config["devices"] = configs_dict.get("gpus", "auto") #
    trainer_config["num_nodes"] = configs_dict.get("num_nodes", 1) #
    trainer_config["gradient_clip_val"] = configs_dict.get("gradient_clip_val", 0.0) #
    exp_name = configs_dict.get("exp_name", configs_dict.get("task_name", "default_xarm_exp")) # Adjusted for xarm

    if "strategy" not in trainer_config or trainer_config["strategy"] == "ddp": #
        trainer_config["strategy"] = DDPStrategy(find_unused_parameters=True) #
    elif trainer_config["strategy"] == "deepspeed_stage_2": # common example
        from lightning.pytorch.strategies import DeepSpeedStrategy
        trainer_config["strategy"] = DeepSpeedStrategy(stage=2) # <--- 修改在这里
        # Add other deepspeed specific configs if needed, e.g. offload_optimizer, offload_parameters

    loggers = None #
    log_dir_path = Path(os.path.join(get_date_str(), exp_name)) # # Renamed log_dir to log_dir_path
    configs_dict["log_dir"] = configs_dict["log_root"] / log_dir_path #
    if isinstance(trainer_config.get("logger"), list): #
        loggers = [] #
        for logger_type in trainer_config.get("logger"): # # Renamed logger to logger_type
            if logger_type == "tensorboard": #
                loggers.append( #
                    TensorBoardLogger(configs_dict["log_dir"].as_posix(), name=exp_name) #
                )
            elif logger_type == "csv": #
                loggers.append(CSVLogger(configs_dict["log_dir"].as_posix(), name=exp_name)) #
            else: #
                raise NotImplementedError #

    trainer_config["logger"] = loggers #

    ckpt_dir_path = Path(os.path.join(get_date_str(), exp_name)) # # Renamed ckpt_dir to ckpt_dir_path
    configs_dict["output_dir"] = configs_dict["output_root"] / ckpt_dir_path #

    configs_dict["log_dir"].mkdir(parents=True, exist_ok=True) #
    configs_dict["output_dir"].mkdir(parents=True, exist_ok=True) #
    if "cache_root" in configs_dict: # Make cache_root optional
        Path(configs_dict["cache_root"]).mkdir(parents=True, exist_ok=True) #
        configs_dict["cache_root"] = Path(configs_dict["cache_root"]).as_posix() #


    configs_dict["log_dir"] = configs_dict["log_dir"].as_posix() #
    configs_dict["output_dir"] = configs_dict["output_dir"].as_posix() #
    
    # These pops might be problematic if other parts of code expect them later.
    # Consider if they should be popped or just keep the full path string.
    # configs_dict.pop("output_root") #
    # configs_dict.pop("log_root") #


    trainer_config["callbacks"] = [ 
        init_setup_callback(configs_dict), 
        init_lr_monitor_callback(), 
        ModelCheckpoint(dirpath=configs_dict["output_dir"], save_top_k=-1, every_n_epochs=configs_dict["trainer"].get("check_val_every_n_epoch",1)), 
    ]

    return trainer_config #


def experiment(variant_dict): # Renamed variant to variant_dict
    # Set seed: os.environ["RANK"] might not be set if not using DDP from torchrun/slurm
    rank = int(os.environ.get("RANK", 0)) 
    seed_everything(variant_dict["seed"] + rank) 
    
    trainer_config_dict = init_trainer_config_fn(variant_dict) # # Renamed trainer_config
    model_load_path = variant_dict.get("model_load_path", None) 

    trainer = Trainer(**trainer_config_dict) 
    variant_dict["gpus"] = trainer.num_devices 
    variant_dict["train_setup"]["precision"] = variant_dict["trainer"]["precision"] 

    if variant_dict.get("fwd_head") is not None: 
        variant_dict["train_setup"]["predict_forward_hand"] = variant_dict["fwd_head"].get( 
            "pred_hand_image", False 
        )

    # Model cloning logic from original main.py
    # Ensure model_path is correct for your setup or exists.
    model_base_path_str = variant_dict.get("model_path", ".vlms/model_default_location")
    model_base_path = Path(model_base_path_str)
    if not model_base_path.exists() and "model_url" in variant_dict: #
        repo_name = variant_dict["model_url"].split("/")[-1] #
        if repo_name.endswith(".git"): repo_name = repo_name[:-4] #
        
        clone_target_parent = model_base_path.parent
        clone_target_parent.mkdir(parents=True, exist_ok=True)
        
        print( #
            f"VLM backbone base path {model_base_path} does not exist, cloning {variant_dict['model']} from {variant_dict['model_url']} into {clone_target_parent}/{repo_name} ..." #
        )
        os.system(f"git clone {variant_dict['model_url']} {clone_target_parent / repo_name}") #
        # Update model_path and model_config to point to the cloned repo
        variant_dict['model_path'] = str(clone_target_parent / repo_name) #
        variant_dict['model_config'] = os.path.join(variant_dict['model_path'], "config.json") #
        # Update tokenizer and vlm paths in config if they were relative to the old model_path
        if "tokenizer" in variant_dict and "pretrained_model_name_or_path" in variant_dict["tokenizer"]:
            variant_dict["tokenizer"]["pretrained_model_name_or_path"] = variant_dict['model_path']
        if "vlm" in variant_dict and "pretrained_model_name_or_path" in variant_dict["vlm"]:
            variant_dict["vlm"]["pretrained_model_name_or_path"] = variant_dict['model_path']


    if variant_dict["model"] == "kosmos": #
        import transformers #
        package_dir = transformers.__path__[0] #
        # Ensure the source tools/modeling_kosmos2.py exists if this block is active
        # os.system( #
        #     f"cp tools/modeling_kosmos2.py {package_dir}/models/kosmos2/modeling_kosmos2.py" #
        # )
        # importlib.reload(transformers) #
        print("Kosmos specific code block skipped as 'tools/modeling_kosmos2.py' might not be available.")


    # Instantiate model
    # BaseTrainer.from_checkpoint will initialize the VLM inside it based on configs
    model = BaseTrainer.from_checkpoint( 
        model_load_path, variant_dict.get("model_load_source", "torch"), variant_dict #
    )

    # Critical: Get the image_processor from the *initialized* model
    # This was a key part missing in the original flow for setting up DataModule's image_fn
    image_preprocess_fn_constructor_arg = model.model.image_processor if hasattr(model.model, 'image_processor') else None
    if image_preprocess_fn_constructor_arg is None and variant_dict["model"] == "paligemma": # Paligemma specific
        from transformers import PaliGemmaProcessor
        try:
            # Attempt to load processor using the model_path from config, which should now be correct
            effective_model_path = variant_dict.get("vlm", {}).get("pretrained_model_name_or_path", variant_dict["model_path"])
            image_preprocess_fn_constructor_arg = PaliGemmaProcessor.from_pretrained(effective_model_path).image_processor
            print(f"Loaded PaliGemmaProcessor's image_processor from {effective_model_path}")
        except Exception as e:
            print(f"Could not load PaliGemmaProcessor for image_fn automatically: {e}. Ensure model_path is correct.")
            raise

    if image_preprocess_fn_constructor_arg is None:
        raise ValueError("Could not determine image_processor from the model. Please check model initialization and config.")

    # Instantiate DataModule
    datamodule = XArmDataModule( 
        train_dataset_config=variant_dict["train_dataset"], 
        val_dataset_config=variant_dict["val_dataset"], 
        batch_size=variant_dict["batch_size"], 
        num_workers=variant_dict["num_workers"], 
        data_root=variant_dict.get("data_root", ""), # example, pass if XArmDataModule uses it
        tokenizer=model.model.tokenizer, 
        tokenizer_config=variant_dict["tokenizer"], 
        fwd_pred_next_n=variant_dict["fwd_pred_next_n"], 
        window_size=variant_dict["window_size"], 
        image_size=variant_dict["image_size"], 
        # image_fn is now constructed inside XArmDataModule if processor is provided
        model_type_for_image_fn=variant_dict["model"], 
        image_processor_for_image_fn=image_preprocess_fn_constructor_arg, # Pass the obtained processor
        # Action processing
        norm_action=variant_dict.get("norm_action", False), 
        norm_min=variant_dict.get("norm_min", -1.0), 
        norm_max=variant_dict.get("norm_max", 1.0), 
        regular_action=variant_dict.get("regular_action", False), 
        x_mean=variant_dict.get("x_mean", 0.0), 
        x_std=variant_dict.get("x_std", 1.0), 
        use_mu_law=variant_dict.get("use_mu_law", False), 
        mu_val=variant_dict.get("mu_val", 255), 
        # Discrete actions
        discrete_action=( 
            False
            if variant_dict.get("act_head") is None 
            else variant_dict["act_head"].get("action_space", "continuous") == "discrete" 
        ),
        n_bin=( 
            256 
            if variant_dict.get("act_head") is None 
            else variant_dict["act_head"].get("n_bin", 256) 
        ),
        min_action_discrete=( # # Renamed from min_action to avoid conflict
            -1.0 
            if variant_dict.get("act_head") is None 
            else variant_dict["act_head"].get("min_action", -1.0) 
        ),
        max_action_discrete=( # # Renamed from max_action
            1.0 
            if variant_dict.get("act_head") is None 
            else variant_dict["act_head"].get("max_action", 1.0) 
        ),
        predict_stop_token=variant_dict.get("predict_stop_token", True), # Added
        tcp_rel=variant_dict.get("tcp_rel", False), 
        model_name=variant_dict.get("model", "paligemma"), 
        weights=variant_dict.get("train_weights", None), 
        seed=variant_dict["seed"] 
    )

    trainer_fit_kwargs = { # # Renamed _kwargs
        "model": model, 
        "datamodule": datamodule, 
        "ckpt_path": variant_dict.get("resume", None), 
    }
    if trainer_fit_kwargs["ckpt_path"] is not None: 
        print(f"Resuming from {variant_dict['resume']}...") 
    
    import torch

    # 👇 Setup data module manually to access datasets
    datamodule.setup(stage="fit")  # 手动触发数据准备阶段

    # 👇 获取一个batch并打印第一条数据
    train_loader = datamodule.train_dataloader()
    first_batch = next(iter(train_loader))

    # 保存到文本文件
    with open("debug_batch_output.txt", "w", encoding="utf-8") as f:
        f.write("🔍 Debug: First training sample keys and shapes:\n")
        if isinstance(first_batch, dict):
            for k, v in first_batch.items():
                if k == "text":
                    f.write(f"{k}: {v[0] if isinstance(v, list) else v}\n")
                elif isinstance(v, torch.Tensor):
                    f.write(f"{k}: shape={v.shape}\n")
                else:
                    f.write(f"{k}: type={type(v)}\n")
        else:
            f.write(str(first_batch) + "\n")




    trainer.fit(**trainer_fit_kwargs) 


def deep_update(d1, d2): 
    for k, v_item in d2.items(): # # Renamed v to v_item
        if isinstance(v_item, dict) and k in d1: 
            if not isinstance(d1[k], dict): # If d1[k] is not a dict, overwrite
                 d1[k] = v_item
            else:
                deep_update(d1[k], v_item) 
        else: 
            d1[k] = v_item 
    return d1 


def load_config_file(config_path_str): # Renamed from load_config, arg
    # Load the main config file
    with open(config_path_str, 'r') as f:
        _config_dict = json.load(f) # # Renamed _config
    
    # Handle parent configs recursively
    # The logic in original main.py for parent config was:
    # config = {}
    # if _config.get("parent", None): deep_update(config, load_config(_config["parent"]))
    # deep_update(config, _config)
    # This means parent is loaded first, then current config updates it.

    final_config = {} #
    if _config_dict.get("parent", None): #
        parent_config_path = Path(config_path_str).parent / _config_dict["parent"]
        # Ensure parent path is absolute or correctly relative to the current config file
        if not parent_config_path.is_absolute():
             # This might need adjustment based on how parent paths are specified
             parent_config_path = Path(os.path.dirname(os.path.abspath(config_path_str))) / _config_dict["parent"]
        
        print(f"Loading parent config from: {parent_config_path}")
        parent_config = load_config_file(parent_config_path.resolve())
        deep_update(final_config, parent_config) #
    
    deep_update(final_config, _config_dict) #
    return final_config #


def update_configs_fn(configs_dict, args_dict): # Renamed from update_configs, args
    configs_dict["raw_config_path"] = args_dict["config"] #
    # Ensure path roots are Path objects before division
    configs_dict["output_root"] = ( #
        Path(configs_dict["output_root"]) / configs_dict["model"] / configs_dict.get("task_name", "default_task") #
    )
    configs_dict["log_root"] = ( #
        Path(configs_dict["log_root"]) / configs_dict["model"] / configs_dict.get("task_name", "default_task") #
    )
    if "cache_root" in configs_dict and configs_dict.get("model"): # Make cache_root update conditional
        configs_dict["cache_root"] = Path(configs_dict["cache_root"]) / configs_dict["model"] #


    for k, v_arg in args_dict.items(): # # Renamed v to v_arg
        if k not in configs_dict: #
            # print(f"{k} not in config. The value is {v_arg}.") #
            configs_dict[k] = v_arg #
        
        if isinstance(v_arg, dict): #
            if k not in configs_dict or not isinstance(configs_dict[k], dict):
                configs_dict[k] = {} # Ensure target is a dict
            for sub_k, sub_v_arg in v_arg.items(): # # Renamed sub_v to sub_v_arg
                if sub_v_arg is not None: # # Check for None before updating
                    configs_dict[k][sub_k] = sub_v_arg #
        else: #
            if v_arg is not None: # # Check for None
                configs_dict[k] = v_arg #
    return configs_dict #

def parse_args_fn(): # Renamed from parse_args
    parser = argparse.ArgumentParser() #

    # Experiment Args
    parser.add_argument("config", type=str, help="Path to the main configuration file.") #
    parser.add_argument("--gpus", default=None, type=int, help="Number of GPUs (overrides config).") #
    parser.add_argument("--num_nodes", default=None, type=int, help="Number of nodes (overrides config).") #
    parser.add_argument("--seed", default=None, type=int, help="Global random seed.") #
    # log_dir and output_dir are usually derived, but can be overridden.
    parser.add_argument("--exp_name", default=None, type=str, help="Experiment name (overrides config task_name for dir naming).") #
    parser.add_argument("--model_load_path", default=None, type=str, help="Path to load pre-trained model weights.") #
    parser.add_argument("--resume", default=None, type=str, help="Path to checkpoint for resuming training.")


    # Training parameters (can override config)
    parser.add_argument("--learning_rate", default=None, type=float) #
    parser.add_argument("--batch_size", default=None, type=int) #
    parser.add_argument("--max_epochs", default=None, type=int) # For trainer group
    parser.add_argument("--max_steps", default=None, type=int) # For trainer group
    parser.add_argument("--weight_decay", default=None, type=float) #

    # Dataset related (can override config values for dataset paths)
    parser.add_argument("--train_data_dir", default=None, type=str, help="Override train_dataset.data_dir")
    parser.add_argument("--val_data_dir", default=None, type=str, help="Override val_dataset.data_dir")
    
    # Specific args mentioned in original parser
    # parser.add_argument("--data_dir", default=None, type=str) # Covered by train/val_data_dir
    # parser.add_argument("--annotation_file", default=None, type=str) # Not in paligemma json
    # parser.add_argument("--data_subfolder", default=None, type=str) # Not in paligemma json
    # parser.add_argument("--task_num", default=None, type=int) # Not in paligemma json
    parser.add_argument("--seq_len", default=None, type=int) # Note: paligemma json has float, but seems like int

    # Loss ratios
    parser.add_argument("--arm_gripper_loss_ratio", default=None, type=float) #
    parser.add_argument("--fwd_loss_ratio", default=None, type=float) #
    parser.add_argument("--fwd_pred_next_n", default=None, type=int) #

    # Boolean flags
    parser.add_argument("--use_multi_modal_emb", default=None, action=argparse.BooleanOptionalAction) #
    parser.add_argument("--no_video_pretrained_model", default=None, action=argparse.BooleanOptionalAction) #
    parser.add_argument("--finetune", default=None, action=argparse.BooleanOptionalAction) #


    # Trainer group args from original main.py
    trainer_parser = parser.add_argument_group("trainer_args") #
    trainer_parser.add_argument("--trainer.strategy", dest='trainer_strategy', default=None, type=str) #
    trainer_parser.add_argument("--trainer.precision", dest='trainer_precision', default=None, type=str) #
    trainer_parser.add_argument("--trainer.gradient_clip_val", dest='trainer_gradient_clip_val', default=None, type=float) #
    # max_epochs already added above, map it to trainer.max_epochs in update_configs_fn
    # max_steps also added above

    # LLM group args (example, if you need to override nested LLM configs)
    # llm_parser = parser.add_argument_group("llm_args") #
    # llm_parser.add_argument("--llm.type", dest='llm_type', default=None, type=str) #
    # llm_parser.add_argument("--llm.n_embd", dest='llm_n_embd', default=None, type=int) #
    
    parsed_args = vars(parser.parse_args()) #
    
    # Restructure for deep_update compatibility (especially for nested trainer args)
    cli_args = {} #
    cli_args["trainer"] = {} #
    # cli_args["llm"] = {} # # Example for other groups

    for k, v_parsed in parsed_args.items(): # # Renamed v to v_parsed
        if v_parsed is None: # Skip None values from CLI not set
            continue
        if k == 'trainer_strategy': cli_args["trainer"]["strategy"] = v_parsed #
        elif k == 'trainer_precision': cli_args["trainer"]["precision"] = v_parsed #
        elif k == 'trainer_gradient_clip_val': cli_args["trainer"]["gradient_clip_val"] = v_parsed #
        elif k == 'max_epochs': cli_args["trainer"]["max_epochs"] = v_parsed #
        elif k == 'max_steps': cli_args["trainer"]["max_steps"] = v_parsed
        # elif k == 'llm_type': cli_args["llm"]["type"] = v_parsed #
        # elif k == 'llm_n_embd': cli_args["llm"]["n_embd"] = v_parsed #
        elif k not in ['config']: # Avoid overriding the main config path itself here
             cli_args[k] = v_parsed #
    
    cli_args["config"] = parsed_args["config"] # Ensure config path is always present
    return cli_args #


if __name__ == "__main__":
    args_cli = parse_args_fn() 

    # Load config files (handles parent configs)
    configs_loaded = load_config_file(args_cli["config"]) #
    
    # Update loaded configs with CLI arguments
    # This creates the final 'variant' dictionary
    final_configs = update_configs_fn(configs_loaded, args_cli) #

    # Special handling for train/val data_dir from CLI
    if args_cli.get("train_data_dir"):
        if "train_dataset" not in final_configs: final_configs["train_dataset"] = {}
        final_configs["train_dataset"]["data_dir"] = args_cli["train_data_dir"]
    if args_cli.get("val_data_dir"):
        if "val_dataset" not in final_configs: final_configs["val_dataset"] = {}
        final_configs["val_dataset"]["data_dir"] = args_cli["val_data_dir"]
    
    # Ensure necessary roots are present before init_trainer_config_fn uses them
    for root_key in ["output_root", "log_root", "cache_root"]:
        if root_key not in final_configs:
            final_configs[root_key] = f"runs_default/{root_key.split('_')[0]}" # Default if not in JSON
            print(f"Warning: '{root_key}' not found in config, defaulting to {final_configs[root_key]}")


    # Initialize DDP
    # Check if WORLD_SIZE is set (indicates DDP environment)
    if "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl") #

    experiment(variant_dict=final_configs) #