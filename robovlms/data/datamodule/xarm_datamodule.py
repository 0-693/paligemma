import copy
import os
import functools # Required for partial

import lightning.pytorch as pl
# from lightning.pytorch.utilities.combined_loader import CombinedLoader # If using multiple datasets
from torch.utils.data import DataLoader # Added
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import SequentialSampler, RandomSampler

import robovlms.data # Assuming this is where XArmDataset will be discoverable
from robovlms.data.weighted_combined_loader import WeightedCombinedLoader #
import robovlms.data.samplers as gr_samplers #
from robovlms.utils.dist_train import get_rank, is_dist #
from robovlms.utils.common import collate_with_none #
# Specific to XArm dataset and its dependencies
from robovlms.data.data_utils import preprocess_image # Assuming this is needed for image_fn

# If XArmDataset is in xarm_dataset.py at the same level as gr_datamodule
# from .xarm_dataset import XArmDataset
# Or adjust path accordingly. For this example, assume it's accessible via robovlms.data
# To make it discoverable, xarm_dataset.py might need to be in a package recognized by robovlms.data
# or explicitly imported. For now, let's use a placeholder getattr.

class XArmDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_dataset_config: dict, # Changed from train_dataset
        val_dataset_config: dict,   # Changed from val_dataset
        batch_size,
        num_workers,
        data_root="", # May not be needed if paths in dataset_config are absolute or relative to a known point
        # Added new arguments required by GRDataModule's _init_dataset call for XArmDataset
        tokenizer=None,
        tokenizer_config=None, # For get_text_function
        fwd_pred_next_n=10,
        window_size=8,
        image_size=224, # For preprocess_image
        image_fn=None, # Will be constructed if None
        model_type_for_image_fn="paligemma", # From main.py, for preprocess_image
        image_processor_for_image_fn=None, # From main.py, for preprocess_image (model.model.image_processor)
        # Action processing related
        norm_action=False, norm_min=-1.0, norm_max=1.0,
        regular_action=False, x_mean=0.0, x_std=1.0,
        use_mu_law=False, mu_val=255,
        # Discrete action related
        discrete_action=False, # from main.py's kwargs
        # action_tokenizer_instance=None, # if pre-initialized
        n_bin=256, min_action_discrete=-1.0, max_action_discrete=1.0,
        predict_stop_token=True,
        tcp_rel=False, # from main.py
        model_name="paligemma", # from main.py
        weights=None, # for WeightedCombinedLoader
        seed=123, # for DistributedSampler
        **kwargs, # Catch-all for other params like discrete_action_history, act_step etc.
    ):
        super().__init__()
        self.train_dataset_config = train_dataset_config
        self.val_dataset_config = val_dataset_config
        self._train_datasets = []
        self._val_datasets = []
        self._train_loader = None
        self._val_loader = None
        self.data_root = data_root #
        self.batch_size = batch_size #
        self.num_workers = num_workers #
        
        # Store additional necessary parameters
        self.tokenizer = tokenizer
        self.tokenizer_config = tokenizer_config # Not directly used in GRDataModule, but XArmDataset might need it
        self.fwd_pred_next_n = fwd_pred_next_n
        self.window_size = window_size
        self.image_size = image_size # Used if image_fn needs to be constructed
        self.model_type_for_image_fn = model_type_for_image_fn
        self.image_processor_for_image_fn = image_processor_for_image_fn

        if image_fn is None and image_processor_for_image_fn is not None:
             self.image_fn = functools.partial(
                preprocess_image, #
                image_processor=self.image_processor_for_image_fn, #
                model_type=self.model_type_for_image_fn #
            )
        elif image_fn is not None:
            self.image_fn = image_fn
        else:
            raise ValueError("Either image_fn or image_processor_for_image_fn must be provided.")

        self.norm_action = norm_action
        self.norm_min = norm_min
        self.norm_max = norm_max
        self.regular_action = regular_action
        self.x_mean = x_mean
        self.x_std = x_std
        self.use_mu_law = use_mu_law
        self.mu_val = mu_val
        self.discrete_action = discrete_action
        # self.action_tokenizer_instance = action_tokenizer_instance # If you create one in main
        self.n_bin = n_bin
        self.min_action_discrete = min_action_discrete
        self.max_action_discrete = max_action_discrete
        self.predict_stop_token = predict_stop_token
        self.tcp_rel = tcp_rel
        self.model_name = model_name # For XArmDataset's model_name param
        self.seed = seed #

        self.kwargs = kwargs # General kwargs
        if weights is not None : self.kwargs['weights'] = weights


    def _check_data_path(self, data_cfg): #
        # print(self.data_root) #
        if data_cfg["type"] == "ConcatDataset": #
            data_cfg["datasets"] = [ #
                self._check_data_path(d) for d in data_cfg["datasets"] #
            ]
        elif "data_dir" in data_cfg and not os.path.isabs(data_cfg["data_dir"]): #
            # If data_root is provided and path is relative, join them.
            # Otherwise, assume data_cfg["data_dir"] is already correct (e.g. absolute, or relative to cwd)
            if self.data_root:
                 data_cfg["data_dir"] = os.path.join(self.data_root, data_cfg["data_dir"]) #
        return data_cfg #

    def _init_dataset(self, dataset_config, batch_size_arg, num_workers_arg, is_training=True): # Renamed args
        dataset_config = self._check_data_path(dataset_config) #

        dataset_config = copy.deepcopy(dataset_config) #
        dataset_type_str = dataset_config.pop("type") #
        
        dataset_config["is_training"] = is_training #
        sampler_config = dataset_config.pop("sampler", None) #

        # Add parameters needed by XArmDataset constructor that were passed to XArmDataModule
        dataset_config["image_fn"] = self.image_fn
        dataset_config["tokenizer"] = self.tokenizer
        dataset_config["fwd_pred_next_n"] = self.fwd_pred_next_n
        dataset_config["window_size"] = self.window_size
        dataset_config["model_name"] = self.model_name # Pass to XArmDataset
        # action processing
        dataset_config["norm_action"] = self.norm_action
        dataset_config["norm_min"] = self.norm_min
        dataset_config["norm_max"] = self.norm_max
        dataset_config["regular_action"] = self.regular_action
        dataset_config["x_mean"] = self.x_mean
        dataset_config["x_std"] = self.x_std
        dataset_config["use_mu_law"] = self.use_mu_law
        dataset_config["mu_val"] = self.mu_val
        # discrete actions
        dataset_config["discrete_action"] = self.discrete_action
        # dataset_config["action_tokenizer_instance"] = self.action_tokenizer_instance
        dataset_config["n_bin"] = self.n_bin
        dataset_config["min_action_discrete"] = self.min_action_discrete
        dataset_config["max_action_discrete"] = self.max_action_discrete
        dataset_config["predict_stop_token"] = self.predict_stop_token
        dataset_config["tcp_rel"] = self.tcp_rel


        # Update with any other kwargs passed that XArmDataset might expect
        dataset_config.update(self.kwargs) #
        
        # Dynamically get the dataset class (e.g., XArmDataset)
        # Assumes xarm_dataset.py is in robovlms.data or similar accessible path
        try:
            dataset_cls = getattr(robovlms.data, dataset_type_str)
        except AttributeError:
            # Fallback if not in robovlms.data, try to import more directly
            # This part is tricky and depends on your project structure.
            # For a robust solution, ensure XArmDataset is part of the robovlms.data package
            # or use a direct import: from path.to.xarm_dataset import XArmDataset
            if dataset_type_str == "XArmDataset":
                from robovlms.data.xarm_dataset import XArmDataset # Direct import
                dataset_cls = XArmDataset
            else:
                raise ValueError(f"Dataset type {dataset_type_str} not found.")

        dataset = dataset_cls(**dataset_config) #

        sampler_cls = None #
        if sampler_config is not None: #
            sampler_type = sampler_config.pop("type") #
            sampler_cls = getattr(gr_samplers, sampler_type, None) #

        if sampler_cls is not None: #
            sampler_config["is_training"] = is_training #
            sampler_config["dataset"] = dataset #
            sampler = sampler_cls(**sampler_config) #
        elif is_dist(): #
            sampler = DistributedSampler( #
                dataset,
                shuffle=is_training, # Shuffle only for training
                drop_last=is_training, # Drop last only for training
                seed=self.seed, #
            )
        elif is_training: #
            sampler = RandomSampler(dataset) #
        else: #
            sampler = SequentialSampler(dataset) #
        
        # Use the collate_fn from the dataset object itself
        collate_function = dataset.collater if hasattr(dataset, "collater") else collate_with_none #

        data_loader = DataLoader( #
            dataset,
            batch_size=batch_size_arg, # Use argument
            num_workers=num_workers_arg, # Use argument
            sampler=sampler,
            drop_last=is_training, # Drop last for training
            collate_fn=collate_function,
            pin_memory=True, #
            # prefetch_factor=3 # # In GRDataModule, but can cause issues with some setups
        )
        return dataset, data_loader #

    # _init_iterable_dataset is removed as XArmDataset is map-style. If you need iterable, adapt from GRDataModule.

    def _init_datasets(self, current_dataset_config, is_training, batch_size_param, num_workers_param): # Renamed args
        # This method is simplified from GRDataModule as we are not handling lists of datasets for XArm yet.
        # If you need to combine XArmDataset with others, the original logic from GRDataModule for lists should be used.
        if isinstance(current_dataset_config, dict): #
            if get_rank() == 0: #
                print("=" * 40) #
                print("Initializing dataloader from config:") #
                for k, v_ in current_dataset_config.items(): print(f"{k}: {v_}") # # Renamed v to v_
                print(f"is_training: {is_training}") #
                print(f"batch_size: {batch_size_param}") #
                print(f"num_workers: {num_workers_param}") #
            
            # Assuming XArmDataset is not an "iterable" type in the sense of GRDataModule's distinction
            return self._init_dataset( #
                current_dataset_config, #
                is_training=is_training, #
                batch_size_arg=batch_size_param, # # Pass renamed arg
                num_workers_arg=num_workers_param, # # Pass renamed arg
            )
        elif isinstance(current_dataset_config, list): # Handle list of dataset configs (like in GRDataModule)
            all_sets_and_loaders = [] #
            # Ensure batch_size_param and num_workers_param are lists of same length as current_dataset_config
            if not (isinstance(batch_size_param, (tuple, list)) and len(batch_size_param) == len(current_dataset_config)):
                 raise ValueError("If dataset_config is a list, batch_size must be a list of same length.")
            if not (isinstance(num_workers_param, (tuple, list)) and len(num_workers_param) == len(current_dataset_config)):
                 raise ValueError("If dataset_config is a list, num_workers must be a list of same length.")

            for i, config_item in enumerate(current_dataset_config): #
                all_sets_and_loaders.append( #
                    self._init_datasets( # Recursive call for each config in the list
                        config_item, #
                        is_training=is_training, #
                        batch_size_param=batch_size_param[i], #
                        num_workers_param=num_workers_param[i], #
                    )
                )
            datasets, dataloaders = zip(*all_sets_and_loaders) #
            if is_training: #
                # Use WeightedCombinedLoader if multiple training datasets
                combined_dataloader = WeightedCombinedLoader( #
                    dataloaders, #
                    "max_size_cycle", #
                    weights=self.kwargs.get("weights", None), #
                )
                return datasets, combined_dataloader #
            else: # For validation, return list of dataloaders
                return datasets, list(dataloaders) #
        else:
            raise TypeError(f"dataset_config must be a dict or list, got {type(current_dataset_config)}")


    def _init_dataset_params(self, is_training, param_name="batch_size"): #
        # This logic is mostly from GRDataModule to handle int or list for batch_size/num_workers
        param_val = getattr(self, param_name) #
        
        config_to_check = self.train_dataset_config if is_training else self.val_dataset_config

        if isinstance(config_to_check, (tuple, list)): # If the config itself is a list of datasets
            if isinstance(param_val, int): # if bs=4 but datasets=[d1, d2], make it [4,4]
                param_val = [param_val] * len(config_to_check) #
            elif isinstance(param_val, (tuple, list)): # if bs=[4,4] and datasets=[d1,d2]
                if len(param_val) != len(config_to_check):
                    raise ValueError(f"{param_name} list length must match dataset_config list length.")
            else:
                raise TypeError(f"{param_name} must be int or list if dataset_config is a list.")
        else: # Config is a single dict
            if not isinstance(param_val, int): # if bs=[4] but dataset=d1, make it 4 (or error)
                if isinstance(param_val, (list, tuple)) and len(param_val) == 1:
                    param_val = param_val[0]
                else:
                    raise ValueError(f"{param_name} must be int if dataset_config is a single dict.")
        return param_val #


    def setup(self, stage: str = None):
        if stage == "fit" or stage is None:
            train_batch_size = self._init_dataset_params(True, "batch_size") #
            train_num_workers = self._init_dataset_params(True, "num_workers") #
            self._train_datasets, self._train_loader = self._init_datasets( #
                self.train_dataset_config, True, train_batch_size, train_num_workers #
            )

            val_batch_size = self._init_dataset_params(False, "batch_size") #
            val_num_workers = self._init_dataset_params(False, "num_workers") #
            self._val_datasets, self._val_loader = self._init_datasets( #
                self.val_dataset_config, False, val_batch_size, val_num_workers #
            )
            if get_rank() == 0 and self._val_loader: #
                 print(f"val_loader type: {type(self._val_loader)}")
                 if isinstance(self._val_loader, list):
                     print(f"val_loader size: {len(self._val_loader)}") #


    def train_dataloader(self): #
        if self._train_loader is None: self.setup("fit") # Ensure setup has run
        return self._train_loader #

    def val_dataloader(self): #
        if self._val_loader is None: self.setup("fit") # Ensure setup has run
        return self._val_loader #

    # Optional: Add test_dataloader if you have test_dataset_config
    # def test_dataloader(self):
    #     # Similar setup as val_dataloader using self.test_dataset_config
    #     pass