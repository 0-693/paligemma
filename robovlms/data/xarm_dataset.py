import io
import logging
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple, Callable, Any

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

from robovlms.data.data_utils import (
    get_text_function,
    normalize_action,
    regularize_action,
    mu_law_companding,
    get_prompt_builder,
)

logger = logging.getLogger(__name__)
Image.MAX_IMAGE_PIXELS = None # To avoid DecompressionBombError for large images in parquet

# Constants from calvin_dataset.py, adjust if needed
MAX_NUM_TOKENS = 256 # Example, adjust based on tokenizer and model
IGNORE_INDEX = -100 # For language modeling labels

class XArmDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        image_fn: Callable,
        tokenizer: Callable,
        window_size: int = 8,
        fwd_pred_next_n: int = 10,
        is_training: bool = True,
        model_name: str = "paligemma",
        # Action processing params (mirroring calvin_dataset)
        norm_action: bool = False,
        norm_min: float = -1.0,
        norm_max: float = 1.0,
        regular_action: bool = False,
        x_mean: float = 0.0,
        x_std: float = 1.0,
        use_mu_law: bool = False,
        mu_val: int = 255,
        # Discrete action params (if used)
        discrete_action: bool = False,
        action_tokenizer_instance=None, # Pass pre-initialized ActionTokenizer
        n_bin: int = 256,
        min_action_discrete: float = -1.0,
        max_action_discrete: float = 1.0,
        predict_stop_token: bool = True,
        # Misc
        task_type: str = "xarm_action", # For data_source field
        rgb_pad: int = -1, # Not implemented in this version for simplicity, but can be added
        gripper_pad: int = -1, # Not implemented for simplicity
        tcp_rel: bool = False, # Not implemented, assumes world frame actions
        **kwargs: Any,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.image_fn = image_fn # Expects a function that takes a list of PIL Images and returns a batched tensor
        self.tokenizer = tokenizer
        self.text_fn = get_text_function(self.tokenizer, model_name)

        self.window_size = window_size
        self.fwd_pred_next_n = fwd_pred_next_n
        self.is_training = is_training
        self.model_name = model_name
        self.task_type = task_type

        # Action processing
        self.norm_action = norm_action
        self.norm_min = norm_min
        self.norm_max = norm_max
        self.regular_action = regular_action
        self.x_mean = x_mean
        self.x_std = x_std
        self.use_mu_law = use_mu_law
        self.mu_val = mu_val

        # Discrete actions
        self.discrete_action = discrete_action
        self.predict_stop_token = predict_stop_token
        self.action_tokenizer = None
        # if self.discrete_action:
        #     if action_tokenizer_instance:
        #         self.action_tokenizer = action_tokenizer_instance
        #     else:
        #         logger.info(f"Initializing ActionTokenizer with bins: {n_bin}, min: {min_action_discrete}, max: {max_action_discrete}")
        #         # self.action_tokenizer = ActionTokenizer(
        #         #     tokenizer=self.tokenizer, # Needs compatible tokenizer
        #         #     bins=n_bin,
        #         #     min_action=min_action_discrete,
        #         #     max_action=max_action_discrete,
        #         # )
        #         # Placeholder if ActionTokenizer is complex or has other dependencies
        #         logger.warning("ActionTokenizer not fully configured in this template. Ensure it's correctly initialized if discrete_action is True.")


        if not self.data_dir.is_dir():
            raise ValueError(f"Data directory {self.data_dir} not found.")

        self.episode_lookup = self._build_episode_lookup()
        if not self.episode_lookup:
            raise RuntimeError(f"No valid sequences found in {self.data_dir}. Check data and window/fwd_pred_next_n settings.")

        logger.info(f"Initialized XArmDataset with {len(self.episode_lookup)} sequences from {self.data_dir}")

    def _build_episode_lookup(self) -> List[Tuple[Path, int, int]]:
        lookup = []
        required_seq_len = self.window_size + self.fwd_pred_next_n

        parquet_files = sorted(list(self.data_dir.glob("*.parquet")))
        if not parquet_files:
            logger.warning(f"No .parquet files found in {self.data_dir}")
            return lookup

        for file_path in parquet_files:
            try:
                # Load only necessary columns to check length and 'is_first', 'is_last', 'is_terminal'
                df_cols = ['is_first', 'is_last', 'is_terminal']
                df = pd.read_parquet(file_path, columns=df_cols)

                # Segment episodes based on 'is_first' and 'is_last'
                episode_starts = df.index[df['is_first']].tolist()
                episode_ends = df.index[df['is_last']].tolist()

                # Ensure there's a matching start and end for each episode
                if len(episode_starts) != len(episode_ends) or len(episode_starts) == 0:
                    logger.warning(f"Skipping {file_path}: Mismatched or no 'is_first'/'is_last' markers.")
                    continue

                for ep_start_idx, ep_end_idx in zip(episode_starts, episode_ends):
                    # +1 to ep_end_idx because iloc slicing is exclusive of the end
                    current_episode_len = ep_end_idx - ep_start_idx + 1
                    
                    if current_episode_len >= required_seq_len:
                        # Iterate through possible start indices within the episode
                        # Max start index for states/images is current_episode_len - required_seq_len
                        for i in range(current_episode_len - required_seq_len + 1):
                            # Check if the sampled window contains a terminal state.
                            # For training, we usually want to avoid sequences that end prematurely
                            # due to a 'is_terminal' flag within the window (unless it's the very end).
                            # Here, we allow it for simplicity, but a more robust dataset might skip.
                            
                            # Store (file_path, actual_start_index_in_df, actual_end_index_in_df_slice)
                            # The start_frame_idx passed to __getitem__ is relative to the file, not the episode segment.
                            lookup.append((file_path, ep_start_idx + i, current_episode_len))
                    else:
                        logger.warning(f"Skipping episode in {file_path} from index {ep_start_idx}: length {current_episode_len} is less than required {required_seq_len}.")

            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
        return lookup

    def __len__(self) -> int:
        return len(self.episode_lookup)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        file_path, start_frame_idx, episode_total_len = self.episode_lookup[idx]
        
        # Length of data to fetch: window_size for current + fwd_pred_next_n for future
        num_frames_to_load = self.window_size + self.fwd_pred_next_n
        
        try:
            episode_df = pd.read_parquet(file_path)
        except Exception as e:
            logger.error(f"Failed to load parquet file {file_path} for index {idx}: {e}")
            raise RuntimeError(f"Could not load data for index {idx}")

        # Slice the DataFrame for the required window
        # df_slice contains `num_frames_to_load` rows
        df_slice = episode_df.iloc[start_frame_idx : start_frame_idx + num_frames_to_load]

        # Padding strategy if the slice is shorter than expected (e.g., at the end of an episode)
        # This should ideally be avoided by _build_episode_lookup.
        # But if it happens, we can pad. For this solution, we assume _build_episode_lookup ensures full slices.
        if len(df_slice) < num_frames_to_load:
            # This indicates an issue with _build_episode_lookup or reaching end of episode.
            # Pad with the last frame's data. This can affect action values if not careful.
            logger.warning(f"Padding required for slice from {file_path} starting at {start_frame_idx}. Slice length: {len(df_slice)}, Expected: {num_frames_to_load}")
            last_row = df_slice.iloc[-1]
            padding_rows = pd.DataFrame([last_row] * (num_frames_to_load - len(df_slice)), columns=df_slice.columns)
            df_slice = pd.concat([df_slice, padding_rows], ignore_index=True)


        raw_images_pil = []
        for img_bytes in df_slice['image_1']:
            raw_images_pil.append(Image.open(io.BytesIO(img_bytes)).convert("RGB"))

        raw_hand_images_pil = []
        if 'image_2' in df_slice and self.image_fn is not None:
            for img_bytes in df_slice['image_2']:
                if pd.isna(img_bytes) or not img_bytes:
                     raw_hand_images_pil.append(Image.new("RGB", raw_images_pil[0].size, (128,128,128))) # Placeholder
                else:
                    raw_hand_images_pil.append(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
        else:
            raw_hand_images_pil = [Image.new("RGB", raw_images_pil[0].size, (128,128,128)) for _ in raw_images_pil]


        # Actions: (num_frames_to_load - 1, 7) or (num_frames_to_load, 7) if action is for current state
        # The current design uses action at row i for transition from state i to i+1.
        # So for N states, there are N-1 actions.
        # df_slice has `num_frames_to_load` states. So we need `num_frames_to_load - 1` actions.
        # However, the required `num_actions_to_load` in collater is `window_size + fwd_pred_next_n - 1`.
        # This means we need actions up to the (num_frames_to_load - 1)th index.
        # The actions_np will have `num_frames_to_load - 1` elements.
        actions_np = np.stack(df_slice['action'].values).astype(np.float32)[:-1]
        
        # If the dataset can contain terminal states, and we want to stop actions there,
        # we would need to mask actions after a terminal state.
        # For now, `is_terminal` is used primarily for episode lookup.
        # The `chunk_mask` will handle action validity in collater.

        # States: (num_frames_to_load, 7)
        states_np = np.stack(df_slice['state'].values).astype(np.float32)
        
        # Language prompt - assuming it's the same for the whole episode/sequence
        lang_prompt = df_slice['prompt'].iloc[0]
        if not lang_prompt or pd.isna(lang_prompt):
            logger.warning(f"Empty or NaN lang_prompt for {file_path} at index {start_frame_idx}. Using default prompt.")
            lang_prompt = "Pick the oreo into the basket." # Default prompt

        # Create a mask for valid frames in the loaded sequence.
        # 'is_terminal' in df_slice needs to be considered for `fwd_mask` and `chunk_mask`.
        # If a frame is terminal, subsequent actions/frames might be invalid.
        # For simplicity, we create a mask that is True for all frames in the slice unless explicitly marked.
        # This needs to align with how `fwd_mask` and `chunk_mask` are computed.
        # Let's use `is_terminal` to determine the effective length of the sequence.
        is_terminal_flags = df_slice['is_terminal'].values
        # The sequence is valid up to and including the first terminal state.
        # All frames/actions after a terminal state are often considered invalid for prediction.
        first_terminal_idx = -1
        if np.any(is_terminal_flags):
            first_terminal_idx = np.where(is_terminal_flags)[0][0]
        
        episode_frame_mask_np = np.ones(num_frames_to_load, dtype=bool)
        if first_terminal_idx != -1:
            # Frames after terminal are invalid.
            episode_frame_mask_np[first_terminal_idx + 1:] = False

        return {
            "raw_images_pil": raw_images_pil, # List of PIL Images [num_frames_to_load]
            "raw_hand_images_pil": raw_hand_images_pil, # List of PIL Images [num_frames_to_load]
            "actions_np": actions_np,       # Numpy array [num_frames_to_load - 1, action_dim]
            "states_np": states_np,         # Numpy array [num_frames_to_load, state_dim]
            "lang": lang_prompt,            # String
            "frame_mask_np": episode_frame_mask_np # Boolean array [num_frames_to_load]
        }

    def collater(self, samples: List[Dict]) -> Dict[str, Any]:
        # Batching the outputs of __getitem__
        batch_size = len(samples)

        # Process images
        all_rgbs_list = [s['raw_images_pil'] for s in samples]
        all_hand_rgbs_list = [s['raw_hand_images_pil'] for s in samples]
        
        flat_rgbs = [img for ep_imgs in all_rgbs_list for img in ep_imgs]
        processed_flat_rgbs = self.image_fn(flat_rgbs)
        all_rgbs_tensor = processed_flat_rgbs.view(batch_size, self.window_size + self.fwd_pred_next_n, *processed_flat_rgbs.shape[1:])

        flat_hand_rgbs = [img for ep_imgs in all_hand_rgbs_list for img in ep_imgs]
        processed_flat_hand_rgbs = self.image_fn(flat_hand_rgbs)
        all_hand_rgbs_tensor = processed_flat_hand_rgbs.view(batch_size, self.window_size + self.fwd_pred_next_n, *processed_flat_hand_rgbs.shape[1:])

        # Process actions
        actions_list_np = [s['actions_np'] for s in samples]
        processed_actions_list = []
        for action_seq_np in actions_list_np:
            action_seq_torch = torch.from_numpy(action_seq_np).float()
            if self.norm_action:
                action_seq_torch = normalize_action(action_seq_torch, self.norm_min, self.norm_max, maintain_last=True)
            if self.regular_action:
                action_seq_torch = regularize_action(action_seq_torch, self.x_mean, self.x_std)
            if self.use_mu_law:
                action_seq_torch = mu_law_companding(action_seq_torch, self.mu_val)
            processed_actions_list.append(action_seq_torch)
        
        # Actions are expected to be (N-1) for N states.
        # The expected length of actions_np from __getitem__ is (window_size + fwd_pred_next_n - 1).
        all_actions_tensor = torch.stack(processed_actions_list) # [BS, (WS+FWD_N)-1, ADIM]

        # Process states (currently not used as 'rel_state' in collater but 'states_np' from getitem)
        states_list_np = [s['states_np'] for s in samples]
        all_states_tensor = torch.stack([torch.from_numpy(st).float() for st in states_list_np]) # [BS, WS+FWD_N, SDIM]
        
        # Process language
        lang_prompts = [s['lang'] for s in samples]
        text_tensors, attention_mask = self.text_fn(lang_prompts)

        # Frame masks
        frame_masks_list_np = [s['frame_mask_np'] for s in samples]
        all_frame_masks_tensor = torch.stack([torch.from_numpy(m).bool() for m in frame_masks_list_np]) # [BS, WS+FWD_N]

        # --- Create model inputs and labels based on window_size and fwd_pred_next_n ---

        # 1. RGB inputs for the current window
        rgb_input = all_rgbs_tensor[:, :self.window_size, ...] # [BS, WS, C, H, W]
        hand_rgb_input = all_hand_rgbs_tensor[:, :self.window_size, ...] # [BS, WS, C, H, W]

        # 2. Action input for the current window
        # The action at index `t` connects state `t` to state `t+1`.
        # So for `window_size` states (0 to WS-1), we need `window_size` actions (0 to WS-1).
        # all_actions_tensor has shape [BS, (WS+FWD_N)-1, ADIM].
        # We need the first `window_size` actions from this tensor.
        action_input = all_actions_tensor[:, :self.window_size, :] # [BS, WS, ADIM]


        # 3. Forward prediction labels (chunks)
        # fwd_rgb_chunk_label is for states/images. Unfold from all_rgbs_tensor.
        # Target chunk for input at time `t` contains frames `t+1` to `t+FWD_N`.
        # This corresponds to slices `[t+1 : t+1+FWD_N]`.
        # The unfold operation on `all_rgbs_tensor` of shape `[BS, WS+FWD_N, C, H, W]`
        # with `size=self.fwd_pred_next_n` and `step=1` will produce `[BS, NumChunks, C, H, W, FWD_N]`.
        # `NumChunks = (WS+FWD_N) - FWD_N + 1 = WS + 1`.
        # We need chunks starting from index 1 (representing frames from t=1 to t=WS+FWD_N-1).
        # So we take slices `[:, 1 : window_size + 1, ...]`
        fwd_rgb_chunk_label_unfold = all_rgbs_tensor.unfold(dimension=1, size=self.fwd_pred_next_n, step=1)
        fwd_rgb_chunk_label = fwd_rgb_chunk_label_unfold.permute(0, 1, 3, 2, 4, 5) # (BS, NumChunks, FWD_N, C, H, W)
        fwd_rgb_chunk_label = fwd_rgb_chunk_label[:, 1 : self.window_size + 1, ...] # [BS, WS, FWD_N, C, H, W]

        fwd_hand_rgb_chunk_label_unfold = all_hand_rgbs_tensor.unfold(dimension=1, size=self.fwd_pred_next_n, step=1)
        fwd_hand_rgb_chunk_label = fwd_hand_rgb_chunk_label_unfold.permute(0, 1, 3, 2, 4, 5)
        fwd_hand_rgb_chunk_label = fwd_hand_rgb_chunk_label[:, 1 : self.window_size + 1, ...] # [BS, WS, FWD_N, C, H, W]

        # 4. Action prediction labels (chunks)
        # all_actions_tensor is [BS, (WS+FWD_N)-1, ADIM]
        # Actions for time `t` to `t+FWD_N-1`.
        # Unfold on `all_actions_tensor` will give `[BS, NumChunks_Act, ADIM, FWD_N]` after permute,
        # where `NumChunks_Act = ((WS+FWD_N)-1) - FWD_N + 1 = WS`.
        # We need all these chunks.
        action_chunk_label_unfold = all_actions_tensor.unfold(dimension=1, size=self.fwd_pred_next_n, step=1)
        action_chunk_label = action_chunk_label_unfold.permute(0, 1, 3, 2) # (BS, WS, FWD_N, ADIM)

        # 5. Masks for chunks
        # fwd_mask: for fwd_rgb_chunk_label.
        # This mask corresponds to the validity of future states (images).
        # Derived from `all_frame_masks_tensor` which has shape `[BS, WS+FWD_N]`.
        # An image chunk for input `t` covers frames `t+1` to `t+FWD_N`.
        # So, the mask for this chunk should be `all_frame_masks_tensor[:, t+1 : t+1+FWD_N]`.
        # This is exactly what the unfold of `all_frame_masks_tensor` gives, sliced.
        fwd_mask_unfold = all_frame_masks_tensor.unfold(dimension=1, size=self.fwd_pred_next_n, step=1) # [BS, WS+1, FWD_N]
        fwd_mask = fwd_mask_unfold[:, 1 : self.window_size + 1, ...] # [BS, WS, FWD_N]
        # The mask for a chunk (sequence of frames) should be True only if all frames in the chunk are valid.
        # If `fwd_mask` is used element-wise for a loss (e.g., L2), it's fine.
        # If it's for a sequence, `.all(dim=-1)` might be needed in the loss function.

        # chunk_mask: for action_chunk_label.
        # This mask corresponds to the validity of future actions.
        # An action at index `j` is for state `j` to `j+1`.
        # If the state `j` is valid, and state `j+1` is valid, then action `j` is valid.
        # For simplicity, let's assume action `j`'s validity depends on `all_frame_masks_tensor[j]`.
        # The actions tensor has length `(WS+FWD_N)-1`. So we need masks for states `0` to `(WS+FWD_N)-2`.
        action_frame_masks = all_frame_masks_tensor[:, :all_actions_tensor.shape[1]] # [BS, (WS+FWD_N)-1]
        chunk_mask_unfold = action_frame_masks.unfold(dimension=1, size=self.fwd_pred_next_n, step=1) # [BS, WS, FWD_N]
        chunk_mask = chunk_mask_unfold # [BS, WS, FWD_N]

        # Discrete action tokenization (if enabled)
        instr_and_action_ids = None
        instr_and_action_labels = None
        instr_and_action_mask = None

        # if self.discrete_action:
        #     logger.warning("Discrete action tokenization in collater is not fully implemented in this template.")
        #     # Fallback: create dummy tensors of expected shape if BaseTrainer requires them
        #     # For this example, let's make dummy tensors of expected shape.
        #     # The actual implementation would involve iterating through samples and using
        #     # self.action_tokenizer and get_prompt_builder.
        #     dummy_seq_len_discrete = MAX_NUM_TOKENS # Example
        #     instr_and_action_ids = torch.zeros((batch_size, self.window_size, dummy_seq_len_discrete), dtype=torch.long)
        #     instr_and_action_labels = torch.ones((batch_size, self.window_size, dummy_seq_len_discrete), dtype=torch.long) * IGNORE_INDEX
        #     instr_and_action_mask = torch.zeros((batch_size, self.window_size, dummy_seq_len_discrete), dtype=torch.bool)


        return {
            "rgb": rgb_input,
            "hand_rgb": hand_rgb_input,
            "action": action_input, # Added new field
            "text": text_tensors,
            "text_mask": attention_mask,
            "fwd_rgb_chunck": fwd_rgb_chunk_label,
            "fwd_hand_rgb_chunck": fwd_hand_rgb_chunk_label,
            "fwd_mask": fwd_mask,
            "action_chunck": action_chunk_label,
            "chunck_mask": chunk_mask,
            # "rel_state": rel_state_input, # Not specified in target output, assuming it's replaced by 'action' and 'rgb' for policy
            "raw_text": lang_prompts,
            "data_source": self.task_type,
            # Optional, if discrete actions are used by the model architecture
            "instr_and_action_ids": instr_and_action_ids,
            "instr_and_action_labels": instr_and_action_labels,
            "instr_and_action_mask": instr_and_action_mask,
        }
