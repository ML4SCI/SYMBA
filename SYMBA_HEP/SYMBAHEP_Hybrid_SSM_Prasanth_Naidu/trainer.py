import os

import torch
import wandb
import numpy as np
from tqdm import tqdm
from torch.amp import GradScaler
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.parallel import DistributedDataParallel as DDP

from .predictor import sequence_accuracy
from .fn_utils import (
    calculate_line_params,
    collate_fn,
    create_mask,
    get_model
)
from .data import Data
from .constants import PAD_IDX


class Trainer():
    """
    Class for training a sequence-to-sequence model.

    Args:
        config (TransformerConfig): Configuration object with training parameters.
        df_train (DataFrame): Training data.
        df_valid (DataFrame): Validation data.
        tokenizer (Tokenizer): Tokenizer for input data.
        src_vocab (Vocab): Source vocabulary.
        tgt_vocab (Vocab): Target vocabulary.

    Attributes:
        scaler (GradScaler): Gradient scaler for automatic mixed precision training.
        dtype (torch.dtype): Data type (float16 if half precision is enabled, else float32).
        local_rank (int): Local rank of the process in distributed training.
        global_rank (int): Global rank of the process in distributed training.
        device (int): CUDA device ID used for training.
        config (TransformerConfig): Configuration object passed during initialization.
        is_master (bool): Flag indicating if this is the master (rank 0) process.
        run (wandb.Run): Weights & Biases run object (only initialized on master).
        dataloaders (dict): Dictionary containing train, validation, and test dataloaders.
        valid_ds (Dataset): Preprocessed valid dataset.
        warmup_steps (int): Number of warmup steps for learning rate scheduling.
        ep_steps (int): Number of steps per epoch.
        root_dir (str): Directory where checkpoints are saved.
        current_epoch (int): The current epoch number.
        best_val_loss (float): Best validation loss observed.
        train_loss_list (list): List of training losses over epochs.
        valid_loss_list (list): List of validation losses over epochs.
        model (nn.Module): Model used for training.
        ddp_model (nn.Module): DistributedDataParallel-wrapped model.
        optimizer (torch.optim.Optimizer): Optimizer used for training.
        warm_scheduler (LambdaLR): Warmup learning rate scheduler.
        lr_scheduler (LambdaLR): Main learning rate scheduler.
        save_freq (int): Frequency (in epochs) at which to save checkpoints.
        test_freq (int): Frequency (in epochs) at which to evaluate test accuracy.
        resume_best (bool): Whether to resume training from the best saved checkpoint.
        save_last (bool): Whether to save the final model at the end of training.
        lr (float): Learning rate to be used for optimization.
        global_step (int): Global step counter used for logging and scheduling.
        ckp_paths (list): List of checkpoint filenames in the root directory (excluding 'best' models).
        save_limit (int): Maximum number of checkpoints to keep.
        is_constant_lr (bool): Flag for using constant learning rate.
    """


    def __init__(self, config, df_train, df_valid, tokenizer, src_vocab, tgt_vocab):
        
        self.scaler = GradScaler()
        self.is_constant_lr = config.is_constant_lr
        
        if config.dtype == 'bfloat16':
            self.dtype = torch.bfloat16
        elif config.dtype == 'float32':
            self.dtype = torch.float32
        elif config.dtype == 'float16':
            self.dtype = torch.float16
    
        
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.global_rank = int(os.environ["RANK"])
        
        if config.debug is not True:
            print(f"PROCESS ID : {int(os.environ['SLURM_PROCID'])} ; TORCH GLOBAL RANK : {self.global_rank} ; TORCH LOCAL RANK : {self.local_rank}")
        
        self.device = self.local_rank
        self.config = config
        self.is_master = self.local_rank == 0

        # Initialize Weights & Biases
                 
        os.makedirs(config.root_dir, exist_ok=True)

        wandb.login()
        self.run = wandb.init(
            project=config.project_name,
            name=f"{config.run_name}_rank{self.global_rank}",
            group=config.run_name,
            dir=config.root_dir,
            config=config.to_dict(),
            resume='allow',
            id=config.run_id
        )
        
        # Initialize dataloaders
        self.dataloaders,self.valid_ds = self._prepare_dataloaders(
            df_train, df_valid, tokenizer, src_vocab, tgt_vocab)

        self.ep_steps = len(self.dataloaders['train'])  
        self.total_steps = self.ep_steps * config.epochs
        self.warmup_steps = int(config.warmup_ratio * self.total_steps)

        self.root_dir = config.root_dir
        self.current_epoch = config.curr_epoch
        
        # Training and valid loss lists
        self.best_val_loss = 1e6
        self.train_loss_list = []
        self.valid_loss_list = []
        
        # Initialize model, optimizer, and schedulers
        self.model, self.ddp_model = self._prepare_model()
        self.optimizer = self._prepare_optimizer()
        self.warm_scheduler, self.lr_scheduler = self._prepare_scheduler()

        self.save_freq = config.save_freq
        self.test_freq = config.test_freq
        self.resume_best = config.resume_best
        self.save_last = config.save_last
        self.lr = config.update_lr
        self.global_step = 0
        self.tgt_vocab = tgt_vocab 
        
        self.ckp_paths = [file for file in os.listdir(config.root_dir) if ('best' not in file and config.model_name in file)]
        self.save_limit = config.save_limit

    def criterion(self, y_pred, y_true):
        """
        Calculate the loss between predicted and true values.

        Args:
            y_pred (Tensor): Predicted values.
            y_true (Tensor): True values.

        Returns:
            Tensor: Loss value.
        """
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
        return loss_fn(y_pred, y_true)

    def _prepare_model(self):
        """
        Initialize and prepare the model for training.

        Returns:
            Model: Initialized model.
        """
        model = get_model(self.config)
        model.to(self.device)
        # ddp_model = DDP(model, device_ids=[self.device])
        ddp_model = DDP(model, device_ids=[self.device], find_unused_parameters=True)
        if self.is_master:
            self.run.watch(ddp_model.module,log_freq=20)
        return model, ddp_model

    def _prepare_optimizer(self):
        """
        Initialize the optimizer.

        Returns:
            Optimizer: Initialized optimizer.
        """
        param_optimizer = list(self.ddp_model.parameters())
        optimizer = torch.optim.AdamW(
            param_optimizer, lr=self.config.optimizer_lr, eps=1e-9, weight_decay = self.config.weight_decay)
        return optimizer

    def _prepare_scheduler(self):
        """
        Initialize the schedulers.

        Returns:
            tuple: A tuple containing:
                - warm_scheduler (LambdaLR or None): The learning rate warm-up scheduler.
                - lr_scheduler (LambdaLR or None): The learning rate decay scheduler.
        """
        start_lr = self.config.optimizer_lr
        end_lr = self.config.end_lr

        if self.warmup_steps:
            m_warm, c_warm = calculate_line_params(
                (0, end_lr), (self.warmup_steps, start_lr))
            
            def lam_warm(step):
                if step >= self.warmup_steps:
                    return 1.0  
                return (m_warm * step + c_warm) / start_lr
            
            warm_scheduler = LambdaLR(self.optimizer, lr_lambda=lam_warm)

        else:
            warm_scheduler = None
        
        if self.is_constant_lr:
            lr_scheduler = None
        else:
            m_decay, c_decay = calculate_line_params(
                (self.warmup_steps, start_lr), (self.total_steps, end_lr))

            def lam_decay(step):
                step = step + self.warmup_steps
                if step <= self.warmup_steps:
                    return 1.0
                return max(0.0, (m_decay * step + c_decay) / start_lr)

            lr_scheduler = LambdaLR(self.optimizer, lr_lambda=lam_decay)

        return warm_scheduler, lr_scheduler

    def _prepare_dataloaders(self, df_train, df_valid, tokenizer, src_vocab, tgt_vocab):
        """
        Prepare dataloaders for training, validation, and testing.

        Returns:
            dict: Dictionary containing train, validation, and test dataloaders.
        """
        datasets = Data.get_data(
            df_train, None, df_valid, self.config, tokenizer,src_vocab, tgt_vocab)
        
        sampler_train = torch.utils.data.DistributedSampler(datasets['train'], num_replicas=self.config.world_size,
                                                            rank=self.device, shuffle=self.config.train_shuffle, seed=self.config.seed)

        train_loader = torch.utils.data.DataLoader(datasets['train'], batch_size=self.config.training_batch_size,
                                                   sampler=sampler_train, num_workers=self.config.num_workers,
                                                   pin_memory=self.config.pin_memory, collate_fn=collate_fn)

        dataloaders = {
            'train': train_loader,
            'valid': torch.utils.data.DataLoader(datasets['valid'],
                                                 batch_size=self.config.valid_batch_size, shuffle=self.config.valid_shuffle,
                                                 num_workers=self.config.num_workers, pin_memory=self.config.pin_memory, collate_fn=collate_fn),
        }
        return dataloaders,datasets['valid']

    def load_model(self, resume=False, epoch=None, lr=None):
        """
        Load the most recent model checkpoint.

        Args:
            resume (bool, optional): Whether to resume training. Defaults to False.
            epoch (int, optional): Load model from a particular epoch
        """
        checkpoint_name = f"{self.config.model_name}_best.pth" if resume else f"{self.config.model_name}_ep{epoch}.pth"
        file = os.path.join(self.root_dir, checkpoint_name)

        device_name = f"cuda:{self.device}"
        state = torch.load(file, map_location=device_name)
        self.model.load_state_dict(state['state_dict'])
        
        if resume or (epoch is not None):
            self.train_loss_list = state['train_loss_list']
            self.valid_loss_list = state['valid_loss_list']
            self.best_val_loss = np.array(self.valid_loss_list).min()
            self.optimizer.load_state_dict(state['optimizer'])
            
            if state['decay_scheduler'] is not None:
                self.lr_scheduler.load_state_dict(state['decay_scheduler'])
            if state['warm_scheduler'] is not None:
                self.warm_scheduler.load_state_dict(state['warm_scheduler'])
            
            if 'scaler' in state:
                self.scaler.load_state_dict(state['scaler'])
            
            self.global_step = state['global_step']

            if epoch == None:
                self.current_epoch = state['epoch']
            
            if lr:
                for g in self.optimizer.param_groups:
                    g['lr'] = lr
                print("Lr_changed :)")

            print(checkpoint_name)
            print("Loaded :)")

    def _train_epoch(self):
        """
        Perform a single training epoch.

        Returns:
            float: Average training loss for the epoch.
        """
        self.ddp_model.train()
        pbar = tqdm(
            self.dataloaders['train'],
            total=len(self.dataloaders['train']),
            disable=not self.is_master
        )
        pbar.set_description(f"[{self.current_epoch + 1}/{self.config.epochs}] Train")

        running_loss = 0.0
        total_samples = 0

        for src, tgt in pbar:
            src = src.to(self.device)
            tgt = tgt.to(self.device)
            batch_size = src.size(0)

            with torch.autocast(device_type='cuda', dtype=self.dtype):
                src_padding_mask, tgt_padding_mask = create_mask(
                    src, tgt[:, :-1]
                )

                logits = self.ddp_model(
                    src, tgt[:, :-1],
                    src_padding_mask, tgt_padding_mask,
                    src_padding_mask
                )

                loss = self.criterion(
                    logits.reshape(-1, logits.shape[-1]),
                    tgt[:, 1:].reshape(-1)
                )

            if (self.global_step % self.config.log_freq == 0):
                self.run.log({'train/loss': loss.item(), 'global_step': self.global_step})

            running_loss += loss.item() * batch_size
            total_samples += batch_size
            avg_loss = running_loss / total_samples
            pbar.set_postfix(loss=avg_loss)

            # Backpropagation
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)

            if self.config.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.ddp_model.parameters(), self.config.clip_grad_norm
                )

            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Compute gradient norm for logging
            grads = [
                param.grad.detach().flatten()
                for param in self.ddp_model.module.parameters()
                if param.grad is not None
            ]
            grad_norm = torch.cat(grads).norm()

            # Learning rate scheduling and logging
            if self.global_step < self.warmup_steps:
                if self.warmup_steps:
                    self.warm_scheduler.step()
            else:
                if not self.is_constant_lr:
                    self.lr_scheduler.step()

            if (self.global_step % self.config.log_freq == 0):
                self.run.log({
                    'train/lr': self.optimizer.param_groups[0]['lr'],
                    'train/epoch': self.global_step / self.ep_steps,
                    'train/grad_norm': grad_norm,
                    'train/scale': self.scaler.get_scale(), 
                    'global_step': self.global_step
                })

            self.global_step += 1

        return avg_loss


    def evaluate(self):
        """
        Evaluate the model on the validation set.

        Returns:
            float: Average validation loss.
        """
        self.ddp_model.eval()
        pbar = tqdm(
            self.dataloaders["valid"],
            total=len(self.dataloaders["valid"]),
            disable=not self.is_master
        )
        pbar.set_description(f"[{self.current_epoch + 1}/{self.config.epochs}] Valid")

        running_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=self.dtype):
                for src, tgt in pbar:
                    src = src.to(self.device)
                    tgt = tgt.to(self.device)
                    batch_size = src.size(0)

                    src_padding_mask, tgt_padding_mask = create_mask(
                        src, tgt[:, :-1]
                    )

                    logits = self.ddp_model(
                        src, tgt[:, :-1],
                        src_padding_mask, tgt_padding_mask,
                        src_padding_mask
                    )

                    loss = self.criterion(
                        logits.reshape(-1, logits.shape[-1]),
                        tgt[:, 1:].reshape(-1)
                    )

                    running_loss += loss.item() * batch_size
                    total_samples += batch_size
                    avg_loss = running_loss / total_samples

        return avg_loss


    def _save_model(self, checkpoint_name):
        """
        Save the model checkpoint.

        Args:
            checkpoint_name (str): Name of the checkpoint file.
        """
        ckp_path = os.path.join(self.root_dir, checkpoint_name)
        state_dict = self.ddp_model.module.state_dict()

        torch.save({
            "epoch": self.current_epoch + 1,
            "state_dict": state_dict,
            "optimizer": self.optimizer.state_dict(),
            "decay_scheduler": self.lr_scheduler.state_dict() if self.lr_scheduler else None,
            "warm_scheduler": self.warm_scheduler.state_dict() if self.warm_scheduler else None,
            "scaler": self.scaler.state_dict(),            
            "train_loss_list": self.train_loss_list,
            "valid_loss_list": self.valid_loss_list,         
            "global_step": self.global_step
        }, ckp_path)

        if "best" not in checkpoint_name:
            self.ckp_paths.append(ckp_path)

        # Remove oldest checkpoints if exceeding save limit
        if len(self.ckp_paths) > self.save_limit:
            oldest = self.ckp_paths.pop(0)
            if os.path.exists(oldest):
                os.remove(oldest)
                print(f"Deleted old checkpoint: {oldest}")


    def _test_seq_acc(self, load_best=True, epochs=None):
        """
        Test sequence accuracy and log the results.

        Args:
            load_best (bool): Whether to load the best model before testing.
            epochs (int or None): Epoch for labeling/testing metadata.
        """
        test_accuracy_seq = sequence_accuracy(
            self.config, self.valid_ds, self.tgt_vocab, load_best, epochs
        )
        self.run.log({
            'test/acc': test_accuracy_seq,
            'global_step': self.global_step
        })
        print(f"Test Accuracy: {round(test_accuracy_seq, 4)}")


    def fit(self):
        """
        Train the model across all epochs.
        """
    
        self.run.define_metric("global_step")
        self.run.define_metric("validation/*", step_metric="global_step")
        self.run.define_metric("train/*", step_metric="global_step")
        self.run.define_metric("test/*", step_metric="global_step")

        if self.current_epoch != 0:
            self.load_model(epoch=self.current_epoch, lr=self.lr)
        elif self.resume_best:
            self.load_model(resume=True, lr=self.lr)

        for self.current_epoch in range(self.current_epoch, self.config.epochs):
            training_loss = self._train_epoch()
            valid_loss = self.evaluate()

            # if self.global_step >= self.warmup_steps and not self.is_constant_lr:
            #     self.lr_scheduler.step(self.current_epoch)

            
            self.run.log({
                'valid/loss': valid_loss,
                'global_step': self.global_step
            })

            self.train_loss_list.append(training_loss)
            self.valid_loss_list.append(valid_loss)

            if self.is_master:
                if valid_loss <= self.best_val_loss:
                    self.best_val_loss = valid_loss
                    self._save_model(f"{self.config.model_name}_best.pth")

                if self.save_freq and (self.current_epoch + 1) % self.save_freq == 0:
                    self._save_model(f"{self.config.model_name}_ep{self.current_epoch + 1}.pth")
                    self._test_seq_acc(load_best=False, epochs=self.current_epoch)

                elif (self.current_epoch + 1) % self.test_freq == 0:
                    self._test_seq_acc()

            torch.distributed.barrier()

            print(
                f"Epoch {self.current_epoch + 1}/{self.config.epochs}, "
                f"Training Loss: {training_loss:.4f}, "
                f"Validation Loss: {valid_loss:.4f}"
            )

        if self.is_master:
            if self.save_last:
                self._save_model(f"{self.config.model_name}_ep{self.current_epoch + 1}.pth")
            self._test_seq_acc(load_best=False, epochs=self.current_epoch)

        wandb.finish()
