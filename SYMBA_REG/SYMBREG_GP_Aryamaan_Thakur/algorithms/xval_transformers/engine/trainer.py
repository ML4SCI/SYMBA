import os

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from .utils import create_mask
from .predictor import Predictor

from .utils import AverageMeter, seed_everything, create_mask, sequence_accuracy

class Trainer:
    """
    Trainer class for training and evaluating a PyTorch model.
    """
    def __init__(self, config, dataloaders):
        """
        Initialize Trainer object.

        Args:
        - config: Configuration object containing training parameters
        - dataloaders: Dictionary containing data loaders for train, validation, and test sets
        """
        self.config = config
        self.device = torch.device(self.config.device)
        self.dataloaders = dataloaders

        seed_everything(self.config.seed)

        self.scaler = torch.cuda.amp.GradScaler()
        if self.config.use_half_precision:
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32

        # Initialize model, optimizer, scheduler, and criterion
        self.model = self.get_model()
        self.model.to(self.device)
        self.optimizer = self.get_optimizer()
        self.scheduler = self.get_scheduler()
        self.criterion = self.get_criterion()

        # Initialize training-related variables
        self.current_epoch = 0
        self.best_accuracy = -1
        self.best_val_loss = 1e6
        self.train_loss_list = []
        self.valid_loss_list = []
        self.valid_accuracy_tok_list = []

        # Create directory for saving logs
        self.logs_dir = os.path.join(self.config.root_dir, self.config.experiment_name)
        os.makedirs(self.logs_dir, exist_ok=True)
        self.config.save(self.logs_dir)

    def get_model(self):
        """
        Initialize and return the model based on the configuration.
        """
        from algorithms.xval_transformers.model.seq2seq import Model
        model = Model(num_encoder_layers=self.config.num_encoder_layers,
                        num_decoder_layers=self.config.num_decoder_layers,
                        emb_size=self.config.embedding_size,
                        nhead=self.config.nhead,
                        src_vocab_size=self.config.src_vocab_size,
                        tgt_vocab_size=self.config.tgt_vocab_size,
                        input_emb_size=self.config.input_emb_size,
                        max_input_points=self.config.max_input_points,
                        )
        
        return model

    def get_optimizer(self):
        """
        Initialize and return the optimizer based on the configuration.
        """
        optimizer_parameters = self.model.parameters()
        
        if self.config.optimizer_type == "sgd":
            optimizer = torch.optim.SGD(optimizer_parameters, lr=self.config.optimizer_lr, momentum=self.config.optimizer_momentum,)
        elif self.config.optimizer_type == "adam":
            optimizer = torch.optim.Adam(optimizer_parameters, lr=self.config.optimizer_lr, eps=1e-8, weight_decay=self.config.optimizer_weight_decay)
        elif self.config.optimizer_type == "adamw":
            optimizer = torch.optim.AdamW(optimizer_parameters, lr=self.config.optimizer_lr, eps=1e-8, weight_decay=self.config.optimizer_weight_decay)
        else:
            raise NotImplementedError
        
        return optimizer
    
    def get_scheduler(self):
        """
        Initialize and return the learning rate scheduler based on the configuration.
        """
        if self.config.scheduler_type == "multi_step":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.config.scheduler_milestones, gamma=self.config.scheduler_gamma)
        elif self.config.scheduler_type == "reduce_lr_on_plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=2)
        elif self.config.scheduler_type == "cosine_annealing_warm_restart":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, self.config.T_0, self.config.T_mult)
        elif self.config.scheduler_type == "cosine_annealing":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.config.T_max)
        elif self.config.scheduler_type == "none":
            scheduler = None
        else:
            raise NotImplementedError
        
        return scheduler

    
    def get_criterion(self):
        """
        Initialize and return the loss function based on the configuration.
        """
        if self.config.criterion == "cross_entropy":
            criterion = torch.nn.CrossEntropyLoss()
        else:
            raise NotImplementedError
        
        return criterion

    def train_one_epoch(self):
        """
        Train the model for one epoch.
        """
        self.model.train()
        pbar = tqdm(self.dataloaders['train'], total=len(self.dataloaders['train']))
        pbar.set_description(f"[{self.current_epoch+1}/{self.config.epochs}] Train")
        running_loss = AverageMeter()
        for (src, num_array, tgt) in pbar:
            src = src.to(self.device)
            num_array = num_array.to(self.device)
            tgt = tgt.to(self.device)

            bs = src.size(0)

            with torch.autocast(device_type='cuda', dtype=self.dtype):
                src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt[:, :-1], self.device)
                logits = self.model(src, num_array, tgt[:, :-1], src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
                #if self.config.use_half_precision:
                #    logits = logits.clamp(min=1e-4)
                loss = self.criterion(logits.reshape(-1, logits.shape[-1]), tgt[:, 1:].reshape(-1))
                            
            running_loss.update(loss.item(), bs)
            pbar.set_postfix(loss=running_loss.avg)
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            if self.config.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            if self.config.scheduler_type in ["cosine_annealing", "cosine_annealing_warm_restarts"]:
                self.scheduler.step()
    
        return running_loss.avg

    def evaluate(self, phase):
        """
        Evaluate the model on validation or test data.

        Args:
        - phase: Phase of evaluation, either "valid" or "test".

        Returns:
        - Tuple containing average token accuracy and average loss.
        """
        self.model.eval()
        
        pbar = tqdm(self.dataloaders[phase], total=len(self.dataloaders[phase]))
        pbar.set_description(f"[{self.current_epoch+1}/{self.config.epochs}] {phase.capitalize()}")
        running_loss = AverageMeter()
        running_acc_tok = AverageMeter()
        
        
        for (src, num_array, tgt) in pbar:
            src = src.to(self.device)
            num_array = num_array.to(self.device)
            tgt = tgt.to(self.device)
            bs = src.size(0)
            
            with torch.autocast(device_type='cuda', dtype=self.dtype):
                with torch.no_grad():
                    src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt[:, :-1], self.device)
                    logits = self.model(src, num_array, tgt[:, :-1], src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
                    loss = self.criterion(logits.reshape(-1, logits.shape[-1]), tgt[:, 1:].reshape(-1))
               
            y_pred = torch.argmax(logits.reshape(-1, logits.shape[-1]), 1)
            correct = (y_pred == tgt[:, 1:].reshape(-1)).cpu().numpy().mean()
            
            running_loss.update(loss.item(), bs)
            running_acc_tok.update(correct, bs)
            
        return running_acc_tok.avg, running_loss.avg

    def train(self):
        """
        Main training loop.
        """
#        torch.autograd.set_detect_anomaly(True)
        start_epoch = self.current_epoch
        for self.current_epoch in range(start_epoch, self.config.epochs):
            training_loss = self.train_one_epoch() 
            valid_accuracy_tok, valid_loss = self.evaluate("valid")
            
            self.train_loss_list.append(round(training_loss, 7))
            self.valid_loss_list.append(round(valid_loss, 7))
            self.valid_accuracy_tok_list.append(round(valid_accuracy_tok, 7))
            
            if self.config.scheduler_type == "multi_step":
                self.scheduler.step()
            elif self.config.scheduler_type == "reduce_lr_on_plateau":
                self.scheduler.step(valid_loss)
                
            if valid_loss<self.best_val_loss:
                self.best_val_loss = valid_loss

            self.save_model("last_checkpoint.pth")

            if valid_accuracy_tok > self.best_accuracy:
                print(f"==> Best Accuracy improved to {round(valid_accuracy_tok, 7)} from {self.best_accuracy}")
                self.best_accuracy = round(valid_accuracy_tok, 7)
                self.save_model("best_checkpoint.pth")
            
            self.log_results()

        
    def save_model(self, file_name):
        """
        Save model checkpoints.
        """
        state_dict = self.model.state_dict()
        torch.save({
                "epoch": self.current_epoch + 1,
                "state_dict": state_dict,
                'optimizer': self.optimizer.state_dict(),
                "train_loss_list": self.train_loss_list,
                "valid_loss_list": self.valid_loss_list,
                "valid_accuracy_tok_list": self.valid_accuracy_tok_list,
            }, os.path.join(self.logs_dir, file_name))

    def log_results(self):
        """
        Log training results to a CSV file.
        """
        data_list = [self.train_loss_list, self.valid_loss_list, self.valid_accuracy_tok_list]
        column_list = ['train_losses', 'valid_losses', 'token_valid_accuracy']
        
        df_data = np.array(data_list).T
        df = pd.DataFrame(df_data, columns=column_list)
        df.to_csv(os.path.join(self.logs_dir, "logs.csv"))
        
    def test_seq_acc(self):
        """
        Evaluate model's sequence accuracy on test data.
        """
        file = os.path.join(self.logs_dir, "best_checkpoint.pth")
        state_dict = torch.load(file, map_location=self.device)['state_dict']
        self.model.load_state_dict(state_dict)
        
        test_accuracy_tok, _ = self.evaluate("test")
        
        predictor = Predictor(self.config)
        
        print("Calculating Sequence Accuracy for predictions (1 example per batch)")
        pbar = tqdm(self.dataloaders["test"], total=len(self.dataloaders["test"]))
        pbar.set_description(f"Test")
        
        y_preds = []
        y_true = []
        for src, num_array, tgt in pbar:
            src = src.to(self.device)
            num_array = num_array.to(self.device)
            tgt = tgt.numpy()
            bs = src.size(0)
            y_pred = predictor.predict(src[0].unsqueeze(0), num_array[0].unsqueeze(0)) #only one example from each batch
            y_preds.append(y_pred.cpu().numpy())
            y_true.append(np.trim_zeros(tgt[0]))
            
        test_accuracy_seq = sequence_accuracy(y_true, y_preds)
        f= open(os.path.join(self.logs_dir, "score.txt"),"w+")
        f.write(f"Token Accuracy = {(round(test_accuracy_tok, 7))}\n")
        f.write(f"Sequence Accuracy = {(round(test_accuracy_seq, 7))}\n")
        f.close()
        print(f"Test Accuracy: {round(test_accuracy_tok, 7)} | Valid Accuracy: {self.best_accuracy}") 
        print(f"Test Sequence Accuracy: {test_accuracy_seq}")

