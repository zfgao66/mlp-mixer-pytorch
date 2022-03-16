import torch
import torch.nn as nn
import logging
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
from copy import deepcopy
import os

logger = logging.getLogger(__name__)
class Base:
    """
    Basic implementation of a general Knowledge Distillation framework

    :param teacher_model (torch.nn.Module): Teacher model
    :param student_model (torch.nn.Module): Student model
    :param train_loader (torch.utils.data.DataLoader): Dataloader for training
    :param val_loader (torch.utils.data.DataLoader): Dataloader for validation/testing
    :param optimizer_teacher (torch.optim.*): Optimizer used for training teacher
    :param optimizer_student (torch.optim.*): Optimizer used for training student
    :param loss_fn (torch.nn.Module): Loss Function used for distillation
    :param temp (float): Temperature parameter for distillation
    :param distil_weight (float): Weight paramter for distillation loss
    :param device (str): Device used for training; 'cpu' for cpu and 'cuda' for gpu
    :param log (bool): True if logging required
    :param logdir (str): Directory for storing logs
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        loss_fn=nn.KLDivLoss(),
        device="cpu",
        log=False,
        logdir="./Experiments",
    ):

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.log = log
        self.logdir = logdir

        if self.log:
            self.writer = SummaryWriter(logdir)

        try:
            torch.Tensor(0).to(device)
            self.device = device
        except:
            logger.info(
                "Either an invalid device or CUDA is not available. Defaulting to CPU."
            )
            self.device = torch.device("cpu")

        try:
            self.model = model.to(self.device)
        except:
            logger.info("Warning!!! Model is NONE.")
        try:
            self.loss_fn = loss_fn.to(self.device)
            self.ce_fn = nn.CrossEntropyLoss().to(self.device)
        except:
            self.loss_fn = loss_fn
            self.ce_fn = nn.CrossEntropyLoss()
            logger.info("Warning: Loss Function can't be moved to device.")

    def train_model(
        self,
        epochs=20,
        save_model=True,
        save_model_pth="./models/model.pt",
    ):
        """
        Function that will be training the teacher

        :param epochs (int): Number of epochs you want to train the teacher
        :param plot_losses (bool): True if you want to plot the losses
        :param save_model (bool): True if you want to save the teacher model
        :param save_model_pth (str): Path where you want to store the teacher model
        """
        self.model.train()
        loss_arr = []
        length_of_dataset = len(self.train_loader.dataset)
        best_acc = 0.0
        self.best_model_weights = deepcopy(self.model.state_dict())

        save_dir = os.path.dirname(save_model_pth)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        logger.info("Training Model... ")

        for ep in range(epochs):
            epoch_loss = 0.0
            correct = 0
            for (data, label) in self.train_loader:
                data = data.to(self.device)
                label = label.to(self.device)
                out = self.model(data)

                if isinstance(out, tuple):
                    out = out[0]

                pred = out.argmax(dim=1, keepdim=True)
                correct += pred.eq(label.view_as(pred)).sum().item()

                loss = self.ce_fn(out, label)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss

            epoch_acc = correct / length_of_dataset

            epoch_val_acc = self.evaluate()

            if epoch_val_acc > best_acc:
                best_acc = epoch_val_acc
                self.best_model_weights = deepcopy(
                    self.model.state_dict()
                )

            # if self.log:
            #     self.writer.add_scalar("Training loss/Teacher", epoch_loss, epochs)
            #     self.writer.add_scalar("Training accuracy/Teacher", epoch_acc, epochs)
            #     self.writer.add_scalar(
            #         "Validation accuracy/Teacher", epoch_val_acc, epochs
            #     )

            loss_arr.append(epoch_loss)
            logger.info(f"Epoch: {ep+1}, Loss: {epoch_loss}, Train Accuracy: {epoch_acc}, Test Accuracy:{epoch_val_acc}")

            self.post_epoch_call(ep)

        self.model.load_state_dict(self.best_model_weights)
        if save_model:
            torch.save(self.model.state_dict(), save_model_pth)
    # def test_model(
    #     self,
    #     epochs=20,
    #     load_model=True,
    #     load_model_pth = './models/models.pt',
    # ):
    #     """
    #     Funtion that test the model.
    #     """
    #     epoch_acc = 
    

    def calculate_kd_loss(self, y_pred_student, y_pred_teacher, y_true):
        """
        Custom loss function to calculate the KD loss for various implementations

        :param y_pred_student (Tensor): Predicted outputs from the student network
        :param y_pred_teacher (Tensor): Predicted outputs from the teacher network
        :param y_true (Tensor): True labels
        """

        raise NotImplementedError

    def _evaluate_model(self, model, verbose=True):
        """
        Evaluate the given model's accuaracy over val set.
        For internal use only.

        :param model (nn.Module): Model to be used for evaluation
        :param verbose (bool): Display Accuracy
        """
        model.eval()
        length_of_dataset = len(self.val_loader.dataset)
        correct = 0
        outputs = []

        with torch.no_grad():
            for data, target in self.val_loader:
                data = data.to(self.device)
                target = target.to(self.device)
                output = model(data)

                if isinstance(output, tuple):
                    output = output[0]
                outputs.append(output)

                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        accuracy = correct / length_of_dataset

        if verbose:
            logger.info("-" * 80)
            logger.info(f"Validation Accuracy: {accuracy}")
        return outputs, accuracy

    def evaluate(self):
        """
        Evaluate method for logger.infoing accuracies of the trained network

        :param teacher (bool): True if you want accuracy of the teacher network
        """
        model = deepcopy(self.model).to(self.device)
        logger.info("Validate model")
        _, accuracy = self._evaluate_model(model)

        return accuracy

    def get_parameters(self):
        """
        Get the number of parameters for the teacher and the student network
        """
        teacher_params = sum(p.numel() for p in self.model.parameters())
        # student_params = sum(p.numel() for p in self.student_model.parameters())

        logger.info("-" * 80)
        logger.info(f"Total parameters for the network are: {teacher_params}")
        # logger.info(f"Total parameters for the student network are: {student_params}")

    def post_epoch_call(self, epoch):
        """
        Any changes to be made after an epoch is completed.

        :param epoch (int) : current epoch number
        :return            : nothing (void)
        """

        pass
