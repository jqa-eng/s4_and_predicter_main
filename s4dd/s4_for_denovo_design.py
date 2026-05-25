import json
import math
import os
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from torch import nn

from . import smiles_utils, torch_callbacks
from .dataloaders import create_dataloader
from .module_library.sequence_model import SequenceModel


class StructuredStateSpaceSequenceModel(nn.Module):
    """A general purpose structured state space sequence (S4) model implemented as a pytorch module."""

    def __init__(
        self,
        model_dim: int,
        state_dim: int,
        n_layers: int,
        n_ssm: int,
        dropout: float,
        learning_rate: float,
        sequence_length: int,
        vocab_size: int,
    ) -> None:
        """Creates a `StructuredStateSpaceSequenceModel` instance.

        Parameters
        ----------
        model_dim : int
            The dimension of the model.
        state_dim : int
            The dimension of the state in recurrent mode.
        n_layers : int
            The number of S4 layers in the model.
        n_ssm : int
            The number of state space models in each layer.
        dropout : float
            The dropout rate.
        learning_rate : float
            The learning rate.
        sequence_length : int
            The length of the sequences.
        vocab_size : int
            The size of the vocabulary.
        """
        super().__init__()
        self.model_dim = model_dim
        self.state_dim = state_dim
        self.n_layers = n_layers
        self.n_ssm = n_ssm
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size

        self.layer_config = [
            {
                "_name_": "s4",
                "d_state": self.state_dim,
                "n_ssm": self.n_ssm,
            },
            {
                "_name_": "s4",
                "d_state": self.state_dim,
                "n_ssm": self.n_ssm,
            },
            {"_name_": "ff"},
        ]
        self.pool_config = {"_name_": "pool", "stride": 1, "expand": None}

        self.embedding = nn.Embedding(self.vocab_size, self.model_dim)
        self.model = SequenceModel(
            d_model=self.model_dim,
            n_layers=self.n_layers,
            transposed=True,
            dropout=self.dropout,
            layer=self.layer_config,
            pool=self.pool_config,
        )
        self.output_embedding = nn.Linear(self.model_dim, self.vocab_size)
        self.recurrent_state = None

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Computes the forward pass of the model. The forward pass consists of embedding the
        input tokens, passing the embeddings through the S4 model (in convolutional mode), and then passing the
        output of the S4 model through a linear layer to get the logits.

        Parameters
        ----------
        batch : torch.Tensor
            A batch of sequences of integers representing the tokens. The input shape is (batch_size, sequence_length, 1).

        Returns
        -------
        torch.Tensor
            The logits of the model.
        """
        batch = self.embedding(batch)
        batch = batch.view(batch.shape[0], self.sequence_length, self.model_dim)
        batch, state = self.model(batch, state=self.recurrent_state)
        self.recurrent_state = state
        batch = self.output_embedding(batch)
        return batch

    def reset_state(self, batch_size: int, device: str = None) -> None:
        """Resets the recurrent state of the model.
        Used in sequential mode before processing a new batch.

        Parameters
        ----------
        batch_size : int
            The batch size.
        device : str
            The device to put the state on, *e.g.,* `"cuda"` or `"cpu"`.
        """
        self.recurrent_state = self.model.default_state(batch_size, device=device)

    def recurrent_step(self, x_t):
        """Computes a single step in the recurrent mode. The internal state of the model is also updated.

        Parameters
        ----------
        x_t : torch.Tensor
            The input token. The input shape is (batch_size, 1).

        Returns
        -------
        torch.Tensor
            The logits resulting from the stepping.
        """
        x_t = self.embedding(x_t).view(x_t.shape[0], 1, self.model_dim)
        x_t = x_t.squeeze(1)
        x_t, state = self.model.step(x_t, state=self.recurrent_state)
        self.recurrent_state = state
        x_t = self.output_embedding(x_t)
        return x_t


class S4forDenovoDesign:
    """A structured state space sequence (S4) model for de novo design."""

    def __init__(
        self,
        model_dim: int = 256,
        state_dim: int = 64,
        n_layers: int = 4,
        n_ssm: int = 1,
        dropout: float = 0.25,
        vocab_size: int = 37,
        sequence_length: int = 99,
        n_max_epochs: int = 400,
        learning_rate: float = 0.001,
        batch_size: int = 2048,
        device: str = "cuda",
    ) -> None:
        """Creates an `S4forDenovoDesign` instance.
        The default configurations are the ones used in the [paper](https://chemrxiv.org/engage/chemrxiv/article-details/65168004ade1178b24567cd3).

        Parameters
        ----------
        model_dim : int
            The number of dimensions used across the model.
        state_dim : int
            The dimension of the state in the recurrent mode.
        n_layers : int
            The number of S4 layers in the model.
        n_ssm : int
            The number of state space models in each layer.
        dropout : float
            The dropout rate.
        vocab_size : int
            The size of the vocabulary.
        sequence_length : int
            The length of the sequences.
        n_max_epochs : int
            The maximum number of epochs to train for.
        learning_rate : float
            The learning rate.
        batch_size : int
            The batch size.
        device : str
            The device to put the model on, *e.g.,* `"cuda"` or `"cpu"`.
        """

        self.model_dim = model_dim
        self.state_dim = state_dim
        self.n_layers = n_layers
        self.n_ssm = n_ssm
        self.dropout = dropout
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.n_max_epochs = n_max_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.device = device

        # These are set during training
        self.token2label = None
        self.label2token = None

        self.s4_model = StructuredStateSpaceSequenceModel(
            model_dim=self.model_dim,
            state_dim=self.state_dim,
            n_layers=self.n_layers,
            n_ssm=self.n_ssm,
            dropout=self.dropout,
            learning_rate=self.learning_rate,
            sequence_length=self.sequence_length,
            vocab_size=self.vocab_size,
        )

    @classmethod
    def from_file(cls, loaddir: str):
        """Loads an `S4forDenovoDesign` instance from a directory.

        Parameters
        ----------
        loaddir : str
            The directory to load the model from.

        Returns
        -------
        S4forDenovoDesign
            The loaded model.
        """
        with open(f"{loaddir}/init_arguments.json", "r") as f:
            properties = json.load(f)
        s4_model = StructuredStateSpaceSequenceModel(
            model_dim=properties["model_dim"],
            state_dim=properties["state_dim"],
            n_layers=properties["n_layers"],
            n_ssm=properties["n_ssm"],
            dropout=properties["dropout"],
            learning_rate=properties["learning_rate"],
            sequence_length=properties["sequence_length"],
            vocab_size=properties["vocab_size"],
        )
        s4_model.load_state_dict(torch.load(f"{loaddir}/model.pt"))
        token2label = properties.pop("token2label")
        label2token = properties.pop("label2token")
        instance = cls(**properties)
        instance.s4_model = s4_model
        instance.s4_model.to(instance.device)
        instance.token2label = token2label
        instance.label2token = {
            int(label): token for label, token in label2token.items()
        }
        return instance

    def _compute_loss(self, loss_fn, X, y):
        X = X.unsqueeze(2).to(self.device)
        y = y.to(self.device)
        logits = self.s4_model(X).permute(0, 2, 1)
        return loss_fn(
            logits,
            y,
        )

    def train(
        self,
        training_molecules_path: str,
        val_molecules_path: str,
        callbacks: List[torch_callbacks.TorchCallback] = None,
    ) -> Dict[str, List[float]]:
        """Trains the model. The inputs are the paths to the training and validation molecules.
        The paths should point either to a .txt file that contains one SMILES per line, or to a zip file with the same structure.
        The optional callbacks can be used to monitor or configure training.
        The training history is returned as a dictionary.

        Parameters
        ----------
        training_molecules_path : str
            The path to the training molecules. Can be a zip file or a text file. Must contain one SMILES string per line.
        val_molecules_path : str
            The path to the validation molecules. Must have the same structure as `training_molecules_path`.
        callbacks : List[torch_callbacks.TorchCallback], optional
            A list of callbacks to use during training. See the documentation of the `torch_callbacks` module for available options.

        Returns
        -------
        Dict[str, List[float]]
            A dictionary containing the training history. The keys are `train_loss` and `val_loss` and the values are lists of the metric values at each epoch.
        """
        self.s4_model = self.s4_model.to(self.device)
        train_dataloader = create_dataloader(
            training_molecules_path,
            batch_size=self.batch_size,
            sequence_length=self.sequence_length + 1,
            num_workers=1,
            shuffle=True,
            token2label=self.token2label,
        )
        self.token2label = train_dataloader.dataset.token2label
        self.label2token = {v: k for k, v in self.token2label.items()}

        val_dataloader = create_dataloader(
            val_molecules_path,
            batch_size=self.batch_size,
            sequence_length=self.sequence_length + 1,
            num_workers=1,
            shuffle=True,
            token2label=self.token2label,
        )
        loss_fn = nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(self.s4_model.parameters(), lr=self.learning_rate)
        history = {"train_loss": list(), "val_loss": list()}
        epoch_train_loss = 0
        for epoch_ix in range(self.n_max_epochs):
            self.s4_model.recurrent_state = None
            # Training
            self.s4_model.train()
            n_train_batches = len(train_dataloader)
            epoch_train_loss = 0
            for X_train, y_train in tqdm.tqdm(train_dataloader):
                optimizer.zero_grad()
                batch_train_loss = self._compute_loss(loss_fn, X_train, y_train)
                epoch_train_loss += batch_train_loss.item()
                batch_train_loss.backward()
                optimizer.step()

            epoch_train_loss = epoch_train_loss / n_train_batches
            history["train_loss"].append(epoch_train_loss)

            # Validation
            self.s4_model.eval()
            n_val_batches = len(val_dataloader)
            epoch_val_loss = 0

            for X_val, y_val in val_dataloader:
                batch_val_loss = self._compute_loss(loss_fn, X_val, y_val)
                epoch_val_loss += batch_val_loss.item()

            epoch_val_loss = epoch_val_loss / n_val_batches
            history["val_loss"].append(epoch_val_loss)

            # Callbacks
            print(
                f"Epoch:{epoch_ix}\tLoss: {epoch_train_loss}, Val Loss: {epoch_val_loss}"
            )
            stop_training = False
            if callbacks is not None:
                for callback in callbacks:
                    callback.on_epoch_end(epoch_ix=epoch_ix, history=history)
                stop_training_flags = [callback.stop_training for callback in callbacks]
                stop_training = stop_training | (sum(stop_training_flags) > 0)
            if stop_training:
                print("Training stopped early. Epoch:", epoch_ix)
                break

            if np.isnan(epoch_train_loss) or np.isnan(epoch_val_loss):
                print("Training diverged. Epoch:", epoch_ix)
                break

        if callbacks is not None:
            for callback in callbacks:
                callback.on_train_end(epoch_ix=epoch_ix, history=history)
        return history

    @torch.no_grad()
    def design_molecules(
        self,
        n_designs: int,
        batch_size: int,
        temperature: float,
        required_prefix_smiles: Optional[str] = None,
    ) -> Tuple[List[str], List[float]]:
        """Designs molecules using the trained model. The number of designs to generate is specified by `n_designs`.
        The designs are generated in batches of size `batch_size`. The temperature is used to control the diversity of the generated designs.
        The designs and their log-likelihoods are returned as a tuple.

        Parameters
        ----------
        n_designs : int
            The number of designs to generate.
        batch_size : int
            The batch size to use during generation.
        temperature : float
            The temperature to use during generation.

        Returns
        -------
        Tuple[List[str], List[float]]
            A tuple containing the generated SMILES strings and their log-likelihoods.
        """
        if self.token2label is None or self.label2token is None:
            raise ValueError("This model is untrained.")

        self.s4_model = self.s4_model.to(self.device)
        for module in self.s4_model.modules():
            if hasattr(module, "setup_step"):
                module.setup_step()
        self.s4_model.eval()

        # Prefix handling logic
        from .smiles_utils import segment_smiles

        prefix_tokens = []
        prefix_labels = []
        consumed = 1  # Default: only [BEG] consumed

        if required_prefix_smiles:
            # Tokenize prefix
            prefix_tokens = ['[BEG]'] + segment_smiles(required_prefix_smiles)

            # Check vocabulary coverage
            missing = [t for t in prefix_tokens if t not in self.token2label]
            if missing:
                raise ValueError(f'Prefix contains OOV tokens: {missing}')

            # Encode prefix
            prefix_labels = [self.token2label[t] for t in prefix_tokens]
            consumed = len(prefix_tokens)

        n_batches = math.ceil(n_designs / batch_size)
        designs, likelihoods = list(), list()
        for batch_idx in range(n_batches):
            if batch_idx == n_batches - 1:
                batch_size = n_designs - batch_idx * batch_size

            # Reset state first
            self.s4_model.reset_state(batch_size, device=self.device)

            # Initialize from prefix or [BEG]
            if required_prefix_smiles:
                # Prefix Replay: feed prefix tokens into model state sequentially
                for tok in prefix_labels[:-1]:
                    _ = self.s4_model.recurrent_step(
                        torch.full((batch_size,), tok, dtype=torch.long, device=self.device)
                    )
                # Last prefix token becomes the starting input
                X_test = torch.full((batch_size,), prefix_labels[-1], dtype=torch.long, device=self.device)
                max_steps = self.sequence_length - consumed
            else:
                X_test = (
                    torch.zeros(batch_size, 1).to(torch.int) + self.token2label["[BEG]"]
                )
                X_test = X_test.to(self.device)
                X_test = X_test[:, 0]
                max_steps = self.sequence_length

            # Initialize constraint trackers
            end_id = self.token2label.get("[END]", None)
            rpar = self.token2label.get(")", None)
            lpar = self.token2label.get("(", None)

            # Track continuation length (for min length constraint)
            cont_len = torch.zeros(batch_size, dtype=torch.int, device=self.device)
            MIN_CONT_TOKENS = 3

            # Track open parentheses count (for parenthesis masking)
            open_paren = torch.zeros(batch_size, dtype=torch.int, device=self.device)

            batch_designs, batch_likelihoods = list(), list()
            for __ in range(max_steps):
                preds = self.s4_model.recurrent_step(X_test)

                # Apply constraints to logits
                # 1) Min continuation length: block [END] until MIN_CONT_TOKENS reached
                if end_id is not None and required_prefix_smiles:
                    not_enough = (cont_len < MIN_CONT_TOKENS)
                    preds[not_enough, end_id] = -1e9

                # 2) Parenthesis masking: block ')' when no open '(' exists
                if rpar is not None:
                    zero_mask = (open_paren == 0)
                    preds[zero_mask, rpar] = -1e9

                softmax_preds = F.softmax(preds, dim=-1).detach().cpu().numpy().tolist()
                preds = preds.detach().cpu().numpy().tolist()
                token_labels, token_likelihoods = list(), list()
                for pred_idx, pred in enumerate(preds):
                    pred_temperature = np.exp(np.array(pred) / temperature).tolist()
                    pred_sum = sum(pred_temperature)
                    pred_normed = [p / pred_sum for p in pred_temperature]
                    probas = np.random.multinomial(1, pred_normed)
                    token_label = np.argmax(probas)
                    token_labels.append(token_label)

                    token_likelihood = softmax_preds[pred_idx][token_label]
                    token_likelihoods.append(token_likelihood)

                batch_designs.append(token_labels)
                batch_likelihoods.append(token_likelihoods)
                X_test = torch.tensor(token_labels).to(self.device)

                # Update constraint trackers
                if required_prefix_smiles:
                    cont_len += 1

                if lpar is not None:
                    open_paren += (X_test == lpar).to(torch.int)
                if rpar is not None:
                    dec = ((X_test == rpar) & (open_paren > 0)).to(torch.int)
                    open_paren -= dec

            designs.append(np.array(batch_designs).T)
            likelihoods.append(np.array(batch_likelihoods).T)

        designs = np.concatenate(designs, axis=0).tolist()

        molecules = [
            [
                self.label2token[label]
                for label in design
                if self.label2token[label] not in ["[BEG]", "[END]", "[PAD]"]
            ]
            for design in designs
        ]

        # Prepend prefix tokens (excluding [BEG]) - they were fed via Prefix Replay but not sampled
        if required_prefix_smiles:
            prefix_tokens_clean = prefix_tokens[1:]  # Remove [BEG]
            molecules = [prefix_tokens_clean + mol for mol in molecules]

        molecule_lens = [
            len(molecule) + 2 for molecule in molecules
        ]  # +2 for [BEG] and [END]
        smiles = ["".join(molecule) for molecule in molecules]
        loglikelihoods = np.log(np.concatenate(likelihoods, axis=0)).tolist()
        mean_loglikelihoods = [
            np.mean(ll[: mol_len - 1])
            for ll, mol_len in zip(loglikelihoods, molecule_lens)
        ]

        return smiles, mean_loglikelihoods

    def rollout(
        self,
        batch_size: int,
        max_len: int,
        required_prefix_smiles: Optional[str] = None,
        temperature: float = 1.0
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[str]]:
        """Gradient-enabled rollout sampling for RL training.

        Parameters
        ----------
        batch_size : int
            The batch size.
        max_len : int
            Maximum sequence length to generate.
        required_prefix_smiles : Optional[str]
            Optional prefix SMILES to prepend to all generated molecules.
        temperature : float
            Sampling temperature.

        Returns
        -------
        Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[str]]
            actions: List of sampled token IDs [T, B]
            logprobs: List of log probabilities [T, B]
            masks: List of validity masks [T, B]
            smiles: List of decoded SMILES strings (with prefix)
        """
        from torch.distributions import Categorical
        from .smiles_utils import segment_smiles

        if self.token2label is None or self.label2token is None:
            raise ValueError("This model is untrained.")

        self.s4_model = self.s4_model.to(self.device)
        for module in self.s4_model.modules():
            if hasattr(module, "setup_step"):
                module.setup_step()

        # KEY: eval mode required by S4 step(), gradients preserved by NOT using torch.no_grad
        self.s4_model.eval()

        # Prefix handling (reuse design_molecules logic)
        prefix_tokens = []
        prefix_labels = []
        consumed = 1

        if required_prefix_smiles:
            prefix_tokens = ['[BEG]'] + segment_smiles(required_prefix_smiles)
            missing = [t for t in prefix_tokens if t not in self.token2label]
            if missing:
                raise ValueError(f'Prefix contains OOV tokens: {missing}')
            prefix_labels = [self.token2label[t] for t in prefix_tokens]
            consumed = len(prefix_tokens)

        # Reset state first
        self.s4_model.reset_state(batch_size, device=self.device)

        # Initialize from prefix or [BEG]
        if required_prefix_smiles:
            # Prefix Replay: feed prefix tokens into model state sequentially
            for tok in prefix_labels[:-1]:
                _ = self.s4_model.recurrent_step(
                    torch.full((batch_size,), tok, dtype=torch.long, device=self.device)
                )
            # Last prefix token becomes the starting input
            X_test = torch.full((batch_size,), prefix_labels[-1], dtype=torch.long, device=self.device)
        else:
            X_test = torch.full((batch_size,), self.token2label["[BEG]"], dtype=torch.long, device=self.device)

        max_steps = min(max_len, self.sequence_length - consumed)

        # Initialize constraint trackers
        end_id = self.token2label.get("[END]", None)
        pad_id = self.token2label.get("[PAD]", -1)
        rpar = self.token2label.get(")", None)
        lpar = self.token2label.get("(", None)

        # Track continuation length (for min length constraint)
        cont_len = torch.zeros(batch_size, dtype=torch.int, device=self.device)
        MIN_CONT_TOKENS = 3

        # Track open parentheses count (for parenthesis masking)
        open_paren = torch.zeros(batch_size, dtype=torch.int, device=self.device)

        actions = []
        logprobs = []
        masks = []

        # Stepwise sampling (NO torch.no_grad)
        for t in range(max_steps):
            # Forward pass
            logits = self.s4_model.recurrent_step(X_test)  # [B, vocab_size]

            # Temperature scaling
            logits = logits / temperature

            # Apply constraints to logits
            # 1) Min continuation length: block [END] until MIN_CONT_TOKENS reached
            if end_id is not None and required_prefix_smiles:
                not_enough = (cont_len < MIN_CONT_TOKENS)
                logits[not_enough, end_id] = -1e9

            # 2) Parenthesis masking: block ')' when no open '(' exists
            if rpar is not None:
                zero_mask = (open_paren == 0)
                logits[zero_mask, rpar] = -1e9

            # Sample
            dist = Categorical(logits=logits)
            action = dist.sample()  # [B]
            logprob = dist.log_prob(action)  # [B]

            # Record
            actions.append(action)
            logprobs.append(logprob)

            # Mask (detect [END]/[PAD])
            mask = (action != end_id) & (action != pad_id) if end_id is not None else (action != pad_id)
            masks.append(mask)

            # Update constraint trackers
            if required_prefix_smiles:
                cont_len += 1

            if lpar is not None:
                open_paren += (action == lpar).to(torch.int)
            if rpar is not None:
                dec = ((action == rpar) & (open_paren > 0)).to(torch.int)
                open_paren -= dec

            # Update input
            X_test = action

        # Decode to SMILES
        actions_np = [a.detach().cpu().numpy() for a in actions]
        actions_transposed = list(zip(*actions_np))  # [B, T]

        smiles_list = []
        for seq in actions_transposed:
            tokens = [self.label2token[int(label)]
                     for label in seq
                     if self.label2token[int(label)] not in ["[BEG]", "[END]", "[PAD]"]]

            # Prepend prefix
            if required_prefix_smiles:
                prefix_clean = prefix_tokens[1:]  # Remove [BEG]
                tokens = prefix_clean + tokens

            smiles_list.append("".join(tokens))

        return actions, logprobs, masks, smiles_list

    @torch.no_grad()
    def compute_molecule_loglikelihoods(
        self, molecules: List[List[str]], batch_size: int
    ) -> List[float]:
        """Computes the log-likelihoods of a list of molecules. The molecules are processed in batches of size `batch_size`.
        The log-likelihoods are returned as a list.

        Parameters
        ----------
        molecules : List[List[str]]
            A list of SMILES strings.
            The input molecules are tokenized and padded (or truncated) internally to the sequence length used during training.
        batch_size : int
            The batch size to use during computation.

        Returns
        -------
        List[float]
            A list of log-likelihoods.
        """
        tokenized_molecules = [
            ["[BEG]"] + smiles_utils.segment_smiles(smiles) + ["[END]"]
            for smiles in molecules
        ]
        padded_molecules = smiles_utils.pad_sequences(
            tokenized_molecules, self.sequence_length + 1, padding_value="[PAD]"
        )
        label_encoded_molecules = [
            [self.token2label[token] for token in tokens] for tokens in padded_molecules
        ]

        self.s4_model = self.s4_model.to(self.device)
        for module in self.s4_model.modules():
            if hasattr(module, "setup_step"):
                module.setup_step()

        self.s4_model.eval()
        n_batches = math.ceil(len(molecules) / batch_size)
        all_sequence_loglikelihoods = list()
        for batch_idx in range(n_batches):
            batch_start_idx = batch_idx * batch_size
            batch_end_idx = (batch_idx + 1) * batch_size
            molecule_batch = label_encoded_molecules[batch_start_idx:batch_end_idx]
            self.s4_model.reset_state(
                batch_size=len(molecule_batch), device=self.device
            )

            batch_loglikelihoods = list()
            for label_idx in range(self.sequence_length):
                labels = [molecule[label_idx] for molecule in molecule_batch]
                X_test = torch.tensor(labels, dtype=torch.int).to(self.device)

                preds = self.s4_model.recurrent_step(X_test)
                softmax_preds = F.softmax(preds, dim=-1).detach().cpu().numpy().tolist()
                log_preds = np.log(softmax_preds)

                next_token_labels = [
                    molecule[label_idx + 1] for molecule in molecule_batch
                ]
                log_likelihoods = [
                    log_pred[nt_label]
                    for nt_label, log_pred in zip(next_token_labels, log_preds)
                ]
                batch_loglikelihoods.append(log_likelihoods)

            batch_loglikelihoods = np.array(batch_loglikelihoods).T.tolist()
            molecule_lengths = [
                len(molecule)
                for molecule in tokenized_molecules[batch_start_idx:batch_end_idx]
            ]
            batch_sequence_loglikelihoods = [
                np.mean(ll[: mol_len - 1])
                for ll, mol_len in zip(batch_loglikelihoods, molecule_lengths)
            ]
            all_sequence_loglikelihoods.extend(batch_sequence_loglikelihoods)

        return all_sequence_loglikelihoods

    def save(self, path: str):
        """Saves the model to a directory. The directory will be created if it does not exist.

        Parameters
        ----------
        path : str
            The directory to save the model to.
        """
        print("Saving model to", path)
        os.makedirs(path, exist_ok=True)
        torch.save(self.s4_model.state_dict(), f"{path}/model.pt")
        properties = {p: v for p, v in self.__dict__.items() if p != "s4_model"}

        with open(f"{path}/init_arguments.json", "w") as f:
            json.dump(properties, f, indent=4)
