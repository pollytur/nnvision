import warnings
from functools import partial

import numpy as np
import torch
from tqdm import tqdm
from collections import Iterable

from neuralpredictors.measures import *
from neuralpredictors import measures as mlmeasures
from neuralpredictors.training import (
    early_stopping,
    MultipleObjectiveTracker,
    eval_state,
    cycle_datasets,
    Exhauster,
    LongCycler,
)
from nnfabrik.utility.nn_helpers import set_random_seed
try:
    from cnexp.lrschedule import CosineAnnealingSchedule, LinearAnnealingSchedule
except:
    pass

from ..utility import measures
from ..utility.measures import get_correlations, get_poisson_loss

try:
    import wandb
except ImportError:
    print("wandb not installed, not logging to wandb")
from sklearn.cluster import KMeans
from torch.nn import KLDivLoss
import math
torch.pi = math.pi
import pickle


# todo - add Nina's loss and wandb tracker here
def nnvision_trainer(
    model,
    dataloaders,
    seed,
    avg_loss=False,
    scale_loss=True,  # trainer args
    loss_function="PoissonLoss",
    stop_function="get_correlations",
    loss_accum_batch_n=None,
    device="cuda",
    verbose=True,
    interval=1,
    patience=5,
    epoch=0,
    lr_init=0.005,  # early stopping args
    max_iter=100,
    maximize=True,
    tolerance=1e-6,
    restore_best=True,
    lr_decay_steps=3,
    lr_decay_factor=0.3,
    min_lr=0.0001,  # lr scheduler args
    cb=None,
    track_training=False,
    return_test_score=False,
    batchping=1000,
    adamw=False,
    wandb_logger=None,
    include_kldivergence=True,
    cluster_number=10,
    alpha=1.0,
    dec_starting_epoch=5,
    kmeans_init=20,
    base_multiplier=1e3,
    use_diag_cov=True,
    # learn_alpha=False,
    exponent=2,
    attention_readout=False,
    save_checkpoint_path=None,
    save_checkpoint_every=5,
    # ignore_neurones=None,
    **kwargs,
):
    """

    Args:
        model:
        dataloaders:
        seed:
        avg_loss:
        scale_loss:
        loss_function:
        stop_function:
        loss_accum_batch_n:
        device:
        verbose:
        interval:
        patience:
        epoch:
        lr_init:
        max_iter:
        maximize:
        tolerance:
        restore_best:
        lr_decay_steps:
        lr_decay_factor:
        min_lr:
        cb:
        track_training:
        **kwargs:

    Returns:

    """

    def get_multiplier(epoch, base_multiplier=4e3):
        """Multiplier to scale KL loss in same order of magnitude as main loss
        To avoid hard peek aat starting epoch we include a warm-up phase s.t. the loss can increase slower
        """
        if epoch < dec_starting_epoch:
            return 0
        else:
            return base_multiplier

    def target_distribution(batch: torch.Tensor, exponent=exponent) -> torch.Tensor:
        """
        Compute the target distribution p_ij, given the batch (q_ij), as in 3.1.3 Equation 3 of
        Xie/Girshick/Farhadi; this is used the KL-divergence loss function.
        p_ij = (q_ij^2/f_j) / sum_j'(q_ij'^2/f_j')  f_j =sum_i(q_ij)

        :param batch: [batch size, number of clusters] Tensor of dtype float
        :return: [batch size, number of clusters] Tensor of dtype float
        """
        weight = (batch**exponent) / torch.sum(batch, 0)
        return (weight.t() / torch.sum(weight, 1)).t()


    def soft_assignments_mult(encoded_features, cluster_centers, sigma, alpha, p=1):
        sigma_inv = 1.0 / sigma  # (K, D)
        diff = encoded_features.T.unsqueeze(1) - cluster_centers.unsqueeze(0)  # (N, K, D)
        norm_sigma = torch.sum(diff * sigma_inv * diff, dim=2)  # (N, K)
        det = torch.sum(torch.log(sigma), dim=1)  # log(det) since sigma is diagonal
        log_gamma_top = torch.lgamma((alpha + p) / 2)
        log_gamma_bottom = torch.lgamma(alpha / 2)
        # Log-density formula for multivariate Student-t
        log_pdf = (
            log_gamma_top
            - log_gamma_bottom
            - 0.5 * det
            - (p / 2) * torch.log(alpha * torch.pi)
            - ((alpha + p) / 2) * torch.log(1 + (norm_sigma / alpha))
        )
        #log_pdf_max = torch.max(log_pdf, dim=1, keepdim=True)[0]  # Get max per row

        log_assignments = log_pdf - torch.logsumexp(log_pdf, dim=1, keepdim=True)
        return torch.exp(log_assignments)  # Convert log-assignments to probabilities

    def EM_t_mult(features, resp, cluster_centers, sigma, alpha, d=1):
        sigma_inv = 1.0 / sigma  # (K,)
        diff = features.T.unsqueeze(1) - cluster_centers.unsqueeze(0)
        norm_sigma = torch.sum((diff**2 * sigma_inv), 2)
        u = ((alpha + d) / (alpha + norm_sigma)).detach()  # ccalculate U shape(N,K)
        
        """ M step """
        numerator = torch.matmul(features, resp * u).T.detach()
        denominator = torch.sum(resp * u, dim=0, keepdim=True).T.detach()
        cluster_centers = numerator / denominator

        weighted_sq_diff = resp.unsqueeze(2) * u.unsqueeze(2) * (diff**2)  # (N, K, D)
        numerator = weighted_sq_diff.sum(dim=0)  # (K,D)
        denominator = torch.sum(resp, dim=0, keepdim=True)  # (K,)
        sigma = (numerator / denominator.T).detach()

        sigma = torch.clamp(sigma, min=1e-4, max=1e4)

        return cluster_centers, sigma

    def EM_t_1D(features, resp, cluster_centers, taus, alpha, d=1):
        norm_squared = torch.sum(
            (features.T.unsqueeze(1) - cluster_centers.unsqueeze(0)) ** 2, dim=2
        )
        u = (alpha + d) / (
            alpha + norm_squared * (taus ** (-1))
        )  # ccalculate U shape(N,K)

        """ M step """
        numerator = torch.matmul(features, resp * u).T.detach()
        denominator = torch.sum(resp * u, dim=0, keepdim=True).T.detach()
        cluster_centers = numerator / denominator

        weighted_sums = torch.sum(resp * u * norm_squared, dim=0)
        taus = (weighted_sums / torch.sum(resp, dim=0, keepdim=True)).detach()
        return cluster_centers, taus

    def soft_assignments_1D(encoded_features, cluster_centers, tau, alpha=1):
        norm_squared = torch.sum(
            (encoded_features.T.unsqueeze(1) - cluster_centers.unsqueeze(0)) ** 2, 2
        )
        assignments = 1.0 / (1.0 + (norm_squared / (alpha * tau)))
        assignments = (assignments ** ((alpha + 1) / 2)) / (tau**1 / 2)
        return assignments / torch.sum(assignments, dim=1, keepdim=True)

    def full_objective(model, dataloader, data_key, *args, **kwargs):
        """

        Args:
            model:
            dataloader:
            data_key:
            *args:

        Returns:

        """
        loss_scale = (
            np.sqrt(len(dataloader[data_key].dataset) / args[0].shape[0])
            if scale_loss
            else 1.0
        )
        preds = model(args[0].to(device), data_key=data_key, **kwargs)
        if "bools" in kwargs:
            preds = preds * kwargs["bools"].to(device)
        resps = args[1].to(device)
        return loss_scale * criterion(preds, resps) + model.regularizer(data_key)


    ##### Model training ####################################################################################################
    model.to(device)
    set_random_seed(seed)
    model.train()

    kldiv_criterion = KLDivLoss(
        size_average=False
    )  # losses are summed for each minibatch

    criterion = getattr(mlmeasures, loss_function)(avg=avg_loss)
    stop_closure = partial(
        getattr(measures, stop_function),
        dataloaders=dataloaders["validation"],
        device=device,
        per_neuron=False,
        avg=True,
    )

    n_iterations = len(LongCycler(dataloaders["train"]))

    if adamw:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr_init)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr_init)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max" if maximize else "min",
        factor=lr_decay_factor,
        patience=patience,
        threshold=tolerance,
        min_lr=min_lr,
        verbose=verbose,
        threshold_mode="abs",
    )

    # set the number of iterations over which you would like to accummulate gradients
    optim_step_count = (
        len(dataloaders["train"].keys())
        if loss_accum_batch_n is None
        else loss_accum_batch_n
    )
    print(f'optim_step_count={optim_step_count}')

    if track_training:
        tracker_dict = dict(
            correlation=partial(
                get_correlations,
                model=model,
                dataloaders=dataloaders["validation"],
                device=device,
                per_neuron=False,
            ),
            poisson_loss=partial(
                get_poisson_loss,
                model=model,
                dataloaders=dataloaders["validation"],
                device=device,
                per_neuron=False,
                avg=False,
            ),
        )
        if hasattr(model, "tracked_values"):
            tracker_dict.update(model.tracked_values)

        
        tracker = MultipleObjectiveTracker(**tracker_dict)
    else:
        tracker = None

    alpha = torch.tensor(alpha, device=device, requires_grad=False)
    kldiv_list = []
    print("Alpha: ", alpha)

    epoch_loss = 0
    epoch_loss_kldiv = 0
    epoch_loss_kldiv_without_scaling = 0
    # train over epochs
    for epoch, val_obj in early_stopping(
        model,
        stop_closure,
        interval=interval,
        patience=patience,
        start=epoch,
        max_iter=max_iter,
        maximize=maximize,
        tolerance=tolerance,
        restore_best=restore_best,
        tracker=tracker,
        scheduler=scheduler,
        lr_decay_steps=lr_decay_steps,
    ):
        epoch_loss_kldiv = 0
        epoch_loss_kldiv_without_scaling = 0
        if include_kldivergence and epoch == dec_starting_epoch:
            cluster_centers_list = []
            kmeans = KMeans(
                n_clusters=cluster_number, n_init=kmeans_init, random_state=seed
            )
            feature_list = []
            # form initial cluster centres
            with torch.no_grad():
                if attention_readout:
                    features = model.readout.all_sessions._features.cpu().detach().squeeze().T.numpy()
                else:
                    for k, readout in model.readout.items():
                        features = readout.features.cpu().detach().squeeze().T.numpy()
                        feature_list.append(np.array(features))

                if not attention_readout:
                    features = np.vstack(feature_list)
                predicted = kmeans.fit_predict(features)

            cluster_centers = torch.tensor(
                kmeans.cluster_centers_, dtype=torch.float, device=device
            )
            if use_diag_cov:
                p = features.shape[1]
                sigma = torch.zeros((cluster_number, p), device=device)
                for k in range(cluster_number):
                    cluster_points = torch.from_numpy(features[predicted == k]).to(
                        device
                    )
                    print(f"Points for cluster {k}: {cluster_points.shape[0]}")
                    if len(cluster_points) > 1:
                        sigma[k] = (
                            torch.var(cluster_points, dim=0, unbiased=True) + 1e-6
                        )
                    else:
                        sigma[k] += 1e-6

            else:
                sigma = torch.zeros(cluster_number, device=device)
                for k in range(cluster_number):
                    cluster_points = torch.from_numpy(features[predicted == k]).to(
                        device
                    )
                    if len(cluster_points) > 0:
                        sigma[k] = torch.mean(
                            torch.sum((cluster_points - cluster_centers[k]) ** 2, 1)
                        )
                sigma = sigma.unsqueeze(0)

        # print the quantities from tracker
        if verbose and tracker is not None:
            print("=======================================")
            for key in tracker.log.keys():
                print(key, tracker.log[key][-1], flush=True)

        # executes callback function if passed in keyword args
        if cb is not None:
            cb()

        # train over batches
        optimizer.zero_grad()
        for batch_no, (data_key, data) in tqdm(
            enumerate(LongCycler(dataloaders["train"])),
            total=n_iterations,
            desc="Epoch {}".format(epoch),
        ):
            if "deeplake" in str(type(data)):
                data_kwargs = data
            else:
                data_kwargs = data._asdict()
            loss = full_objective(
                model, dataloaders["train"], data_key, *list(data)[:2], **data_kwargs
            )
            loss.backward()
            if (batch_no + 1) % optim_step_count == 0:
                if include_kldivergence and epoch >= dec_starting_epoch:
                    kldiv_loss = torch.zeros(1).to(device)
                    if not attention_readout:
                        feature_list = []
                        for k, readout in model.readout.items():
                            features = readout.features.squeeze()
                            feature_list.append(features)
                        feature_list = torch.cat(feature_list, dim=1)
                    else:
                        feature_list = model.readout.all_sessions._features.squeeze()
                    if use_diag_cov:
                        # if torch.isnan(sigma).any() or torch.isnan(cluster_centers).any() or torch.isnan(feature_list).any():
                        #     print(f'batch_no={batch_no}, torch.isnan(sigma).any()={torch.isnan(sigma).any()},  alpha={alpha}, p={p}')
                        #     print(f'batch_no={batch_no}, torch.isnan(cluster_centers).any()={torch.isnan(cluster_centers).any()}')
                        #     print(f'batch_no={batch_no}, torch.isnan(feature_list).any() = {torch.isnan(feature_list).any()}')
                        q = soft_assignments_mult(
                            feature_list, cluster_centers, sigma, alpha, p
                        )
                    else:
                        q = soft_assignments_1D(
                            feature_list, cluster_centers, sigma, alpha
                        )
                    # if torch.isnan(q).any() or torch.isnan(q).all():
                    #     print(f'batch_no={batch_no}, torch.isnan(q).any()={torch.isnan(q).any()}, torch.isnan(q).all()={torch.isnan(q).all()}')
                    target = target_distribution(q, exponent)
                    # if torch.isnan(target).any():
                    #     print(f'batch_no={batch_no}, torch.isnan(target).any()={torch.isnan(target).any()}')
                    target = target.clamp(min=1e-10)
                    q = q.clamp(min=1e-10)

                    kldiv_loss = get_multiplier(epoch, base_multiplier) * (
                        kldiv_criterion(q.log(), target)
                    )
                    # To avoid underflow issues when computing this quantity, this loss expects the argument input in the log-space.
                    # https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html
                    kldiv_loss.backward()
                    epoch_loss_kldiv += kldiv_loss.detach()
                    epoch_loss_kldiv_without_scaling += (
                        kldiv_loss.detach() / get_multiplier(epoch, base_multiplier)
                    )
                    epoch_loss += kldiv_loss.detach()
                    with torch.no_grad():
                        cluster_centers_list.append(cluster_centers.cpu().detach())
                        kldiv_list.append(
                            kldiv_loss.cpu() / get_multiplier(epoch, base_multiplier)
                        )

                    if use_diag_cov:
                        cluster_centers, sigma = EM_t_mult(
                            feature_list, q, cluster_centers, sigma, alpha, p
                        )
                    else:
                        cluster_centers, sigma = EM_t_1D(
                            feature_list, q, cluster_centers, sigma, alpha
                        )
                optimizer.step()
                optimizer.zero_grad()
            if (batch_no % batchping == 0) and (cb is not None):
                cb()
        if wandb_logger is not None:
            tracker_info = tracker.asdict(make_copy=True)
            wandb.log(
                {
                    "train_loss": loss.item(),
                    "lr": optimizer.param_groups[0]["lr"],
                    "epoch": epoch,
                    "Batch": batch_no,
                    "val_corr": tracker_info["correlation"][-1],
                    "val_poisson_loss": tracker_info["poisson_loss"][-1],
                    "Epoch Train loss Kullback-Leibler-divergence": epoch_loss_kldiv,
                    "Epoch Train loss KL without scaling main": epoch_loss_kldiv_without_scaling,
                }
            )
        if save_checkpoint_path is not None and epoch > 0 and epoch % save_checkpoint_every == 0:
            torch.save(model.state_dict(), f'{save_checkpoint_path}epoch_{epoch}.pth')

    ##### Model evaluation ####################################################################################################
    model.eval()
    tracker.finalize() if track_training else None

    # Compute avg validation and test correlation
    validation_correlation = get_correlations(
        model, dataloaders["validation"], device=device, as_dict=False, per_neuron=False
    )
    if return_test_score:
        test_correlation = get_correlations(
            model, dataloaders["test"], device=device, as_dict=False, per_neuron=False
        )

    # return the whole tracker output as a dict
    output = {k: v for k, v in tracker.log.items()} if track_training else {}
    output["validation_corr"] = validation_correlation

    if include_kldivergence:
        if not attention_readout:
            soft_assignments_list = []
            for k, readout in model.readout.items():
                features = readout.features.detach().squeeze()
                soft_assignments_list.append(
                    soft_assignments_mult(features, cluster_centers, sigma, alpha, p)
                )
            predicted = torch.cat(soft_assignments_list).max(1)[1]
        else:
            features = model.readout.all_sessions._features.squeeze()
            predicted = soft_assignments_mult(features, cluster_centers, sigma, alpha, p).max(1)[1]
        # append final cluster_centers
        cluster_centers_list.append(cluster_centers.cpu().detach().numpy())
        cluster_centers_np = np.array(cluster_centers_list)
        print("Alpha: ", alpha)
        output['cluster_centers_np'] = cluster_centers_np
        output['predicted'] = predicted


    score = (
        np.mean(test_correlation)
        if return_test_score
        else np.mean(validation_correlation)
    )
    if wandb_logger is not None:
        wandb.finish()
    if save_checkpoint_path is not None:
            torch.save(model.state_dict(), f'{save_checkpoint_path}best.pth')
            with open(f'{save_checkpoint_path}output_dict.pkl', 'wb') as f:
                pickle.dump(output, f)
    return score, output, model.state_dict()


def finetune_trainer(
    model,
    dataloaders,
    seed,
    avg_loss=False,
    scale_loss=True,  # trainer args
    loss_function="PoissonLoss",
    stop_function="get_correlations",
    loss_accum_batch_n=None,
    device="cuda",
    verbose=True,
    interval=1,
    patience=5,
    epoch=0,
    lr_init=0.005,  # early stopping args
    max_iter=100,
    maximize=True,
    tolerance=1e-6,
    restore_best=True,
    lr_decay_steps=3,
    lr_decay_factor=0.3,
    min_lr=0.0001,  # lr scheduler args
    cb=None,
    track_training=False,
    return_test_score=False,
    fine_tune="sequential",
    **kwargs,
):
    def full_objective(model, dataloader, data_key, *args):

        loss_scale = (
            np.sqrt(len(dataloader[data_key].dataset) / args[0].shape[0])
            if scale_loss
            else 1.0
        )
        return loss_scale * criterion(
            model(args[0].to(device), data_key=data_key), args[1].to(device)
        ) + model.regularizer(data_key)

    ##### Model training ####################################################################################################
    model.to(device)
    set_random_seed(seed)
    model.train()

    criterion = getattr(mlmeasures, loss_function)(avg=avg_loss)
    stop_closure = partial(
        getattr(measures, stop_function),
        dataloaders=dataloaders["validation"],
        device=device,
        per_neuron=False,
        avg=True,
    )

    n_iterations = len(LongCycler(dataloaders["train"]))

    # set the number of iterations over which you would like to accummulate gradients
    optim_step_count = (
        len(dataloaders["train"].keys())
        if loss_accum_batch_n is None
        else loss_accum_batch_n
    )

    if track_training:
        tracker_dict = dict(
            correlation=partial(
                get_correlations,
                model=model,
                dataloaders=dataloaders["validation"],
                device=device,
                per_neuron=False,
            ),
            poisson_loss=partial(
                get_poisson_loss,
                model=model,
                dataloaders=dataloaders["validation"],
                device=device,
                per_neuron=False,
                avg=False,
            ),
        )
        if hasattr(model, "tracked_values"):
            tracker_dict.update(model.tracked_values)
        tracker = MultipleObjectiveTracker(**tracker_dict)
    else:
        tracker = None

    if fine_tune == "sequential":
        parameters_to_train = [model.readout.parameters(), model.parameters()]
    elif fine_tune == "full":
        parameters_to_train = [model.parameters()]
    elif fine_tune == "core":
        parameters_to_train = [model.core.parameters()]
    elif fine_tune == "readout":
        parameters_to_train = [model.readout.parameters()]

    for i, parameters in enumerate(parameters_to_train):
        if isinstance(lr_init, Iterable):
            lr = lr_init[i]
        print(f"training with lr = {lr}")
        optimizer = torch.optim.Adam(parameters, lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max" if maximize else "min",
            factor=lr_decay_factor,
            patience=patience,
            threshold=tolerance,
            min_lr=min_lr,
            verbose=verbose,
            threshold_mode="abs",
        )
        # train over epochs
        for epoch, val_obj in early_stopping(
            model,
            stop_closure,
            interval=interval,
            patience=patience,
            start=0,
            max_iter=max_iter,
            maximize=maximize,
            tolerance=tolerance,
            restore_best=restore_best,
            tracker=tracker,
            scheduler=scheduler,
            lr_decay_steps=lr_decay_steps,
        ):

            # print the quantities from tracker
            if verbose and tracker is not None:
                print("=======================================")
                for key in tracker.log.keys():
                    print(key, tracker.log[key][-1], flush=True)

            # executes callback function if passed in keyword args
            if cb is not None:
                cb()

            # train over batches
            optimizer.zero_grad()
            for batch_no, (data_key, data) in tqdm(
                enumerate(LongCycler(dataloaders["train"])),
                total=n_iterations,
                desc="Epoch {}".format(epoch),
            ):

                loss = full_objective(model, dataloaders["train"], data_key, *data)
                loss.backward()
                if (batch_no + 1) % optim_step_count == 0:
                    optimizer.step()
                    optimizer.zero_grad()

    ##### Model evaluation ####################################################################################################
    model.eval()
    tracker.finalize() if track_training else None

    # Compute avg validation and test correlation
    validation_correlation = get_correlations(
        model, dataloaders["validation"], device=device, as_dict=False, per_neuron=False
    )
    test_correlation = get_correlations(
        model, dataloaders["test"], device=device, as_dict=False, per_neuron=False
    )

    # return the whole tracker output as a dict
    output = {k: v for k, v in tracker.log.items()} if track_training else {}
    output["validation_corr"] = validation_correlation

    score = (
        np.mean(test_correlation)
        if return_test_score
        else np.mean(validation_correlation)
    )
    return score, output, model.state_dict()


def multistep_trainer(
    model,
    dataloaders,
    seed,
    avg_loss=False,
    scale_loss=True,  # trainer args
    loss_function="PoissonLoss",
    stop_function="get_correlations",
    loss_accum_batch_n=None,
    device="cuda",
    interval=1,
    patience=5,
    maximize=True,
    tolerance=1e-4,
    restore_best=True,
    lr_decay_steps=4,
    cb=None,
    track_training=False,
    return_test_score=False,
    lr1=5e-4,
    lr2=1e-5,
    n1=200,
    n2=200,
    disable_tqdm=True,
    **kwargs,
):
    def full_objective(model, dataloader, data_key, *args, **kwargs):
        """

        Args:
            model:
            dataloader:
            data_key:
            *args:

        Returns:

        """
        loss_scale = (
            np.sqrt(len(dataloader[data_key].dataset) / args[0].shape[0])
            if scale_loss
            else 1.0
        )
        preds = model(args[0].to(device), data_key=data_key, **kwargs)
        if "bools" in kwargs:
            preds = preds * kwargs["bools"].to(device)
        resps = args[1].to(device)
        return loss_scale * criterion(preds, resps) + model.regularizer(data_key)

    model.to(device)
    set_random_seed(seed)
    model.train()

    criterion = getattr(mlmeasures, loss_function)(avg=avg_loss)
    stop_closure = partial(
        getattr(measures, stop_function),
        dataloaders=dataloaders["validation"],
        device=device,
        per_neuron=False,
        avg=True,
    )

    n_iterations = len(LongCycler(dataloaders["train"]))

    # set the number of iterations over which you would like to accummulate gradients
    optim_step_count = (
        len(dataloaders["train"].keys())
        if loss_accum_batch_n is None
        else loss_accum_batch_n
    )

    # Step 1: train readout
    tracker = None
    model.core.requires_grad_(False)
    model.readout.requires_grad_(True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr1)
    scheduler = CosineAnnealingSchedule(opt=optimizer, n_epochs=n1)

    # train over epochs
    for epoch, val_obj in early_stopping(
        model,
        stop_closure,
        interval=interval,
        patience=patience,
        start=0,
        max_iter=n1,
        maximize=maximize,
        tolerance=tolerance,
        restore_best=restore_best,
        tracker=tracker,
        scheduler=scheduler,
        lr_decay_steps=lr_decay_steps,
    ):
        # executes callback function if passed in keyword args
        if cb is not None:
            cb()
        # train over batches
        optimizer.zero_grad()
        for batch_no, (data_key, data) in tqdm(
            enumerate(LongCycler(dataloaders["train"])),
            total=n_iterations,
            desc="Epoch {}".format(epoch),
            disable=disable_tqdm,
        ):

            loss = full_objective(
                model, dataloaders["train"], data_key, *data[:2], **data._asdict()
            )
            loss.backward()
            if (batch_no + 1) % optim_step_count == 0:
                optimizer.step()
                optimizer.zero_grad()

    # Step 2: finetune entire model
    model.core.requires_grad_(True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr2)
    scheduler = CosineAnnealingSchedule(opt=optimizer, n_epochs=n2)

    # train over epochs
    for epoch, val_obj in early_stopping(
            model,
            stop_closure,
            interval=interval,
            patience=patience,
            start=0,
            max_iter=n2,
            maximize=maximize,
            tolerance=tolerance,
            restore_best=restore_best,
            tracker=tracker,
            scheduler=scheduler,
            lr_decay_steps=lr_decay_steps,
    ):
        # executes callback function if passed in keyword args
        if cb is not None:
            cb()
        # train over batches
        optimizer.zero_grad()
        for batch_no, (data_key, data) in tqdm(
                enumerate(LongCycler(dataloaders["train"])),
                total=n_iterations,
                desc="Epoch {}".format(epoch),
                disable=disable_tqdm,
        ):

            loss = full_objective(model, dataloaders["train"], data_key, *data)
            loss.backward()
            if (batch_no + 1) % optim_step_count == 0:
                optimizer.step()
                optimizer.zero_grad()

    ##### Model evaluation ####################################################################################################
    model.eval()
    tracker.finalize() if track_training else None

    # Compute avg validation and test correlation
    validation_correlation = get_correlations(
        model, dataloaders["validation"], device=device, as_dict=False, per_neuron=False
    )
    test_correlation = get_correlations(
        model, dataloaders["test"], device=device, as_dict=False, per_neuron=False
    )

    # return the whole tracker output as a dict
    output = {k: v for k, v in tracker.log.items()} if track_training else {}
    output["validation_corr"] = validation_correlation

    score = (
        np.mean(test_correlation)
        if return_test_score
        else np.mean(validation_correlation)
    )
    return score, output, model.state_dict()


def shared_readout_trainer(model, dataloaders, seed, uid=None, cb=None):
    score = 0
    output = 0
    model_state = model.state_dict()

    return score, output, model_state
