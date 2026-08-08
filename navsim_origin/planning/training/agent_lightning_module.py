import pytorch_lightning as pl
import torch
from torch import Tensor
from typing import Dict, Tuple, List

from navsim.common.dataclasses import Trajectory
from navsim.agents.abstract_agent import AbstractAgent


def _rowwise_isin(tensor_1: torch.Tensor, target_tensor: torch.Tensor) -> torch.Tensor:
    matches = tensor_1[:, None] == target_tensor
    return torch.sum(matches, dim=1, dtype=torch.bool)


class AgentLightningModule(pl.LightningModule):
    """Pytorch lightning wrapper for learnable agent."""

    def __init__(self, agent: AbstractAgent, for_viz=False):
        super().__init__()
        self.agent = agent
        self.checkpoint_file = None
        self.for_viz = for_viz

    def _step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], logging_prefix: str) -> Tensor:
        features, targets = batch
        prediction = self.agent.forward(features)
        loss_dict = self.agent.compute_loss(features, targets, prediction)

        if isinstance(loss_dict, dict):
            for key, value in loss_dict.items():
                self.log(
                    f"{logging_prefix}/{key}",
                    value,
                    on_step=True,
                    on_epoch=False,
                    prog_bar=True,
                    sync_dist=True,
                )
            return loss_dict["loss"]
        return loss_dict

    def training_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int) -> Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int):
        if "drivor" in self.agent.name() or "DrivoR" in self.agent.name():
            features, targets = batch
            predictions = self.agent.forward(features)
            all_chosen_trajectories = predictions["trajectory"][:, None]
            all_proposed_trajectories = predictions["proposals"]
            final_score, _, proposal_scores, l2, trajectory_scores = self.agent.compute_score(
                targets, all_chosen_trajectories
            )
            _, best_score, all_proposal_scores, _, _ = self.agent.compute_score(targets, all_proposed_trajectories)
            mean_score = proposal_scores.mean()

            logging_prefix = "val"
            if "pdm_score" in predictions:
                pdm_score = predictions["pdm_score"]
                best_pred_score_values = pdm_score[torch.arange(len(pdm_score)), torch.argmax(pdm_score, dim=1)]
                score_error = torch.abs(best_pred_score_values - proposal_scores).mean()
                self.log(
                    f"{logging_prefix}/score_error",
                    score_error,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    sync_dist=True,
                )

                best_pred_score_index = torch.argmax(pdm_score, dim=1)
                best_real_score_index = torch.argmax(all_proposal_scores, dim=1)
                score_hit_rate = torch.mean(best_pred_score_index == best_real_score_index, dtype=torch.float32)

                best_possible_scores = all_proposal_scores[torch.arange(len(all_proposal_scores)), best_real_score_index]
                best_actual_scores = all_proposal_scores[torch.arange(len(all_proposal_scores)), best_pred_score_index]
                lost_score = torch.mean(best_possible_scores - best_actual_scores)
                self.log(
                    f"{logging_prefix}/score_hit_rate",
                    score_hit_rate,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    sync_dist=True,
                )
                self.log(
                    f"{logging_prefix}/lost_score",
                    lost_score,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    sync_dist=True,
                )

                top_5_indices_real = torch.topk(all_proposal_scores, k=5, dim=1).indices
                top_5_score_hit_rate = _rowwise_isin(best_pred_score_index, top_5_indices_real).mean(dtype=torch.float32)
                self.log(
                    f"{logging_prefix}/top_5_score_hit_rate",
                    top_5_score_hit_rate,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    sync_dist=True,
                )

            self.log(f"{logging_prefix}/score", final_score, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log(f"{logging_prefix}/best_score", best_score, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log(f"{logging_prefix}/mean_score", mean_score, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log(f"{logging_prefix}/l2", l2, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log(f"{logging_prefix}/collision", trajectory_scores[:, 0].mean(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log(f"{logging_prefix}/dac", trajectory_scores[:, 1].mean(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log(f"{logging_prefix}/progress", trajectory_scores[:, 2].mean(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log(f"{logging_prefix}/ttc", trajectory_scores[:, 3].mean(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log(f"{logging_prefix}/comfort", trajectory_scores[:, 4].mean(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

            return final_score
        return self._step(batch, "val")

    def configure_optimizers(self):
        return self.agent.get_optimizers()

    def predict_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int):
        return self.predict_step_drivor(batch, batch_idx)

    def predict_step_drivor(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor], List[str]], batch_idx: int):
        if len(batch) == 3:
            features, targets, tokens = batch
        else:
            features, targets = batch
            tokens = features["scenario_token"]
        self.agent.eval()
        with torch.no_grad():
            predictions = self.agent.forward(features)
            poses = predictions["trajectory"]
            if self.for_viz:
                all_proposed_trajectories = predictions["proposal_list"]
                final_trajectories = predictions["proposals"]
                can_compute_scores = (
                    getattr(self.agent, "test_metric_cache_paths", None) is not None
                    or getattr(self.agent, "train_metric_cache_paths", None) is not None
                )
                if can_compute_scores:
                    _, _, final_scores, _, _ = self.agent.compute_score(targets, final_trajectories)
                else:
                    final_scores = torch.zeros(
                        final_trajectories.shape[0], final_trajectories.shape[1], dtype=torch.float32
                    )
                ego_status = features["ego_status"]
                gt_bev_maps = targets.get("bev_semantic_map")
                pred_bev_logits = predictions.get("bev_semantic_map")
                pred_bev_maps = pred_bev_logits.argmax(dim=1) if pred_bev_logits is not None else None
                scene_names = targets.get("scene_name")
        result = {}
        for index, (pose, token) in enumerate(zip(poses.cpu().numpy(), tokens)):
            proposal = Trajectory(pose)
            if self.for_viz:
                proposal_list = [proposal_list[index].cpu().numpy() for proposal_list in all_proposed_trajectories]
                result[token] = {
                    "trajectory": proposal,
                    "all_proposals": proposal_list,
                    "all_proposal_scores": final_scores[index],
                    "high_level_command": ego_status[index],
                    "gt_bev_semantic_map": gt_bev_maps[index].cpu().numpy() if gt_bev_maps is not None else None,
                    "pred_bev_semantic_map": pred_bev_maps[index].cpu().numpy() if pred_bev_maps is not None else None,
                    "scene_name": scene_names[index] if scene_names is not None else None,
                }
            else:
                result[token] = {"trajectory": proposal}
        return result
