import torch
import torch.nn.functional as F

from .base import BaseExplainer, hamming_distance


class HeuristicExplainer(BaseExplainer):
    """
    Simple embedding-based heuristic baseline.

    Bring-in:
        Replace the j most recent items with the nearest
        neighbours of the target item in embedding space.

    Push-out:
        Replace the j most recent items with items that are
        far away from the original top-1 recommendation
        in embedding space.

    Try j = 1, ..., 5 and stop at the first valid CF.
    """

    def explain(
        self,
        original_seq,
        original_rec,
        target_rec=None,
        rank_constraint=None,
    ):
        self.reset_query_count()

        constraint = self.resolve_rank_constraint(
            target_rec=target_rec,
            rank_constraint=rank_constraint,
        )

        if constraint is None:
            return None, {
                "method": "heuristic",
                "valid": False,
                "queries_used": 0,
                "top_k_cfs": [],
            }

        # Filter out the embedding positions
        positions = [
            p for p, item in enumerate(original_seq)
            if item != 0
        ]

        max_attempts = min(5, self.max_edits, len(positions), self.query_budget)

        # Retrieve embedding and normalize them like trace to keep things consistent
        with torch.no_grad():
            embeddings = self.model.item_emb.weight[
                1:self.num_items + 1
            ].detach().float()

            embeddings = F.normalize(embeddings, dim=-1)

        if constraint.kind == "bring_in":
            reference_item = constraint.item

        else:
            reference_item = original_rec

        target_embedding = embeddings[reference_item - 1]

        # Because we normalized the embeddings, the cosine similarity is equivalant with performing the dot product
        similarities = embeddings @ target_embedding

        seen_items = list(original_seq)
        forbidden_items = list(self.forbidden_items)
        padding_token = [0]

        excluded = seen_items + forbidden_items + padding_token

        for item in excluded:
            # We added this branch to check whether the item id is valid, otherwise an error can occur when indexing
            if 1 <= item <= self.num_items:
                similarities[item - 1] = -float("inf")

        if constraint.kind == "bring_in":
            # Sort the embeddings with highest similarity first w.r.t the target
            candidate_order = torch.argsort(
                similarities,
                descending=True,
            )

        else:
            # Sort the embeddings with lowest similarity first w.r.t the the top-1 recomendation
            candidate_order = torch.argsort(
                similarities,
                descending=False,
            )

        # Convert the tensor back to actual ids ([0, self.num_items - 1] -> [1, self.num_items])
        candidates = []
        for i in candidate_order:
            if torch.isfinite(similarities[i]):
                candidates.append(int(i) + 1)

        for j in range(1, max_attempts + 1):

            if self.budget_remaining() <= 0 or len(candidates) < j:
                break

            cf_seq = list(original_seq)
            # Grap here the most recent position. Use negative indexing because most recent position is last item in the list
            edited_positions = positions[-j:]

            # Apply reserve to get most recent item at the front of the list
            for pos, item in zip(reversed(edited_positions), candidates[:j]):
                cf_seq[pos] = item

            scores = self.get_scores(cf_seq)
            self.mask_scores_inplace(scores, cf_seq)

            if constraint.is_satisfied(scores):
                edit_distance = hamming_distance(
                    original_seq,
                    cf_seq,
                )

                return cf_seq, {
                    "method": "heuristic",
                    "valid": True,
                    "edit_distance": edit_distance,
                    "edits": j,
                    "queries_used": self.query_count,
                    "target_rec": (
                        constraint.item
                        if constraint.kind == "bring_in"
                        else None
                    ),
                    "top_k_cfs": [
                        {
                            "seq": cf_seq,
                            "edit_distance": edit_distance,
                            "valid": True,
                        }
                    ],
                }

        return None, {
            "method": "heuristic",
            "valid": False,
            "edit_distance": 0,
            "edits": 0,
            "queries_used": self.query_count,
            "target_rec": (
                constraint.item
                if constraint.kind == "bring_in"
                else None
            ),
            "top_k_cfs": [],
        }