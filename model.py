"""STCRN model used in the main experiments."""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContextualRefractionModulation(nn.Module):
    """Construct context-dependent refracted traffic representations."""

    def __init__(self, dataset: str, num_nodes: int, spatial_dim: int):
        super().__init__()
        self.time_of_day_size = 288
        self.day_of_week_size = 7
        sampling_intervals = {
            "PEMS04": 5,
            "PEMS07": 5,
            "PEMS08": 5,
            "XMBRT": 5,
            "HZMETRO": 5,
            "BJMETRO": 10,
        }
        try:
            self.sampling_interval = sampling_intervals[dataset]
        except KeyError as error:
            raise ValueError(f"Unsupported dataset: {dataset}") from error

        # E^X, E^D, E^W, and the node identity embedding table E_bar^N.
        self.traffic_embedding_projection = nn.Linear(1, spatial_dim)
        self.time_of_day_embedding = nn.Parameter(
            torch.empty(self.time_of_day_size, spatial_dim)
        )
        self.day_of_week_embedding = nn.Parameter(
            torch.empty(self.day_of_week_size, spatial_dim)
        )
        self.node_identity_embedding = nn.Parameter(
            torch.empty(num_nodes, spatial_dim)
        )
        nn.init.xavier_uniform_(self.time_of_day_embedding)
        nn.init.xavier_uniform_(self.day_of_week_embedding)
        nn.init.xavier_uniform_(self.node_identity_embedding)

    def forward(self, traffic_history, timestamps):
        batch_size, _, num_nodes = traffic_history.shape
        day_of_week = timestamps[..., 2]
        hour = timestamps[..., 3]
        minute = timestamps[..., 4]

        time_of_day_index = (
            (hour * 60.0 + minute) / self.sampling_interval
        ).long()
        time_of_day_index = time_of_day_index.clamp(
            0, self.time_of_day_size - 1
        )
        time_of_day_index = time_of_day_index.unsqueeze(-1).expand(
            -1, -1, num_nodes
        )
        day_of_week_index = day_of_week.long().clamp(
            0, self.day_of_week_size - 1
        )
        day_of_week_index = day_of_week_index.unsqueeze(-1).expand(
            -1, -1, num_nodes
        )

        time_of_day_context = self.time_of_day_embedding[time_of_day_index]
        day_of_week_context = self.day_of_week_embedding[day_of_week_index]
        node_identity_context = self.node_identity_embedding.view(
            1, 1, num_nodes, -1
        ).expand(batch_size, -1, -1, -1)
        traffic_embedding = self.traffic_embedding_projection(
            traffic_history.unsqueeze(-1)
        )

        # Equations (5) and (6): E^C and the refracted representation H.
        contextual_embedding = (
            node_identity_context
            * time_of_day_context
            * day_of_week_context
        )
        refracted_representation = traffic_embedding * contextual_embedding
        return refracted_representation, contextual_embedding


class ContextRefractiveRelationLearner(nn.Module):
    """Learn A_t and H_t^A from refracted representations."""

    def __init__(self, spatial_dim: int, num_nodes: int):
        super().__init__()
        self.relation_embedding_mlp = nn.Sequential(
            OrderedDict(
                [
                    ("linear_1", nn.Linear(spatial_dim, 16)),
                    ("sigmoid_1", nn.Sigmoid()),
                    ("linear_2", nn.Linear(16, 2)),
                    ("sigmoid_2", nn.Sigmoid()),
                    ("linear_3", nn.Linear(2, spatial_dim)),
                ]
            )
        )

        # E^P and the node-adaptive parameter generators in Equation (9).
        self.node_parameter_embedding = nn.Parameter(
            torch.randn(num_nodes, spatial_dim)
        )
        nn.init.xavier_uniform_(self.node_parameter_embedding)
        self.node_weight_generator = nn.Linear(
            spatial_dim, spatial_dim * spatial_dim, bias=False
        )
        self.node_bias_generator = nn.Linear(
            spatial_dim, spatial_dim, bias=False
        )

    def node_adaptive_parameters(self):
        num_nodes, spatial_dim = self.node_parameter_embedding.shape
        node_weights = self.node_weight_generator(
            self.node_parameter_embedding
        ).view(num_nodes, spatial_dim, spatial_dim)
        node_biases = self.node_bias_generator(self.node_parameter_embedding)
        return node_weights, node_biases

    def forward(
        self,
        refracted_representation,
        contextual_embedding,
        node_weights,
        node_biases,
    ):
        # Equations (7) and (8): E_t^G and A_t.
        relation_embedding = torch.tanh(
            self.relation_embedding_mlp(refracted_representation)
            * contextual_embedding
        )
        dynamic_relation_matrix = F.relu(
            torch.bmm(
                relation_embedding,
                relation_embedding.transpose(1, 2),
            )
        )
        dynamic_relation_matrix = F.softmax(
            dynamic_relation_matrix, dim=-1
        )

        # Equation (10): node-adaptive representation H_t^A.
        node_adaptive_representation = torch.einsum(
            "bni,nio->bno", refracted_representation, node_weights
        ) + node_biases
        node_adaptive_representation = F.gelu(
            node_adaptive_representation
        )
        return dynamic_relation_matrix, node_adaptive_representation


class GraphAttentionScoring(nn.Module):
    """Compute the base attention coefficient matrix S_t."""

    def __init__(self, spatial_dim: int):
        super().__init__()
        self.feature_projection = nn.Linear(
            spatial_dim, spatial_dim, bias=False
        )
        self.source_scoring = nn.Linear(spatial_dim, 1, bias=False)
        self.target_scoring = nn.Linear(spatial_dim, 1, bias=False)
        self.leaky_relu = nn.LeakyReLU(0.2)

        nn.init.xavier_uniform_(self.feature_projection.weight)
        nn.init.xavier_uniform_(self.source_scoring.weight)
        nn.init.xavier_uniform_(self.target_scoring.weight)

    def forward(self, node_adaptive_representation):
        # Equations (11)-(13): Z_t, source/target scores, and S_t.
        projected_representation = self.feature_projection(
            node_adaptive_representation
        )
        source_scores = self.source_scoring(projected_representation)
        target_scores = self.target_scoring(projected_representation)
        attention_coefficients = F.softmax(
            self.leaky_relu(
                source_scores + target_scores.transpose(-1, -2)
            ),
            dim=-1,
        )
        return projected_representation, attention_coefficients


class RelationCalibration(nn.Module):
    """Calibrate S_t with the context-refractive relation matrix A_t."""

    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(0.1)

    def forward(
        self,
        projected_representation,
        attention_coefficients,
        dynamic_relation_matrix,
    ):
        # Equation (14): relation-calibrated coefficients S_tilde_t.
        calibrated_coefficients = (
            attention_coefficients * dynamic_relation_matrix
        )
        calibrated_coefficients = self.dropout(calibrated_coefficients)
        return F.elu(
            torch.bmm(calibrated_coefficients, projected_representation)
        )


class ContextRefractiveGraphAttention(nn.Module):
    """Learn and aggregate context-dependent spatial relations."""

    def __init__(self, spatial_dim: int, temporal_dim: int, num_nodes: int):
        super().__init__()
        self.relation_learner = ContextRefractiveRelationLearner(
            spatial_dim, num_nodes
        )
        self.input_dropout = nn.Dropout(0.1)
        self.graph_attention_scoring = GraphAttentionScoring(spatial_dim)
        self.relation_calibration = RelationCalibration()
        self.message_dropout = nn.Dropout(0.1)
        self.spatial_projection = nn.Linear(spatial_dim, temporal_dim)
        self.skip_projection = nn.Linear(spatial_dim, temporal_dim)
        self.output_norm = nn.LayerNorm(temporal_dim)

    def forward(self, refracted_representation, contextual_embedding):
        _, history_length, _, _ = refracted_representation.shape
        spatial_representations = []
        node_weights, node_biases = (
            self.relation_learner.node_adaptive_parameters()
        )

        for time_step in range(history_length):
            refracted_step = refracted_representation[:, time_step]
            contextual_step = contextual_embedding[:, time_step]

            relation_matrix, node_adaptive_representation = (
                self.relation_learner(
                    refracted_step,
                    contextual_step,
                    node_weights,
                    node_biases,
                )
            )
            attention_input = self.input_dropout(
                node_adaptive_representation
            )
            projected_representation, attention_coefficients = (
                self.graph_attention_scoring(attention_input)
            )
            calibrated_message = self.relation_calibration(
                projected_representation,
                attention_coefficients,
                relation_matrix,
            )
            spatial_representation = self.spatial_projection(
                self.message_dropout(calibrated_message)
            )
            spatial_representation = self.output_norm(
                self.skip_projection(refracted_step)
                + spatial_representation
            )
            spatial_representations.append(spatial_representation)

        # H^S in Equation (15).
        return torch.stack(spatial_representations, dim=1)


class SpectralConvolution(nn.Module):
    """Learnable spectral transformation in Equation (19)."""

    def __init__(self, feature_dim: int):
        super().__init__()
        self.real_imaginary_projection = nn.Conv1d(
            feature_dim * 2, feature_dim * 2, 1, bias=False
        )
        self.batch_norm = nn.BatchNorm1d(feature_dim * 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, temporal_features):
        history_length = temporal_features.size(-1)
        spectrum = torch.fft.rfft(temporal_features, norm="ortho")
        real_imaginary = torch.cat([spectrum.real, spectrum.imag], dim=1)
        transformed_spectrum = self.relu(
            self.batch_norm(
                self.real_imaginary_projection(real_imaginary)
            )
        )
        real_part, imaginary_part = torch.chunk(
            transformed_spectrum, 2, dim=1
        )
        complex_spectrum = torch.complex(real_part, imaginary_part)
        return torch.fft.irfft(
            complex_spectrum, n=history_length, norm="ortho"
        )


class FrequencyConvolution(nn.Module):
    """Frequency convolution block for spectral-temporal refinement."""

    def __init__(self, temporal_dim: int):
        super().__init__()
        self.feature_expansion = nn.Conv1d(
            temporal_dim, temporal_dim * 2, 1
        )
        self.local_branch_3 = nn.Sequential(
            nn.Conv1d(
                temporal_dim,
                temporal_dim,
                kernel_size=3,
                padding=1,
                groups=temporal_dim,
            ),
            nn.GELU(),
            nn.Conv1d(temporal_dim, temporal_dim, kernel_size=1),
        )
        self.local_branch_5 = nn.Sequential(
            nn.Conv1d(
                temporal_dim,
                temporal_dim,
                kernel_size=5,
                padding=2,
                groups=temporal_dim,
            ),
            nn.GELU(),
            nn.Conv1d(temporal_dim, temporal_dim, kernel_size=1),
        )
        self.spectral_convolution = SpectralConvolution(temporal_dim * 2)
        self.feature_restoration = nn.Conv1d(
            temporal_dim * 2, temporal_dim, 1
        )

    def forward(self, spatial_representation):
        batch_size, history_length, num_nodes, temporal_dim = (
            spatial_representation.shape
        )
        temporal_features = spatial_representation.permute(
            0, 2, 3, 1
        ).reshape(batch_size * num_nodes, temporal_dim, history_length)
        temporal_features = self.feature_expansion(temporal_features)

        branch_3, branch_5 = torch.split(
            temporal_features, temporal_dim, dim=1
        )
        branch_3 = self.local_branch_3(branch_3)
        branch_5 = self.local_branch_5(branch_5)
        local_temporal_features = torch.cat(
            [branch_3, branch_5], dim=1
        )
        frequency_enhanced_features = (
            self.spectral_convolution(local_temporal_features)
            + local_temporal_features
        )
        refined_features = self.feature_restoration(
            frequency_enhanced_features
        )
        return refined_features.reshape(
            batch_size, num_nodes, temporal_dim, history_length
        ).permute(0, 3, 1, 2)


class SpectralTemporalRefinement(nn.Module):
    """Refine spatial representations in the time and frequency domains."""

    def __init__(self, temporal_dim: int):
        super().__init__()
        self.frequency_convolution = FrequencyConvolution(temporal_dim)
        self.frequency_norm = nn.LayerNorm(temporal_dim)
        self.frequency_dropout = nn.Dropout(0.1)
        self.feed_forward = nn.Sequential(
            nn.Linear(temporal_dim, temporal_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(temporal_dim * 2, temporal_dim),
        )
        self.output_norm = nn.LayerNorm(temporal_dim)
        self.output_dropout = nn.Dropout(0.1)

    def forward(self, spatial_representation):
        frequency_features = self.frequency_convolution(
            spatial_representation
        )
        refined_representation = self.frequency_norm(
            spatial_representation
            + self.frequency_dropout(frequency_features)
        )
        feed_forward_features = self.feed_forward(
            refined_representation
        )
        # H^R in Equation (22).
        return self.output_norm(
            refined_representation
            + self.output_dropout(feed_forward_features)
        )


class PredictionHead(nn.Module):
    """Two-stage regression projection in Equation (23)."""

    def __init__(self, input_length: int, output_length: int, temporal_dim: int):
        super().__init__()
        self.horizon_projection = nn.Linear(input_length, output_length)
        self.target_projection = nn.Linear(temporal_dim, 1)

    def forward(self, refined_representation):
        prediction = self.horizon_projection(
            refined_representation.permute(0, 2, 3, 1)
        )
        prediction = self.target_projection(
            prediction.permute(0, 3, 1, 2)
        )
        return prediction.squeeze(-1)


class STCRN(nn.Module):
    """Spatial-Temporal Context-Refractive Network (STCRN)."""

    def __init__(self, args):
        super().__init__()
        input_length = getattr(args, "input_len", 12)
        output_length = getattr(args, "output_len", 12)
        num_nodes = getattr(args, "num_nodes", 170)
        spatial_dim = getattr(args, "spatial_hidden_dim", 32)
        temporal_dim = getattr(args, "temporal_hidden_dim", 64)

        self.contextual_refraction_modulation = ContextualRefractionModulation(
            args.dataset, num_nodes, spatial_dim
        )
        self.context_refractive_graph_attention = (
            ContextRefractiveGraphAttention(
                spatial_dim, temporal_dim, num_nodes
            )
        )
        self.spectral_temporal_refinement = SpectralTemporalRefinement(
            temporal_dim
        )
        self.prediction_head = PredictionHead(
            input_length, output_length, temporal_dim
        )

    def parameter_count(self):
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    def forward(self, traffic_history, timestamps):
        refracted_representation, contextual_embedding = (
            self.contextual_refraction_modulation(
                traffic_history, timestamps
            )
        )
        spatial_representation = self.context_refractive_graph_attention(
            refracted_representation, contextual_embedding
        )
        refined_representation = self.spectral_temporal_refinement(
            spatial_representation
        )
        return self.prediction_head(refined_representation)
