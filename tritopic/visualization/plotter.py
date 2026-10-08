"""
Topic Visualization
====================

Interactive visualizations for topic models using:
- UMAP / PaCMAP for dimensionality reduction
- Plotly for interactive plots
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class TopicVisualizer:
    """
    Visualize topics and documents.
    
    Provides various visualization methods for exploring topic models.
    
    Parameters
    ----------
    method : str
        Dimensionality reduction method: "umap" or "pacmap"
    random_state : int
        Random seed for reproducibility.
    """
    
    def __init__(
        self,
        method: Literal["umap", "pacmap"] = "umap",
        random_state: int = 42,
    ):
        self.method = method
        self.random_state = random_state
        
        self._reducer = None
        self._reduced_embeddings = None
    
    def _reduce_dimensions(
        self,
        embeddings: np.ndarray,
        n_components: int = 2,
    ) -> np.ndarray:
        """Reduce embeddings to 2D for visualization."""
        if self.method == "umap":
            from umap import UMAP
            
            reducer = UMAP(
                n_components=n_components,
                n_neighbors=15,
                min_dist=0.1,
                metric="cosine",
                random_state=self.random_state,
            )
        else:  # pacmap
            try:
                from pacmap import PaCMAP
                reducer = PaCMAP(
                    n_components=n_components,
                    random_state=self.random_state,
                )
            except ImportError:
                # Fallback to UMAP
                from umap import UMAP
                reducer = UMAP(
                    n_components=n_components,
                    random_state=self.random_state,
                )
        
        self._reducer = reducer
        self._reduced_embeddings = reducer.fit_transform(embeddings)
        
        return self._reduced_embeddings
    
    def plot_documents(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        documents: list[str] | None = None,
        topics: list | None = None,
        show_outliers: bool = True,
        interactive: bool = True,
        title: str = "Topic Document Map",
        width: int = 900,
        height: int = 700,
        **kwargs,
    ) -> go.Figure:
        """
        Plot documents in 2D space colored by topic.
        
        Parameters
        ----------
        embeddings : np.ndarray
            Document embeddings.
        labels : np.ndarray
            Topic assignments.
        documents : list[str], optional
            Document texts for hover info.
        topics : list[TopicInfo], optional
            Topic info for labels.
        show_outliers : bool
            Whether to show outlier documents.
        interactive : bool
            Create interactive Plotly figure.
        title : str
            Plot title.
        width, height : int
            Figure dimensions.
            
        Returns
        -------
        fig : go.Figure
            Plotly figure.
        """
        # Reduce dimensions
        coords = self._reduce_dimensions(embeddings, n_components=2)
        
        # Prepare data
        mask = np.ones(len(labels), dtype=bool)
        if not show_outliers:
            mask = labels != -1
        
        x = coords[mask, 0]
        y = coords[mask, 1]
        topic_labels = labels[mask]
        
        # Create hover text
        if documents:
            hover_texts = []
            for i, idx in enumerate(np.where(mask)[0]):
                doc = documents[idx]
                # Truncate long documents
                if len(doc) > 200:
                    doc = doc[:200] + "..."
                topic_id = labels[idx]
                
                # Get topic label if available
                topic_name = f"Topic {topic_id}"
                if topics:
                    for t in topics:
                        if t.topic_id == topic_id:
                            topic_name = t.label or f"Topic {topic_id}"
                            break
                
                hover_texts.append(f"<b>{topic_name}</b><br>{doc}")
        else:
            hover_texts = [f"Topic {l}" for l in topic_labels]
        
        # Create color mapping
        unique_labels = sorted(np.unique(topic_labels))
        n_topics = len([l for l in unique_labels if l != -1])
        
        # Use a good colorscale
        colors = px.colors.qualitative.Set2 + px.colors.qualitative.Set3
        color_map = {}
        color_idx = 0
        for label in unique_labels:
            if label == -1:
                color_map[-1] = "lightgray"
            else:
                color_map[label] = colors[color_idx % len(colors)]
                color_idx += 1
        
        point_colors = [color_map[l] for l in topic_labels]
        
        # Create figure
        fig = go.Figure()
        
        # Add scatter for each topic (for legend)
        for label in unique_labels:
            topic_mask = topic_labels == label
            
            # Get topic name
            topic_name = "Outliers" if label == -1 else f"Topic {label}"
            if topics:
                for t in topics:
                    if t.topic_id == label:
                        topic_name = t.label or f"Topic {label}"
                        break
            
            fig.add_trace(go.Scatter(
                x=x[topic_mask],
                y=y[topic_mask],
                mode="markers",
                name=topic_name,
                marker=dict(
                    color=color_map[label],
                    size=6 if label != -1 else 4,
                    opacity=0.7 if label != -1 else 0.3,
                ),
                text=[hover_texts[i] for i in np.where(topic_mask)[0]],
                hovertemplate="%{text}<extra></extra>",
            ))
        
        # Update layout
        fig.update_layout(
            title=dict(text=title, font=dict(size=16)),
            width=width,
            height=height,
            xaxis=dict(title=f"{self.method.upper()} 1", showgrid=False),
            yaxis=dict(title=f"{self.method.upper()} 2", showgrid=False),
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02,
            ),
            template="plotly_white",
        )
        
        return fig
    
    def plot_topics(
        self,
        topics: list,
        n_keywords: int = 8,
        title: str = "Topics Overview",
        width: int = 900,
        height: int = None,
    ) -> go.Figure:
        """
        Plot topics as horizontal bar charts of keywords.
        
        Parameters
        ----------
        topics : list[TopicInfo]
            Topic information objects.
        n_keywords : int
            Number of keywords to show per topic.
        title : str
            Plot title.
        width, height : int
            Figure dimensions.
            
        Returns
        -------
        fig : go.Figure
            Plotly figure.
        """
        # Filter out outliers and sort by size
        valid_topics = [t for t in topics if t.topic_id != -1]
        valid_topics = sorted(valid_topics, key=lambda t: -t.size)
        
        n_topics = len(valid_topics)
        if height is None:
            height = max(400, 80 * n_topics)
        
        # Create subplots
        fig = make_subplots(
            rows=n_topics,
            cols=1,
            subplot_titles=[
                f"{t.label or f'Topic {t.topic_id}'} (n={t.size})"
                for t in valid_topics
            ],
            vertical_spacing=0.08,
        )
        
        colors = px.colors.qualitative.Set2
        
        for i, topic in enumerate(valid_topics):
            keywords = topic.keywords[:n_keywords]
            scores = topic.keyword_scores[:n_keywords]
            
            # Normalize scores
            max_score = max(scores) if scores else 1
            scores = [s / max_score for s in scores]
            
            fig.add_trace(
                go.Bar(
                    x=scores[::-1],
                    y=keywords[::-1],
                    orientation="h",
                    marker_color=colors[i % len(colors)],
                    showlegend=False,
                ),
                row=i + 1,
                col=1,
            )
        
        fig.update_layout(
            title=dict(text=title, font=dict(size=16)),
            width=width,
            height=height,
            template="plotly_white",
        )
        
        return fig
    
    def plot_hierarchy(
        self,
        topic_embeddings: np.ndarray,
        topics: list,
        title: str = "Topic Hierarchy",
        width: int = 800,
        height: int = 500,
    ) -> go.Figure:
        """
        Plot topic hierarchy as a dendrogram.
        
        Parameters
        ----------
        topic_embeddings : np.ndarray
            Centroid embeddings for each topic.
        topics : list[TopicInfo]
            Topic information objects.
        title : str
            Plot title.
        width, height : int
            Figure dimensions.
            
        Returns
        -------
        fig : go.Figure
            Plotly figure.
        """
        from scipy.cluster.hierarchy import linkage, dendrogram
        from scipy.spatial.distance import pdist
        
        # Filter valid topics
        valid_topics = [t for t in topics if t.topic_id != -1]
        
        if len(valid_topics) < 2:
            # Not enough topics for hierarchy
            fig = go.Figure()
            fig.add_annotation(
                text="Need at least 2 topics for hierarchy",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
            )
            return fig
        
        # Compute linkage
        distances = pdist(topic_embeddings, metric="cosine")
        Z = linkage(distances, method="ward")
        
        # Create dendrogram
        labels = [t.label or f"Topic {t.topic_id}" for t in valid_topics]
        
        # Use scipy's dendrogram to get coordinates
        dendro = dendrogram(Z, labels=labels, no_plot=True)
        
        # Create plotly figure
        fig = go.Figure()
        
        # Add lines for dendrogram
        icoord = dendro["icoord"]
        dcoord = dendro["dcoord"]
        
        for xs, ys in zip(icoord, dcoord):
            fig.add_trace(go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                line=dict(color="#636EFA", width=2),
                showlegend=False,
            ))
        
        # Add labels
        fig.update_layout(
            title=dict(text=title, font=dict(size=16)),
            width=width,
            height=height,
            xaxis=dict(
                ticktext=dendro["ivl"],
                tickvals=list(range(5, len(dendro["ivl"]) * 10, 10)),
                tickangle=45,
            ),
            yaxis=dict(title="Distance"),
            template="plotly_white",
        )
        
        return fig
    
    def plot_topic_similarity(
        self,
        topic_embeddings: np.ndarray,
        topics: list,
        title: str = "Topic Similarity",
        width: int = 600,
        height: int = 600,
    ) -> go.Figure:
        """
        Plot topic similarity as a heatmap.
        
        Parameters
        ----------
        topic_embeddings : np.ndarray
            Centroid embeddings for each topic.
        topics : list[TopicInfo]
            Topic information objects.
        title : str
            Plot title.
        width, height : int
            Figure dimensions.
            
        Returns
        -------
        fig : go.Figure
            Plotly figure.
        """
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Filter valid topics
        valid_topics = [t for t in topics if t.topic_id != -1]
        
        # Compute similarity matrix
        sim_matrix = cosine_similarity(topic_embeddings)
        
        # Labels
        labels = [t.label or f"Topic {t.topic_id}" for t in valid_topics]
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=sim_matrix,
            x=labels,
            y=labels,
            colorscale="RdBu",
            zmid=0.5,
            text=np.round(sim_matrix, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
            hovertemplate="Similarity: %{z:.3f}<extra></extra>",
        ))
        
        fig.update_layout(
            title=dict(text=title, font=dict(size=16)),
            width=width,
            height=height,
            xaxis=dict(tickangle=45),
            template="plotly_white",
        )
        
        return fig
    
    def plot_topic_over_time(
        self,
        labels: np.ndarray,
        timestamps: list,
        topics: list | None = None,
        title: str = "Topics Over Time",
        width: int = 900,
        height: int = 500,
    ) -> go.Figure:
        """
        Plot topic distribution over time.
        
        Parameters
        ----------
        labels : np.ndarray
            Topic assignments.
        timestamps : list
            Timestamps for each document.
        topics : list[TopicInfo], optional
            Topic information for labels.
        title : str
            Plot title.
        width, height : int
            Figure dimensions.
            
        Returns
        -------
        fig : go.Figure
            Plotly figure.
        """
        import pandas as pd
        
        # Create dataframe
        df = pd.DataFrame({
            "topic": labels,
            "timestamp": pd.to_datetime(timestamps),
        })
        
        # Filter outliers
        df = df[df["topic"] != -1]
        
        # Group by time and topic
        df["period"] = df["timestamp"].dt.to_period("M").dt.to_timestamp()
        counts = df.groupby(["period", "topic"]).size().unstack(fill_value=0)
        
        # Normalize to percentages
        counts = counts.div(counts.sum(axis=1), axis=0) * 100
        
        # Create figure
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set2
        
        for i, topic_id in enumerate(counts.columns):
            topic_name = f"Topic {topic_id}"
            if topics:
                for t in topics:
                    if t.topic_id == topic_id:
                        topic_name = t.label or f"Topic {topic_id}"
                        break
            
            fig.add_trace(go.Scatter(
                x=counts.index,
                y=counts[topic_id],
                name=topic_name,
                mode="lines",
                stackgroup="one",
                line=dict(color=colors[i % len(colors)]),
            ))
        
        fig.update_layout(
            title=dict(text=title, font=dict(size=16)),
            width=width,
            height=height,
            xaxis=dict(title="Time"),
            yaxis=dict(title="Topic Share (%)", range=[0, 100]),
            template="plotly_white",
        )

        return fig


def plot_topic_overlap(
    overlap_matrix: "pd.DataFrame",
    topics: list | None = None,
    title: str = "Topic Overlap (Soft Assignment Co-occurrence)",
    width: int = 650,
    height: int = 600,
) -> go.Figure:
    """Heatmap of topic co-occurrence from soft assignments.

    Parameters
    ----------
    overlap_matrix : pd.DataFrame
        Symmetric co-occurrence matrix from ``model.topic_overlap_matrix()``.
    topics : list[TopicInfo], optional
        Topic info for labels.
    title : str
        Plot title.
    width, height : int
        Figure dimensions.

    Returns
    -------
    fig : go.Figure
    """
    labels = list(overlap_matrix.columns)
    if topics:
        labels = [t.label or f"Topic {t.topic_id}" for t in topics]

    z = overlap_matrix.values.astype(float)

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=labels,
        y=labels,
        colorscale="Blues",
        text=z.astype(int),
        texttemplate="%{text}",
        textfont={"size": 10},
        hovertemplate="Co-occurrence: %{z}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        width=width,
        height=height,
        xaxis=dict(tickangle=45),
        template="plotly_white",
    )

    return fig


def plot_hierarchy_tree(
    hierarchy: "TopicHierarchy",
    title: str = "Topic Hierarchy",
    width: int = 900,
    height: int = 700,
) -> go.Figure:
    """Visualize a :class:`TopicHierarchy` as a tree diagram.

    Uses a simple top-down layout: levels are spaced vertically and nodes
    within each level are spread horizontally.  Edges connect parents to
    children.

    Parameters
    ----------
    hierarchy : TopicHierarchy
        Hierarchy object from :meth:`TriTopic.build_hierarchy`.
    title : str
        Plot title.
    width, height : int
        Figure dimensions.

    Returns
    -------
    fig : go.Figure
    """
    fig = go.Figure()

    # Assign (x, y) coordinates per node
    node_pos: dict[str, tuple[float, float]] = {}
    level_colors = px.colors.qualitative.Set2

    for level_idx, level_nodes in enumerate(hierarchy.levels):
        y = -level_idx  # top-down
        n = len(level_nodes)
        for i, node in enumerate(level_nodes):
            x = (i - (n - 1) / 2)  # center around 0
            node_pos[node.node_id] = (x, y)

    # Draw edges first (as lines)
    for level_nodes in hierarchy.levels:
        for node in level_nodes:
            if node.parent is not None and node.parent.node_id in node_pos:
                px0, py0 = node_pos[node.parent.node_id]
                px1, py1 = node_pos[node.node_id]
                fig.add_trace(go.Scatter(
                    x=[px0, px1], y=[py0, py1],
                    mode="lines",
                    line=dict(color="lightgray", width=1.5),
                    showlegend=False,
                    hoverinfo="skip",
                ))

    # Draw nodes
    for level_idx, level_nodes in enumerate(hierarchy.levels):
        color = level_colors[level_idx % len(level_colors)]
        xs, ys, texts, hovers = [], [], [], []
        for node in level_nodes:
            x, y = node_pos[node.node_id]
            xs.append(x)
            ys.append(y)
            kw_str = ", ".join(node.keywords[:3]) if node.keywords else ""
            texts.append(f"{node.node_id}<br>n={node.size}")
            hovers.append(
                f"<b>{node.label or node.node_id}</b><br>"
                f"Size: {node.size}<br>"
                f"Keywords: {kw_str}"
            )

        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode="markers+text",
            marker=dict(size=18, color=color, line=dict(width=1, color="white")),
            text=texts,
            textposition="bottom center",
            textfont=dict(size=9),
            hovertext=hovers,
            hovertemplate="%{hovertext}<extra></extra>",
            name=f"Level {level_idx} (res={hierarchy.resolution_levels[level_idx]:.3f})",
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        width=width,
        height=height,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        template="plotly_white",
    )

    return fig
