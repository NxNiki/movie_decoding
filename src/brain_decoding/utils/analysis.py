import os
import re
import warnings
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sb
from matplotlib.patches import Patch

from brain_decoding.dataloader.patients import Events
from scripts.save_clusterless import PREDICTION_FS

PREDICTION_VALUE_THRESH = 0.5
SLEEP_SCORE_FS = 1 / 30
SLEEP_SCORE_OFFSET = 0
SLEEP_STAGE_THRESH = 600
SLEEP_STAGE_COLORMAP = ["Blues", "Reds", "Greens", "Purples"]
SECONDS_PER_HOUR = 3600

CONCEPT_LABELS = ["White House", "CIA", "Hostage", "Handcuff", "Jack", "Bill", "Fayed", "Amar"]


def plot_concept_graph(annotation_file: str, labels: List[str], figure_name: str = None):
    data = np.load(annotation_file)
    if data.shape[0] < data.shape[1]:
        data = data.transpose()

    co_occurrence_matrix = np.dot(data.transpose(), data)
    total_occurrences = data.sum(axis=0)
    co_occurrence_df = pd.DataFrame(co_occurrence_matrix, index=labels, columns=labels)

    g = nx.Graph()
    # Add nodes with size attribute based on total occurrences
    for i, concept in enumerate(co_occurrence_df.columns):
        g.add_node(concept, size=total_occurrences[i])

    # Add edges with weights (number of co-occurrences)
    for i, concept1 in enumerate(co_occurrence_df.columns):
        for j, concept2 in enumerate(co_occurrence_df.columns):
            if i < j:  # Avoid duplicate edges and self-loops
                weight = co_occurrence_df.loc[concept1, concept2]
                if weight > 0:  # Add edge only if there's a co-occurrence
                    g.add_edge(concept1, concept2, weight=weight)

    pos = nx.circular_layout(g)
    # pos = nx.spring_layout(g, k=1, seed=42)
    # Find isolated nodes (nodes with degree 0)
    # isolated_nodes = [node for node, degree in dict(g.degree()).items() if degree == 0]
    # for node in isolated_nodes:
    #     pos[node] = np.array([0.1, 0.3])

    plt.figure(figsize=(12, 10))
    node_sizes = [g.nodes[concept]["size"] for concept in g.nodes]
    node_color = (0.4, 0.6, 1.0, 0.6)  # Light blue with 60% opacity (RGBA)
    nx.draw_networkx_nodes(g, pos, node_size=node_sizes, node_color=[node_color] * len(node_sizes), edgecolors=None)
    node_labels = {concept: (f"{concept}\n" f"({int(total_occurrences[i])})") for i, concept in enumerate(g.nodes)}
    nx.draw_networkx_labels(g, pos, labels=node_labels, font_size=7, font_color="black")

    edges = g.edges(data=True)
    nx.draw_networkx_edges(
        g, pos, edgelist=edges, width=[np.log2(edge_data["weight"]) / 5 for _, _, edge_data in edges], edge_color="gray"
    )
    edge_labels = {(u, v): f"{int(data['weight'])}" for u, v, data in edges}
    nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels, font_size=8, font_color="gray")

    if figure_name:
        title = os.path.basename(figure_name)
        plt.title(re.sub(r"\.[a-z]*", "", title))
        plt.savefig(figure_name, format="png", bbox_inches="tight")
    plt.show()


def concept_frequency(concept_file: str, concept_label: List[str]) -> Tuple[Dict[str, int], np.ndarray[float]]:
    """
    Count the frequency of each concept in the concept file and calculate the weight

    Parameters:
    - concept_file: a .npy file for an n_time by n_concepts array. 1 if concept appears at a time frame.
    - concept_label:
    """
    concept = np.load(concept_file)
    concept_count = np.sum(concept, axis=1)
    res = {}
    weight = []
    for i, label in enumerate(concept_label):
        res[label] = int(concept_count[i])
        weight.append(1 / concept_count[i])

    weight = np.array(weight)
    weight = weight / np.sum(weight)
    return res, weight


def filter_predictions(predictions: np.ndarray, labels: List[str], thresh: float) -> Tuple:
    concept_index = np.nanmean(predictions, axis=0) > thresh
    predictions = predictions[:, concept_index]
    labels = np.compress(concept_index, np.array(labels)).tolist()
    return predictions, labels


def add_label_count(labels: List[str], labels_count: Dict[str, int]) -> List[str]:
    labels_with_count = []
    for label in labels:
        labels_with_count.append(f"{label} ({labels_count[label]})")
    return labels_with_count


def prediction_curve(
    predictions: np.ndarray,
    sleep_score: pd.DataFrame,
    labels: List[str],
    save_file_name: str,
    sampling_frequency: int = PREDICTION_FS,
) -> None:
    """
    Plot prediction curves with background colors representing sleep stages and a legend.

    Parameters:
    - predictions (np.ndarray): n_samples by n_concepts array of predictions.
    - sleep_score (pd.DataFrame): n by 2 DataFrame with sleep stage (column 0) and start index (column 1).
    - labels (List[str]): List of labels for each prediction curve.
    - save_file_name (str): The file path to save the plot.

    Returns:
    - None: The function saves the figure to the specified output file.
    """

    y_min = np.nanmin(predictions)
    y_max = np.nanmax(predictions)

    if np.isnan(y_min):
        warnings.warn("all values in prediction are nan.")
        return

    palette = sb.color_palette("husl", n_colors=predictions.shape[1])
    fig_height = 2 * len(labels)
    fig, axes = plt.subplots(nrows=predictions.shape[1], ncols=1, figsize=(20, fig_height), sharex=True)
    # Loop through each prediction curve
    for i in range(predictions.shape[1]):
        # Calculate time in hours
        time = np.arange(predictions.shape[0]) / sampling_frequency / SECONDS_PER_HOUR

        # Plot the prediction curve with time in hours
        sb.lineplot(
            x=time,
            y=predictions[:, i],
            ax=axes[i],
            color=palette[i],
            linewidth=1.5,
        )
        # Plot the mean curve with a dashed line
        sb.lineplot(
            x=time,
            y=np.mean(predictions[:, i]),
            ax=axes[i],
            color="#808080",
            linestyle="--",
        )
        axes[i].set_ylim([y_min, y_max])
        axes[i].set_title(labels[i], fontsize=14)

    if sleep_score is not None:
        # Assign a unique color for each unique sleep stage
        unique_stages = sleep_score["Score"].unique()
        stage_colors = sb.color_palette("Set2", len(unique_stages))
        stage_color_map = dict(zip(unique_stages, stage_colors))  # Map sleep stages to colors

        for i in range(predictions.shape[1]):
            # Add background color based on sleep_score start_index
            for j in range(len(sleep_score)):
                start = sleep_score.iloc[j]["start_index"] / sampling_frequency / SECONDS_PER_HOUR
                end = (
                    sleep_score.iloc[j + 1]["start_index"] / sampling_frequency / SECONDS_PER_HOUR
                    if j < len(sleep_score) - 1
                    else predictions.shape[0] / sampling_frequency / SECONDS_PER_HOUR
                )

                if 0 <= start < predictions.shape[0] / sampling_frequency / SECONDS_PER_HOUR:
                    color = stage_color_map[sleep_score.iloc[j]["Score"]]
                    axes[i].axvspan(xmin=start, xmax=end, color=color, alpha=0.3)

        # Create custom legend for the background colors
        legend_elements = [Patch(facecolor=stage_color_map[stage], label=stage, alpha=0.3) for stage in unique_stages]
        plt.legend(handles=legend_elements, loc="upper right", title="Sleep Stages")

    fig.supylabel("Activation", fontsize=14)
    plt.xlabel("Time (hours)", fontsize=14)
    plt.tight_layout()

    plt.savefig(save_file_name)
    plt.show()


def stage_box_plot(
    predictions: np.ndarray,
    sleep_score: pd.DataFrame,
    labels: List[str],
    save_figure_name: str,
    prediction_thresh: Optional[float],
) -> None:
    """
    Plot box plots with swarms overlaid for each sleep stage, with a separate subplot for each label.
    Limit the number of swarm points per stage for performance improvement and add stage length to the label.

    Parameters:
    - predictions (np.ndarray): n by m array of predictions.
    - sleep_score (pd.DataFrame): n by 2 DataFrame with sleep stage (column 0) and start index (column 1).
    - labels (List[str]): List of labels for each prediction column.
    - save_figure_name (str): The file path to save the plot.
    - prediction_thresh Optional[float]: select concepts (column of predictions) with mean prediction larger than the
        threshold.

    Returns:
    - None: The function saves the figure with subplots to the specified output file.
    """

    if sleep_score is None:
        warnings.warn("sleep_score is None!")
        return

    if prediction_thresh:
        predictions = predictions[:, np.nanmean(predictions, axis=0) > prediction_thresh]
        labels = labels[np.nanmean(predictions, axis=0) > prediction_thresh]

    n_samples, n_labels = predictions.shape

    # Create subplots for each label (column of predictions)
    fig, axes = plt.subplots(n_labels, 1, figsize=(12, 3 * n_labels), sharex=True)

    # If there's only one label, we need to convert axes to an iterable
    if n_labels == 1:
        axes = [axes]

    # Loop through each label (column of predictions)
    for i, label in enumerate(labels):
        # Overwrite the combined DataFrame for memory efficiency
        combined_df_list = []
        show_legend = True if i == 0 else False

        for stage_data in prediction_iterator(
            predictions[:, i], sleep_score, PREDICTION_VALUE_THRESH, SLEEP_STAGE_THRESH
        ):
            combined_df_list.append(pd.DataFrame(stage_data))

        combined_df = pd.concat(combined_df_list, axis=0)
        # Sample a maximum of n points per stage for the swarmplot
        combined_df_sample = (
            combined_df.groupby("Stage")
            .apply(lambda x: x.sample(n=min(len(x), 200), random_state=42))
            .reset_index(drop=True)
        )

        # Create a color palette for the stages
        unique_stages = combined_df["Stage Label"].unique()
        palette = sb.color_palette("Set2", len(unique_stages))
        stage_color_map = dict(zip(unique_stages, palette))

        # Plot the violin/box plot for this label on its respective axis
        ax = sb.boxplot(
            x="Stage",
            y="Value(>.5)",
            data=combined_df,
            hue="Stage Label",
            palette=stage_color_map,
            linewidth=1.5,
            color="none",
            width=0.7,
            notch=True,
            ax=axes[i],
            dodge=False,
            legend=False,
        )
        # Overlay the swarmplot with limited points
        ax = sb.swarmplot(
            x="Stage",
            y="Value(>.5)",
            data=combined_df_sample,
            hue="Stage Label",
            palette=stage_color_map,
            size=2,
            dodge=False,
            legend=show_legend,
            ax=axes[i],
        )

        if show_legend:
            ax.legend(
                borderaxespad=0.0,
                loc="right",
                columnspacing=1.2,
                frameon=False,
                markerscale=5,
                handlelength=0.1,
                prop={"size": 10},
                title="",
                bbox_to_anchor=(1, 1.1),
                ncol=2,
            )

        # change boxplot edge color:
        for i, artist in enumerate(ax.patches):
            # Set the linecolor on the artist to the facecolor, and set the facecolor to None
            col = artist.get_facecolor()
            artist.set_edgecolor(col)
            artist.set_facecolor("None")

            # Each box has 6 associated Line2D objects (to make the whiskers, fliers, etc.)
            # Loop over them here, and use the same colour as above
            for j in range(i * 6, i * 6 + 6):
                line = ax.lines[j]
                line.set_color(col)
                line.set_mfc(col)
                line.set_mec(col)

        # sb.violinplot(x='Stage', y='Value(>.5)', data=combined_df, hue='Stage Label', palette=stage_color_map,
        #            linewidth=1.5, facecolor="none", ax=axes[i], inner=None, dodge=False, legend=False)

        # Hide the right and top spines
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

        # Set the title for each subplot
        ax.set_ylabel(label, fontsize=12)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=45)

    # Add overall figure label
    plt.tight_layout()

    # Save the figure
    plt.savefig(save_figure_name)
    plt.show()


def correlation_heatmap(data: np.ndarray, column_labels: List[str], save_figure_name: str) -> None:
    """
    Calculate the correlation among the columns of the data array and plot a heatmap with the
    distribution of correlation values in a subplot.

    Parameters:
    - data (np.ndarray): n by m array where n is the number of samples and m is the number of columns.
    - column_labels (List[str]): A list of labels for each column.
    - save_figure_name (str): The file path to save the heatmap image.

    Returns:
    - None: The function saves the figure to the specified output file.
    """
    # Calculate the correlation matrix
    corr_matrix = np.corrcoef(data, rowvar=False)
    v_min, v_max = -1, 1

    # Flatten the correlation matrix and exclude the diagonal (correlation of a variable with itself)
    corr_values = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]

    # Create a figure with 2 subplots: 1 for the heatmap, 1 for the histogram
    fig, (ax_heatmap, ax_hist) = plt.subplots(1, 2, figsize=(14, 8), gridspec_kw={"width_ratios": [2.5, 1.5]})

    # Plot the heatmap on the first subplot
    sb.heatmap(
        corr_matrix,
        annot=True,
        fmt=".3f",
        cmap="coolwarm",
        xticklabels=column_labels,
        center=0,
        vmin=v_min,
        vmax=v_max,
        cbar=False,
        annot_kws={"size": 12},
        yticklabels=column_labels,
        ax=ax_heatmap,
    )

    file_name = os.path.splitext(os.path.basename(save_figure_name))[0]
    ax_heatmap.set_title(file_name)

    ax_heatmap.set_xticklabels(ax_heatmap.get_xticklabels(), rotation=45, horizontalalignment="right", fontsize=12)
    ax_heatmap.set_yticklabels(ax_heatmap.get_yticklabels(), rotation=0, fontsize=12)

    # Plot the distribution of correlation values on the second subplot
    ax_hist.hist(corr_values, bins=10, color="gray", edgecolor="black")
    ax_hist.set_title("Correlation Value Distribution")
    ax_hist.set_xlabel("Correlation")
    ax_hist.set_ylabel("Frequency")

    # Save the figure
    plt.tight_layout()
    plt.savefig(save_figure_name, bbox_inches="tight")
    plt.show()


def correlation_heatmap_by_stage(
    predictions: np.ndarray[float], labels: List[str], sleep_score: pd.DataFrame, result_path: str
) -> None:
    for i, (i_stage, stage_label, start_index, end_index) in enumerate(
        sleep_stage_iterator(sleep_score, predictions.shape[0], SLEEP_STAGE_THRESH)
    ):
        predictions_stage = predictions[start_index:end_index, :]
        stage_label = stage_label.replace("/", "")
        file_extension = os.path.splitext(result_path)[1]
        figure_name = result_path.replace(file_extension, f"_{i_stage}_{stage_label}{file_extension}")
        correlation_heatmap(predictions_stage, labels, figure_name)


def multi_facet_correlation_heatmap(
    predictions: np.ndarray[float], labels: List[str], sleep_score: pd.DataFrame, result_path: str
) -> None:
    """
    Plots a faceted grid of heatmaps for a 3D correlation matrix.
    Each layer of the 3D matrix is represented as a separate heatmap for better clarity.
    Histograms are created for each unique label combining correlation values with the same label.

    Args:
        predictions (np.ndarray): model activation samples by labels.
        labels (list): List of labels for each dimension (length N).
        sleep_score: N by 2 dataframe. 1st column: sleep stage, 2nd column start index corresponding predictions.
        result_path: the full path of result file.
    """

    if sleep_score is None:
        warnings.warn("sleep score is None")
        return

    sleep_stages, correlation_matrix, sleep_stages_index = get_correlation_matrix_by_stage(predictions, sleep_score)

    num_label, _, num_stage = correlation_matrix.shape
    if len(sleep_stages) != num_stage:
        raise ValueError("The length of labels must match the dimensions of the correlation matrix.")

    # Identify unique labels and assign a color map to each
    unique_labels = list(set(sleep_stages))
    colormap_dict = {}
    for i, unique_label in enumerate(unique_labels):
        colormap_dict[unique_label] = SLEEP_STAGE_COLORMAP[i]

    # Organize plots in a grid with 5 heatmaps per row with additional histgram
    num_rows = num_stage // 5 + 1  # Calculate the number of rows needed (5 per row)
    # fig, axes = plt.subplots(num_rows, 5, figsize=(25, 5 * num_rows), constrained_layout=True)
    fig, axes = plt.subplots(num_rows, 5, figsize=(25, 5 * num_rows))

    # Flatten axes for easier iteration if there are multiple rows
    if num_rows > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    # Create a dictionary to collect correlation values by label
    correlation_by_label = defaultdict(list)

    # Iterate through each layer and create a heatmap
    for stage in range(num_stage):
        ax = axes[stage]

        # Determine the colormap based on the first label in each layer
        cmap = colormap_dict.get(sleep_stages[stage], "viridis")

        # Create the heatmap
        sb.heatmap(
            correlation_matrix[:, :, stage],
            annot=False,
            fmt=".2f",
            cmap=cmap,
            cbar=False,
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
            square=True,
        )

        # Annotate only the lower triangle
        for i in range(num_label):
            for j in range(i):  # Only for lower triangle
                ax.text(
                    j + 0.5,
                    i + 0.5,  # Offset for centering text
                    f"{correlation_matrix[i, j, stage]:.2f}",
                    ha="center",
                    va="center",
                    color="black" if correlation_matrix[i, j, stage] < 0 else "white",
                )

        ax.set_title(
            f"Stage {sleep_stages_index[stage]}: {sleep_stages[stage]}", fontdict={"fontsize": 13, "weight": 500}
        )
        ax.tick_params(axis="x", rotation=45, labelsize=12)
        ax.tick_params(axis="y", rotation=45, labelsize=12)

        # Collect correlation values by unique label
        for i in range(num_label):
            for j in range(i + 1, num_label):
                correlation_by_label[sleep_stages[stage]].append(correlation_matrix[i, j, stage])

    ax_hist = axes[num_stage]
    for idx, label in enumerate(unique_labels):
        ax_hist.hist(
            correlation_by_label[label],
            density=True,
            bins=20,
            color=sb.color_palette(colormap_dict[label])[3],
            label=f"{label}: ({len(correlation_by_label[label])})",
            alpha=0.7,
            edgecolor=None,
        )

    ax_hist.set_aspect(0.35)
    ax_hist.legend(title="", frameon=False, handletextpad=-6, handlelength=6.5, loc="best", fontsize=10)
    ax_hist.set_title(f"Histogram of Correlations", fontdict={"fontsize": 13, "weight": 500})
    ax_hist.set_xlabel("Correlation Value")
    ax_hist.set_ylabel("Density")

    # Remove any empty subplots
    for idx in range(num_stage + 1, num_rows * 5):
        fig.delaxes(axes[idx])

    plt.subplots_adjust(wspace=0.05, hspace=0.05)

    plt.tight_layout()
    # plt.savefig(result_path, bbox_inches="tight")
    plt.savefig(result_path)
    plt.show()


def get_correlation_matrix_by_stage(
    predictions: np.ndarray[float],
    sleep_score: pd.DataFrame,
) -> Tuple[List[str], np.ndarray[float, Any], List[int]]:
    """
    Calculate correlation matrices for different sleep stages.

    Args:
        predictions (np.ndarray): A 2D array of predictions (samples x features).
        sleep_score (pd.DataFrame): DataFrame containing sleep stage labels and start/end indices.

    Returns:
        Tuple[List[str], np.ndarray, List[int]]: A list of stage labels and a 3D numpy array of correlation matrices.
    """

    correlation_matrices = []
    sleep_stages = []
    sleep_stages_index = []
    for i, (i_stage, stage_label, start_index, end_index) in enumerate(
        sleep_stage_iterator(sleep_score, predictions.shape[0], SLEEP_STAGE_THRESH)
    ):
        predictions_stage = predictions[start_index:end_index, :]
        sleep_stages.append(stage_label.replace("/", "-"))
        sleep_stages_index.append(i_stage)
        corr_matrix = np.corrcoef(predictions_stage, rowvar=False)
        correlation_matrices.append(corr_matrix[:, :, np.newaxis])

    correlation_matrix = np.concatenate(correlation_matrices, axis=2)
    return sleep_stages, correlation_matrix, sleep_stages_index


def prediction_heatmap(predictions: np.ndarray[float], events: Events, title: str, file_path: str):
    fig, ax = plt.subplots(figsize=(4, 8))
    heatmap = ax.imshow(predictions, cmap="viridis", aspect="auto", interpolation="none")

    for concept_i, concept_vocalizations in enumerate(events):
        if not len(concept_vocalizations) > 0:
            continue
        for concept_vocalization in concept_vocalizations:
            t = 4 * concept_vocalization / 1000
            ax.axhline(
                y=t,
                color="red",
                linestyle="-",
                alpha=0.6,
                xmin=concept_i / len(events),
                xmax=(concept_i + 1) / len(events),
            )

    cbar = plt.colorbar(heatmap)
    cbar.ax.tick_params(labelsize=10)
    tick_positions = np.arange(0, len(predictions), 15 * 4)  # 15 seconds * 100 samples per second
    tick_labels = [int(pos * 0.25) for pos in tick_positions]
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels)
    ax.set_xticks(np.arange(0, predictions.shape[1], 1))
    ax.set_xticklabels(events.events_name, rotation=80)

    ax.set_ylabel("Time (s)")
    ax.set_xlabel("Concept")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(file_path)
    plt.show()


def smooth_data(data: np.ndarray[float], window_size: int = 5) -> np.ndarray[float, Any]:
    return np.convolve(data, np.ones(window_size) / window_size, mode="valid")


def smooth_columns(data: np.ndarray[float], window_size: int = 5) -> np.ndarray[float, Any]:
    n_rows = data.shape[0]
    smoothed_data = np.zeros((n_rows - window_size + 1, data.shape[1]))  # Adjust size for smoothing

    # Smoothing each column
    for i in range(data.shape[1]):
        smoothed_data[:, i] = smooth_data(data[:, i], window_size=window_size)

    return smoothed_data


def combine_continuous_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine rows with continuous same values in the 'Score' column and keep the first value in the 'start_index' column.
    This is used to combine continuous sleep stages in sleep score.

    Parameters:
    - df (pd.DataFrame): A DataFrame with 'Score' and 'start_index' columns.

    Returns:
    - pd.DataFrame: A new DataFrame with combined rows, keeping the first 'start_index' value for each group.
    """

    # Create a mask to identify where the 'Score' changes
    df["group"] = (df["Score"] != df["Score"].shift()).cumsum()
    combined_df = df.groupby("group").agg({"Score": "first", "start_index": "first"}).reset_index(drop=True)
    combined_df = combined_df[["Score", "start_index"]]

    return combined_df


def sleep_stage_iterator(sleep_score: pd.DataFrame, last_index: int, duration_thresh: int) -> Tuple[str, int, int]:
    for i in range(len(sleep_score)):
        stage_label = sleep_score.iloc[i]["Score"]
        start_index = sleep_score.iloc[i]["start_index"]

        if start_index > last_index:
            break

        if i < len(sleep_score) - 1:
            next_start_index = sleep_score.iloc[i + 1]["start_index"]
        else:
            next_start_index = last_index

        if next_start_index - start_index > duration_thresh * PREDICTION_FS:
            yield i, stage_label, start_index, next_start_index


def prediction_iterator(
    prediction: np.ndarray,
    sleep_score: pd.DataFrame,
    value_thresh: float,
    length_thresh: int,
) -> Dict:
    """
    Creates an iterator to loop through slices of the prediction array based on start indices in the sleep_score
    DataFrame.

    Args:
        prediction (np.ndarray): A 1D numpy array of prediction values.
        sleep_score (pd.DataFrame): A DataFrame with 'Score' (str) and 'start_index' (int) columns.
        value_thresh: remove prediction values less than this value.
        length_thresh (seconds): ignore sleep stage shorter than this value.

    Yields:
        Iterator[Tuple[str, np.ndarray]]: An iterator that yields a tuple with a label and the corresponding
                                          slice of the prediction array.
    """
    for i, (i_stage, label, start_index, end_index) in enumerate(
        sleep_stage_iterator(sleep_score, len(prediction), length_thresh)
    ):
        stage_data = prediction[start_index:end_index]
        stage_data = stage_data[stage_data > value_thresh]  # Filter values greater than 0.5
        # Calculate stage length (duration in seconds)
        stage_length = (end_index - start_index) / PREDICTION_FS
        stage_label = f"Stage: {i_stage} ({stage_length:.1f} sec)"

        yield {
            "Stage": [stage_label] * len(stage_data),
            "Value(>.5)": stage_data,
            "Label": [label] * len(stage_data),
            "Stage Label": [sleep_score.iloc[i_stage]["Score"]] * len(stage_data),
        }


def read_sleep_score(filename: str, signal_fs: float = PREDICTION_FS) -> pd.DataFrame:
    sleep_score = pd.read_csv(filename, header=0)
    print(
        f"shape of sleep_score: {sleep_score.shape}, "
        f"duration: {sleep_score.shape[0] / SLEEP_SCORE_FS / SECONDS_PER_HOUR} hours"
    )
    sleep_score["start_index"] = [
        int(i * signal_fs / SLEEP_SCORE_FS + SLEEP_SCORE_OFFSET) for i in range(sleep_score.shape[0])
    ]
    sleep_score = combine_continuous_scores(sleep_score)

    print(f"shape of sleep_score after merge: {sleep_score.shape}")

    return sleep_score


def load_prediction(activation_file: str) -> np.ndarray[float]:
    predictions = np.load(activation_file)
    print(
        f"shape of predictions: {predictions.shape}, duration: {predictions.shape[0] / PREDICTION_FS / SECONDS_PER_HOUR} hours"
    )

    return predictions


if __name__ == "__main__":
    from brain_decoding.config.file_path import TWILIGHT_LABEL_PATH
    from brain_decoding.param.param_data import TWILIGHT_LABELS

    plot_concept_graph(TWILIGHT_LABEL_PATH, TWILIGHT_LABELS)
