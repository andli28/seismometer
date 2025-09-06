import logging
from typing import Any, Optional

import numpy as np
import pandas as pd
from IPython.display import HTML, SVG, IFrame, display
from pandas.io.formats.style import Styler

import seismometer.plot as plot
from seismometer.controls.decorators import disk_cached_html_and_df_segment
from seismometer.core.autometrics import AutomationManager, store_call_parameters
from seismometer.core.decorators import export
from seismometer.data import get_cohort_data, get_cohort_performance_data, metric_apis
from seismometer.data import pandas_helpers as pdh
from seismometer.data.filter import FilterRule
from seismometer.data.performance import (
    STATNAMES,
    BinaryClassifierMetricGenerator,
    assert_valid_performance_metrics_df,
    calculate_bin_stats,
    calculate_eval_ci,
    default_cohort_summaries,
    get_cohort_data,
    get_cohort_performance_data,
)
from .data import pandas_helpers as pdh
from .data import score_target_cohort_summaries
from .data.decorators import export
from .data.filter import FilterRule
from .data.timeseries import create_metric_timeseries
from .html import template
from .html.iframe import load_as_iframe
from .report.auditing import fairness_audit_altair
from .report.profiling import ComparisonReportWrapper, SingleReportWrapper
from .seismogram import Seismogram

logger = logging.getLogger("seismometer")


# region Reports
@export
def feature_alerts(exclude_cols: Optional[list[str]] = None):
    """
    Generates (or loads from disk) the `ydata-profiling` report feature quality alerts.

    Note: Does not regenerate the report if one exists on disk.

    Parameters
    ----------
    exclude_cols : Optional[list[str]], optional
        Columns to exclude from profiling. If None, defaults to excluding the identifiers in the dataset,
        by default None.
    """
    sg = Seismogram()
    exclude_cols = exclude_cols or sg.entity_keys

    SingleReportWrapper(
        df=sg.dataframe,
        output_path=sg.config.output_dir,
        exclude_cols=exclude_cols,
        title="Feature Report",
        alert_config=sg.alert_config,
    ).display_alerts()


def feature_summary(exclude_cols: Optional[list[str]] = None, inline: bool = False):
    """
    Generates (or loads from disk) the `ydata-profiling` report.

    Note: Does not regenerate the report if one exists on disk.

    Parameters
    ----------
    exclude_cols : Optional[list[str]], optional
        Columns to exclude from profiling. If None, defaults to excluding the identifiers in the dataset,
        by default None.
    inline : bool, optional
        If True, shows the `ydata-profiling` report inline, by default False; displaying a link instead.
    """
    sg = Seismogram()
    exclude_cols = exclude_cols or sg.entity_keys

    SingleReportWrapper(
        df=sg.dataframe,
        output_path=sg.config.output_dir,
        exclude_cols=exclude_cols,
        title="Feature Report",
        alert_config=sg.alert_config,
    ).display_report(inline)


@export
def cohort_comparison_report(exclude_cols: list[str] = None):
    """
    Generates (or loads from disk) the `ydata-profiling` report stratified by the selected cohort variable.

    Note: Does not regenerate the report if one exists on disk.

    Parameters
    ----------
    exclude_cols : Optional[list[str]], optional
        Columns to exclude from profiling. If None, defaults to excluding the identifiers in the dataset,
        by default None.
    """
    sg = Seismogram()
    from .controls.cohort_comparison import ComparisonReportGenerator

    comparison_selections = ComparisonReportGenerator(sg.available_cohort_groups, exclude_cols=exclude_cols)
    comparison_selections.show()


@export
def target_feature_summary(exclude_cols: list[str] = None, inline=False):
    """
    Generates (or loads from disk) the `ydata-profiling` report stratified by the target variable.

    Note: Does not regenerate the report if one exists on disk.

    Parameters
    ----------
    exclude_cols : Optional[list[str]], optional
        Columns to exclude from profiling. If None, defaults to excluding the identifiers in the dataset,
        by default None.
    inline : bool, optional
        True to show the `ydata-profiling` report inline, by default False; displaying a link instead.
    """
    sg = Seismogram()

    exclude_cols = exclude_cols or sg.entity_keys
    positive_target = FilterRule.eq(sg.target, 1)
    negative_target = ~positive_target

    negative_target_df = negative_target.filter(sg.dataframe)
    positive_target_df = positive_target.filter(sg.dataframe)

    if negative_target_df.empty:
        logger.warning("No comparison report generated. The negative target has no data to profile.")
        return

    if positive_target_df.empty:
        logger.warning("No comparison report generated. The positive target has no data to profile.")
        return

    wrapper = ComparisonReportWrapper(
        l_df=negative_target_df,
        r_df=positive_target_df,
        output_path=sg.output_path,
        l_title=f"{sg.target}=0",
        r_title=f"{sg.target}=1",
        exclude_cols=exclude_cols,
        base_title="Target Comparison Report",
    )

    wrapper.display_report(inline)


@export
def fairness_audit(metric_list: Optional[list[str]] = None, fairness_threshold: float = 1.25) -> HTML:
    """
    Displays the Aequitas fairness audit for a set of sensitive groups and metrics.

    Parameters
    ----------
    metric_list : Optional[list[str]
        The list of metrics to use in Aequitas. Chosen from:
            "tpr",
            "tnr",
            "for",
            "fdr",
            "fpr",
            "fnr",
            "npv",
            "ppr",
            "precision",
            "pprev".
    fairness_threshold : float
        The maximum ratio between sensitive groups before differential performance is considered a 'failure'.
    """
    sg = Seismogram()

    sensitive_groups = sg.cohort_cols
    metric_list = metric_list or ["tpr", "fpr", "pprev"]

    return generate_fairness_audit(
        sensitive_groups,
        sg.target,
        sg.output,
        sg.thresholds[0],
        per_context=True,
        metric_list=metric_list,
        fairness_threshold=fairness_threshold,
    )


@export
def generate_fairness_audit(
    cohort_columns: list[str],
    target_column: str,
    score_column: str,
    score_threshold: float,
    per_context: bool = False,
    metric_list: Optional[list[str]] = None,
    fairness_threshold: float = 1.25,
) -> HTML | IFrame | Any:
    """
    Generates the Aequitas fairness audit for a set of sensitive groups and metrics.

    Parameters
    ----------
    cohort_columns : list[str]
        cohort columns to investigate
    target_column : str
        target column to use
    score_column : str
        score column to use
    score_threshold : float
        threshold at which a score predicts a positive event
    per_context : bool, optional
        if scores should be grouped within a context, by default False
    metric_list : Optional[list[str]], optional
        list of metrics to evaluate, by default None
    fairness_threshold : float, optional
        allowed deviation from the default class within a cohort, by default 1.25

    Returns
    -------
    HTML | IFrame | Altair Chart
        IFrame holding the HTML of the audit
    """
    sg = Seismogram()
    path = "aequitas_{cohorts}_with_{target}_and_{score}_gt_{threshold}_metrics_{metrics}_ratio_{ratio}".format(
        cohorts="_".join(cohort_columns),
        target=target_column,
        score=score_column,
        threshold=score_threshold,
        metrics="_".join(metric_list),
        ratio=fairness_threshold,
    )
    if per_context:
        path += "_grouped"
    fairness_path = sg.config.output_dir / (slugify(path) + ".html")
    height = 200 + 100 * len(metric_list)

    if NotebookHost.supports_iframe() and fairness_path.exists():
        return load_as_iframe(fairness_path, height=height)

    target = pdh.event_value(target_column)
    data = (
        pdh.event_score(
            sg.data(),
            sg.entity_keys,
            score=sg.output,
            ref_event=sg.predict_time,
            aggregation_method=sg.event_aggregation_method(sg.target),
        )
        if per_context
        else sg.data()
    )

    data = data[[target, score_column] + cohort_columns]
    data = FilterRule.isin(target, (0, 1)).filter(data)
    if len(data.index) < sg.censor_threshold:
        return template.render_censored_plot_message(sg.censor_threshold)

    try:
        altair_plot = fairness_audit_altair(
            data, cohort_columns, score_column, target, score_threshold, metric_list, fairness_threshold
        )
    except CensoredResultException as error:
        return template.render_censored_data_message(error.message)

    if NotebookHost.supports_iframe():
        altair_plot.save(fairness_path, format="html")
        return load_as_iframe(fairness_path, height=height)

    return altair_plot


# endregion

# region notebook IPWidgets


@export
def cohort_list():
    """
    Displays an exhaustive list of available cohorts for analysis.
    """
    sg = Seismogram()
    from ipywidgets import Output, VBox

    from .controls.selection import MultiSelectionListWidget

    options = sg.available_cohort_groups

    comparison_selections = MultiSelectionListWidget(options, title="Cohort")
    output = Output()

    def on_widget_value_changed(*args):
        output.clear_output(wait=True)
        with output:
            display("Recalculating...")
            output.clear_output(wait=True)
            html = _cohort_list_details(comparison_selections.value)
            display(html)

    comparison_selections.observe(on_widget_value_changed, "value")

    # get initial value
    on_widget_value_changed()

    return VBox(children=[comparison_selections, output])


@disk_cached_html_segment
def _cohort_list_details(cohort_dict: dict[str, tuple[Any]]) -> HTML:
    """
    Generates a HTML table of cohort details.

    Parameters
    ----------
    cohort_dict : dict[str, tuple[Any]]
        dictionary of cohort columns and values used to subselect a population for evaluation


    Returns
    -------
    HTML
        able indexed by targets, with counts of unique entities, and mean values of the output columns.
    """
    from .data.filter import filter_rule_from_cohort_dictionary

    sg = Seismogram()
    rule = filter_rule_from_cohort_dictionary(cohort_dict)
    data = rule.filter(sg.dataframe)
    cohort_count = data[sg.entity_keys[0]].nunique()
    if cohort_count < sg.censor_threshold:
        return template.render_censored_plot_message(sg.censor_threshold)

    cfg = sg.config
    target_cols = [pdh.event_value(x) for x in cfg.targets]
    intervention_cols = [pdh.event_value(x) for x in cfg.interventions]
    outcome_cols = [pdh.event_value(x) for x in cfg.outcomes]
    groups = data[cfg.entity_keys + cfg.output_list + intervention_cols + outcome_cols + target_cols].groupby(
        target_cols
    )
    aggregation = {cfg.entity_id: ["count", "nunique"]}
    if len(cfg.context_id):
        aggregation[cfg.context_id] = "nunique"
    # add in other keys for aggregation
    aggregation.update({k: "mean" for k in cfg.output_list + intervention_cols + outcome_cols})
    title = "Summary"
    html_table = groups.agg(aggregation).to_html()
    return template.render_title_message(title, html_table)


# endregion


# region plot accessors
@export
def plot_cohort_hist():
    """Display a histogram plot of predicted probabilities for all cohorts in the selected attribute."""
    sg = Seismogram()
    cohort_col = sg.selected_cohort[0]
    subgroups = sg.selected_cohort[1]
    censor_threshold = sg.censor_threshold
    return _plot_cohort_hist(sg.data(), sg.target, sg.output, cohort_col, subgroups, censor_threshold)


@disk_cached_html_and_df_segment
@export
def plot_cohort_group_histograms(
    cohort_col: str, subgroups: list[str], target_column: str, score_column: str
) -> tuple[HTML, pd.DataFrame]:
    """
    Generate a histogram plot of predicted probabilities for each subgroup in a cohort.

    Parameters
    ----------
    cohort_col : str
        column for cohort splits
    subgroups : list[str]
        column values to split by
    target_column : str
        target column
    score_column : str
        score column

    Returns
    -------
    tuple[HTML, pd.DataFrame]
        html visualization of the histogram and the data used to generate it
    """
    sg = Seismogram()
    target_column = pdh.event_value(target_column)
    target_data = FilterRule.isin(target_column, (0, 1)).filter(sg.dataframe)
    return _plot_cohort_hist(
        target_data,
        target_column,
        score_column,
        cohort_col,
        subgroups,
        sg.censor_threshold,
    )


def _plot_cohort_hist(
    dataframe: pd.DataFrame,
    target: str,
    output: str,
    cohort_col: str,
    subgroups: list[str],
    censor_threshold: int = 10,
) -> tuple[HTML, pd.DataFrame]:
    """
    Creates an HTML segment displaying a histogram of predicted probabilities for each cohort.

    Parameters
    ----------
    dataframe : pd.DataFrame
        data source
    target : str
        event value for the target
    output : str
        score column
    cohort_col : str
        column for cohort splits
    subgroups : list[str]
        column values to split by
    censor_threshold : int
        minimum rows to allow in a plot, by default 10.

    Returns
    -------
    HTML
        A stacked set of histograms for each selected subgroup in the cohort.
    """
    cData = get_cohort_data(dataframe, cohort_col, proba=output, true=target, splits=subgroups)

    # filter by groups by size
    cCount = cData["cohort"].value_counts()
    good_groups = cCount.loc[cCount > censor_threshold].index
    cData = cData.loc[cData["cohort"].isin(good_groups)]

    if len(cData.index) == 0:
        return template.render_censored_plot_message(censor_threshold), cData

    bins = np.histogram_bin_edges(cData["pred"], bins=20)
    try:
        svg = plot.cohorts_vertical(cData, plot.histogram_stacked, func_kws={"show_legend": False, "bins": bins})
        title = f"Predicted Probabilities by {cohort_col}"
        return template.render_title_with_image(title, svg), cData
    except Exception as error:
        return template.render_title_message("Error", f"Error: {error}"), pd.DataFrame()


@export
def plot_leadtime_enc(score=None, ref_time=None, target_event=None):
    """Displays the amount of time that a high prediction gives before an event of interest.

    Parameters
    ----------
    score : Optional[str], optional
        The name of the score column to use, by default None; uses sg.output.
    ref_time : Optional[str], optional
        The reference time used in the visualization, by default None; uses sg.predict_time.
    target_event : Optional[str], optional
        The name of the target column to use, by default None; uses sg.target.
    """
    sg = Seismogram()
    # Assume sg.dataframe has scores/events with appropriate windowed event
    cohort_col = sg.selected_cohort[0]
    subgroups = sg.selected_cohort[1]
    censor_threshold = sg.censor_threshold
    x_label = "Lead Time (hours)"

    score = score or sg.output
    ref_time = ref_time or sg.predict_time
    target_event = pdh.event_value(target_event) or sg.target
    target_zero = pdh.event_time(target_event) or sg.time_zero
    max_hours = sg.event_aggregation_window_hours(target_event)
    threshold = sg.thresholds[0]
    target_data = FilterRule.isin(target_event, (0, 1)).filter(sg.dataframe)

    return _plot_leadtime_enc(
        target_data,
        sg.entity_keys,
        target_event,
        target_zero,
        score,
        threshold,
        ref_time,
        cohort_col,
        subgroups,
        max_hours,
        x_label,
        censor_threshold,
    )


@store_call_parameters(cohort_col="cohort_col", subgroups="subgroups")
@disk_cached_html_and_df_segment
@export
def plot_cohort_lead_time(
    cohort_col: str, subgroups: list[str], event_column: str, score_column: str, threshold: float
) -> tuple[HTML, pd.DataFrame]:
    """
    Plots a lead times between the first positive prediction give an threshold and an event.

    Parameters
    ----------
    cohort_col : str
        cohort column name
    subgroups : list[str]
        subgroups of interest in the cohort column
    event_column : str
        event column name
    score_column : str
        score column name
    threshold : float
        _description_

    Returns
    -------
    HTML
        _description_
    """
    sg = Seismogram()
    x_label = "Lead Time (hours)"
    event_value = pdh.event_value(event_column)
    event_time = pdh.event_time(event_column)
    target_data = FilterRule.isin(event_value, (0, 1)).filter(sg.dataframe)
    max_hours = sg.event_aggregation_window_hours(event_value)

    return _plot_leadtime_enc(
        target_data,
        sg.entity_keys,
        event_value,
        event_time,
        score_column,
        threshold,
        sg.predict_time,
        cohort_col,
        subgroups,
        max_hours,
        x_label,
        sg.censor_threshold,
    )


def _plot_leadtime_enc(
    dataframe: pd.DataFrame,
    entity_keys: str,
    target_event: str,
    target_zero: str,
    score: str,
    threshold: list[float],
    ref_time: str,
    cohort_col: str,
    subgroups: list[any],
    max_hours: int,
    x_label: str,
    censor_threshold: int = 10,
) -> tuple[HTML, pd.DataFrame]:
    """
    HTML Plot of time between prediction and target event.

    Parameters
    ----------
    dataframe : pd.DataFrame
        source data
    target_event : str
        event column
    target_zero : str
        event value
    threshold : str
        score thresholds
    score : str
        score column
    ref_time : str
        prediction time
    entity_keys : str
        entity key column
    cohort_col : str
        cohort column name
    subgroups : list[any]
        cohort groups from the cohort column
    x_label : str
        label for the x axis of the plot
    max_hours : _type_
        max horizon time
    censor_threshold : int
        minimum rows to allow in a plot.

    Returns
    -------
    tuple[HTML, pd.DataFrame]
        Lead time plot and the data used to generate it
    """
    if target_event not in dataframe:
        msg = f"Target event ({target_event}) not found in dataset. Cannot plot leadtime."
        logger.error(msg)
        return template.render_title_message("Error", msg), pd.DataFrame()

    if target_zero not in dataframe:
        msg = f"Target event time-zero ({target_zero}) not found in dataset. Cannot plot leadtime."
        logger.error(msg)
        return template.render_title_message("Error", msg), pd.DataFrame()

    summary_data = dataframe[dataframe[target_event] == 1]
    if len(summary_data.index) == 0:
        msg = f"No positive events ({target_event}=1) were found"
        logger.error(msg)
        return template.render_title_message("Error", msg), pd.DataFrame()

    cohort_mask = summary_data[cohort_col].isin(subgroups)
    threshold_mask = summary_data[score] > threshold

    # summarize to first score
    summary_data = pdh.event_score(
        summary_data[cohort_mask & threshold_mask],
        entity_keys,
        score=score,
        ref_event=target_zero,
        aggregation_method="first",
    )
    if summary_data is not None and len(summary_data) > censor_threshold:
        summary_data = summary_data[[target_zero, ref_time, cohort_col]]
    else:
        return template.render_censored_plot_message(censor_threshold), pd.DataFrame()

    # filter by group size
    counts = summary_data[cohort_col].value_counts()
    good_groups = counts.loc[counts > censor_threshold].index
    summary_data = summary_data.loc[summary_data[cohort_col].isin(good_groups)]

    if len(summary_data.index) == 0:
        return template.render_censored_plot_message(censor_threshold), summary_data

    # Truncate to minute but plot hour
    summary_data[x_label] = (summary_data[ref_time] - summary_data[target_zero]).dt.total_seconds() // 60 / 60

    title = f'Lead Time from {score.replace("_", " ")} to {(target_zero).replace("_", " ")}'
    rows = summary_data[cohort_col].nunique()
    svg = plot.leadtime_violin(summary_data, x_label, cohort_col, xmax=max_hours, figsize=(9, 1 + rows))
    return template.render_title_with_image(title, svg), summary_data


@store_call_parameters(cohort_col="cohort_col", subgroups="subgroups")
@disk_cached_html_and_df_segment
@export
def plot_cohort_evaluation(
    cohort_col: str,
    subgroups: list[str],
    target_column: str,
    score_column: str,
    thresholds: list[float],
    per_context: bool = False,
) -> tuple[HTML, pd.DataFrame]:
    """
    Plots model performance metrics split by on a cohort attribute.

    Parameters
    ----------
    cohort_col : str
        cohort column name
    subgroups : list[str]
        subgroups of interest in the cohort column
    target_column : str
        target column
    score_column : str
        score column
    thresholds : list[float]
        thresholds to highlight
    per_context : bool, optional
        if scores should be grouped, by default False

    Returns
    -------
    HTML
        an html visualization of the model performance metrics
    """
    sg = Seismogram()
    target_event = pdh.event_value(target_column)
    target_data = FilterRule.isin(target_event, (0, 1)).filter(sg.dataframe)
    return _plot_cohort_evaluation(
        target_data,
        sg.entity_keys,
        target_event,
        score_column,
        thresholds,
        cohort_col,
        subgroups,
        sg.censor_threshold,
        per_context,
    )


def _plot_cohort_evaluation(
    dataframe: pd.DataFrame,
    entity_keys: list[str],
    target: str,
    output: str,
    thresholds: list[float],
    cohort_col: str,
    subgroups: list[str],
    censor_threshold: int = 10,
    per_context_id: bool = False,
    aggregation_method: str = "max",
    ref_time: str = None,
) -> tuple[HTML, pd.DataFrame]:
    """
    Plots model performance metrics split by on a cohort attribute.

    Parameters
    ----------
    dataframe : pd.DataFrame
        source data
    entity_keys : list[str]
        columns to use for aggregation
    target : str
        target value
    output : str
        score column
    thresholds : list[float]
        model thresholds
    cohort_col : str
        cohort column name
    subgroups : list[str]
        subgroups of interest in the cohort column
    censor_threshold : int
        minimum rows to allow in a plot, by default 10
    per_context_id : bool, optional
        if true, aggregate scores for each context, by default False
    aggregation_method : str, optional
        method to reduce multiple scores into a single value before calculation of performance, by default "max"
        ignored if per_context_id is False
    ref_time : str, optional
        reference time column used for aggregation when per_context_id is True and aggregation_method is time-based

    Returns
    -------
    HTML
        _description_
    """
    data = (
        pdh.event_score(
            dataframe, entity_keys, score=output, ref_event=ref_time, aggregation_method=aggregation_method
        )
        if per_context_id
        else dataframe
    )

    plot_data = get_cohort_performance_data(
        data, cohort_col, proba=output, true=target, splits=subgroups, censor_threshold=censor_threshold
    )
    try:
        assert_valid_performance_metrics_df(plot_data)
    except ValueError:
        return template.render_censored_plot_message(censor_threshold), plot_data
    svg = plot.cohort_evaluation_vs_threshold(plot_data, cohort_feature=cohort_col, highlight=thresholds)
    title = f"Model Performance Metrics on {cohort_col} across Thresholds"
    return template.render_title_with_image(title, svg), plot_data


@store_call_parameters(cohort_dict="cohort_dict")
@disk_cached_html_and_df_segment
@export
def plot_model_evaluation(
    cohort_dict: dict[str, tuple[Any]],
    target_column: str,
    score_column: str,
    thresholds: list[float],
    per_context: bool = False,
) -> tuple[HTML, pd.DataFrame]:
    """
    Generates a 2x3 plot showing the performance of a model.

    This includes the ROC, recall vs predicted condition prevalence, calibration,
    PPV vs sensitivity, sensitivity/specificity/ppv, and a histogram.

    Parameters
    ----------
    cohort_dict : dict[str, tuple[Any]]
        dictionary of cohort columns and values used to subselect a population for evaluation
    target_column : str
        target column
    score_column : str
        score column
    thresholds : list[float]
        thresholds to highlight
    per_context : bool, optional
        if scores should be grouped, by default False

    Returns
    -------
    HTML
        an html visualization of the model evaluation metrics
    """
    sg = Seismogram()
    cohort_filter = FilterRule.from_cohort_dictionary(cohort_dict)
    data = cohort_filter.filter(sg.dataframe)
    target_event = pdh.event_value(target_column)
    target_data = FilterRule.isin(target_event, (0, 1)).filter(data)
    return _model_evaluation(
        target_data,
        sg.entity_keys,
        target_column,
        target_event,
        score_column,
        thresholds,
        sg.censor_threshold,
        per_context,
        sg.event_aggregation_method(sg.target),
        sg.predict_time,
    )


def _model_evaluation(
    dataframe: pd.DataFrame,
    entity_keys: list[str],
    target_event: str,
    target: str,
    score_col: str,
    thresholds: Optional[list[float]],
    censor_threshold: int = 10,
    per_context_id: bool = False,
    aggregation_method: str = "max",
    ref_time: Optional[str] = None,
    cohort: dict = {},
) -> tuple[HTML, pd.DataFrame]:
    """
    plots common model evaluation metrics

    Parameters
    ----------
    dataframe : pd.DataFrame
        source data
    entity_keys : list[str]
        columns to use for aggregation
    target_event : str
        target event name
    target : str
        target column
    score_col : str
        score column
    thresholds : Optional[list[float]]
        model thresholds
    censor_threshold : int, optional
        minimum rows to allow in a plot, by default 10
    per_context_id : bool, optional
        report only the max score for a given entity context, by default False
    aggregation_method : str, optional
        method to reduce multiple scores into a single value before calculation of performance, by default "max"
        ignored if per_context_id is False
    ref_time : str, optional
        reference time column used for aggregation when per_context_id is True and aggregation_method is time-based

    Returns
    -------
    HTML
        Plot of model evaluation metrics
    """
    data = (
        pdh.event_score(
            dataframe, entity_keys, score=score_col, ref_event=ref_time, aggregation_method=aggregation_method
        )
        if per_context_id
        else dataframe
    )

    # Validate
    requirements = FilterRule.isin(target, (0, 1)) & FilterRule.notna(score_col)
    data = requirements.filter(data)
    if len(data.index) < censor_threshold:
        return template.render_censored_plot_message(censor_threshold), data
    if (lcount := data[target].nunique()) != 2:
        return (
            template.render_title_message(
                "Evaluation Error", f"Model Evaluation requires exactly two classes but found {lcount}"
            ),
            data,
        )
    stats = calculate_bin_stats(data[target], data[score_col])
    ci_data = calculate_eval_ci(stats, data[target], data[score_col], conf=0.95)
    title = f"Overall Performance for {target_event} (Per {'Encounter' if per_context_id else 'Observation'})"
    svg = plot.evaluation(
        stats,
        ci_data=ci_data,
        truth=data[target],
        output=data[score_col].values,
        show_thresholds=True,
        highlight=thresholds,
    )
    return template.render_title_with_image(title, svg), data


def plot_trend_intervention_outcome() -> HTML:
    """
    Plots two timeseries based on selectors; an outcome and then an intervention.

    Makes use of the cohort selectors as well intervention and outcome selectors for which data to use.
    Uses the configuration for comparison_time as the reference time for both plots.
    """
    sg = Seismogram()
    if not sg.outcome or not sg.intervention:
        return HTML("No outcome or intervention configured.")
    return _plot_trend_intervention_outcome(
        sg.dataframe,
        sg.entity_keys,
        sg.selected_cohort[0],
        sg.selected_cohort[1],
        sg.outcome,
        sg.intervention,
        sg.comparison_time or sg.predict_time,
        sg.censor_threshold,
    )


@disk_cached_html_and_df_segment
@export
def plot_intervention_outcome_timeseries(
    cohort_col: str,
    subgroups: list[str],
    outcome: str,
    intervention: str,
    reference_time_col: str,
    censor_threshold: int = 10,
) -> tuple[HTML, pd.DataFrame]:
    """
    Plots two timeseries based on an outcome and an intervention.

    cohort_col : str
        column name for the cohort to split on
    subgroups : list[str]
        values of interest in the cohort column
    outcome : str
        outcome event time column
    intervention : str
        intervention event time column
    reference_time_col : str
        reference time column for alignment
    censor_threshold : int, optional
        minimum rows to allow in a plot, by default 10

    Returns
    -------
    HTML
        Plot of two timeseries
    """
    sg = Seismogram()
    return _plot_trend_intervention_outcome(
        sg.dataframe,
        sg.entity_keys,
        cohort_col,
        subgroups,
        outcome,
        intervention,
        reference_time_col,
        censor_threshold,
    )


def _plot_trend_intervention_outcome(
    dataframe: pd.DataFrame,
    entity_keys: list[str],
    cohort_col: str,
    subgroups: list[str],
    outcome: str,
    intervention: str,
    reftime: str,
    censor_threshold: int = 10,
) -> tuple[HTML, pd.DataFrame]:
    """
    Plots two timeseries based on selectors; an outcome and then an intervention.

    Parameters
    ----------
    dataframe : pd.DataFrame
        data source
    entity_keys : list[str]
        columns to use for aggregation
    cohort_col : str
        column name for the cohort to split on
    subgroups : list[str]
        values of interest in the cohort column
    outcome : str
        model score
    intervention : str
        intervention event time column
    reftime : str
        reference time column for alignment
    censor_threshold : int, optional
        minimum rows to allow in a plot, by default 10

    Returns
    -------
    HTML
        Plot of two timeseries
    """
    time_bounds = (dataframe[reftime].min(), dataframe[reftime].max())  # Use the full time range
    show_legend = True
    outcome_plot, intervention_plot = None, None

    try:
        intervention_col = pdh.event_value(intervention)
        intervention_svg = _plot_ts_cohort(
            dataframe,
            entity_keys,
            intervention_col,
            cohort_col,
            subgroups,
            reftime=reftime,
            time_bounds=time_bounds,
            boolean_event=True,
            show_legend=show_legend,
            censor_threshold=censor_threshold,
        )
        intervention_plot = template.render_title_with_image("Intervention: " + intervention, intervention_svg)
        show_legend = False
    except IndexError:
        intervention_plot = template.render_title_message(
            "Missing Intervention", f"No intervention timeseries plotted; No events with name {intervention}."
        )

    try:
        outcome_col = pdh.event_value(outcome)
        outcome_svg = _plot_ts_cohort(
            dataframe,
            entity_keys,
            outcome_col,
            cohort_col,
            subgroups,
            reftime=reftime,
            time_bounds=time_bounds,
            show_legend=show_legend,
            censor_threshold=censor_threshold,
        )
        outcome_plot = template.render_title_with_image("Outcome: " + outcome, outcome_svg)
    except IndexError:
        outcome_plot = template.render_title_message(
            "Missing Outcome", f"No outcome timeseries plotted; No events with name {outcome}."
        )

    return HTML(outcome_plot.data + intervention_plot.data), dataframe


def _plot_ts_cohort(
    dataframe: pd.DataFrame,
    entity_keys: list[str],
    event_col: str,
    cohort_col: str,
    subgroups: list[str],
    reftime: str,
    *,
    time_bounds: Optional[str] = None,
    boolean_event: bool = False,
    plot_counts: bool = False,
    show_legend: bool = False,
    ylabel: str = None,
    censor_threshold: int = 10,
) -> SVG:
    """
    Plot a single timeseries given a full dataframe and parameters.

    Given a dataframe like sg.dataframe and the relevant column names, will filter data to the values,
    cohorts, and timeframe before plotting.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The input data.
    entity_keys : list[str]
        A list of column names to use for summarizing the data.
    event_col : str
        The column name of value, used as y-axis.
    cohort_col : str
        The column name of the cohort, used to style separate lines.
    subgroups : list[str]
        The specific cohorts to plot, restricts cohort_col.
    reftime : str
        The column name of the reference time.
    time_bounds : Optional[str], optional
        An optional tuple (min, max) of times to bound the data to, by default None.
        If present the data is reduced to times within bounds prior to summarization.
    boolean_event : bool, optional
        A flag when set indicates the event is boolean so negative values are filtered out, by default False.
    plot_counts : bool, optional
        A flag when set will add an additional axis showing count of data at each point, by default False.
    show_legend : bool, optional
        A flag when set will show the legend on the plot, by default False.
    save : bool, optional
        A flag when set will save the plot to disk, by default False.
    ylabel : Optional[str], optional
        The label for the y-axis, by default None; will derive the label from the column name.
    censor_threshold : int, optional
        The minimum number of values for a given time that are needed to not be filtered, by default 10.
    """

    keep_columns = [cohort_col, reftime, event_col]

    cohort_msk = dataframe[cohort_col].isin(subgroups)
    plotdata = create_metric_timeseries(
        dataframe[cohort_msk][entity_keys + keep_columns],
        reftime,
        event_col,
        entity_keys,
        cohort_col,
        time_bounds=time_bounds,
        boolean_event=boolean_event,
        censor_threshold=censor_threshold,
    )

    if plotdata.empty:
        # all groups have been censored.
        return template.render_censored_plot_message(censor_threshold)

    counts = None
    if plot_counts:
        counts = plotdata.groupby([reftime, cohort_col]).count()

    if ylabel is None:
        ylabel = pdh.event_name(event_col)
    plotdata = plotdata.rename(columns={event_col: ylabel})

    # plot
    return plot.compare_series(
        plotdata,
        cohort_col=cohort_col,
        ref_str=reftime,
        event_col=event_col,
        ylabel=ylabel,
        counts=counts,
        show_legend=show_legend,
    )


@store_call_parameters(cohort_dict="cohort_dict")
@disk_cached_html_and_df_segment
@export
def plot_model_score_comparison(
    cohort_dict: dict[str, tuple[Any]], target: str, scores: tuple[str], *, per_context: bool
) -> tuple[HTML, pd.DataFrame]:
    """
    Gets the required data dictionary for the info template.

    Parameters
    ----------
    plot_help : bool
        If True, displays additional information about available plotting utilities, by default False.

    Returns
    -------
    dict[str, str | list[str]]
        The data dictionary.
    """
    sg = Seismogram()

    cohort_filter = FilterRule.from_cohort_dictionary(cohort_dict)
    data = cohort_filter.filter(sg.dataframe)
    target_event = pdh.event_value(target)
    dataframe = FilterRule.isin(target_event, (0, 1)).filter(data)

    # Need a dataframe with three columns, ScoreName, Score, and Target
    # Index - one copy of the index for each score name (to allow three columns of real scores.)
    # ScoreName - the cohort column (which score was used)
    # Score - the score value
    # Target - the target value

    data = []
    for score in scores:
        if per_context:
            one_score_data = pdh.event_score(
                dataframe, sg.entity_keys, score=score, ref_event=target_event, aggregation_method="max"
            )[[score, target_event]]
        else:
            one_score_data = dataframe[[score, target_event]].copy()
        one_score_data["ScoreName"] = score
        one_score_data.rename(columns={score: "Score"}, inplace=True)
        data.append(one_score_data)
    data = pd.concat(data, axis=0, ignore_index=True)
    data["ScoreName"] = data["ScoreName"].astype("category")

    plot_data = get_cohort_performance_data(
        data,
        "ScoreName",
        proba=data["Score"],
        true=data[target_event],
        splits=list(scores),
        censor_threshold=sg.censor_threshold,
    )
    recorder = metric_apis.OpenTelemetryRecorder(metric_names=STATNAMES, name="Model Score Comparison")
    recorder.log_by_column(
        df=plot_data, col_name="Threshold", cohorts={"cohort": scores}, base_attributes={"Target Column": target}
    )
    try:
        assert_valid_performance_metrics_df(plot_data)
    except ValueError:
        return template.render_censored_plot_message(sg.censor_threshold), plot_data
    svg = plot.cohort_evaluation_vs_threshold(plot_data, cohort_feature="ScoreName")
    title = f"Model Metrics: {', '.join(scores)} vs {target}"
    return template.render_title_with_image(title, svg), plot_data


@disk_cached_html_and_df_segment
@export
def plot_model_target_comparison(
    cohort_dict: dict[str, tuple[Any]], targets: tuple[str], score: str, *, per_context: bool
) -> tuple[HTML, pd.DataFrame]:
    """
    Displays information about the dataset

    Parameters
    ----------
    plot_help : bool, optional
        If True, displays additional information about available plotting utilities, by default False.
    """
    sg = Seismogram()

    cohort_filter = FilterRule.from_cohort_dictionary(cohort_dict)
    source_data = cohort_filter.filter(sg.dataframe)

    # Need a dataframe with three columns, ScoreName, Score, and Target
    # Index - one copy of the index for each score name (to allow three columns of real scores.)
    # TargetName - the cohort column (which target was used)
    # Score - the score value
    # Target - the target value

    data = []
    for target in targets:
        target_event = pdh.event_value(target)
        dataframe = FilterRule.isin(target_event, (0, 1)).filter(source_data)
        if per_context:
            one_score_data = pdh.event_score(
                dataframe, sg.entity_keys, score=score, ref_event=target_event, aggregation_method="max"
            )[[score, target_event]]
        else:
            one_score_data = dataframe[[score, target_event]].copy()
        one_score_data["TargetName"] = target
        one_score_data.rename(columns={target_event: "Target"}, inplace=True)
        data.append(one_score_data)
    data = pd.concat(data, axis=0, ignore_index=True)
    data["TargetName"] = data["TargetName"].astype("category")

    plot_data = get_cohort_performance_data(
        data,
        "TargetName",
        proba=data[score],
        true=data["Target"],
        splits=list(targets),
        censor_threshold=sg.censor_threshold,
    )
    recorder = metric_apis.OpenTelemetryRecorder(metric_names=STATNAMES, name="Model Score Comparison")
    recorder.log_by_column(
        df=plot_data, col_name="Threshold", cohorts={"cohort": targets}, base_attributes={"Score Column": score}
    )
    try:
        assert_valid_performance_metrics_df(plot_data)
    except ValueError:
        return template.render_censored_plot_message(sg.censor_threshold), plot_data
    svg = plot.cohort_evaluation_vs_threshold(plot_data, cohort_feature="ScoreName")
    title = f"Model Metrics: {', '.join(targets)} vs {score}"
    return template.render_title_with_image(title, svg), plot_data


# region Explore Any Metric (NNT, etc)
@store_call_parameters(cohort_dict="cohort_dict")
@disk_cached_html_and_df_segment
@export
def plot_binary_classifier_metrics(
    metric_generator: BinaryClassifierMetricGenerator,
    metrics: str | list[str],
    cohort_dict: dict[str, tuple[Any]],
    target: str,
    score_column: str,
    *,
    per_context: bool = False,
    table_only: bool = False,
) -> tuple[HTML, pd.DataFrame]:
    """
    Gets the summary levels for the cohort summary tables.

    Parameters
    ----------
    selected_attribute : str
        The name of the current attribute to generate summary levels for.
    by_target : bool
        If True, adds an additional summary table to break down the population by target prevalence.
    by_score : bool
        If True, adds an additional summary table to break down the population by model output.

    Returns
    -------
    tuple[list[str], list[str], list[str]]
        groupby_groups: The levels in the dataframe to groupby when summarizing
        grab_groups: The columns in the dataframe to grab to summarize
        index_rename: The display names for the indices
    """
    sg = Seismogram()

    score_bins = sg.score_bins()
    cut_bins = pd.cut(sg.dataframe[sg.output], score_bins)

    groupby_groups = [selected_attribute]
    grab_groups = [selected_attribute]
    index_rename = ["Cohort"]

    if by_score:
        groupby_groups.append(cut_bins)
        grab_groups.append(sg.output)
        index_rename.append(sg.output)

    if by_target:
        groupby_groups.append(sg.target)
        grab_groups.append(sg.target)
        index_rename.append(sg.target.strip("_Value"))

    return groupby_groups, grab_groups, index_rename


def _style_score_target_cohort_summaries(df: pd.DataFrame, index_rename: list[str], cohort: str) -> Styler:
    """
    Adds required styling to a multiple summary level cohort summary dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        The output of score_target_cohort_summaries.
    index_rename : list[str]
        The display names to put in the index from _score_target_levels_and_index().
    cohort : str
        The display name of the cohort.

    Returns
    -------
    Styler
        Styled dataframe.
    """
    df.index = df.index.rename(index_rename)
    style = df.style.format(precision=2)
    style = style.format_index(precision=2)
    return style.set_caption(f"Counts by {cohort}")


def binary_classifier_metric_evaluation(
    metric_generator: BinaryClassifierMetricGenerator,
    metrics: str | list[str],
    dataframe: pd.DataFrame,
    entity_keys: list[str],
    target: str,
    score_col: str,
    censor_threshold: int = 10,
    per_context_id: bool = False,
    aggregation_method: str = "max",
    ref_time: str = None,
    table_only: bool = False,
) -> tuple[HTML, pd.DataFrame]:
    """
    plots common model evaluation metrics

    Parameters
    ----------
    by_target : bool
        If True, adds an additional summary table to break down the population by target prevalence.
    by_score : bool
        If True, adds an additional summary table to break down the population by model output.

    Returns
    -------
    HTML
        Plot of model evaluation metrics
    """
    data = pdh.get_model_scores(
        dataframe,
        entity_keys,
        score_col=score_col,
        ref_time=ref_time,
        ref_event=target,
        aggregation_method=aggregation_method,
        per_context_id=per_context_id,
    )
    # Validate
    requirements = FilterRule.isin(target, (0, 1)) & FilterRule.notna(score_col)
    data = requirements.filter(data)
    if len(data.index) < censor_threshold:
        return template.render_censored_plot_message(censor_threshold), data
    if (lcount := data[target].nunique()) != 2:
        return (
            template.render_title_message(
                "Evaluation Error", f"Model Evaluation requires exactly two classes but found {lcount}"
            ),
            data,
        )
    if isinstance(metrics, str):
        metrics = [metrics]
    stats = metric_generator.calculate_binary_stats(data, target, score_col, metrics)[0]
    recorder = metric_apis.OpenTelemetryRecorder(name="Binary Classifier Evaluations", metric_names=metrics)
    attributes = {"score_col": score_col, "target": target}
    am = AutomationManager()
    for metric in metrics:
        log_all = am.get_metric_config(metric)["log_all"]
        if log_all:
            recorder.populate_metrics(attributes=attributes, metrics={metric: stats[metric].to_dict()})
    if table_only:
        return HTML(stats[metrics].T.to_html()), data
    return plot.binary_classifier.plot_metric_list(stats, metrics), data


# endregion
