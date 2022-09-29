import sys

from viskit.core import AttrDict

sys.path.append('.')
import matplotlib
import os

matplotlib.use('Agg')
import flask  # import Flask, render_template, send_from_directory
from viskit import core
import sys
import argparse
import json
import numpy as np
from plotly import tools
import plotly.offline as po
import plotly.graph_objs as go

named_colors = [
    'dodgerblue',
    'darkorange',
    'green',
    'cyan',
    'magenta',
    'orange',
    'yellow',
    'black',
    'blue',
    'brown',
    'lime',
    'pink',
    'purple',
]


def flatten(xs):
    return [x for y in xs for x in y]


def sliding_mean(data_array, window=5):
    data_array = np.array(data_array)
    new_list = []
    for i in range(len(data_array)):
        indices = list(range(max(i - window + 1, 0),
                             min(i + window + 1, len(data_array))))
        avg = 0
        for j in indices:
            avg += data_array[j]
        avg /= float(len(indices))
        new_list.append(avg)

    return np.array(new_list)


import itertools

app = flask.Flask(__name__, static_url_path='/static')

exps_data = None
plottable_keys = None
distinct_params = None


@app.route('/js/<path:path>')
def send_js(path):
    return flask.send_from_directory('js', path)


@app.route('/css/<path:path>')
def send_css(path):
    return flask.send_from_directory('css', path)

def create_bar_chart(
        plot_lists,
        use_median=False,
        plot_width=None,
        plot_height=None,
        title=None,
        value_i=-1,
    ):
    """
    plot_lists is a list of lists.
    Each outer list represents different y-axis attributes.
    Each inner list represents different experiments to run, within that y-axis
    attribute.
    Each plot is an AttrDict which should have the elements used below.
    """

    x_axis = [(subplot['plot_key'], subplot['means']) for plot_list in plot_lists for subplot in plot_list if subplot['x_key']]
    plot_lists = [[subplot for subplot in plot_list] for plot_list in plot_lists if not plot_list[0]['x_key']]
    xlabel = x_axis[0][0] if len(x_axis) else 'iteration'

    p25, p50, p75 = [], [], []
    num_y_axes = len(plot_lists)
    fig = tools.make_subplots(
        rows=num_y_axes,
        cols=1,
        print_grid=False,
        shared_xaxes=True,
    )
    fig.layout.update(
        width=plot_width,
        height=plot_height,
        title=title,
        barmode='group',
    )
    all_plot_keys = []
    for plot_list in plot_lists:
        all_plot_keys.append(plot_list[0].plot_key)
    traces = []
    num_exps = len(plot_lists[0])
    for y_idx, plot_list in enumerate(plot_lists):
        traces = []
        y_idx_plotly = y_idx + 1
        for plt_idx, plt in enumerate(plot_list):
            if use_median:
                value = plt.percentile50[value_i]
                error = plt.percentile75[value_i] - value
                error_minus = value - plt.percentile25[value_i]
            else:
                value = np.mean(plt.means)
                error = plt.stds[value_i]
                error_minus = plt.stds[value_i]
            # convert numpy scalar to number
            # value = value.item()
            # error = error.item()
            # error_minus = error_minus.item()
            trace = go.Bar(
                x=[plt.legend],
                y=[value],
                # TODO: implement this correctly. I should give the option of
                # choosing another field as the error bar for this field.
                # Currently, this uses the own field to compute std. This might
                # be correct, but often will be misleading (e.g. "std of mean"
                # vs "mean of std" if each trial measures its own mean/std).
                # error_y=dict(
                    # type='data',
                    # symmetric=False,
                    # array=[error],
                    # arrayminus=[error_minus],
                    # visible=True,
                # ),
                name=plt.legend,
                showlegend=y_idx==0,
                legendgroup=plt.legend,
                marker=dict(
                    color=named_colors[plt_idx % len(named_colors)],
                ),
            )
            fig.append_trace(trace, y_idx_plotly, 1)
        fig['layout']['yaxis{}'.format(y_idx_plotly)].update(
            title=plt.plot_key,
        )

    fig_div = po.plot(
        fig,
        output_type='div',
        include_plotlyjs=False,
    )
    if "footnote" in plot_list[0]:
        footnote = "<br />".join([
            r"<span><b>%s</b></span>: <span>%s</span>" % (
                plt.legend, plt.footnote)
            for plt in plot_list
        ])
        return r"%s<div>%s</div>" % (fig_div, footnote)
    else:
        return fig_div

def make_plot(
        plot_lists,
        use_median=False,
        plot_width=None,
        plot_height=None,
        title=None,
    ):
    """
    plot_lists is a list of lists.
    Each outer list represents different y-axis attributes.
    Each inner list represents different experiments to run, within that y-axis
    attribute.
    Each plot is an AttrDict which should have the elements used below.
    """

    x_axis = [(subplot['plot_key'], subplot['means']) for plot_list in plot_lists for subplot in plot_list if subplot['x_key']]
    plot_lists = [[subplot for subplot in plot_list] for plot_list in plot_lists if not plot_list[0]['x_key']]
    xlabel = x_axis[0][0] if len(x_axis) else 'iteration'

    p25, p50, p75 = [], [], []
    num_y_axes = len(plot_lists)
    fig = tools.make_subplots(rows=num_y_axes, cols=1, print_grid=False)
    fig['layout'].update(
        width=plot_width,
        height=plot_height,
        title=title,
    )

    for y_idx, plot_list in enumerate(plot_lists):
        for idx, plt in enumerate(plot_list):
            color = core.color_defaults[idx % len(core.color_defaults)]
            if use_median:
                p25.append(np.mean(plt.percentile25))
                p50.append(np.mean(plt.percentile50))
                p75.append(np.mean(plt.percentile75))
                if x_axis:
                    x = list(x_axis[idx][1])
                else:
                    x = list(range(len(plt.percentile50)))
                y = list(plt.percentile50)
                y_upper = list(plt.percentile75)
                y_lower = list(plt.percentile25)
            else:
                if x_axis:
                    x = list(x_axis[idx][1])
                else:
                    x = list(range(len(plt.means)))
                y = list(plt.means)
                y_upper = list(plt.means + plt.stds)
                y_lower = list(plt.means - plt.stds)

            errors = go.Scatter(
                x=x + x[::-1],
                y=y_upper + y_lower[::-1],
                fill='tozerox',
                fillcolor=core.hex_to_rgb(color, 0.2),
                line=go.scatter.Line(color=core.hex_to_rgb(color, 0)),
                showlegend=False,
                legendgroup=plt.legend,
                hoverinfo='none',
            )
            values = go.Scatter(
                x=x,
                y=y,
                name=plt.legend,
                legendgroup=plt.legend,
                line=dict(color=core.hex_to_rgb(color)),
                hoverlabel=dict(namelength=-1),
                hoverinfo='all',
            )
            # plotly is 1-indexed like matplotlib for subplots
            y_idx_plotly = y_idx + 1
            fig.append_trace(values, y_idx_plotly, 1)
            fig.append_trace(errors, y_idx_plotly, 1)
            title = plt.plot_key
            if len(title) > 30:
                title_parts = title.split('/')
                title = "<br />/".join(
                    title_parts[:-1]
                    + [r"<b>{}</b>".format(t) for t in title_parts[-1:]]
                )
            fig['layout']['yaxis{}'.format(y_idx_plotly)].update(
                title=title,
            )
            fig['layout']['xaxis{}'.format(y_idx_plotly)].update(
                title=xlabel,
            )

    fig_div = po.plot(fig, output_type='div', include_plotlyjs=False)
    if "footnote" in plot_list[0]:
        footnote = "<br />".join([
            r"<span><b>%s</b></span>: <span>%s</span>" % (
                plt.legend, plt.footnote)
            for plt in plot_list
        ])
        return r"%s<div>%s</div>" % (fig_div, footnote)
    else:
        return fig_div


def make_plot_eps(plot_list, use_median=False, counter=0):
    import matplotlib.pyplot as _plt
    f, ax = _plt.subplots(figsize=(8, 5))
    for idx, plt in enumerate(plot_list):
        color = core.color_defaults[idx % len(core.color_defaults)]
        if use_median:
            x = list(range(len(plt.percentile50)))
            y = list(plt.percentile50)
            y_upper = list(plt.percentile75)
            y_lower = list(plt.percentile25)
        else:
            x = list(range(len(plt.means)))
            y = list(plt.means)
            y_upper = list(plt.means + plt.stds)
            y_lower = list(plt.means - plt.stds)
        plt.legend = plt.legend.replace('rllab.algos.trpo.TRPO', 'TRPO')
        plt.legend = plt.legend.replace('rllab.algos.vpg.VPG', 'REINFORCE')
        plt.legend = plt.legend.replace('rllab.algos.erwr.ERWR', 'ERWR')
        plt.legend = plt.legend.replace('sandbox.rein.algos.trpo_vime.TRPO',
                                        'TRPO+VIME')
        plt.legend = plt.legend.replace('sandbox.rein.algos.vpg_vime.VPG',
                                        'REINFORCE+VIME')
        plt.legend = plt.legend.replace('sandbox.rein.algos.erwr_vime.ERWR',
                                        'ERWR+VIME')
        plt.legend = plt.legend.replace('0.0001', '1e-4')
        #         plt.legend = plt.legend.replace('0.001', 'TRPO+VIME')
        #         plt.legend = plt.legend.replace('0', 'TRPO')
        #         plt.legend = plt.legend.replace('0.005', 'TRPO+L2')

        if idx == 0:
            plt.legend = 'TRPO (0.0)'
        if idx == 1:
            plt.legend = 'TRPO+VIME (103.7)'
        if idx == 2:
            plt.legend = 'TRPO+L2 (0.0)'

        ax.fill_between(
            x, y_lower, y_upper, interpolate=True, facecolor=color,
            linewidth=0.0, alpha=0.3)
        if idx == 2:
            ax.plot(x, y, color=color, label=plt.legend, linewidth=2.0,
                    linestyle="--")
        else:
            ax.plot(x, y, color=color, label=plt.legend, linewidth=2.0)
        ax.grid(True)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        if counter == 1:
            #             ax.set_xlim([0, 120])
            ax.set_ylim([-3, 60])
            #             ax.set_xlim([0, 80])

            loc = 'upper left'
        elif counter == 2:
            ax.set_ylim([-0.04, 0.4])

            #             ax.set_ylim([-0.1, 0.4])
            ax.set_xlim([0, 2000])
            loc = 'upper left'
        elif counter == 3:
            #             ax.set_xlim([0, 1000])
            loc = 'lower right'
        elif counter == 4:
            #             ax.set_xlim([0, 800])
            #             ax.set_ylim([0, 2])
            loc = 'lower right'
        leg = ax.legend(loc=loc, prop={'size': 12}, ncol=1)
        for legobj in leg.legendHandles:
            legobj.set_linewidth(5.0)

        def y_fmt(x, y):
            return str(int(np.round(x / 1000.0))) + 'K'

        import matplotlib.ticker as tick
        #         ax.xaxis.set_major_formatter(tick.FuncFormatter(y_fmt))
        _plt.savefig('tmp' + str(counter) + '.pdf', bbox_inches='tight')


def summary_name(exp, selector=None):
    # if selector is not None:
    #     exclude_params = set([x[0] for x in selector._filters])
    # else:
    #     exclude_params = set()
    # rest_params = set([x[0] for x in distinct_params]).difference(exclude_params)
    # if len(rest_params) > 0:
    #     name = ""
    #     for k in rest_params:
    #         name += "%s=%s;" % (k.split(".")[-1], str(exp.flat_params.get(k, "")).split(".")[-1])
    #     return name
    return exp.params["exp_name"]


def check_nan(exp):
    return all(
        not np.any(np.isnan(vals)) for vals in list(exp.progress.values()))

def get_plot_instruction(
        plot_keys,
        x_keys=None,
        split_keys=None,
        group_keys=None,
        best_filter_key=None,
        filters=None,
        exclusions=None,
        use_median=False,
        only_show_best=False,
        best_based_on_final=False,
        gen_eps=False,
        only_show_best_sofar=False,
        best_is_lowest=False,
        clip_plot_value=None,
        plot_width=None,
        plot_height=None,
        filter_nan=False,
        smooth_curve=False,
        custom_filter=None,
        legend_post_processor=None,
        normalize_error=False,
        make_bar_chart=False,
        value_i=-1,  # TODO: add option to set value_i
        custom_series_splitter=None,
):
    if x_keys is None:
        x_keys = []
    if x_keys:
        assert len(x_keys) == 1
        if x_keys[0] is None:
            x_keys = []
        plot_keys = x_keys + plot_keys

    """
    A custom filter might look like
    "lambda exp: exp.flat_params['algo_params_base_kwargs.batch_size'] == 64"
    """
    if filter_nan:
        nonnan_exps_data = list(filter(check_nan, exps_data))
        selector = core.Selector(nonnan_exps_data)
    else:
        selector = core.Selector(exps_data)
    if legend_post_processor is None:
        legend_post_processor = lambda x: x
    if filters is None:
        filters = dict()
    if exclusions is None:
        exclusions = []
    if split_keys is None:
        split_keys = []
    if group_keys is None:
        group_keys = []
    if plot_height is None:
        plot_height = 300 * len(plot_keys)
    for k, v in filters.items():
        selector = selector.where(k, str(v))
    for k, v in exclusions:
        selector = selector.where_not(k, str(v))
    if custom_filter is not None:
        selector = selector.custom_filter(custom_filter)

    if len(split_keys) > 0:
        split_selectors, split_titles = split_by_keys(
            selector, split_keys, distinct_params
        )
    else:
        split_selectors = [selector]
        split_titles = ["Plot"]
    plots = []
    counter = 1
    print("Plot_keys:", plot_keys)
    print("X keys:", x_keys)
    print("split_keys:", split_keys)
    print("group_keys:", group_keys)
    print("filters:", filters)
    print("exclusions:", exclusions)
    for split_selector, split_title in zip(split_selectors, split_titles):
        if custom_series_splitter is not None:
            exps = split_selector.extract()
            splitted_dict = dict()
            for exp in exps:
                key = custom_series_splitter(exp)
                if key not in splitted_dict:
                    splitted_dict[key] = list()
                splitted_dict[key].append(exp)
            splitted = list(splitted_dict.items())
            group_selectors = [core.Selector(list(x[1])) for x in splitted]
            group_legends = [x[0] for x in splitted]
        else:
            if len(group_keys) > 0:
                group_selectors, group_legends = split_by_keys(
                    split_selector, group_keys, distinct_params
                )
            else:
                group_selectors = [split_selector]
                group_legends = [split_title]
        list_of_list_of_plot_dicts = []
        for plot_ind, plot_key in enumerate(plot_keys):
            to_plot = []
            for group_selector, group_legend in zip(group_selectors, group_legends):
                filtered_data = group_selector.extract()
                if len(filtered_data) == 0:
                    continue
                if (best_filter_key
                        and best_filter_key not in group_keys
                        and best_filter_key not in split_keys):
                    selectors = split_by_key(
                        group_selector, best_filter_key, distinct_params
                    )
                    scores = [
                        get_selector_score(plot_key, selector, use_median, best_based_on_final)
                        for selector in selectors
                    ]

                    if np.isfinite(scores).any():
                        if best_is_lowest:
                            best_idx = np.nanargmin(scores)
                        else:
                            best_idx = np.nanargmax(scores)

                        best_selector = selectors[best_idx]
                        filtered_data = best_selector.extract()
                        print("For split '{0}', group '{1}':".format(
                            split_title,
                            group_legend,
                        ))
                        print("    best '{0}': {1}".format(
                            best_filter_key,
                            dict(best_selector._filters)[best_filter_key]
                        ))

                if only_show_best or only_show_best_sofar:
                    # Group by seed and sort.
                    # -----------------------

                    filtered_params = core.extract_distinct_params(
                        filtered_data, l=0)
                    filtered_params2 = [p[1] for p in filtered_params]
                    filtered_params_k = [p[0] for p in filtered_params]
                    product_space = list(itertools.product(
                        *filtered_params2
                    ))
                    data_best_regret = None
                    best_regret = np.inf if best_is_lowest else -np.inf
                    kv_string_best_regret = None
                    for idx, params in enumerate(product_space):
                        selector = core.Selector(exps_data)
                        for k, v in zip(filtered_params_k, params):
                            selector = selector.where(k, str(v))
                        data = selector.extract()
                        if len(data) > 0:
                            progresses = [
                                exp.progress.get(plot_key, np.array([np.nan]))
                                for exp in data
                            ]
                            sizes = list(map(len, progresses))
                            max_size = max(sizes)
                            progresses = [
                                np.concatenate(
                                    [ps, np.ones(max_size - len(ps)) * np.nan])
                                for ps in progresses]

                            if best_based_on_final:
                                progresses = np.asarray(progresses)[:, -1]
                            if only_show_best_sofar:
                                if best_is_lowest:
                                    progresses = np.min(np.asarray(progresses),
                                                        axis=1)
                                else:
                                    progresses = np.max(np.asarray(progresses),
                                                        axis=1)
                            if use_median:
                                medians = np.nanmedian(progresses, axis=0)
                                regret = np.mean(medians)
                            else:
                                means = np.nanmean(progresses, axis=0)
                                regret = np.mean(means)
                            distinct_params_k = [p[0] for p in distinct_params]
                            distinct_params_v = [
                                v for k, v in zip(filtered_params_k, params) if
                                k in distinct_params_k]
                            distinct_params_kv = [
                                (k, v) for k, v in
                                zip(distinct_params_k, distinct_params_v)]
                            distinct_params_kv_string = str(
                                distinct_params_kv).replace('), ', ')\t')
                            print(
                                '{}\t{}\t{}'.format(regret, len(progresses),
                                                    distinct_params_kv_string))
                            if best_is_lowest:
                                change_regret = regret < best_regret
                            else:
                                change_regret = regret > best_regret
                            if change_regret:
                                best_regret = regret
                                best_progress = progresses
                                data_best_regret = data
                                kv_string_best_regret = distinct_params_kv_string

                    print(group_selector._filters)
                    print('best regret: {}'.format(best_regret))
                    # -----------------------
                    if np.isfinite(best_regret):
                        progresses = [
                            exp.progress.get(plot_key, np.array([np.nan])) for
                            exp in data_best_regret]
                        #                         progresses = [progress[:500] for progress in progresses ]
                        sizes = list(map(len, progresses))
                        # more intelligent:
                        max_size = max(sizes)
                        progresses = [
                            np.concatenate(
                                [ps, np.ones(max_size - len(ps)) * np.nan]) for
                            ps in progresses]
                        legend = '{} (mu: {:.3f}, std: {:.5f})'.format(
                            group_legend, best_regret, np.std(best_progress))
                        window_size = np.maximum(
                            int(np.round(max_size / float(1000))), 1)
                        statistics = get_statistics(
                            progresses, use_median, normalize_error,
                        )
                        statistics = process_statistics(
                            statistics,
                            smooth_curve,
                            clip_plot_value,
                            window_size,
                        )
                        to_plot.append(
                            AttrDict(
                                legend=legend_post_processor(legend),
                                plot_key=plot_key,
                                **statistics
                            )
                        )
                        if len(to_plot) > 0 and len(data) > 0:
                            to_plot[-1]["footnote"] = "%s; e.g. %s" % (
                                kv_string_best_regret,
                                data[0].params.get("exp_name", "NA"))
                        else:
                            to_plot[-1]["footnote"] = ""
                else:
                    progresses = [
                        exp.progress.get(plot_key, np.array([np.nan])) for exp
                        in filtered_data
                    ]
                    sizes = list(map(len, progresses))
                    # more intelligent:
                    max_size = max(sizes)
                    progresses = [
                        np.concatenate(
                            [ps, np.ones(max_size - len(ps)) * np.nan]) for ps
                        in progresses]
                    window_size = np.maximum(
                        int(np.round(max_size / float(100))),
                        1,
                    )

                    statistics = get_statistics(
                        progresses, use_median, normalize_error,
                    )
                    statistics = process_statistics(
                        statistics,
                        smooth_curve,
                        clip_plot_value,
                        window_size,
                    )
                    to_plot.append(
                        AttrDict(
                            legend=legend_post_processor(group_legend),
                            plot_key=plot_key,
                            x_key=plot_key in x_keys and plot_ind == 0,
                            **statistics
                        )
                    )
            if len(to_plot) > 0:
                list_of_list_of_plot_dicts.append(to_plot)

        if len(list_of_list_of_plot_dicts) > 0 and not gen_eps:
            fig_title = split_title
            if make_bar_chart:
                plots.append(create_bar_chart(
                    list_of_list_of_plot_dicts,
                    use_median=use_median, title=fig_title,
                    plot_width=plot_width, plot_height=plot_height,
                    value_i=value_i,
                ))
            else:
                plots.append(make_plot(
                    list_of_list_of_plot_dicts,
                    use_median=use_median, title=fig_title,
                    plot_width=plot_width, plot_height=plot_height
                ))

        if gen_eps:
            make_plot_eps(to_plot, use_median=use_median, counter=counter)
        counter += 1
    return "\n".join(plots)


def shorten_key(key):
    """
    Convert a dot-map string like "foo.bar.baz" into "f.b.baz"
    """
    *heads, tail = key.split(".")
    new_key_builder = []
    for subkey in heads:
        if len(subkey) > 0:
            new_key_builder.append(subkey[0])
    new_key_builder.append(tail)
    return ".".join(new_key_builder)


def get_selector_score(key, selector, use_median, best_based_on_final):
    """
    :param key: Thing to measure (e.g. Average Returns, Loss, etc.)
    :param selector: Selector instance
    :param use_median: Use the median? Else use the mean
    :param best_based_on_final: Only look at the final value? Else use all
    values.
    :return: A single number that gives the score of `key` inside `selector`
    """
    data = selector.extract()
    if best_based_on_final:
        values = [
            exp.progress.get(key, np.array([np.nan]))[-1]
            for exp in data
        ]
    else:
        values = np.concatenate([
            exp.progress.get(key, np.array([np.nan]))
            for exp in data
        ] or [[np.nan]])

    if len(values) == 0 or not np.isfinite(values).all():
        return np.nan
    if use_median:
        return np.nanpercentile(values, q=50, axis=0)
    else:
        return np.nanmean(values)


def get_statistics(progresses, use_median, normalize_errors):
    """
    Get some dictionary of statistics (e.g. the median, mean).
    :param progresses:
    :param use_median:
    :param normalize_errors:
    :return:
    """
    if use_median:
        return dict(
            percentile25=np.nanpercentile(progresses, q=25, axis=0),
            percentile50=np.nanpercentile(progresses, q=50, axis=0),
            percentile75=np.nanpercentile(progresses, q=75, axis=0),
        )
    else:
        stds = np.nanstd(progresses, axis=0)
        if normalize_errors:
            stds /= np.sqrt(np.sum((1. - np.isnan(progresses)), axis=0))
        return dict(
            means=np.nanmean(progresses, axis=0),
            stds=stds,
        )


def process_statistics(
        statistics,
        smooth_curve,
        clip_plot_value,
        window_size
):
    """
    Smoothen and clip time-series data.
    """
    clean_statistics = {}
    for k, v in statistics.items():
        clean_statistics[k] = v
        if smooth_curve:
            clean_statistics[k] = sliding_mean(v, window=window_size)
        if clip_plot_value is not None:
            clean_statistics[k] = np.clip(
                clean_statistics[k],
                -clip_plot_value,
                clip_plot_value,
            )
    return clean_statistics


def get_possible_values(distinct_params, key):
    return [vs for k, vs in distinct_params if k == key][0]


def split_by_key(selector, key, distinct_params):
    """
    Return a list of selectors based on this selector.
    Each selector represents one distinct value of `key`.
    """
    values = get_possible_values(distinct_params, key)
    return [selector.where(key, v) for v in values]


def split_by_keys(base_selector, keys, distinct_params):
    """
    Return a list of selectors based on the base_selector.
    Each selector represents one distinct set of values for each key in `keys`.

    :param base_selector:
    :param keys:
    :param distinct_params:
    :return:
    """
    list_of_key_and_unique_value = [
        [
            (key, v)
            for v in get_possible_values(distinct_params, key)
        ]
        for key in keys
    ]
    """
    elements of list_of_key_and_unique_value should look like:
        - [(color, red), (color, blue), (color, green), ...]
        - [(season, spring), (season, summer), (season, fall), ...]
    We now take the cartesian product so that we get all the
    combinations, like:
        - [(color, red), (season, spring)]
        - [(color, blue), (season, spring)]
        - ...
    """
    selectors = []
    descriptions = []
    for key_and_value_list in itertools.product(
            *list_of_key_and_unique_value
    ):
        selector = None
        keys = []
        for key, value in key_and_value_list:
            keys.append(key)
            if selector is None:
                selector = base_selector.where(key, value)
            else:
                selector = selector.where(key, value)
        selectors.append(selector)
        descriptions.append(", ".join([
            "{0}={1}".format(
                shorten_key(key),
                value,
            )
            for key, value in key_and_value_list
        ]))
    return selectors, descriptions

def parse_float_arg(args, key):
    x = args.get(key, "")
    try:
        return float(x)
    except Exception:
        return None


@app.route("/plot_div")
def plot_div():
    args = flask.request.args
    plot_keys_json = args.get("plot_keys")
    plot_keys = json.loads(plot_keys_json)
    x_keys_json = args.get("x_keys")
    x_keys = json.loads(x_keys_json)
    split_keys_json = args.get("split_keys", "[]")
    split_keys = json.loads(split_keys_json)
    group_keys_json = args.get("group_keys", "[]")
    group_keys = json.loads(group_keys_json)
    best_filter_key = args.get("best_filter_key", "")
    filters_json = args.get("filters", "{}")
    filters = json.loads(filters_json)
    exclusions_json = args.get("exclusions", "{}")
    exclusions = json.loads(exclusions_json)
    if len(best_filter_key) == 0:
        best_filter_key = None
    use_median = args.get("use_median", "") == 'True'
    gen_eps = args.get("eps", "") == 'True'
    only_show_best = args.get("only_show_best", "") == 'True'
    best_based_on_final = args.get("best_based_on_final", "") == 'True'
    only_show_best_sofar = args.get("only_show_best_sofar", "") == 'True'
    best_is_lowest = args.get("best_is_lowest", "") == 'True'
    normalize_error = args.get("normalize_error", "") == 'True'
    make_bar_chart = args.get("make_bar_chart", "") == 'True'
    filter_nan = args.get("filter_nan", "") == 'True'
    smooth_curve = args.get("smooth_curve", "") == 'True'
    clip_plot_value = parse_float_arg(args, "clip_plot_value")
    plot_width = parse_float_arg(args, "plot_width")
    plot_height = parse_float_arg(args, "plot_height")
    custom_filter = args.get("custom_filter", None)
    custom_series_splitter = args.get("custom_series_splitter", None)
    if custom_filter is not None and len(custom_filter.strip()) > 0:
        custom_filter = safer_eval(custom_filter)

    else:
        custom_filter = None
    legend_post_processor = args.get("legend_post_processor", None)
    if legend_post_processor is not None and len(
            legend_post_processor.strip()) > 0:
        legend_post_processor = safer_eval(legend_post_processor)
    else:
        legend_post_processor = None
    if custom_series_splitter is not None and len(
            custom_series_splitter.strip()) > 0:
        custom_series_splitter = safer_eval(custom_series_splitter)
    else:
        custom_series_splitter = None

    plot_div = get_plot_instruction(
        plot_keys=plot_keys,
        x_keys=x_keys,
        split_keys=split_keys,
        filter_nan=filter_nan,
        group_keys=group_keys,
        best_filter_key=best_filter_key,
        filters=filters,
        exclusions=exclusions,
        use_median=use_median,
        gen_eps=gen_eps,
        only_show_best=only_show_best,
        best_based_on_final=best_based_on_final,
        only_show_best_sofar=only_show_best_sofar,
        best_is_lowest=best_is_lowest,
        clip_plot_value=clip_plot_value,
        plot_width=plot_width,
        plot_height=plot_height,
        smooth_curve=smooth_curve,
        custom_filter=custom_filter,
        legend_post_processor=legend_post_processor,
        normalize_error=normalize_error,
        make_bar_chart=make_bar_chart,
        custom_series_splitter=custom_series_splitter,
    )
    return plot_div


def safer_eval(some_string):
    """
    Not full-proof, but taking advice from:

    https://nedbatchelder.com/blog/201206/eval_really_is_dangerous.html
    """
    if "__" in some_string or "import" in some_string:
        raise Exception("string to eval looks suspicious")
    return eval(some_string, {'__builtins__': {}})

@app.route("/")
def index():
    if "AverageReturn" in plottable_keys:
        plot_keys = ["AverageReturn"]
    elif 'training/return-average' in plottable_keys:
        plot_keys = ['training/return-average']
    elif len(plottable_keys) > 0:
        plot_keys = plottable_keys[0:1]
    else:
        plot_keys = None
    plot_div = get_plot_instruction(plot_keys=plot_keys)
    return flask.render_template(
        "main.html",
        plot_div=plot_div,
        plot_keys=plot_keys,
        group_keys=[],
        plottable_keys=plottable_keys,
        distinct_param_keys=[str(k) for k, v in distinct_params],
        distinct_params=dict([(str(k), list(map(str, v)))
                              for k, v in distinct_params]),
    )


@app.route("/reload-data", methods=['POST'])
def reload():
    reload_data()
    return 'Reloaded'


def reload_data():
    global exps_data
    global plottable_keys
    global distinct_params
    exps_data = core.load_exps_data(
        args.data_paths,
        args.data_filename,
        args.params_filename,
        args.disable_variant,
    )
    plottable_keys = list(
        set(flatten(list(exp.progress.keys()) for exp in exps_data)))
    plottable_keys = sorted([k for k in plottable_keys if k is not None])
    distinct_params = sorted(core.extract_distinct_params(exps_data))


def main():
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument("data_paths", type=str, nargs='*')
    parser.add_argument("--prefix", type=str, nargs='?', default="???")
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--disable-variant", default=False, action='store_true')
    parser.add_argument("--data-filename",
                        default='progress.csv',
                        help='name of data file.')
    parser.add_argument("--params-filename",
                        default='params.json',
                        help='name of params file.')
    args = parser.parse_args(sys.argv[1:])

    # load all folders following a prefix
    if args.prefix != "???":
        args.data_paths = []
        dirname = os.path.dirname(args.prefix)
        subdirprefix = os.path.basename(args.prefix)
        for subdirname in os.listdir(dirname):
            path = os.path.join(dirname, subdirname)
            if os.path.isdir(path) and (subdirprefix in subdirname):
                args.data_paths.append(path)
    print("Importing data from {path}...".format(path=args.data_paths))
    reload_data()
    port = args.port
    try:
        print("View http://localhost:%d in your browser" % port)
        app.run(host='0.0.0.0', port=port, debug=args.debug)
    except OSError as e:
        if e.strerror == 'Address already in use':
            print("Port {} is busy. Try specifying a different port with ("
                  "e.g.) --port=5001".format(port))

if __name__ == "__main__":
    main()
