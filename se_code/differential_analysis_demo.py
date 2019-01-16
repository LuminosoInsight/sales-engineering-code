"""
Tools for differential analytics on projects.  It is probably easiest (and
most flexible) to use them from an ipython prompt or a jupyter notebook, but
a command-line interface exposing most of the functionality is also provided.
"""

import argparse
import matplotlib.pyplot as plt
import matplotlib.dates as plt_dates
import matplotlib.lines as plt_lines
import numpy as np
import sys

from api_utils_app.luminoso_client_holder import LuminosoClientHolder
from scipy.special import erf, erfcinv


def compare_term_vectors(vectors0, vectors1, normalize=True):
    """
    Compute the similarity between the vectors for the same terms from
    the two given embeddings (dataframes).  If normalize is set to True,
    L^2 normalize the term vectors for each embedding before computing
    the similarity (dot product), yielding cosine similarity; otherwise
    compute similarity as Euclidean norm.
    """
    common_terms = vectors0.index.intersection(vectors1.index)
    common_vectors0 = vectors0.loc[common_terms].values
    common_vectors1 = vectors1.loc[common_terms].values

    # If one embedding has fewer columns (features) than the other, pad the
    # shorter one with zeros to make up the difference.  This leaves cosine
    # similarities in the padded frame unaltered.
    dimension_diff = common_vectors0.shape[1] - common_vectors1.shape[1]
    if dimension_diff < 0:
        common_vectors0 = np.hstack(
            [
                common_vectors0,
                np.zeros(
                    (common_vectors0.shape[0], -dimension_diff),
                    dtype=common_vectors0.dtype,
                ),
            ]
        )
    elif dimension_diff > 0:
        common_vectors1 = np.hstack(
            [
                common_vectors1,
                np.zeros(
                    (common_vectors1.shape[0], dimension_diff),
                    dtype=common_vectors1.dtype,
                ),
            ]
        )

    # Normalize (rows of) both embeddings if requested.
    if normalize:
        norms0 = np.sqrt(np.sum(common_vectors0 * common_vectors0, axis=1))
        norms1 = np.sqrt(np.sum(common_vectors1 * common_vectors1, axis=1))
        common_vectors0n = common_vectors0 / norms0.reshape((len(norms0), 1))
        common_vectors1n = common_vectors1 / norms1.reshape((len(norms1), 1))
    else:
        common_vectors0n = common_vectors0
        common_vectors1n = common_vectors1

    # Rotate first embedding to match second, and, just to be safe,
    # renormalize it.
    U, _sigma, Vt = np.linalg.svd(common_vectors0n.T.dot(common_vectors1n))
    xform = U.dot(Vt)
    common_vectors0nx = common_vectors0n.dot(xform)
    if normalize:
        norms0 = np.sqrt(np.sum(common_vectors0nx * common_vectors0nx, axis=1))
        common_vectors0nx /= norms0.reshape((len(norms0), 1))

    # Now cosine similarity is just dot product.
    if normalize:
        similarities = np.sum(common_vectors0nx * common_vectors1n, axis=1)
    else:
        similarities = np.linalg.norm(
            common_vectors0nx - common_vectors1n, axis=1, ord=2
        )

    return list(common_terms), similarities


def jaccard_index(set0, set1):
    """
    Compute the Jaccard index of the two sets, defined as the cardinality
    of the intersection divided by the cardinality of the union (and by
    fiat equal to one if the union is empty).
    """
    union_cardinality = len(set0 | set1)
    if union_cardinality < 1:
        return 1.0
    index = len(set0 & set1) / union_cardinality
    return index


def plot_multiple_lines(caption, ordinates, abscissas, time_plot=False):
    """
    Make and show a plot of several dependent variables (abscissas) against
    a single independent variable (ordinate).  Handle the ordinate as time
    data if time_plot=True (default is False).
    """
    for _name, values in abscissas.items():
        assert values.shape == (len(ordinates),)
    fig, axs = plt.subplots()
    if time_plot:  # if the x axis is time data, treat it specially
        ordinates2 = ordinates.astype("O")  # convert to datetime.datetimes
        fig.autofmt_xdate()
        axs.xaxis_date()
        axs.fmt_xdata = plt_dates.DateFormatter("%Y-%m-%d")
    else:
        ordinates2 = ordinates
    markers = ["*", "+", "^"]
    colors = ["red", "green", "blue", "cyan"]
    for i_abscissa, (name, values) in enumerate(abscissas.items()):
        line = plt_lines.Line2D(
            ordinates2,
            values,
            marker=markers[i_abscissa % len(markers)],
            color=colors[i_abscissa % len(colors)],
            label=name,
        )
        axs.add_line(line)
    axs.autoscale()
    axs.legend()
    fig.suptitle(caption)
    plt.show()


def get_date_from_document(doc):
    """
    Pulls the (or a) date from the metadata of the given document, and
    returns it (as a numpy datetime64).
    """
    dates = [m["value"] for m in doc.get("metadata", []) if m["name"].lower() == "date"]
    if len(dates) < 1:
        return None
    date = dates[0]  # if multiple dates just pick the first
    date = np.datetime64(date)
    return date


def subset_study(
    project_holder,
    filter0=None,
    search0=None,
    filter1=None,
    search1=None,
    get_data_kwargs=None,
    comparison_kwargs=None,
    caption=None,
    output_file=None,
):
    """
    Compare vectors, top concepts, etc., from two projects built from
    subsets of the documents of the given project, as defined by the two
    given document filters and concept selectors.
    """
    docs0_args = {}
    if filter0 is not None:
        docs0_args["filter"] = filter0
    if search0 is not None:
        docs0_args["search"] = search0

    docs1_args = {}
    if filter1 is not None:
        docs1_args["filter"] = filter1
    if search1 is not None:
        docs1_args["search"] = search1

    docs0 = project_holder.get_docs(**docs0_args)
    docs1 = project_holder.get_docs(**docs1_args)
    project0_name = "Tmp project from {}, filter 0".format(project_holder.project_name)
    project1_name = "Tmp project from {}, filter 1".format(project_holder.project_name)
    project0 = project_holder.new_project_from_docs(project0_name, docs=docs0)
    project1 = project_holder.new_project_from_docs(project1_name, docs=docs1)

    get_data_kwargs = get_data_kwargs or {}
    data0 = ProjectData(project0, **get_data_kwargs)
    data1 = ProjectData(project1, **get_data_kwargs)
    project_holder.delete_project(project0.project_id)
    project_holder.delete_project(project1.project_id)

    comparison_kwargs = comparison_kwargs or {}
    comparison = ProjectDataComparison(data0, data1, **comparison_kwargs)

    data0.print_summary()
    data1.print_summary()
    comparison.print_summary()
    comparison.show(caption=caption)
    data0.print_summary(output_file=output_file, append=False)
    data1.print_summary(output_file=output_file, append=True)
    comparison.print_summary(output_file=output_file, append=True)

    result = dict(data0=data0, data1=data1, comparison=comparison)
    return result


def get_project_time_windows(
    project_holder,
    time_step=np.timedelta64(1, "W"),
    window_length=np.timedelta64(30, "D"),
):
    """
    Given a project client holder, a time step and window length (numpy
    timedelta64's, defaults are one week and 30 days), generate a sequence of
    time windows, the i-th starting i time steps after the first date of any
    document in the project and having duration equal to the given window
    length.  Return a sequence of dicts, one for each window, with two
    values:  the pair of endpoints of the window, and a list of the documents
    from the project having dates within that time span.  The time step and
    window length arguments are np.timedelta64's and default to one week and
    30 days.
    """
    if time_step <= np.timedelta64(0, dtype=time_step.dtype):
        raise ValueError("Time step {} is not positive.".format(time_step))
    if window_length <= np.timedelta64(0, dtype=window_length.dtype):
        raise ValueError("Window length {} is not positive.".format(window_length))

    # Get all the dated documents, in order by date.
    docs = [
        doc
        for doc in project_holder.get_docs()
        if get_date_from_document(doc) is not None
    ]
    docs.sort(key=get_date_from_document)
    print("Making windows from a total of {} documents.".format(len(docs)))

    if len(docs) < 1:
        return  # No docs with dates means nothing to generate.

    # Make a list of start and end times for time windows of the specified
    # length (the given window size times the given interval duration),
    # starting on times separated by the given interval (so overlapping if
    # the window size is more than one interval).  This is complicated by
    # the existence of intervals (e.g. months) that are not of constant
    # duration (e.g. 28-31 days).
    start_time = get_date_from_document(docs[0])
    grand_end_time = get_date_from_document(docs[-1])
    time_windows = []
    while True:
        end_time = start_time + window_length
        time_windows.append((start_time, end_time))
        if end_time > grand_end_time:
            break
        start_time = start_time + time_step

    # Generate the sequence of lists of documents whose dates lie in those
    # time windows.  Each window is treated as a half-open interval, i.e.
    # (t0, t1) corresponds to documents whose date t satisfies t0 <= t < t1.
    dates = [get_date_from_document(doc) for doc in docs]
    i0 = 0  # Used to avoid repeatedly searching the front of the list of docs.
    for t0, t1 in time_windows:
        this_window_docs = []
        for date, doc in zip(dates[i0:], docs[i0:]):
            if date < t0:
                i0 += 1  # this doc can be skipped in the succeeding windows
            elif date < t1:
                this_window_docs.append(doc)
            else:
                break
        result = dict(interval=(t0, t1), documents=this_window_docs)
        yield result


def longitudinal_study(
    project_holder,
    time_step=np.timedelta64(1, "W"),
    window_length=np.timedelta64(30, "D"),
    verbose=True,
    show_plots=True,
    summary_percentile=50,
    get_data_kwargs=None,
    comparison_kwargs=None,
    output_file=None,
):
    """
    Compute and plot changes over time in the vectors and top concepts of
    a series of projects constructed from the given one by restricting to
    documents from time windows generated from the given time step and
    window length (as in get_project_time_windows).
    """

    def by_twos(iterator):
        try:
            last = next(iterator)
        except StopIteration:
            return
        for value in iterator:
            yield last, value
            last = value

    def window_to_data(window):
        t0, t1 = window["interval"]
        docs = window["documents"]
        if len(docs) < 1:
            print("Time window {} to {} had no data; skipping.".format(t0, t1))
            return None
        start_date = get_date_from_document(docs[0])
        end_date = get_date_from_document(docs[-1])
        data = dict(
            nominal_start_date=t0,
            nominal_end_date=t1,
            start_date=start_date,
            end_date=end_date,
            docs=docs,
        )
        return data

    project_name = project_holder.project_name
    project_id = project_holder.project_id
    caption = "{} (id {})".format(project_name, project_id)
    caption += "\n time step {}, window length {}".format(
        str(time_step), str(window_length)
    )  # format chokes on np.timedelta64
    caption += "\n{}th percentile of absolute change".format(summary_percentile)

    # If vectors are normalized, the similarity is cosine similarity,
    # and higher values are better; otherwise higher values are (as
    # with everything else but Jaccard similarity of concept sets)
    # worse.  So when vectors are normalized we take the opposite
    # percentile of their similarities, to get a comparably bad
    # representative value.
    normalize_vectors = comparison_kwargs.get("normalize_vectors", True)
    if normalize_vectors:
        vector_summary_percentile = 100 - summary_percentile
        vector_msg = "vector similarity metric is cosine similarity"
    else:
        vector_summary_percentile = summary_percentile
        vector_msg = "vector difference metric is Euclidean distance"

    vector_summary_quantile = vector_summary_percentile / 100
    summary_quantile = summary_percentile / 100

    print(caption)
    print(vector_msg)
    if output_file is not None:
        with open(output_file, "wt", encoding="utf-8") as fp:
            fp.write(caption + "\n" + vector_msg + "\n\n")

    data_dicts = (
        window_to_data(window)
        for window in get_project_time_windows(
            project_holder, time_step=time_step, window_length=window_length
        )
    )
    good_data_dicts = (d for d in data_dicts if d is not None)

    times = []
    relevance_range = [np.inf, -np.inf]
    similarities = {
        "concept": [],
        "vector": [],
        "relevance": [],
        "association": [],
        "confidence": [],
        "confidence-p": [],
        "impact": [],
        "importance": [],
    }

    comparison_kwargs = comparison_kwargs or {}
    percentiles = comparison_kwargs.get("percentiles", (10, 50, 90))
    percentiles = set(percentiles)
    percentiles.add(summary_percentile)
    percentiles.add(vector_summary_percentile)
    percentiles = tuple(sorted(percentiles))
    comparison_kwargs["percentiles"] = percentiles

    for raw_data0, raw_data1 in by_twos(good_data_dicts):
        times.append(raw_data1["end_date"])
        msg = "{} ({})".format(project_name, project_id)
        msg += "\nNominal {} to {} vs {} to {}".format(
            raw_data0["nominal_start_date"],
            raw_data0["nominal_end_date"],
            raw_data1["nominal_start_date"],
            raw_data1["nominal_end_date"],
        )
        msg += "\n{} to {} vs {} to {}".format(
            raw_data0["start_date"],
            raw_data0["end_date"],
            raw_data1["start_date"],
            raw_data1["end_date"],
        )
        print(msg)

        project0_name = "Tmp project from {}, {} to {}".format(
            project_id, raw_data0["start_date"], raw_data0["end_date"]
        )
        project1_name = "Tmp project from {}, {} to {}".format(
            project_id, raw_data1["start_date"], raw_data1["end_date"]
        )

        project0 = project_holder.new_project_from_docs(
            project0_name, docs=raw_data0["docs"]
        )
        project1 = project_holder.new_project_from_docs(
            project1_name, docs=raw_data1["docs"]
        )
        data0 = ProjectData(project0, **get_data_kwargs)
        data1 = ProjectData(project1, **get_data_kwargs)
        project_holder.delete_project(project0.project_id)
        project_holder.delete_project(project1.project_id)
        comparison = ProjectDataComparison(data0, data1, **comparison_kwargs)

        if output_file is not None:
            if len(times) < 1:
                data0.print_summary(output_file=output_file, append=True)
            data1.print_summary(output_file=output_file, append=True)
            comparison.print_summary(output_file=output_file, append=True)
        if verbose:
            if len(times) < 1:
                data0.print_summary()
            data1.print_summary()
            comparison.print_summary()
        if show_plots:
            comparison.show(caption=msg)

        similarities["concept"].append(comparison.concept["similarity"])

        similarities["vector"].append(
            comparison.vector.get_quantile(vector_summary_quantile)
        )

        relevance_range[0] = np.min(
            [
                relevance_range[0],
                np.min(list(data0.relevances.values())),
                np.min(list(data1.relevances.values())),
            ]
        )
        relevance_range[1] = np.max(
            [
                relevance_range[1],
                np.max(list(data0.relevances.values())),
                np.max(list(data1.relevances.values())),
            ]
        )

        similarities["relevance"].append(
            comparison.relevance.get_quantile(summary_quantile)
        )
        similarities["association"].append(
            comparison.association.get_quantile(summary_quantile)
        )
        if comparison.score_driver is not None:
            for field in ["confidence", "confidence-p", "importance", "impact"]:
                similarities[field].append(
                    comparison.score_driver.get(field).get_quantile(summary_quantile)
                )

    result = {
        key: np.array(value) for key, value in similarities.items() if len(value) > 0
    }  # exclude empty data (e.g. score drivers)
    times = np.array(times)

    # relevance can have a very different scale from the rest....
    relevance_msg = "relevance min: {}, max: {}".format(
        relevance_range[0], relevance_range[1]
    )
    plot_multiple_lines(
        caption + "\n" + relevance_msg,
        times,
        dict(relevance=result["relevance"]),
        time_plot=True,
    )
    plot_multiple_lines(
        caption + "\n" + vector_msg,
        times,
        {key: value for key, value in result.items() if key != "relevance"},
        time_plot=True,
    )

    print(relevance_msg)
    if output_file is not None:
        with open(output_file, "at", encoding="utf-8") as fp:
            fp.write(relevance_msg)

    result.update(relevance_range=relevance_range)
    result.update(times=times)
    return result


class DiscrepancyData:
    """
    Helper class holding distributional data about differences between
    project statistics.
    """

    def __init__(
        self,
        cdf=None,
        quantiles=None,
        max_discrepancies=None,
        min_discrepancies=None,
        scale=None,
        max_value=None,
        min_value=None,
        bigger_diff_is_more_discrepancy=True,
    ):
        self.cdf = cdf
        self.quantiles = quantiles
        self.max_discrepancies = max_discrepancies
        self.min_discrepancies = min_discrepancies
        self.scale = scale
        self.max_value = max_value
        self.min_value = min_value
        self.bigger_diff_is_more_discrepancy = bigger_diff_is_more_discrepancy

    def plot(
        self, caption=None, name="", x_label="", y_label="", x_lim=None, note=None
    ):
        """
        Plot the cdf data, given as a list of pairs (x, Pr[X < x]).
        """
        if self.cdf is None:
            print("Skipping plot; no data.")
            return
        fig, axs = plt.subplots()
        xs = [x for x, y in self.cdf]
        ys = [y for x, y in self.cdf]
        axs.plot(xs, ys)
        if x_lim is None:
            axs.set_xlim([np.min(xs), np.max(xs)])
        else:
            axs.set_xlim(x_lim)
        axs.set_ylim([0.0, 1.0])
        axs.set_xlabel(x_label)
        axs.set_ylabel(y_label)
        if caption is None:
            caption = ""
        else:
            caption = "{}\n".format(caption)
        caption = "{}CDF of absolute differences in {}\n{}".format(
            caption,
            name,
            "\n".join(["quantile {} is {}".format(q, v) for q, v in self.quantiles]),
        )
        if not self.bigger_diff_is_more_discrepancy:
            caption = "{}\n(smaller values indicate larger discrepancies)".format(
                caption
            )
        if note is not None:
            caption = "{}\n{}".format(caption, note)
        fig.suptitle(caption)
        plt.show()

    def print_summary(self, name, output_file):
        direction = "bigger" if self.bigger_diff_is_more_discrepancy else "smaller"
        output_file.write(
            ("Discrepancies in {} " "({} values indicate bigger changes).\n").format(
                name, direction
            )
        )
        if self.quantiles is not None and len(self.quantiles) > 0:
            output_file.write("Quantiles of {}:\n".format(name))
            for q, v in self.quantiles:
                output_file.write("  quantile {} is {}\n".format(q, v))
        else:
            output_file.write("No quantiles for {}.\n".format(name))

        if self.max_discrepancies is not None and len(self.max_discrepancies) > 0:
            output_file.write(
                "Maximum {} discrepancies of {} are:\n".format(
                    len(self.max_discrepancies), name
                )
            )
            if len(self.max_discrepancies[0]) > 2:
                for discrepancy in self.max_discrepancies:
                    output_file.write(
                        "  {} vs {} for {}.\n".format(
                            discrepancy[1], discrepancy[2], discrepancy[0]
                        )
                    )
            else:
                for discrepancy in self.max_discrepancies:
                    output_file.write(
                        "  {} for {}.\n".format(discrepancy[1], discrepancy[0])
                    )
        else:
            output_file.write("No maximum discrepancies for {}.\n".format(name))

        if self.min_discrepancies is not None and len(self.min_discrepancies) > 0:
            output_file.write(
                "Minimum {} discrepancies of {} are:\n".format(
                    len(self.min_discrepancies), name
                )
            )
            if len(self.min_discrepancies[0]) > 2:
                for discrepancy in self.min_discrepancies:
                    output_file.write(
                        "  {} vs {} for {}.\n".format(
                            discrepancy[1], discrepancy[2], discrepancy[0]
                        )
                    )
            else:
                for discrepancy in self.min_discrepancies:
                    output_file.write(
                        "  {} for {}.\n".format(discrepancy[1], discrepancy[0])
                    )
        else:
            output_file.write("No minimum discrepancies for {}.\n".format(name))

        if self.max_value is not None:
            output_file.write(
                "Maximum value for {} is {}.\n".format(name, self.max_value)
            )
        if self.min_value is not None:
            output_file.write(
                "Minimum value for {} is {}.\n".format(name, self.min_value)
            )

    @classmethod
    def from_values(
        cls,
        values0,
        values1,
        terms,
        scale=None,
        percentiles=(10, 50, 90),
        n_extreme_values=5,
        bigger_diff_is_more_discrepancy=True,
        **kwargs
    ):
        assert len(values0) == len(values1)
        assert len(values0) == len(terms)
        max_value = max([np.max(values0), np.max(values1)])
        min_value = min([np.min(values0), np.min(values1)])
        diffs = np.abs(np.array(values1) - np.array(values0))
        sign = -1 if bigger_diff_is_more_discrepancy else 1
        permutation = np.argsort(sign * diffs)  # minus to sort descending
        max_discrepancies = [
            (terms[i], values0[i], values1[i]) for i in permutation[:n_extreme_values]
        ]
        min_discrepancies = [
            (terms[i], values0[i], values1[i]) for i in permutation[-n_extreme_values:]
        ]
        if scale is None:
            scale = np.max(
                [
                    np.percentile(np.abs(values0), 90),
                    np.percentile(np.abs(values1), 90),
                    diffs[permutation[0]],
                ]
            )
        percentiles = np.sort(percentiles)
        diff_percentiles = np.percentile(diffs, percentiles)
        quantiles = [(p / 100, v) for p, v in zip(percentiles, diff_percentiles)]
        cdf = []
        fractions = np.linspace(0.0, 1.0, 51)
        for fraction in fractions:
            diff_size = fraction * scale
            n_less = sum([1 for i in range(diffs.shape[0]) if diffs[i] <= diff_size])
            ratio_less = n_less / diffs.shape[0]
            cdf.append((diff_size, ratio_less))
        return cls(
            cdf=cdf,
            quantiles=quantiles,
            max_discrepancies=max_discrepancies,
            min_discrepancies=min_discrepancies,
            scale=scale,
            max_value=max_value,
            min_value=min_value,
            bigger_diff_is_more_discrepancy=bigger_diff_is_more_discrepancy,
            **kwargs
        )

    @classmethod
    def from_diffs(
        cls,
        diffs,
        terms,
        scale=None,
        percentiles=(10, 50, 90),
        n_extreme_values=5,
        bigger_diff_is_more_discrepancy=True,
        **kwargs
    ):
        assert len(diffs) == len(terms)
        sign = -1 if bigger_diff_is_more_discrepancy else 1
        permutation = np.argsort(sign * diffs)  # minus to sort descending
        max_discrepancies = [
            (terms[i], diffs[i]) for i in permutation[:n_extreme_values]
        ]
        min_discrepancies = [
            (terms[i], diffs[i]) for i in permutation[-n_extreme_values:]
        ]
        percentiles = np.sort(percentiles)
        diff_percentiles = np.percentile(diffs, percentiles)
        quantiles = [(p / 100, v) for p, v in zip(percentiles, diff_percentiles)]
        cdf = []
        fractions = np.linspace(0.0, 1.0, 51)
        for fraction in fractions:
            diff_size = fraction * scale
            n_less = sum([1 for i in range(diffs.shape[0]) if diffs[i] <= diff_size])
            ratio_less = n_less / diffs.shape[0]
            cdf.append((diff_size, ratio_less))
        return cls(
            cdf=cdf,
            quantiles=quantiles,
            max_discrepancies=max_discrepancies,
            min_discrepancies=min_discrepancies,
            scale=scale,
            max_value=None,
            min_value=None,
            bigger_diff_is_more_discrepancy=bigger_diff_is_more_discrepancy,
            **kwargs
        )

    def get_quantile(self, quantile_location):
        loc, val = [(q, v) for q, v in self.quantiles if q >= quantile_location][0]
        return val


class AssociationDiscrepancyData(DiscrepancyData):
    def __init__(
        self,
        set_similarity,
        n_concepts,
        cdf=None,
        quantiles=None,
        max_discrepancies=None,
        min_discrepancies=None,
        scale=None,
        max_value=None,
        min_value=None,
        bigger_diff_is_more_discrepancy=True,
    ):
        super().__init__(
            cdf=cdf,
            quantiles=quantiles,
            max_discrepancies=max_discrepancies,
            min_discrepancies=min_discrepancies,
            scale=scale,
            max_value=max_value,
            min_value=min_value,
        )
        self.set_similarity = set_similarity
        self.n_concepts = n_concepts

    def print_summary(self, name, output_file):
        super().print_summary(name, output_file)
        output_file.write(
            "Jaccard index of association concepts is {} (of {}).\n".format(
                self.set_similarity, self.n_concepts
            )
        )

    def plot(
        self, caption=None, name="", x_label="", y_label="", x_lim=None, note=None
    ):
        if note is None:
            note = ""
        else:
            note = "{}\n"
        note += "Jaccard index of association concept sets is {} (of {})".format(
            self.set_similarity, self.n_concepts
        )
        super().plot(
            caption=caption, name=name, x_label=x_label, x_lim=x_lim, note=note
        )


class ProjectData:
    """
    The data we choose to extract from a project for comparison with other projects.
    """

    def __init__(
        self,
        project_holder,
        score_field=None,
        n_concepts=100,
        concept_selector=None,
        n_association_concepts=20,
        concept_association_selector=None,
    ):
        """
        Extract and save data from the project (holder).
        """
        self.project_id = project_holder.project_id
        self.project_name = project_holder.project_name
        self.score_field = score_field

        self.embedding = project_holder.get_term_vectors()

        if concept_selector is None:
            concept_selector = dict(type="top", limit=n_concepts)
        self.concept_selector = concept_selector
        self.concepts = project_holder.get_concept_term_ids(
            concept_selector=self.concept_selector
        )

        self.relevances = {
            t["term_id"]: t["relevance"]
            for t in project_holder.client.get("terms", term_ids=self.concepts)
        }

        if concept_association_selector is None:
            concept_association_selector = dict(
                type="top", limit=n_association_concepts
            )
        self.concept_association_selector = concept_association_selector
        self.associations = project_holder.get_concept_associations(
            concept_selector=concept_association_selector
        )

        if score_field is None:
            self.score_drivers = None
        else:
            self.score_drivers = project_holder.client.get(
                "concepts/score_drivers",
                score_field=score_field,
                concept_selector=concept_selector,
            )

    def print_summary(self, output_file=None, append=False):
        """
        Write a summary of the data to a file (defaults to stdout).  The
        output file may be specified as a file object, or a string
        (representing a path).  If a string is given, the 'append' argument
        may be used to specify whether to write with append mode or not.
        """
        if output_file is None:
            self.print_summary(output_file=sys.stdout)
        elif isinstance(output_file, str):
            with open(output_file, "at" if append else "wt", encoding="utf-8") as fp:
                self.print_summary(output_file=fp)
        else:
            output_file.write(
                "Project: {} ({}).\n".format(self.project_name, self.project_id)
            )
            output_file.write(
                "Selector used for concepts: {}\n".format(self.concept_selector)
            )
            output_file.write(
                "Selector used for concept/concept associations: {}\n".format(
                    self.concept_association_selector
                )
            )
            output_file.write("Score field: {}.\n".format(self.score_field))
            output_file.write("Concepts (with relevances):\n")
            for c in self.concepts:
                output_file.write("  {} ({})\n".format(c, self.relevances[c]))
            output_file.write("Concept/concept associations:\n")
            assocs, term_ids = self.associations
            keys = sorted(assocs.keys())
            for term0, term1 in keys:
                output_file.write(
                    "  {}/{}: {}\n".format(term0, term1, assocs[term0, term1])
                )
            if self.score_drivers is not None:
                output_file.write(
                    "Score drivers for field {}:\n".format(self.score_field)
                )
                for sd in self.score_drivers:
                    output_file.write(
                        "  {}: confidence {}, impact {}, importance {}\n".format(
                            sd["exact_term_ids"][0],
                            sd["confidence"],
                            sd["impact"],
                            sd["importance"],
                        )
                    )
            output_file.write("\n")


class ProjectDataComparison:
    """
    Class of object holding a summary description of the difference between
    two ProjectData instances.
    """

    def __init__(
        self,
        project0_data,
        project1_data,
        n_extreme_values=5,
        normalize_vectors=True,
        percentiles=(10, 50, 90),
        score_field_scale=None,
        n_vector_bins=100,
    ):
        """
        Compute and (optionally) plot similarity comparisons between the
        vectors, top concepts, and (if possible) score drivers between the
        two given projects.
        """
        self.project0_name = project0_data.project_name
        self.project0_id = project0_data.project_id
        self.project1_name = project1_data.project_name
        self.project1_id = project1_data.project_id

        # Compare the common vectors between the projects.
        emb0 = project0_data.embedding
        emb1 = project1_data.embedding
        common_terms, vector_similarities = compare_term_vectors(
            emb0, emb1, normalize=normalize_vectors
        )
        self.vector = DiscrepancyData.from_diffs(
            vector_similarities,
            common_terms,
            scale=1.0,
            percentiles=percentiles,
            n_extreme_values=n_extreme_values,
            bigger_diff_is_more_discrepancy=not normalize_vectors,
        )

        # Compare the top concepts between the projects.
        concepts0 = set(project0_data.concepts)
        concepts1 = set(project1_data.concepts)
        assert len(concepts0) == len(concepts1)
        concepts0only = sorted(
            concepts0 - concepts1, key=lambda c: -project0_data.relevances[c]
        )
        concepts1only = sorted(
            concepts1 - concepts0, key=lambda c: -project1_data.relevances[c]
        )
        concepts0only = concepts0only[:n_extreme_values]
        concepts1only = concepts1only[:n_extreme_values]
        self.concept = dict(
            similarity=jaccard_index(concepts0, concepts1),
            n_concepts=len(concepts0),
            concepts0only=concepts0only,
            concepts1only=concepts1only,
        )

        # Compare the term relevances of the top concepts.
        common_concepts = sorted(concepts0 & concepts1)
        relevances0 = [project0_data.relevances[c] for c in common_concepts]
        relevances1 = [project1_data.relevances[c] for c in common_concepts]
        self.relevance = DiscrepancyData.from_values(
            relevances0,
            relevances1,
            common_concepts,
            percentiles=percentiles,
            n_extreme_values=n_extreme_values,
        )

        # Compare the concept-concept associations.
        associations0, term_ids0 = project0_data.associations
        associations1, term_ids1 = project1_data.associations
        term_ids0 = set(term_ids0)
        term_ids1 = set(term_ids1)
        assert len(term_ids0) == len(term_ids1)
        common_assoc_concepts = sorted(term_ids0 & term_ids1)
        if len(common_assoc_concepts) < 1:
            print("Warning: no common concepts found for association comparison.")
            self.association = AssociationDiscrepancyData(
                set_similarity=jaccard_index(term_ids0, term_ids1),
                n_concepts=len(term_ids0),
            )
        else:
            pairs = [
                (c0, c1) for c0 in common_assoc_concepts for c1 in common_assoc_concepts
            ]
            assocs0 = [associations0[(c0, c1)] for c0, c1 in pairs]
            assocs1 = [associations1[(c0, c1)] for c0, c1 in pairs]
            self.association = AssociationDiscrepancyData.from_values(
                assocs0,
                assocs1,
                pairs,
                n_extreme_values=n_extreme_values,
                percentiles=percentiles,
                set_similarity=jaccard_index(term_ids0, term_ids1),
                n_concepts=len(term_ids0),
            )

        # Compare the score drivers between the projects (if possible).
        self.score_driver = None
        score_drivers0 = project0_data.score_drivers
        score_drivers1 = project1_data.score_drivers
        if score_drivers0 is not None and score_drivers1 is not None:
            score_drivers0_dict = {
                c: sd
                for c in common_concepts
                for sd in score_drivers0
                if sd["exact_term_ids"] == [c]
            }
            score_drivers1_dict = {
                c: sd
                for c in common_concepts
                for sd in score_drivers1
                if sd["exact_term_ids"] == [c]
            }
            score_driver_pairs = [
                (score_drivers0_dict[c], score_drivers1_dict[c])
                for c in common_concepts
            ]
            self.score_driver = {}
            sqrt2 = np.sqrt(2.0)  # take end of scale for t-values as 0.01 p-value
            for field, scale in [
                ("impact", score_field_scale),
                ("confidence", sqrt2 * erfcinv(0.01)),
                ("importance", 1.0),
            ]:
                values0 = [sd0[field] for sd0, sd1 in score_driver_pairs]
                values1 = [sd1[field] for sd0, sd1 in score_driver_pairs]
                if field == "confidence":
                    # Convert t-values to 1-(p-values), i.e. estimates of the
                    # probability that this concept actually impacts the score.
                    # Since we don't know the number of degrees of freedom of the
                    # relevant t-distribution, assume it is large enough that
                    # the t-distribution is approximately standard normal.  (This
                    # is likely with our data).  Then the conversion (using a two-
                    # sided test, and letting Z denote a standard normal variate)
                    # is t -> Pr[|Z| < |t|] = erf(|t|/sqrt(2)), but to better
                    # see changes from positive to negative t-values we preserve
                    # the sign on the p-value (i.e. t -> erf(t/sqrt(2))).
                    values0p = [erf(t / sqrt2) for t in values0]
                    values1p = [erf(t / sqrt2) for t in values1]
                    self.score_driver["confidence-p"] = DiscrepancyData.from_values(
                        values0p,
                        values1p,
                        common_concepts,
                        n_extreme_values=n_extreme_values,
                        scale=1.0,
                        percentiles=percentiles,
                    )
                self.score_driver[field] = DiscrepancyData.from_values(
                    values0,
                    values1,
                    common_concepts,
                    n_extreme_values=n_extreme_values,
                    scale=scale,
                    percentiles=percentiles,
                )

    def show(self, caption=None):
        """
        Display a series of plots containing the comparison data.
        """
        if caption is None:
            vectors_caption = ""
        else:
            vectors_caption = "{}".format(caption)
        vectors_caption += "\nconcept similarity {} (of {})".format(
            self.concept["similarity"], self.concept["n_concepts"]
        )
        if self.vector.bigger_diff_is_more_discrepancy:
            name = "\nvector difference (Euclidean distance)"
        else:
            name = "\nvector similarity (cosine simiarity)"
        self.vector.plot(
            caption=vectors_caption,
            name=name,
            x_label=name,
            y_label="fraction of terms",
        )

        self.relevance.plot(
            caption=caption,
            name="term relevances",
            x_label="absolute difference in relevance",
            y_label="fraction of terms",
        )

        self.association.plot(
            caption=caption,
            name="concept associations",
            x_lim=[0.0, 1.0],
            x_label="absolute difference in association",
            y_label="fraction of concept pairs",
        )

        if self.score_driver is not None:
            for field, data in self.score_driver.items():
                x_lim = None if data.scale is None else [0.0, data.scale]
                data.plot(
                    caption=caption,
                    name=field,
                    x_lim=x_lim,
                    x_label="absolute difference in {}".format(field),
                    y_label="fraction of concepts",
                )

    def print_summary(self, output_file=None, append=False):
        """
        Write a summary of the comparison data to a file (defaults to stdout).
        The output file may be specified as a file object, or a string
        (representing a path).  If a string is given, the 'append' argument
        may be used to specify whether to write in append mode or not.
        """
        if output_file is None:
            self.print_summary(output_file=sys.stdout)
        elif isinstance(output_file, str):
            with open(output_file, "at" if append else "wt", encoding="utf-8") as fp:
                self.print_summary(output_file=fp)
        else:
            output_file.write(
                "Comparison of {} ({}) to {} ({}).\n".format(
                    self.project0_name,
                    self.project0_id,
                    self.project1_name,
                    self.project1_id,
                )
            )
            if self.vector.bigger_diff_is_more_discrepancy:
                msg = "Euclidean distance of common vectors"
            else:
                msg = "cosine similarity of common vectors"
            self.vector.print_summary(msg, output_file)

            output_file.write(
                "Jaccard index of top concepts is {} (of {}).\n".format(
                    self.concept["similarity"], self.concept["n_concepts"]
                )
            )
            output_file.write(
                "Most relevant concepts in {} not in {} are:\n".format(
                    self.project0_id, self.project1_id
                )
            )
            for c in self.concept["concepts0only"]:
                output_file.write("  {}\n".format(c))
            output_file.write(
                "Most relevant concepts in {} not in {} are:\n".format(
                    self.project1_id, self.project0_id
                )
            )
            for c in self.concept["concepts1only"]:
                output_file.write("  {}\n".format(c))

            self.relevance.print_summary(
                "absolute differences in term relevance", output_file
            )

            self.association.print_summary(
                "absolute differences in concept associations", output_file
            )

            if self.score_driver is not None:
                for field, data in self.score_driver.items():
                    name = "absolute differences in {}".format(field)
                    data.print_summary(name, output_file)

            output_file.write("\n")


def main(args):
    """
    Perform a differential study as directed by the arguments in args.
    """
    print("Starting {} study for project {}.".format(args.study_kind, args.project_id))
    client = LuminosoClientHolder.from_root_url(url=args.root_url)
    project = client.get_project(args.project_id)
    get_data_kwargs = dict(
        score_field=args.score_field,
        n_concepts=args.n_concepts,
        n_association_concepts=args.n_assoc_concepts,
    )
    get_data_kwargs = {k: v for k, v in get_data_kwargs.items() if v is not None}
    comparison_kwargs = dict(
        normalize_vectors=args.normalize_vectors,
        score_field_scale=args.score_field_scale,
        n_extreme_values=args.n_extreme_values,
    )
    comparison_kwargs = {k: v for k, v in comparison_kwargs.items() if v is not None}
    filter0 = [eval(d) for d in args.filter] if args.filter is not None else None
    filter1 = [eval(d) for d in args.filter1] if args.filter1 is not None else None
    search0 = eval(args.search) if args.search is not None else None
    search1 = eval(args.search1) if args.search1 is not None else None
    if args.study_kind == "subset":
        subset_study(
            project,
            filter0=filter0,
            search0=search0,
            filter1=filter1,
            search1=search1,
            get_data_kwargs=get_data_kwargs,
            comparison_kwargs=comparison_kwargs,
            caption=project.project_name,
            output_file=args.output_file,
        )
    else:
        longitudinal_study(
            project,
            time_step=np.timedelta64(args.time_step, "D"),
            window_length=np.timedelta64(args.window_length, "D"),
            verbose=args.verbose,
            show_plots=args.show_plots,
            get_data_kwargs=get_data_kwargs,
            comparison_kwargs=comparison_kwargs,
            output_file=args.output_file,
        )


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description=(
            "Perform differential analytics on a given project. "
            "Either a comparison between two subsets of the project's "
            "documents (a 'subset study') may be done, or a series of "
            "comparisons between documents dated within successive time "
            "windows (a 'longintudinal study'; requires the documents to "
            "have date metadata)."
        )
    )
    argparser.add_argument("project_id", help="The id of a project to analyze.")
    argparser.add_argument(
        "-r",
        "--root-url",
        default="https://analytics.luminoso.com/api/v5",
        help=(
            "The root URL used to construct a V5 API client to access "
            "projects. (Note that the V5 client requires you to have set "
            "up authentication tokens in the standard location to read "
            "projects, and that you will need permission to create projects "
            "as well to use this tool.)"
        ),
    )
    argparser.add_argument(
        "-k",
        "--study-kind",
        default="subset",
        choices=["subset", "longitudinal"],
        help="The kind of study to perform (subset or longitudinal).",
    )
    argparser.add_argument(
        "--filter",
        action="append",
        help=(
            "A document filter entry (see the API documentation), e.g. "
            "{'name': 'Rating', 'maximum': 3}.  Zero or more entries may "
            "be specified, by repeating this option, to create a composite "
            "filter.  Used only to define the first subset of a subset study."
        ),
    )
    argparser.add_argument(
        "--search",
        default=None,
        help=(
            "A JSON-encoded concept used (along with any document filter given) "
            "to select the first subset of documents of a subset study.  "
            "(Optional)."
        ),
    )
    argparser.add_argument(
        "--filter1",
        action="append",
        help=(
            "A document filter entry.  Zero or more entries may be given, by "
            "repeating this option, to create a second composite filter, used "
            "to define the second subset of a subset study."
        ),
    )
    argparser.add_argument(
        "--search1",
        default=None,
        help=(
            "A second JSON-encoded concept, used to select the second subset of "
            "a subset study."
        ),
    )
    argparser.add_argument(
        "--time-step",
        type=int,
        default=7,
        help=(
            "Number of days by which to offset successive document windows "
            "for longitudinal studies."
        ),
    )
    argparser.add_argument(
        "--window-length",
        type=int,
        default=30,
        help="Length in days of each time window of a longintudinal study.",
    )
    argparser.add_argument(
        "--n-concepts",
        type=int,
        default=20,
        help="Number of (top) concepts to compare.",
    )
    argparser.add_argument(
        "--n-assoc-concepts",
        type=int,
        default=5,
        help="Number of concepts retrieved to compare concept-concept associations.",
    )
    argparser.add_argument(
        "--score-field",
        default=None,
        help="Document metadata field to be used to compute score drivers. (Optional.)",
    )
    argparser.add_argument(
        "--score-field-scale",
        type=float,
        default=None,
        help=(
            "Maximum value attained by the score field.  (Optional, if not "
            "given an approximate value will be deduced from the data.)"
        ),
    )
    argparser.add_argument(
        "--use-nonnormal-vectors",
        dest="normalize_vectors",
        action="store_false",
        help=(
            "Compare similarity of vectors by Euclidean distance instead "
            "of cosine similarity.  (Optional, default is to use cosine "
            "similarity.)"
        ),
    )
    argparser.add_argument(
        "--n-extreme-values",
        type=int,
        default=5,
        help=(
            "Number of examples of most extreme differences to report "
            "for each statistic compared."
        ),
    )
    argparser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Output additional messages to stdout.",
    )
    argparser.add_argument(
        "--show-plots",
        action="store_true",
        help="Display additional plots during computation.",
    )
    argparser.add_argument(
        "-f",
        "--output-file",
        default=None,
        help="Path to a file to which to write output (optional).",
    )
    args = argparser.parse_args()
    main(args)
