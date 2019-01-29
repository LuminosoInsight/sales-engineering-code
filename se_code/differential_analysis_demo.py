"""
Tools for differential analytics on projects.  It is probably easiest (and
most flexible) to use them from an ipython prompt or a jupyter notebook, but
a command-line interface exposing most of the functionality is also provided.
"""

import argparse
import csv
import matplotlib.pyplot as plt
import matplotlib.dates as plt_dates
import matplotlib.lines as plt_lines
import numpy as np
import pandas as pd
import re
import sys

from api_utils_app.luminoso_client_holder import LuminosoClientHolder


SCORE_DRIVER_METRICS = {"impact", "confidence", "importance"}
DERIVED_METRICS = {"impact_CI_lower_bound", "impact_CI_upper_bound"}
ALL_METRICS = sorted(SCORE_DRIVER_METRICS | DERIVED_METRICS)


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
    whole_project_data_kwargs=None,
    subproject_data_kwargs=None,
    report_file=None,
):
    """
    Compare score drivers from two projects built from
    subsets of the documents of the given project, as defined by the two
    given document filters and concept selectors.
    """
    whole_project_data_kwargs = whole_project_data_kwargs or {}
    subproject_data_kwargs = subproject_data_kwargs or {}
    project_data = ProjectData(project_holder, **whole_project_data_kwargs)
    score_drivers = project_data.score_drivers  # get 'em for the whole project
    if (
        subproject_data_kwargs.get("score_drivers") is None
        and subproject_data_kwargs.get("concept_selector") is None
    ):
        subproject_data_kwargs.update(score_drivers=score_drivers)
    subproject_data_kwargs.update(score_field=project_data.score_field)

    project0_name = "Tmp project from {}, subset 0".format(project_holder.project_name)
    project1_name = "Tmp project from {}, subset 1".format(project_holder.project_name)

    data0 = ProjectData.from_filter_and_search(
        project_holder,
        filter=filter0,
        search=search0,
        project_name=project0_name,
        tag=0,
        **subproject_data_kwargs
    )
    data1 = ProjectData.from_filter_and_search(
        project_holder,
        filter=filter1,
        search=search1,
        project_name=project1_name,
        tag=1,
        **subproject_data_kwargs
    )

    caption = (
        "Subset comparison of impact for driver(s) of {}, project: {} ({})\n"
        "Filter for subset zero: {}\n"
        "Search for subset zero: {}\n"
        "Filter for subset one: {}\n"
        "Search for subset one: {}\n"
    ).format(
        project_data.score_field,
        project_holder.project_name,
        project_holder.project_id,
        filter0,
        search0,
        filter1,
        search1,
    )

    msg = (
        caption
        + "Requested (whole project) score drivers:\n"
        + "\n".join(["  {}".format(sd) for sd in score_drivers])
    )
    msg += "\n(from concept selector {})".format(project_data.concept_selector)

    if report_file is not None:
        with open(report_file, "wt", encoding="utf-8") as fp:
            fp.write(msg + "\n")
            data0.print_summary(output_file=fp)
            data1.print_summary(output_file=fp)

    print(msg)
    data0.print_summary()
    data1.print_summary()

    result = ProjectDataSequence(score_drivers=score_drivers, is_time_series=False)
    result.append(data0)
    result.append(data1)
    result.plot(caption=caption, x_label="Subset", y_label="impact")
    return result


def get_project_time_windows(
    project_holder,
    start_date=None,
    end_date=None,
    time_step=np.timedelta64(1, "W"),
    window_length=np.timedelta64(30, "D"),
):
    """
    Given a project client holder, a start and end date (numpy datetime64's,
    default values are None, which implies the date of the first document of
    the project, and the day after the date of the last), a time step and a
    window length (numpy timedelta64's, defaults are one week and 30 days),
    generate a sequence of time windows, the i-th starting i time steps after
    the start date and having duration equal to the given window length.  The
    The last window will start on or before the end date (if given).

    Return a sequence of dicts, one for each window, with two values:  the
    pair of endpoints of the window, and a list of the documents from the
    project having dates within that time span.
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
    if start_date is None:
        start_time = get_date_from_document(docs[0])
    else:
        start_time = start_date
    if end_date is None:
        grand_end_time = get_date_from_document(docs[-1])
    else:
        grand_end_time = end_date

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
    start_date=None,
    end_date=None,
    time_step=np.timedelta64(1, "W"),
    window_length=np.timedelta64(30, "D"),
    verbose=True,
    whole_project_data_kwargs=None,
    subproject_data_kwargs=None,
    report_file=None,
):
    """
    Compute and plot changes over time in the score drivers of
    a series of projects constructed from the given one by restricting to
    documents from time windows generated from the given time step and
    window length (as in get_project_time_windows).
    """

    def window_to_data(window):
        t0, t1 = window["interval"]
        docs = window["documents"]
        if len(docs) < 1:
            print("Warning: time window {} to {} had no data.".format(t0, t1))
        data = dict(nominal_start_date=t0, nominal_end_date=t1, docs=docs)
        return data

    project_name = project_holder.project_name
    project_id = project_holder.project_id
    whole_project_data_kwargs = whole_project_data_kwargs or {}
    subproject_data_kwargs = subproject_data_kwargs or {}

    project_data = ProjectData(project_holder, **whole_project_data_kwargs)
    score_drivers = project_data.score_drivers  # get 'em for the whole project
    if (
        subproject_data_kwargs.get("score_drivers") is None
        and subproject_data_kwargs.get("concept_selector") is None
    ):
        subproject_data_kwargs.update(score_drivers=score_drivers)
    subproject_data_kwargs.update(score_field=project_data.score_field)

    msg = "{} (id {})".format(project_name, project_id)
    msg += "\nTime step {}, window length {}\n".format(
        str(time_step), str(window_length)
    )  # format chokes on np.timedelta64
    msg += "Requested (whole project) score drivers:\n" + "\n".join(
        ["  {}".format(sd) for sd in score_drivers]
    )
    msg += "\n(from concept selector {})".format(project_data.concept_selector)

    print(msg)
    if report_file is not None:
        with open(report_file, "wt", encoding="utf-8") as fp:
            fp.write(msg + "\n\n")

    subproject_data_sequence = ProjectDataSequence(score_drivers=score_drivers)

    for window in get_project_time_windows(
        project_holder,
        start_date=start_date,
        end_date=end_date,
        time_step=time_step,
        window_length=window_length,
    ):
        start_date, end_date = window["interval"]
        msg = "{} ({})".format(project_name, project_id)
        msg += "\nTime from {} to {}, {} documents".format(
            start_date, end_date, len(window["documents"])
        )
        print(msg)

        subproject_name = "Tmp project from {}, {} to {}".format(
            project_id, start_date, end_date
        )

        data = ProjectData.from_docs(
            project_holder,
            docs=window["documents"],
            tag=end_date,
            project_name=subproject_name,
            **subproject_data_kwargs
        )
        subproject_data_sequence.append(data)

        if report_file is not None:
            data.print_summary(output_file=report_file, append=True)
        if verbose:
            data.print_summary()

    caption = (
        "Time vs impact of driver(s) on {}, project: {} ({})\n"
        "Time step {}\n"
        "Time window length {}\n"
    ).format(
        project_data.score_field,
        project_name,
        project_id,
        str(time_step),
        str(window_length),
    )
    subproject_data_sequence.plot(
        caption=caption, x_label="Final date of window", y_label="impact"
    )
    return subproject_data_sequence


class ProjectData:
    """
    The data we choose to extract from a project for comparison with other
    projects.
    """

    def __init__(
        self,
        project_holder,
        score_drivers=None,
        score_field="score",
        concept_selector=None,
        tag=None,
    ):
        """
        Extract and save data from the project (holder).
        """
        self.tag = tag
        if project_holder is None:
            self.project_id = None
            self.project_name = None
            self.document_count = 0
        else:
            self.project_id = project_holder.project_id
            self.project_name = project_holder.project_name
            self.document_count = project_holder.get_project_info(
                project_id=self.project_id
            )[0]["document_count"]
        self.score_field = score_field

        if score_drivers is not None and len(score_drivers) > 0:
            self.concept_selector = dict(
                type="specified", concepts=[{"texts": sd} for sd in score_drivers]
            )
        elif concept_selector is not None:
            self.concept_selector = concept_selector
        else:
            self.concept_selector = dict(type="top", limit=5)

        if project_holder is None:
            all_score_driver_concepts = []
        else:
            all_score_driver_concepts = project_holder.client.get(
                "concepts/score_drivers",
                score_field=self.score_field,
                concept_selector=self.concept_selector,
            )

        if score_drivers is not None and len(score_drivers) > 0:
            self.score_drivers = score_drivers
        else:
            self.score_drivers = [c["texts"] for c in all_score_driver_concepts]
        self.score_drivers.sort()

        self.score_driver_values = {}
        for metric in ALL_METRICS:
            self.score_driver_values[metric] = np.full(
                (len(self.score_drivers),), np.nan
            )

        for i_driver, score_driver in enumerate(self.score_drivers):
            score_driver_concepts = [
                c for c in all_score_driver_concepts if score_driver == c["texts"]
            ]

            if len(score_driver_concepts) > 1:
                print(
                    "Warning:  driver {} for {} was found multiple times in {}.".format(
                        score_driver, self.score_field, self.project_id
                    )
                )

            if len(score_driver_concepts) > 0:
                for metric in SCORE_DRIVER_METRICS:
                    value = score_driver_concepts[0].get(metric, np.nan)
                    self.score_driver_values[metric][i_driver] = float(value)
                # also save the lower and upper bounds of a 95% confidence
                # interval for the impact, using the normal 1.96 formula
                impact = self.score_driver_values["impact"][i_driver]
                confidence = self.score_driver_values["confidence"][i_driver]
                estimated_stdev = impact / confidence  # stdev of impact
                self.score_driver_values["impact_CI_lower_bound"][i_driver] = (
                    impact - 1.96 * estimated_stdev
                )
                self.score_driver_values["impact_CI_upper_bound"][i_driver] = (
                    impact + 1.96 * estimated_stdev
                )

    @classmethod
    def from_docs(cls, project_holder, docs, project_name=None, **kwargs):
        """
        Make a ProjectData instance from an iterable of documents.
        """
        if project_name is None:
            project_name = "Tmp project from {}".format(project_holder.project_name)

        # We could just try to make a new project from the given docs and
        # catch the error if no docs were given, but that is a pain through
        # the web API as it leaves an empty project on the server.  So we
        # explicitly check for that condition, even though that is tricky
        # too as we want to accomodate the case in which docs is a generator.
        try:
            doc0 = next(doc for doc in docs)
        except StopIteration:
            no_docs = True
        else:
            no_docs = False

            def true_docs():
                yield doc0
                yield from docs

        if no_docs:
            return cls(None, **kwargs)
        else:
            new_project = project_holder.new_project_from_docs(
                project_name=project_name, docs=true_docs()
            )
            result = cls(project_holder=new_project, **kwargs)
            project_holder.delete_project(new_project.project_id)
            return result

    @classmethod
    def from_filter_and_search(
        cls, project_holder, filter=None, search=None, project_name=None, **kwargs
    ):
        """
        Make a ProjectData instance from the subproject of the given project
        (holder) defined by the given filter and search.
        """
        if project_name is None:
            project_name = "Tmp project from {}".format(project_holder.project_id)

        docs_args = {}
        if filter is not None:
            docs_args["filter"] = filter
        if search is not None:
            docs_args["search"] = search

        docs = project_holder.get_docs(**docs_args)
        project_data = ProjectData.from_docs(
            project_holder, docs=docs, project_name=project_name, **kwargs
        )
        return project_data

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
            output_file.write("Tag is {}.\n".format(self.tag))
            output_file.write("Document count is {}.\n".format(self.document_count))
            for metric in ALL_METRICS:
                for driver, value in zip(
                    self.score_drivers, self.score_driver_values[metric]
                ):
                    output_file.write(
                        "Score driver {} for {} has {} {:.3f}.\n".format(
                            driver, self.score_field, metric, value
                        )
                    )


class ProjectDataSequence:
    def __init__(self, score_drivers, is_time_series=True):
        self.project_data_list = []
        self.score_drivers = score_drivers
        self.is_time_series = is_time_series
        self._score_driver_values = None

    def append(self, project_data):
        self.project_data_list.append(project_data)
        self._score_driver_values = None  # force recalculation

    @property
    def score_driver_values(self):
        if self._score_driver_values is None:
            score_driver_strings = [str(driver) for driver in self.score_drivers]
            self._score_driver_values = {}
            for metric in ALL_METRICS:
                score_driver_frame = pd.DataFrame(
                    index=score_driver_strings,
                    data=np.empty((len(self.score_drivers), 0)),
                )
                for project_data in self.project_data_list:
                    project_score_driver_strings = [
                        str(driver) for driver in project_data.score_drivers
                    ]
                    frame = pd.DataFrame(
                        index=project_score_driver_strings,
                        data=project_data.score_driver_values[metric],
                    )
                    score_driver_frame.insert(
                        len(score_driver_frame.columns),
                        len(score_driver_frame.columns),
                        frame,
                    )
                for driver_string in score_driver_strings:
                    self._score_driver_values[
                        (driver_string, metric)
                    ] = score_driver_frame.loc[driver_string].values
        return self._score_driver_values

    def plot(self, caption=None, x_label=None, y_label=None):
        if len(self.project_data_list) < 1:
            print("Attempt to plot zero-length project data sequence, skipping.")
            return

        ordinates = np.array([data.tag for data in self.project_data_list])
        fig, axs = plt.subplots()
        if self.is_time_series:  # if the x axis is time data, treat it specially
            ordinates = ordinates.astype("O")  # convert to datetime.datetimes
            fig.autofmt_xdate()
            axs.xaxis_date()
            axs.fmt_xdata = plt_dates.DateFormatter("%Y-%m-%d")
        markers = ["o", "v", "^", "s", "D"]
        colors = ["red", "green", "blue", "cyan", "magenta", "brown"]
        if self.is_time_series:
            linestyle = "solid"
            count_linestyle = "dotted"
            offsets = [np.timedelta64(0, "D").astype("O")]
        else:
            linestyle = ""  # no lines between markers
            count_linestyle = ""
            max_offset = 0.25 / (len(self.score_drivers) + 1)
            offsets = np.linspace(-max_offset, max_offset, num=len(self.score_drivers))
        for i_driver, driver in enumerate(self.score_drivers):
            marker = markers[i_driver % len(markers)]
            color = colors[i_driver % len(colors)]
            offset = offsets[i_driver % len(offsets)]
            values = self.score_driver_values[(str(driver), "impact")]
            ci_lower_bounds = self.score_driver_values[
                (str(driver), "impact_CI_lower_bound")
            ]
            ci_upper_bounds = self.score_driver_values[
                (str(driver), "impact_CI_upper_bound")
            ]
            line = plt_lines.Line2D(
                ordinates + offset,
                values,
                marker=marker,
                color=color,
                label=str(driver),
                linestyle=linestyle,
            )
            axs.add_line(line)
            for ordinate, lower, upper in zip(
                ordinates, ci_lower_bounds, ci_upper_bounds
            ):
                error_bar = plt_lines.Line2D(
                    [ordinate + offset, ordinate + offset],
                    [lower, upper],
                    marker="_",
                    color=color,
                    linestyle="solid",
                )
                axs.add_line(error_bar)

        axs.autoscale(axis="y", tight=False)
        min_ordinate = np.min(ordinates)
        max_ordinate = np.max(ordinates)
        if self.is_time_series:
            x0 = min_ordinate - 0.05 * (max_ordinate - min_ordinate)
            x1 = max_ordinate + 0.05 * (max_ordinate - min_ordinate)
            axs.set_xlim(left=x0, right=x1)
        else:
            x0 = min_ordinate - 0.25 * (max_ordinate - min_ordinate)
            x1 = max_ordinate + 0.25 * (max_ordinate - min_ordinate)
            axs.set_xlim(left=x0, right=x1)
            axs.set_xticks([min_ordinate, max_ordinate])
        axs.legend()
        if x_label is not None:
            axs.set_xlabel(x_label)
        if y_label is not None:
            axs.set_ylabel(y_label)
        if caption is not None:
            fig.suptitle(caption)

        axs_counts = axs.twinx()  # overlay plot of document counts
        counts = [data.document_count for data in self.project_data_list]
        axs_counts.plot(
            ordinates,
            counts,
            color="black",
            marker="x",
            linestyle=count_linestyle,
            label="document count",
        )
        axs_counts.legend()
        plt.show()

    def write_csv(self, path):
        # Since this is CSV, we keep commas out of the fieldnames.
        sanitized_score_drivers = [
            re.sub(",", "", str(score_driver)) for score_driver in self.score_drivers
        ]
        fieldnames = ["tag", "document_count"] + [
            "{}/{}".format(sanitized, metric)
            for sanitized in sanitized_score_drivers
            for metric in ALL_METRICS
        ]
        with open(path, "wt", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writeheader()
            for i_row, project_data in enumerate(self.project_data_list):
                row = dict(
                    tag=str(project_data.tag),
                    document_count=str(project_data.document_count),
                )
                for sanitzed, score_driver in zip(
                    sanitized_score_drivers, self.score_drivers
                ):
                    for metric in ALL_METRICS:
                        fieldname = "{}/{}".format(score_driver, metric)
                        value = self.score_driver_values[(str(score_driver), metric)][
                            i_row
                        ]
                        row.update({fieldname: value})
                writer.writerow(row)


def main(args):
    """
    Perform a differential study as directed by the arguments in args.
    """
    print("Starting {} study for project {}.".format(args.study_kind, args.project_id))
    client = LuminosoClientHolder.from_root_url(url=args.root_url)
    project = client.get_project(args.project_id)

    if args.score_driver is None:
        score_drivers = None
    else:
        score_drivers = [eval(sd) for sd in args.score_driver]

    whole_project_data_kwargs = dict(
        score_field=args.score_field,
        score_drivers=score_drivers,
        concept_selector=eval(args.score_driver_selector),
    )

    if args.subproject_driver_selector is None:
        subproject_driver_selector = None
    else:
        subproject_driver_selector = eval(args.subproject_driver_selector)

    subproject_data_kwargs = dict(concept_selector=subproject_driver_selector)

    filter0 = [eval(d) for d in args.filter] if args.filter is not None else None
    filter1 = [eval(d) for d in args.filter1] if args.filter1 is not None else None
    search0 = eval(args.search) if args.search is not None else None
    search1 = eval(args.search1) if args.search1 is not None else None

    if args.start_date is not None:
        start_date = np.datetime64(args.start_date)
    else:
        start_date = None
    if args.end_date is not None:
        end_date = np.datetime64(args.end_date)
    else:
        end_date = None

    if args.study_kind == "subset":
        sequence = subset_study(
            project,
            filter0=filter0,
            search0=search0,
            filter1=filter1,
            search1=search1,
            whole_project_data_kwargs=whole_project_data_kwargs,
            subproject_data_kwargs=subproject_data_kwargs,
            report_file=args.report_file,
        )
    else:
        sequence = longitudinal_study(
            project,
            start_date=start_date,
            end_date=end_date,
            time_step=np.timedelta64(args.time_step, "D"),
            window_length=np.timedelta64(args.window_length, "D"),
            verbose=args.verbose,
            whole_project_data_kwargs=whole_project_data_kwargs,
            subproject_data_kwargs=subproject_data_kwargs,
            report_file=args.report_file,
        )
    if args.csv_file is not None:
        sequence.write_csv(args.csv_file)


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
            "as well to use this tool.  Defaults to luminoso analytics.)"
        ),
    )
    argparser.add_argument(
        "-k",
        "--study-kind",
        default="longitudinal",
        choices=["subset", "longitudinal"],
        help=(
            "The kind of study to perform (subset or longitudinal). "
            "(Default is longitudinal.)"
        ),
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
            "a subset study. (Optional.)"
        ),
    )
    argparser.add_argument(
        "--start-date",
        default=None,
        help=(
            "Date (YYYY-MM-DD) on which to start document windows for "
            "longintudinal studies.  (Defaults to the earliest document "
            "date in the project.)"
        ),
    )
    argparser.add_argument(
        "--end-date",
        default=None,
        help=(
            "Date (YYYY-MM-DD) on which to end document windows for "
            "longintudinal studies.  (The final window may extend past "
            "this date but will start on or before it.  Defaults to "
            "the last date of a document in the project.)"
        ),
    )
    argparser.add_argument(
        "--time-step",
        type=int,
        default=7,
        help=(
            "Number of days by which to offset successive document windows "
            "for longitudinal studies. (Default 7.)"
        ),
    )
    argparser.add_argument(
        "--window-length",
        type=int,
        default=30,
        help=(
            "Length in days of each time window of a longintudinal study. "
            "(Default 30.)"
        ),
    )
    argparser.add_argument(
        "--score-field",
        default="score",
        help=(
            "Document metadata field to be used to compute score drivers. "
            "(Defaults to 'score'.)"
        ),
    )
    argparser.add_argument(
        "--score-driver",
        action="append",
        help=(
            'Concept texts (e.g. ["red", "pink"]) of a score driver to '
            "compare.  This argument may be repeated to specify multiple "
            "drivers.  If none are given, the selector given as the "
            "score-driver-selector argument will be applied to the whole "
            "project to find score drivers."
        ),
    )
    argparser.add_argument(
        "--score-driver-selector",
        default="{'type': 'top', 'limit': 5}",
        help=(
            "If no score driver arguments are given, this concept selector "
            "will be used to find the score driver concepts from the whole "
            "project.  (Defaults to top 5 concepts.)"
        ),
    )
    argparser.add_argument(
        "--subproject-driver-selector",
        default=None,
        help=(
            "If given, selects the score-driver concepts from the subprojects.  "
            "If not specified, the score drivers for the whole project are used."
        ),
    )
    argparser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Output additional messages to stdout.",
    )
    argparser.add_argument(
        "-f",
        "--report-file",
        default=None,
        help="Path to a file to which to write a summary report (optional).",
    )
    argparser.add_argument(
        "-c",
        "--csv-file",
        default=None,
        help=(
            "Path to which to write a .csv file containing score driver "
            "metrics (optional)."
        ),
    )
    args = argparser.parse_args()
    main(args)
