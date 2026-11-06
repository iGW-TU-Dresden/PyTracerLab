"""Reading tracer input time series and observations from text files.

The central function is :func:`read_series`. It reads simple delimited text
files (e.g. CSV files exported from a spreadsheet) with one row per date and
one column per tracer::

    Date, 3H, 3H_sd
    1963, 1450, 45
    1985-06, 52.1, 1.6
    1994.5, 18.3, 0.6

File format
-----------
- The first column holds the date, every further column one tracer.
- A header line with the column names is optional but recommended; it may start
  with ``#``. Further lines starting with ``#`` are treated as comments.
- Delimiter: comma, semicolon, tab or whitespace (detected automatically). With
  semicolon, tab or whitespace delimiters, a decimal comma is accepted as well.
- Dates: ``YYYY``, ``YYYY-MM``, ``YYYY-MM-DD`` (also with ``/``),
  ``DD.MM.YYYY``, ``MM.YYYY`` or a decimal year (e.g. ``1985.5``). A time of
  day after the date (e.g. ``1985-06-15 00:00:00``) is ignored. A date given as
  a year only refers to 1 January of that year.
- Missing values: empty cell, ``nan``, ``NA``, ``n/a`` or ``-``.
- A column named ``<tracer>_sd`` holds the measurement uncertainty (standard
  deviation, same unit as the tracer) of the column ``<tracer>``.

Tracer input time series must have regular time steps without gaps (e.g.
monthly or yearly). Observations can either be given for every time step of
the input series (``nan`` where nothing was observed) or only at the sampling
dates; in the latter case they are assigned to the time step of the input
series that contains the sampling date (use ``align_to``).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

__all__ = ["DataFileError", "TracerSeries", "read_series"]

PathLike = Union[str, "os.PathLike[str]"]

_MISSING = {"", "nan", "na", "n/a", "-", "--", "null", "none"}
# candidate delimiters in order of preference; None = any whitespace
_DELIMITERS: Tuple[Optional[str], ...] = ("\t", ";", ",", None)
_SD_SUFFIX = "_sd"
_DATE_FORMATS_TEXT = (
    "YYYY, YYYY-MM, YYYY-MM-DD, DD.MM.YYYY, MM.YYYY or a decimal year (e.g. 1985.5)"
)

_RE_YEAR = re.compile(r"^(\d{4})$")
_RE_DECIMAL_YEAR = re.compile(r"^(\d{4})\.(\d+)$")
_RE_YM = re.compile(r"^(\d{4})[-/](\d{1,2})$")
_RE_YMD = re.compile(r"^(\d{4})[-/](\d{1,2})[-/](\d{1,2})$")
_RE_DMY = re.compile(r"^(\d{1,2})\.(\d{1,2})\.(\d{4})$")
_RE_MY = re.compile(r"^(\d{1,2})\.(\d{4})$")


class DataFileError(ValueError):
    """A data file could not be read.

    The message names the file, the line and the value that caused the
    problem, and what was expected instead.
    """


@dataclass
class TracerSeries:
    """Time series of one or more tracers as returned by :func:`read_series`.

    Attributes
    ----------
    times : ndarray of datetime.datetime
        Dates of the time steps, shape ``(N,)``.
    values : ndarray
        Tracer values, shape ``(N,)`` for one tracer or ``(N, K)`` for ``K``
        tracers (the same convention as
        :class:`~PyTracerLab.model.model.Model`); ``nan`` where no value is
        available.
    tracers : list of str
        Tracer names from the header line (or ``tracer_1``, ``tracer_2``, ...
        if the file has no header).
    uncertainty : ndarray or None
        Measurement uncertainties (standard deviations) from ``<tracer>_sd``
        columns, same shape as ``values`` and ``nan`` where not given; ``None``
        if the file has no uncertainty columns.
    step : str or None
        Regular time step of the series, e.g. ``"1 month"`` or ``"1 year"``.
    """

    times: np.ndarray
    values: np.ndarray
    tracers: List[str]
    uncertainty: Optional[np.ndarray] = None
    step: Optional[str] = None

    @property
    def n_tracers(self) -> int:
        """Number of tracers (columns of ``values``)."""
        return 1 if np.ndim(self.values) == 1 else int(np.shape(self.values)[1])

    def __len__(self) -> int:
        return int(np.shape(self.values)[0])


# --------------------------------------------------------------------------
# low-level helpers
# --------------------------------------------------------------------------
def _read_lines(path: PathLike) -> List[str]:
    """Read all lines of a text file; tolerate BOMs and legacy encodings."""
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            with open(path, "r", encoding=encoding, newline="") as f:
                return f.read().splitlines()
        except UnicodeDecodeError:
            continue
    raise DataFileError(
        f"{_name(path)}: the file could not be decoded as text."
    )  # pragma: no cover


def _name(path: PathLike) -> str:
    return os.path.basename(os.fspath(path))


def _clean(token: str) -> str:
    return token.strip().strip('"').strip("'").strip()


def _split(line: str, delimiter: Optional[str]) -> List[str]:
    parts = line.split() if delimiter is None else line.split(delimiter)
    return [_clean(p) for p in parts]


def _parse_date(token: str) -> Optional[datetime]:
    """Parse a date token; return ``None`` if it is not a supported date."""
    tok = _clean(token)
    if not tok:
        return None
    # ignore a time of day after the date (e.g. spreadsheet exports)
    tok = re.split(r"[ T]", tok, maxsplit=1)[0]
    try:
        m = _RE_YEAR.match(tok)
        if m:
            return datetime(int(m.group(1)), 1, 1)
        m = _RE_DECIMAL_YEAR.match(tok)
        if m:
            year = int(m.group(1))
            frac = float("0." + m.group(2))
            start = datetime(year, 1, 1)
            n_days = (datetime(year + 1, 1, 1) - start).days
            return start + timedelta(days=frac * n_days)
        m = _RE_YM.match(tok)
        if m:
            return datetime(int(m.group(1)), int(m.group(2)), 1)
        m = _RE_YMD.match(tok)
        if m:
            return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        m = _RE_DMY.match(tok)
        if m:
            return datetime(int(m.group(3)), int(m.group(2)), int(m.group(1)))
        m = _RE_MY.match(tok)
        if m:
            return datetime(int(m.group(2)), int(m.group(1)), 1)
    except ValueError:  # e.g. month 13
        return None
    return None


def _is_missing(token: str) -> bool:
    return token.strip().lower() in _MISSING


def _detect_data_line(line: str) -> Optional[Tuple[Optional[str]]]:
    """Return ``(delimiter,)`` if ``line`` is a data line, else ``None``.

    A data line has at least two fields and a date in the first field. The
    delimiter is wrapped in a 1-tuple to distinguish the whitespace delimiter
    (``None``) from "not a data line".
    """
    if line.lstrip().startswith("#"):
        return None
    for delim in _DELIMITERS:
        fields = _split(line, delim)
        if len(fields) >= 2 and _parse_date(fields[0]) is not None:
            return (delim,)
    return None


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------
@dataclass
class _Parsed:
    times: List[datetime]
    date_tokens: List[str]
    line_numbers: List[int]
    tracers: List[str]
    values: np.ndarray  # (N, K)
    uncertainty: Optional[np.ndarray]  # (N, K) or None


def _parse_file(path: PathLike) -> _Parsed:
    name = _name(path)
    lines = _read_lines(path)
    entries = [(i + 1, ln) for i, ln in enumerate(lines) if ln.strip()]
    if not entries:
        raise DataFileError(f"{name}: the file is empty.")

    # locate the first data line and the delimiter
    first_idx = None
    delimiter: Optional[str] = None
    for j, (_, ln) in enumerate(entries):
        found = _detect_data_line(ln)
        if found is not None:
            first_idx, delimiter = j, found[0]
            break
    if first_idx is None:
        lineno, ln = entries[min(1, len(entries) - 1)]
        raise DataFileError(
            f"{name}: no data rows found. Each data row must start with a date "
            f"({_DATE_FORMATS_TEXT}) followed by at least one value, separated by "
            f"comma, semicolon or tab (e.g. line {lineno}: '{ln.strip()}')."
        )

    # header = last non-data line before the first data line (leading '#' removed)
    header: List[str] = []
    if first_idx > 0:
        header_line = entries[first_idx - 1][1].strip().lstrip("#").strip()
        header = _split(header_line, delimiter)

    # tokenise data rows (skip comment lines)
    rows: List[Tuple[int, List[str]]] = []
    for lineno, ln in entries[first_idx:]:
        if ln.lstrip().startswith("#"):
            continue
        rows.append((lineno, _split(ln, delimiter)))

    # columns: as many as the longest row or the header, minus unnamed empty
    # trailing columns (e.g. from a trailing delimiter)
    names = [h for h in header[1:]]
    while names and not names[-1]:  # trailing delimiter in the header line
        names.pop()
    n_val = max(len(tokens) for _, tokens in rows) - 1
    while n_val > 0:
        j = n_val - 1
        unnamed = j >= len(names) or not names[j]
        empty = all(len(tokens) <= j + 1 or _is_missing(tokens[j + 1]) for _, tokens in rows)
        if unnamed and empty:
            n_val -= 1
        else:
            break
    n_cols = max(n_val, len(names))
    if n_cols == 0:
        raise DataFileError(f"{name}: the file contains dates but no value columns.")
    names = names[:n_cols] + [""] * (n_cols - len(names))
    names = [nm if nm else f"tracer_{j + 1}" for j, nm in enumerate(names)]

    # convert
    times: List[datetime] = []
    date_tokens: List[str] = []
    line_numbers: List[int] = []
    data = np.full((len(rows), n_cols), np.nan)
    for r, (lineno, tokens) in enumerate(rows):
        t = _parse_date(tokens[0])
        if t is None:
            raise DataFileError(
                f"{name}, line {lineno}: could not read the date '{tokens[0]}'. "
                f"Expected {_DATE_FORMATS_TEXT}."
            )
        times.append(t)
        date_tokens.append(tokens[0])
        line_numbers.append(lineno)
        for j, tok in enumerate(tokens[1 : n_cols + 1]):
            if _is_missing(tok):
                continue
            if delimiter != ",":
                tok = tok.replace(",", ".")
            try:
                data[r, j] = float(tok)
            except ValueError:
                raise DataFileError(
                    f"{name}, line {lineno}, column '{names[j]}': could not read the value "
                    f"'{tokens[j + 1]}' as a number. Use an empty cell or 'nan' for "
                    f"missing values."
                ) from None

    # split tracer columns and uncertainty columns ('<tracer>_sd')
    tracer_cols = [j for j, nm in enumerate(names) if not nm.lower().endswith(_SD_SUFFIX)]
    sd_cols = [j for j, nm in enumerate(names) if nm.lower().endswith(_SD_SUFFIX)]
    if not tracer_cols:
        raise DataFileError(f"{name}: the file contains only uncertainty ('_sd') columns.")
    tracers = [names[j] for j in tracer_cols]
    values = data[:, tracer_cols]
    uncertainty = None
    if sd_cols:
        uncertainty = np.full_like(values, np.nan)
        lower = [t.lower() for t in tracers]
        for j in sd_cols:
            base = names[j][: -len(_SD_SUFFIX)].lower()
            if base not in lower:
                raise DataFileError(
                    f"{name}: the uncertainty column '{names[j]}' has no matching tracer "
                    f"column '{names[j][: -len(_SD_SUFFIX)]}'."
                )
            col = data[:, j]
            bad = np.where(~np.isnan(col) & (col <= 0.0))[0]
            if bad.size:
                r = int(bad[0])
                raise DataFileError(
                    f"{name}, line {line_numbers[r]}, column '{names[j]}': uncertainties "
                    f"must be positive (found {col[r]:g})."
                )
            uncertainty[:, lower.index(base)] = col

    return _Parsed(times, date_tokens, line_numbers, tracers, values, uncertainty)


# --------------------------------------------------------------------------
# time steps
# --------------------------------------------------------------------------
def _month_index(t: datetime) -> int:
    return t.year * 12 + t.month - 1


def _step_of(
    times: Sequence[datetime],
    name: str,
    line_numbers: Optional[Sequence[int]] = None,
    date_tokens: Optional[Sequence[str]] = None,
) -> Tuple[str, int]:
    """Return the regular step of ``times`` as ``("month", n)`` or ``("day", n)``.

    Raises :class:`DataFileError` if the time steps are not regular.
    """
    if len(times) < 2:
        raise DataFileError(f"{name}: at least two time steps are required.")
    months = np.array([_month_index(t) for t in times])
    days = np.array([t.toordinal() for t in times])
    dm, dd = np.diff(months), np.diff(days)
    if dm[0] > 0 and np.all(dm == dm[0]):
        return ("month", int(dm[0]))
    if dd[0] > 0 and np.all(dd == dd[0]):
        return ("day", int(dd[0]))

    # build a helpful message: first non-increasing or irregular step
    def where(i: int) -> str:
        if line_numbers is None or date_tokens is None:
            return f"between {times[i]:%Y-%m-%d} and {times[i + 1]:%Y-%m-%d}"
        return (
            f"lines {line_numbers[i]} and {line_numbers[i + 1]} "
            f"({date_tokens[i]} -> {date_tokens[i + 1]})"
        )

    hint = (
        " PyTracerLab requires tracer input time series with regular time steps (e.g. "
        "monthly or yearly) and without gaps; please fill gaps or resample the series. "
        "If this file contains observations, read it with "
        "read_series(path, align_to=<input series>)."
    )
    if np.any(dd <= 0):
        i = int(np.where(dd <= 0)[0][0])
        raise DataFileError(f"{name}: the dates are not in increasing order at {where(i)}." + hint)
    diffs = dm if np.all(dm > 0) else dd
    vals, counts = np.unique(diffs, return_counts=True)
    regular = vals[np.argmax(counts)]
    i = int(np.where(diffs != regular)[0][0])
    raise DataFileError(f"{name}: irregular time step at {where(i)}." + hint)


def _step_label(step: Tuple[str, int]) -> str:
    kind, n = step
    if kind == "month" and n % 12 == 0:
        k = n // 12
        return f"{k} year" if k == 1 else f"{k} years"
    unit = kind
    return f"{n} {unit}" if n == 1 else f"{n} {unit}s"


def _step_text(t: datetime, step: Tuple[str, int]) -> str:
    kind, n = step
    if kind == "month":
        return f"{t:%Y}" if n % 12 == 0 else f"{t:%Y-%m}"
    return f"{t:%Y-%m-%d}"


def _squeeze(arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if arr is None:
        return None
    return arr[:, 0].copy() if arr.shape[1] == 1 else arr


def _as_2d(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    return arr.reshape(-1, 1) if arr.ndim == 1 else arr


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------
def read_series(path: PathLike, align_to: Optional[TracerSeries] = None) -> TracerSeries:
    """Read a tracer input time series or observations from a text file.

    See the module documentation (:mod:`PyTracerLab.model.io`) for the file
    format.

    Parameters
    ----------
    path : str or path-like
        File to read.
    align_to : TracerSeries, optional
        Tracer input time series to which the observations in ``path`` are
        aligned. Each observation is assigned to the time step of
        ``align_to`` that contains its date, so the file may list only the
        sampling dates. The result has the same length as ``align_to`` and
        ``nan`` where nothing was observed, as required for the
        ``target_series`` of :class:`~PyTracerLab.model.model.Model`. Without
        ``align_to`` the file is read as it is and must have regular time
        steps (as required for tracer input time series).

    Returns
    -------
    TracerSeries
        Dates, values, tracer names, optional uncertainties and time step.

    Raises
    ------
    DataFileError
        If the file cannot be read. The message names the line and the value
        that caused the problem.

    Examples
    --------
    >>> import PyTracerLab as lab
    >>> inp = lab.read_series("tritium_input.csv")
    >>> obs = lab.read_series("tritium_observations.csv", align_to=inp)
    >>> m = lab.Model(dt=1.0, lambda_=lab.decay_constant("3H"),
    ...               input_series=inp.values, target_series=obs.values)
    """
    name = _name(path)
    p = _parse_file(path)

    if align_to is None:
        step = _step_of(p.times, name, p.line_numbers, p.date_tokens)
        return TracerSeries(
            times=np.array(p.times, dtype=object),
            values=_squeeze(p.values),
            tracers=p.tracers,
            uncertainty=_squeeze(p.uncertainty),
            step=_step_label(step),
        )

    # ---- align observations to the time steps of the input series ----
    ref_times = list(align_to.times)
    ref_step = _step_of(ref_times, "input series (align_to)")
    n_ref = len(ref_times)
    k_ref = _as_2d(align_to.values).shape[1]
    n_tr = p.values.shape[1]
    if n_tr != k_ref:
        raise DataFileError(
            f"{name}: the file has {n_tr} tracer column(s) ({', '.join(p.tracers)}) but the "
            f"input series has {k_ref} ({', '.join(align_to.tracers)})."
        )

    kind, n = ref_step
    if kind == "month":
        origin = _month_index(ref_times[0])

        def index_of(t: datetime) -> int:
            return (_month_index(t) - origin) // n

    else:
        origin = ref_times[0].toordinal()

        def index_of(t: datetime) -> int:
            return (t.toordinal() - origin) // n

    values = np.full((n_ref, n_tr), np.nan)
    uncertainty = None if p.uncertainty is None else np.full((n_ref, n_tr), np.nan)
    source_line = np.zeros((n_ref, n_tr), dtype=int)
    for r, t in enumerate(p.times):
        row = p.values[r]
        if np.all(np.isnan(row)):
            continue  # nothing observed at this date
        lineno = p.line_numbers[r]
        i = index_of(t)
        if not 0 <= i < n_ref:
            raise DataFileError(
                f"{name}, line {lineno}: the date {p.date_tokens[r]} lies outside the period "
                f"of the input series ({_step_text(ref_times[0], ref_step)} to "
                f"{_step_text(ref_times[-1], ref_step)})."
            )
        for k in range(n_tr):
            if np.isnan(row[k]):
                continue
            if not np.isnan(values[i, k]):
                raise DataFileError(
                    f"{name}, lines {source_line[i, k]} and {lineno}: two observations of "
                    f"{p.tracers[k]} fall into the same time step "
                    f"({_step_text(ref_times[i], ref_step)}) of the input series. Please "
                    f"combine them into one value per time step (e.g. their mean)."
                )
            values[i, k] = row[k]
            source_line[i, k] = lineno
            if uncertainty is not None:
                uncertainty[i, k] = p.uncertainty[r, k]

    return TracerSeries(
        times=np.array(ref_times, dtype=object),
        values=_squeeze(values),
        tracers=p.tracers,
        uncertainty=_squeeze(uncertainty),
        step=align_to.step,
    )
