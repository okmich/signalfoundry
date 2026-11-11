"""Which finding kinds remediation drops, holds, or treats as a failed refetch verification; which registry rows cover what."""
from __future__ import annotations

from .enums import FindingKind, WindowKind

#: Dropped whether or not the registry lists them: the rows themselves are wrong, no judgment needed. Each such finding
#: carries the exact defective rows (``Finding.rows``), and only those rows are dropped.
MECHANICAL = frozenset({FindingKind.MISSING_STAMP, FindingKind.OFF_GRID, FindingKind.FUTURE_STAMP, FindingKind.CLOCK_OVERLAP,
                        FindingKind.OHLC_VIOLATION, FindingKind.NON_FINITE, FindingKind.NON_POSITIVE, FindingKind.OUT_OF_SESSION,
                        FindingKind.REPEATED_BLOCK})
#: Need a human decision (recorded in the registry) before anything is dropped.
JUDGMENT = frozenset({FindingKind.CLOCK_MISALIGNED, FindingKind.EVENT_STAMP_DRIFT, FindingKind.BAD_PRICES})
#: An unregistered CRITICAL/MAJOR finding of these kinds makes the audit verdict REVIEW.
REVIEW_KINDS = frozenset({FindingKind.CLOCK_MISALIGNED, FindingKind.EVENT_STAMP_DRIFT, FindingKind.BAD_PRICES, FindingKind.REPEATED_BLOCK,
                          FindingKind.CLOCK_OVERLAP, FindingKind.OHLC_VIOLATION, FindingKind.NON_FINITE, FindingKind.NON_POSITIVE,
                          FindingKind.OVERFULL, FindingKind.INTERLEAVE})
# Remediation holds (keeps, flags, status REVIEW) unregistered CRITICAL/MAJOR JUDGMENT findings whatever their action,
# and also any MAJOR REVIEW_KINDS finding still present after remediation. A symbol the audit sends to review can't come
# out VERIFIED with the finding still there. A CRITICAL that survives is FAILED_VERIFY.

#: A refetch is rejected if splicing it creates a new or worse finding of one of these kinds near the window.
VERIFY_REJECT = MECHANICAL | JUDGMENT | {FindingKind.DUPLICATE_STAMP, FindingKind.NON_MONOTONIC, FindingKind.OVERFULL,
                                         FindingKind.INTERLEAVE}
#: Findings about absent bars, and the registry rows that record absent bars. A missing-data row (outage / hole) can
#: only register missing-data findings: it must not silence a clock or price finding in the same window.
MISSING_DATA_KINDS = frozenset({FindingKind.OUTAGE, FindingKind.HOLE, FindingKind.THIN_NO_TICK})
MISSING_DATA_WINDOWS = frozenset({WindowKind.OUTAGE, WindowKind.HOLE})
