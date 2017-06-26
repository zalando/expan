Change Log
==========

`v0.5.3 <https://github.com/zalando/expan/tree/v0.5.3>`__ (2017-06-26)
----------------------------------------------------------------------

`Full
Changelog <https://github.com/zalando/expan/compare/v0.5.2...v0.5.3>`__

**Implemented enhancements:**

-  Weighted KPIs is only implemented in regular delta
   `#114 <https://github.com/zalando/expan/issues/114>`__

**Fixed bugs:**

-  Assumption of nan when computing weighted KPIs
   `#119 <https://github.com/zalando/expan/issues/119>`__
-  Weighted KPIs is only implemented in regular delta
   `#114 <https://github.com/zalando/expan/issues/114>`__
-  Percentiles value is lost during computing group\_sequential\_delta
   `#108 <https://github.com/zalando/expan/issues/108>`__

**Closed issues:**

-  Failing early stopping unit tests
   `#85 <https://github.com/zalando/expan/issues/85>`__

**Merged pull requests:**

-  OCTO-1804: Optimize the loading of .stan model in expan.
   `#126 <https://github.com/zalando/expan/pull/126>`__
   (`daryadedik <https://github.com/daryadedik>`__)
-  Test travis python version
   `#125 <https://github.com/zalando/expan/pull/125>`__
   (`shansfolder <https://github.com/shansfolder>`__)
-  OCTO-1619 Cleanup ExpAn code
   `#124 <https://github.com/zalando/expan/pull/124>`__
   (`shansfolder <https://github.com/shansfolder>`__)
-  OCTO-1748: Make number of iterations as a method argument in
   \_bayes\_sampling
   `#123 <https://github.com/zalando/expan/pull/123>`__
   (`daryadedik <https://github.com/daryadedik>`__)
-  OCTO-1615 Use Python builtin logging instead of our own debugging.py
   `#122 <https://github.com/zalando/expan/pull/122>`__
   (`shansfolder <https://github.com/shansfolder>`__)
-  OCTO-1711 Support weighted KPIs in early stopping
   `#121 <https://github.com/zalando/expan/pull/121>`__
   (`shansfolder <https://github.com/shansfolder>`__)
-  Fixed a few bugs `#120 <https://github.com/zalando/expan/pull/120>`__
   (`shansfolder <https://github.com/shansfolder>`__)
-  OCTO-1614 cleanup module structure
   `#115 <https://github.com/zalando/expan/pull/115>`__
   (`shansfolder <https://github.com/shansfolder>`__)
-  OCTO-1677 : fix missing .stan files
   `#113 <https://github.com/zalando/expan/pull/113>`__
   (`gbordyugov <https://github.com/gbordyugov>`__)
-  Bump version 0.5.1 -> 0.5.2
   `#112 <https://github.com/zalando/expan/pull/112>`__
   (`mkolarek <https://github.com/mkolarek>`__)

`v0.5.2 <https://github.com/zalando/expan/tree/v0.5.2>`__ (2017-05-11)
----------------------------------------------------------------------

`Full
Changelog <https://github.com/zalando/expan/compare/v0.5.1...v0.5.2>`__

**Merged pull requests:**

-  OCTO-1502 support \*\*kwargs for four delta functions
   `#111 <https://github.com/zalando/expan/pull/111>`__
   (`shansfolder <https://github.com/shansfolder>`__)
-  new version 0.5.1
   `#107 <https://github.com/zalando/expan/pull/107>`__
   (`mkolarek <https://github.com/mkolarek>`__)

`v0.5.1 <https://github.com/zalando/expan/tree/v0.5.1>`__ (2017-04-20)
----------------------------------------------------------------------

`Full
Changelog <https://github.com/zalando/expan/compare/v0.5.0...v0.5.1>`__

**Implemented enhancements:**

-  Derived KPIs are passed to Experiment.fixed\_horizon\_delta() but
   never used in there
   `#96 <https://github.com/zalando/expan/issues/96>`__

**Merged pull requests:**

-  OCTO-1501: bugfix in Results.to\_json()
   `#105 <https://github.com/zalando/expan/pull/105>`__
   (`gbordyugov <https://github.com/gbordyugov>`__)
-  OCTO-1502 removed variant\_subset parameter...
   `#104 <https://github.com/zalando/expan/pull/104>`__
   (`gbordyugov <https://github.com/gbordyugov>`__)
-  OCTO-1540 cleanup handling of derived kpis
   `#102 <https://github.com/zalando/expan/pull/102>`__
   (`shansfolder <https://github.com/shansfolder>`__)
-  OCTO-1540: cleanup of derived kpi handling in Experiment.delta() and
   … `#97 <https://github.com/zalando/expan/pull/97>`__
   (`gbordyugov <https://github.com/gbordyugov>`__)
-  Merge dev to master for v0.5.0
   `#94 <https://github.com/zalando/expan/pull/94>`__
   (`mkolarek <https://github.com/mkolarek>`__)

`v0.5.0 <https://github.com/zalando/expan/tree/v0.5.0>`__ (2017-04-05)
----------------------------------------------------------------------

`Full
Changelog <https://github.com/zalando/expan/compare/v0.4.5...v0.5.0>`__

**Implemented enhancements:**

-  Bad code duplication in experiment.py
   `#81 <https://github.com/zalando/expan/issues/81>`__
-  pip == 8.1.0 requirement
   `#76 <https://github.com/zalando/expan/issues/76>`__

**Fixed bugs:**

-  Experiment.sga() assumes features and KPIs are merged in self.metrics
   `#87 <https://github.com/zalando/expan/issues/87>`__
-  pctile can be undefined in ``Results.to\_json\(\)``
   `#78 <https://github.com/zalando/expan/issues/78>`__

**Closed issues:**

-  Results.to\_json() => TypeError: Object of type 'UserWarning' is not
   JSON serializable
   `#77 <https://github.com/zalando/expan/issues/77>`__
-  Rethink Results structure
   `#66 <https://github.com/zalando/expan/issues/66>`__

**Merged pull requests:**

-  new dataframe tree traverser in to\_json()
   `#92 <https://github.com/zalando/expan/pull/92>`__
   (`gbordyugov <https://github.com/gbordyugov>`__)
-  updated requirements.txt to have 'greater than' dependencies instead
   … `#89 <https://github.com/zalando/expan/pull/89>`__
   (`mkolarek <https://github.com/mkolarek>`__)
-  pip version requirement
   `#88 <https://github.com/zalando/expan/pull/88>`__
   (`gbordyugov <https://github.com/gbordyugov>`__)
-  Test `#86 <https://github.com/zalando/expan/pull/86>`__
   (`s4826 <https://github.com/s4826>`__)
-  merging in categorical binning
   `#84 <https://github.com/zalando/expan/pull/84>`__
   (`gbordyugov <https://github.com/gbordyugov>`__)

`v0.4.5 <https://github.com/zalando/expan/tree/v0.4.5>`__ (2017-02-10)
----------------------------------------------------------------------

`Full
Changelog <https://github.com/zalando/expan/compare/v0.4.4...v0.4.5>`__

**Fixed bugs:**

-  Numbers cannot appear in variable names for derived metrics
   `#58 <https://github.com/zalando/expan/issues/58>`__

`v0.4.4 <https://github.com/zalando/expan/tree/v0.4.4>`__ (2017-02-09)
----------------------------------------------------------------------

`Full
Changelog <https://github.com/zalando/expan/compare/v0.4.3...v0.4.4>`__

**Implemented enhancements:**

-  Add argument assume\_normal and treatment\_cost to
   calculate\_prob\_uplift\_over\_zero() and
   prob\_uplift\_over\_zero\_single\_metric()
   `#26 <https://github.com/zalando/expan/issues/26>`__
-  host intro slides (from the ipython notebook) somewhere for public
   viewing `#10 <https://github.com/zalando/expan/issues/10>`__

**Closed issues:**

-  migrate issues from github enterprise
   `#20 <https://github.com/zalando/expan/issues/20>`__

`v0.4.3 <https://github.com/zalando/expan/tree/v0.4.3>`__ (2017-02-07)
----------------------------------------------------------------------

`Full
Changelog <https://github.com/zalando/expan/compare/v0.4.2...v0.4.3>`__

**Closed issues:**

-  coverage % is misleading
   `#23 <https://github.com/zalando/expan/issues/23>`__

`v0.4.2 <https://github.com/zalando/expan/tree/v0.4.2>`__ (2016-12-08)
----------------------------------------------------------------------

`Full
Changelog <https://github.com/zalando/expan/compare/v0.4.1...v0.4.2>`__

**Fixed bugs:**

-  frequency table in the chi square test doesn't respect the order of
   categories `#56 <https://github.com/zalando/expan/issues/56>`__

`v0.4.1 <https://github.com/zalando/expan/tree/v0.4.1>`__ (2016-10-18)
----------------------------------------------------------------------

`Full
Changelog <https://github.com/zalando/expan/compare/v0.4.0...v0.4.1>`__

`v0.4.0 <https://github.com/zalando/expan/tree/v0.4.0>`__ (2016-08-19)
----------------------------------------------------------------------

`Full
Changelog <https://github.com/zalando/expan/compare/v0.3.4...v0.4.0>`__

**Closed issues:**

-  Support 'overall ratio' metrics (e.g. conversion rate/return rate) as
   opposed to per-entity ratios
   `#44 <https://github.com/zalando/expan/issues/44>`__

`v0.3.4 <https://github.com/zalando/expan/tree/v0.3.4>`__ (2016-08-08)
----------------------------------------------------------------------

`Full
Changelog <https://github.com/zalando/expan/compare/v0.3.3...v0.3.4>`__

**Closed issues:**

-  perform trend analysis cumulatively
   `#31 <https://github.com/zalando/expan/issues/31>`__
-  Python3 `#21 <https://github.com/zalando/expan/issues/21>`__

`v0.3.3 <https://github.com/zalando/expan/tree/v0.3.3>`__ (2016-08-02)
----------------------------------------------------------------------

`Full
Changelog <https://github.com/zalando/expan/compare/v0.3.2...v0.3.3>`__

`v0.3.2 <https://github.com/zalando/expan/tree/v0.3.2>`__ (2016-08-02)
----------------------------------------------------------------------

`Full
Changelog <https://github.com/zalando/expan/compare/v0.3.1...v0.3.2>`__

`v0.3.1 <https://github.com/zalando/expan/tree/v0.3.1>`__ (2016-07-15)
----------------------------------------------------------------------

`Full
Changelog <https://github.com/zalando/expan/compare/v0.3.0...v0.3.1>`__

`v0.3.0 <https://github.com/zalando/expan/tree/v0.3.0>`__ (2016-06-23)
----------------------------------------------------------------------

`Full
Changelog <https://github.com/zalando/expan/compare/v0.2.5...v0.3.0>`__

**Implemented enhancements:**

-  Add P(uplift>0) as a statistic
   `#2 <https://github.com/zalando/expan/issues/2>`__

`v0.2.5 <https://github.com/zalando/expan/tree/v0.2.5>`__ (2016-05-30)
----------------------------------------------------------------------

`Full
Changelog <https://github.com/zalando/expan/compare/v0.2.4...v0.2.5>`__

**Implemented enhancements:**

-  Implement \_\_version\_\_
   `#14 <https://github.com/zalando/expan/issues/14>`__

**Closed issues:**

-  upload full documentation!
   `#1 <https://github.com/zalando/expan/issues/1>`__

`v0.2.4 <https://github.com/zalando/expan/tree/v0.2.4>`__ (2016-05-16)
----------------------------------------------------------------------

`Full
Changelog <https://github.com/zalando/expan/compare/v0.2.3...v0.2.4>`__

**Closed issues:**

-  No module named experiment and test\_data
   `#13 <https://github.com/zalando/expan/issues/13>`__

`v0.2.3 <https://github.com/zalando/expan/tree/v0.2.3>`__ (2016-05-06)
----------------------------------------------------------------------

`Full
Changelog <https://github.com/zalando/expan/compare/v0.2.2...v0.2.3>`__

`v0.2.2 <https://github.com/zalando/expan/tree/v0.2.2>`__ (2016-05-06)
----------------------------------------------------------------------

`Full
Changelog <https://github.com/zalando/expan/compare/v0.2.1...v0.2.2>`__

`v0.2.1 <https://github.com/zalando/expan/tree/v0.2.1>`__ (2016-05-06)
----------------------------------------------------------------------

`Full
Changelog <https://github.com/zalando/expan/compare/v0.2.0...v0.2.1>`__

`v0.2.0 <https://github.com/zalando/expan/tree/v0.2.0>`__ (2016-05-06)
----------------------------------------------------------------------

\* *This Change Log was automatically generated by
`github\_changelog\_generator <https://github.com/skywinder/Github-Changelog-Generator>`__*
