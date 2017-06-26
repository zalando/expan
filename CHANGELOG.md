# Change Log

## [v0.5.3](https://github.com/zalando/expan/tree/v0.5.3) (2017-06-26)
[Full Changelog](https://github.com/zalando/expan/compare/v0.5.2...v0.5.3)

**Implemented enhancements:**

- Weighted KPIs is only implemented in regular delta [\#114](https://github.com/zalando/expan/issues/114)

**Fixed bugs:**

- Assumption of nan when computing weighted KPIs  [\#119](https://github.com/zalando/expan/issues/119)
- Weighted KPIs is only implemented in regular delta [\#114](https://github.com/zalando/expan/issues/114)
- Percentiles value is lost during computing group\_sequential\_delta [\#108](https://github.com/zalando/expan/issues/108)

**Closed issues:**

- Failing early stopping unit tests [\#85](https://github.com/zalando/expan/issues/85)

**Merged pull requests:**

- OCTO-1804: Optimize the loading of .stan model in expan. [\#126](https://github.com/zalando/expan/pull/126) ([daryadedik](https://github.com/daryadedik))
- Test travis python version [\#125](https://github.com/zalando/expan/pull/125) ([shansfolder](https://github.com/shansfolder))
- OCTO-1619 Cleanup ExpAn code [\#124](https://github.com/zalando/expan/pull/124) ([shansfolder](https://github.com/shansfolder))
- OCTO-1748: Make number of iterations as a method argument in \_bayes\_sampling [\#123](https://github.com/zalando/expan/pull/123) ([daryadedik](https://github.com/daryadedik))
- OCTO-1615 Use Python builtin logging instead of our own debugging.py [\#122](https://github.com/zalando/expan/pull/122) ([shansfolder](https://github.com/shansfolder))
- OCTO-1711 Support weighted KPIs in early stopping [\#121](https://github.com/zalando/expan/pull/121) ([shansfolder](https://github.com/shansfolder))
- Fixed a few bugs [\#120](https://github.com/zalando/expan/pull/120) ([shansfolder](https://github.com/shansfolder))
- OCTO-1614 cleanup module structure [\#115](https://github.com/zalando/expan/pull/115) ([shansfolder](https://github.com/shansfolder))
- OCTO-1677 : fix missing .stan files [\#113](https://github.com/zalando/expan/pull/113) ([gbordyugov](https://github.com/gbordyugov))
- Bump version 0.5.1 -\> 0.5.2 [\#112](https://github.com/zalando/expan/pull/112) ([mkolarek](https://github.com/mkolarek))

## [v0.5.2](https://github.com/zalando/expan/tree/v0.5.2) (2017-05-11)
[Full Changelog](https://github.com/zalando/expan/compare/v0.5.1...v0.5.2)

**Merged pull requests:**

- OCTO-1502 support \*\*kwargs for four delta functions [\#111](https://github.com/zalando/expan/pull/111) ([shansfolder](https://github.com/shansfolder))
- new version 0.5.1 [\#107](https://github.com/zalando/expan/pull/107) ([mkolarek](https://github.com/mkolarek))

## [v0.5.1](https://github.com/zalando/expan/tree/v0.5.1) (2017-04-20)
[Full Changelog](https://github.com/zalando/expan/compare/v0.5.0...v0.5.1)

**Implemented enhancements:**

- Derived KPIs are passed to Experiment.fixed\_horizon\_delta\(\) but never used in there [\#96](https://github.com/zalando/expan/issues/96)

**Merged pull requests:**

- OCTO-1501: bugfix in Results.to\_json\(\) [\#105](https://github.com/zalando/expan/pull/105) ([gbordyugov](https://github.com/gbordyugov))
- OCTO-1502 removed variant\_subset parameter... [\#104](https://github.com/zalando/expan/pull/104) ([gbordyugov](https://github.com/gbordyugov))
- OCTO-1540 cleanup handling of derived kpis [\#102](https://github.com/zalando/expan/pull/102) ([shansfolder](https://github.com/shansfolder))
- OCTO-1540: cleanup of derived kpi handling in Experiment.delta\(\) and … [\#97](https://github.com/zalando/expan/pull/97) ([gbordyugov](https://github.com/gbordyugov))
- Merge dev to master for v0.5.0 [\#94](https://github.com/zalando/expan/pull/94) ([mkolarek](https://github.com/mkolarek))

## [v0.5.0](https://github.com/zalando/expan/tree/v0.5.0) (2017-04-05)
[Full Changelog](https://github.com/zalando/expan/compare/v0.4.5...v0.5.0)

**Implemented enhancements:**

- Bad code duplication in experiment.py [\#81](https://github.com/zalando/expan/issues/81)
- pip == 8.1.0 requirement [\#76](https://github.com/zalando/expan/issues/76)

**Fixed bugs:**

- Experiment.sga\(\) assumes features and KPIs are merged in self.metrics [\#87](https://github.com/zalando/expan/issues/87)
- pctile can be undefined in `Results.to\_json\(\)` [\#78](https://github.com/zalando/expan/issues/78)

**Closed issues:**

- Results.to\_json\(\) =\> TypeError: Object of type 'UserWarning' is not JSON serializable [\#77](https://github.com/zalando/expan/issues/77)
- Rethink Results structure [\#66](https://github.com/zalando/expan/issues/66)

**Merged pull requests:**

- new dataframe tree traverser in to\_json\(\) [\#92](https://github.com/zalando/expan/pull/92) ([gbordyugov](https://github.com/gbordyugov))
- updated requirements.txt to have 'greater than' dependencies instead … [\#89](https://github.com/zalando/expan/pull/89) ([mkolarek](https://github.com/mkolarek))
- pip version requirement [\#88](https://github.com/zalando/expan/pull/88) ([gbordyugov](https://github.com/gbordyugov))
- Test [\#86](https://github.com/zalando/expan/pull/86) ([s4826](https://github.com/s4826))
- merging in categorical binning [\#84](https://github.com/zalando/expan/pull/84) ([gbordyugov](https://github.com/gbordyugov))

## [v0.4.5](https://github.com/zalando/expan/tree/v0.4.5) (2017-02-10)
[Full Changelog](https://github.com/zalando/expan/compare/v0.4.4...v0.4.5)

**Fixed bugs:**

- Numbers cannot appear in variable names for derived metrics [\#58](https://github.com/zalando/expan/issues/58)

## [v0.4.4](https://github.com/zalando/expan/tree/v0.4.4) (2017-02-09)
[Full Changelog](https://github.com/zalando/expan/compare/v0.4.3...v0.4.4)

**Implemented enhancements:**

- Add argument assume\_normal and treatment\_cost to calculate\_prob\_uplift\_over\_zero\(\) and prob\_uplift\_over\_zero\_single\_metric\(\) [\#26](https://github.com/zalando/expan/issues/26)
- host intro slides \(from the ipython notebook\) somewhere for public viewing [\#10](https://github.com/zalando/expan/issues/10)

**Closed issues:**

- migrate issues from github enterprise [\#20](https://github.com/zalando/expan/issues/20)

## [v0.4.3](https://github.com/zalando/expan/tree/v0.4.3) (2017-02-07)
[Full Changelog](https://github.com/zalando/expan/compare/v0.4.2...v0.4.3)

**Closed issues:**

- coverage % is misleading [\#23](https://github.com/zalando/expan/issues/23)

## [v0.4.2](https://github.com/zalando/expan/tree/v0.4.2) (2016-12-08)
[Full Changelog](https://github.com/zalando/expan/compare/v0.4.1...v0.4.2)

**Fixed bugs:**

- frequency table in the chi square test doesn't respect the order of categories [\#56](https://github.com/zalando/expan/issues/56)

## [v0.4.1](https://github.com/zalando/expan/tree/v0.4.1) (2016-10-18)
[Full Changelog](https://github.com/zalando/expan/compare/v0.4.0...v0.4.1)

## [v0.4.0](https://github.com/zalando/expan/tree/v0.4.0) (2016-08-19)
[Full Changelog](https://github.com/zalando/expan/compare/v0.3.4...v0.4.0)

**Closed issues:**

- Support 'overall ratio' metrics \(e.g. conversion rate/return rate\) as opposed to per-entity ratios [\#44](https://github.com/zalando/expan/issues/44)

## [v0.3.4](https://github.com/zalando/expan/tree/v0.3.4) (2016-08-08)
[Full Changelog](https://github.com/zalando/expan/compare/v0.3.3...v0.3.4)

**Closed issues:**

- perform trend analysis cumulatively [\#31](https://github.com/zalando/expan/issues/31)
- Python3 [\#21](https://github.com/zalando/expan/issues/21)

## [v0.3.3](https://github.com/zalando/expan/tree/v0.3.3) (2016-08-02)
[Full Changelog](https://github.com/zalando/expan/compare/v0.3.2...v0.3.3)

## [v0.3.2](https://github.com/zalando/expan/tree/v0.3.2) (2016-08-02)
[Full Changelog](https://github.com/zalando/expan/compare/v0.3.1...v0.3.2)

## [v0.3.1](https://github.com/zalando/expan/tree/v0.3.1) (2016-07-15)
[Full Changelog](https://github.com/zalando/expan/compare/v0.3.0...v0.3.1)

## [v0.3.0](https://github.com/zalando/expan/tree/v0.3.0) (2016-06-23)
[Full Changelog](https://github.com/zalando/expan/compare/v0.2.5...v0.3.0)

**Implemented enhancements:**

- Add P\(uplift\>0\) as a statistic [\#2](https://github.com/zalando/expan/issues/2)

## [v0.2.5](https://github.com/zalando/expan/tree/v0.2.5) (2016-05-30)
[Full Changelog](https://github.com/zalando/expan/compare/v0.2.4...v0.2.5)

**Implemented enhancements:**

- Implement \_\_version\_\_ [\#14](https://github.com/zalando/expan/issues/14)

**Closed issues:**

- upload full documentation! [\#1](https://github.com/zalando/expan/issues/1)

## [v0.2.4](https://github.com/zalando/expan/tree/v0.2.4) (2016-05-16)
[Full Changelog](https://github.com/zalando/expan/compare/v0.2.3...v0.2.4)

**Closed issues:**

- No module named experiment and test\_data [\#13](https://github.com/zalando/expan/issues/13)

## [v0.2.3](https://github.com/zalando/expan/tree/v0.2.3) (2016-05-06)
[Full Changelog](https://github.com/zalando/expan/compare/v0.2.2...v0.2.3)

## [v0.2.2](https://github.com/zalando/expan/tree/v0.2.2) (2016-05-06)
[Full Changelog](https://github.com/zalando/expan/compare/v0.2.1...v0.2.2)

## [v0.2.1](https://github.com/zalando/expan/tree/v0.2.1) (2016-05-06)
[Full Changelog](https://github.com/zalando/expan/compare/v0.2.0...v0.2.1)

## [v0.2.0](https://github.com/zalando/expan/tree/v0.2.0) (2016-05-06)


\* *This Change Log was automatically generated by [github_changelog_generator](https://github.com/skywinder/Github-Changelog-Generator)*