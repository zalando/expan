# Change Log

## [Unreleased](https://github.com/zalando/expan/tree/HEAD)

[Full Changelog](https://github.com/zalando/expan/compare/v0.5.0...HEAD)

**Implemented enhancements:**

- Derived KPIs are passed to Experiment.fixed\_horizon\_delta\(\) but never used in there [\#96](https://github.com/zalando/expan/issues/96)

**Merged pull requests:**

- updated CONTRIBUTING.rst with deployment flow [\#106](https://github.com/zalando/expan/pull/106) ([mkolarek](https://github.com/mkolarek))
- OCTO-1501: bugfix in Results.to\_json\(\) [\#105](https://github.com/zalando/expan/pull/105) ([gbordyugov](https://github.com/gbordyugov))
- OCTO-1502 removed variant\_subset parameter... [\#104](https://github.com/zalando/expan/pull/104) ([gbordyugov](https://github.com/gbordyugov))
- OCTO-1540 cleanup handling of derived kpis [\#102](https://github.com/zalando/expan/pull/102) ([shansfolder](https://github.com/shansfolder))
- OCTO-1540: cleanup of derived kpi handling in Experiment.delta\(\) and … [\#97](https://github.com/zalando/expan/pull/97) ([gbordyugov](https://github.com/gbordyugov))
- Small refactoring [\#95](https://github.com/zalando/expan/pull/95) ([shansfolder](https://github.com/shansfolder))
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

- updated requirements.txt to have 'greater than' dependencies instead … [\#89](https://github.com/zalando/expan/pull/89) ([mkolarek](https://github.com/mkolarek))
- pip version requirement [\#88](https://github.com/zalando/expan/pull/88) ([gbordyugov](https://github.com/gbordyugov))
- merging in categorical binning [\#84](https://github.com/zalando/expan/pull/84) ([gbordyugov](https://github.com/gbordyugov))
- Add documentation of the weighting logic [\#83](https://github.com/zalando/expan/pull/83) ([jbao](https://github.com/jbao))
- Merge to\_json\(\) changes [\#75](https://github.com/zalando/expan/pull/75) ([mkolarek](https://github.com/mkolarek))
- Feature/early stopping [\#73](https://github.com/zalando/expan/pull/73) ([jbao](https://github.com/jbao))

## [v0.4.5](https://github.com/zalando/expan/tree/v0.4.5) (2017-02-10)
[Full Changelog](https://github.com/zalando/expan/compare/v0.4.4...v0.4.5)

**Fixed bugs:**

- Numbers cannot appear in variable names for derived metrics [\#58](https://github.com/zalando/expan/issues/58)

**Merged pull requests:**

- Feature/results and to json refactor [\#74](https://github.com/zalando/expan/pull/74) ([mkolarek](https://github.com/mkolarek))
- regex fix, see https://github.com/zalando/expan/issues/58 [\#70](https://github.com/zalando/expan/pull/70) ([gbordyugov](https://github.com/gbordyugov))

## [v0.4.4](https://github.com/zalando/expan/tree/v0.4.4) (2017-02-09)
[Full Changelog](https://github.com/zalando/expan/compare/v0.4.3...v0.4.4)

**Implemented enhancements:**

- Add argument assume\_normal and treatment\_cost to calculate\_prob\_uplift\_over\_zero\(\) and prob\_uplift\_over\_zero\_single\_metric\(\) [\#26](https://github.com/zalando/expan/issues/26)
- host intro slides \(from the ipython notebook\) somewhere for public viewing [\#10](https://github.com/zalando/expan/issues/10)

**Closed issues:**

- migrate issues from github enterprise [\#20](https://github.com/zalando/expan/issues/20)

**Merged pull requests:**

- Feature/results and to json refactor [\#71](https://github.com/zalando/expan/pull/71) ([mkolarek](https://github.com/mkolarek))
- new to\_json\(\) functionality and improved vim support [\#67](https://github.com/zalando/expan/pull/67) ([mkolarek](https://github.com/mkolarek))

## [v0.4.3](https://github.com/zalando/expan/tree/v0.4.3) (2017-02-07)
[Full Changelog](https://github.com/zalando/expan/compare/v0.4.2...v0.4.3)

**Closed issues:**

- coverage % is misleading [\#23](https://github.com/zalando/expan/issues/23)

**Merged pull requests:**

- Vim modelines [\#63](https://github.com/zalando/expan/pull/63) ([gbordyugov](https://github.com/gbordyugov))
- 0.4.2 release [\#60](https://github.com/zalando/expan/pull/60) ([mkolarek](https://github.com/mkolarek))

## [v0.4.2](https://github.com/zalando/expan/tree/v0.4.2) (2016-12-08)
[Full Changelog](https://github.com/zalando/expan/compare/v0.4.1...v0.4.2)

**Fixed bugs:**

- frequency table in the chi square test doesn't respect the order of categories [\#56](https://github.com/zalando/expan/issues/56)

**Merged pull requests:**

- OCTO-1143 Review outlier filtering [\#59](https://github.com/zalando/expan/pull/59) ([domheger](https://github.com/domheger))
- Workaround to fix \#56 [\#57](https://github.com/zalando/expan/pull/57) ([jbao](https://github.com/jbao))

## [v0.4.1](https://github.com/zalando/expan/tree/v0.4.1) (2016-10-18)
[Full Changelog](https://github.com/zalando/expan/compare/v0.4.0...v0.4.1)

**Merged pull requests:**

- small doc cleanup [\#55](https://github.com/zalando/expan/pull/55) ([jbao](https://github.com/jbao))
- Add comments to cli.py [\#54](https://github.com/zalando/expan/pull/54) ([igusher](https://github.com/igusher))
- Feature/octo 545 add consolidate documentation [\#53](https://github.com/zalando/expan/pull/53) ([mkolarek](https://github.com/mkolarek))

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