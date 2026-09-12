# Changelog

## [0.26.0](https://github.com/dryvist/mlx-benchmarks/compare/v0.25.2...v0.26.0) (2026-09-12)


### Features

* **benchmarks:** record and publish the reasoning effort every run used ([#245](https://github.com/dryvist/mlx-benchmarks/issues/245)) ([507a253](https://github.com/dryvist/mlx-benchmarks/commit/507a25320d2d652fcf1837d0e48e954dbe8a1b96))

## [0.25.2](https://github.com/dryvist/mlx-benchmarks/compare/v0.25.1...v0.25.2) (2026-09-11)


### Documentation

* add benchmark platform roadmap ([#250](https://github.com/dryvist/mlx-benchmarks/issues/250)) ([93f58f3](https://github.com/dryvist/mlx-benchmarks/commit/93f58f3dc72ec9c274dbccc320b12bd6e2a0c6f9))

## [0.25.1](https://github.com/dryvist/mlx-benchmarks/compare/v0.25.0...v0.25.1) (2026-09-11)


### Documentation

* **coding-replay:** record that the current task set is saturated ([5e533f4](https://github.com/dryvist/mlx-benchmarks/commit/5e533f433fd80395ad1a8c6373e8dbc8dd3bb016))
* **coding-replay:** record that the current task set is saturated ([95a96dd](https://github.com/dryvist/mlx-benchmarks/commit/95a96dd307b326f55d2b02ca41fc036d24ead21a))
* record the external benchmark landscape as a guide, not a result ([7c164b9](https://github.com/dryvist/mlx-benchmarks/commit/7c164b901fa7f95d4d8be6ffdfd27fb5bd78f000))
* record the external benchmark landscape as a guide, not a result ([#246](https://github.com/dryvist/mlx-benchmarks/issues/246)) ([7ba4fd5](https://github.com/dryvist/mlx-benchmarks/commit/7ba4fd5c653551c5f1e6d00de3608e799ac7331f))

## [0.25.0](https://github.com/dryvist/mlx-benchmarks/compare/v0.24.5...v0.25.0) (2026-09-09)


### Features

* **coding-replay:** record whether a run had the endpoint to itself ([#242](https://github.com/dryvist/mlx-benchmarks/issues/242)) ([bd8d2df](https://github.com/dryvist/mlx-benchmarks/commit/bd8d2df93f40ae919f1e041b0a072148eddfba1b))

## [0.24.5](https://github.com/dryvist/mlx-benchmarks/compare/v0.24.4...v0.24.5) (2026-09-07)


### Bug Fixes

* **coding-replay:** publish rows under the served model, not "unknown" ([#240](https://github.com/dryvist/mlx-benchmarks/issues/240)) ([0a01f5f](https://github.com/dryvist/mlx-benchmarks/commit/0a01f5fd900f8b5d42325126edd6d1bd54760ff0))

## [0.24.4](https://github.com/dryvist/mlx-benchmarks/compare/v0.24.3...v0.24.4) (2026-09-07)


### Bug Fixes

* **coding-replay:** strip any provider prefix, not only the literal mlx/ ([#238](https://github.com/dryvist/mlx-benchmarks/issues/238)) ([2a014d3](https://github.com/dryvist/mlx-benchmarks/commit/2a014d3b163e6ee40e50345d7077cc6032572f15))

## [0.24.3](https://github.com/dryvist/mlx-benchmarks/compare/v0.24.2...v0.24.3) (2026-09-07)


### Bug Fixes

* **coding-replay:** re-wait on a 429, and attribute it correctly ([#235](https://github.com/dryvist/mlx-benchmarks/issues/235)) ([7c3355f](https://github.com/dryvist/mlx-benchmarks/commit/7c3355fa322efb2635e936574ecf34ed0ff5b90e))

## [0.24.2](https://github.com/dryvist/mlx-benchmarks/compare/v0.24.1...v0.24.2) (2026-09-07)


### Bug Fixes

* **coding-replay:** confine the agent with PWD, not just cwd ([#233](https://github.com/dryvist/mlx-benchmarks/issues/233)) ([2da8776](https://github.com/dryvist/mlx-benchmarks/commit/2da87764711859e6076fe40c2645a15a34c448f5))

## [0.24.1](https://github.com/dryvist/mlx-benchmarks/compare/v0.24.0...v0.24.1) (2026-09-07)


### Bug Fixes

* **coding-replay:** bound the slot wait by wall clock, and abort when none opens ([#231](https://github.com/dryvist/mlx-benchmarks/issues/231)) ([068b5e9](https://github.com/dryvist/mlx-benchmarks/commit/068b5e959eaf480b47f00807f287cc8cd029e987))

## [0.24.0](https://github.com/dryvist/mlx-benchmarks/compare/v0.23.2...v0.24.0) (2026-09-06)


### Features

* **coding-replay:** keep the agent transcript so a zero is explainable ([#228](https://github.com/dryvist/mlx-benchmarks/issues/228)) ([e0352b2](https://github.com/dryvist/mlx-benchmarks/commit/e0352b226825dd626da7dd8b764dfbe5a96354ef))


### Bug Fixes

* **coding-replay:** pin an editing agent, or the benchmark scores the operator ([#229](https://github.com/dryvist/mlx-benchmarks/issues/229)) ([3c98e36](https://github.com/dryvist/mlx-benchmarks/commit/3c98e36d9b2e52e4cb1e6e859f30663fd3f64db9))

## [0.23.2](https://github.com/dryvist/mlx-benchmarks/compare/v0.23.1...v0.23.2) (2026-09-06)


### Bug Fixes

* **coding-replay:** abort when the agentic CLI never ran ([#226](https://github.com/dryvist/mlx-benchmarks/issues/226)) ([70539a3](https://github.com/dryvist/mlx-benchmarks/commit/70539a3106c85402bab5413d79eaed7a04ab049b))

## [0.23.1](https://github.com/dryvist/mlx-benchmarks/compare/v0.23.0...v0.23.1) (2026-09-06)


### Documentation

* **coding-replay:** --tasks-json takes a JSON array, not JSON Lines ([#224](https://github.com/dryvist/mlx-benchmarks/issues/224)) ([7e08a6d](https://github.com/dryvist/mlx-benchmarks/commit/7e08a6d6375a47b6fe72f25e05651c69b2373efd))

## [0.23.0](https://github.com/dryvist/mlx-benchmarks/compare/v0.22.2...v0.23.0) (2026-09-06)


### Features

* **coding-replay:** add a real-PR-replay benchmark suite ([#222](https://github.com/dryvist/mlx-benchmarks/issues/222)) ([b05aa3f](https://github.com/dryvist/mlx-benchmarks/commit/b05aa3f3fb553797901c06ef7b89b5e4f94dd334))

## [0.22.2](https://github.com/dryvist/mlx-benchmarks/compare/v0.22.1...v0.22.2) (2026-09-02)


### Documentation

* **journal:** Lane A latency rows for the Studio tier, 2026-09-01 ([#219](https://github.com/dryvist/mlx-benchmarks/issues/219)) ([5aa5195](https://github.com/dryvist/mlx-benchmarks/commit/5aa5195dbcf4576182e32aa75cb47aaee067994e))

## [0.22.1](https://github.com/dryvist/mlx-benchmarks/compare/v0.22.0...v0.22.1) (2026-09-01)


### Documentation

* **configs:** read the endpoint's concurrency limit instead of pinning it ([#217](https://github.com/dryvist/mlx-benchmarks/issues/217)) ([005e83b](https://github.com/dryvist/mlx-benchmarks/commit/005e83b6c3a721dc4fc827337898ac8d052f3009))

## [0.22.0](https://github.com/dryvist/mlx-benchmarks/compare/v0.21.3...v0.22.0) (2026-08-31)


### Features

* index clean-reset 128k evidence for the viewer ([f8d7e04](https://github.com/dryvist/mlx-benchmarks/commit/f8d7e047c28dea4a8866a7fd3d8e9d1d5ecf42d9))

## [0.21.3](https://github.com/dryvist/mlx-benchmarks/compare/v0.21.2...v0.21.3) (2026-08-31)


### Documentation

* record clean-reset Qwen3.8 128k evidence ([96cb8e7](https://github.com/dryvist/mlx-benchmarks/commit/96cb8e746ee1663f227174f59c050e6af04916f0))

## [0.21.2](https://github.com/dryvist/mlx-benchmarks/compare/v0.21.1...v0.21.2) (2026-08-31)


### Bug Fixes

* make throughput stream timeout configurable ([b27c598](https://github.com/dryvist/mlx-benchmarks/commit/b27c5987e168f62912b1aabb522bdb6e7f08ffde))

## [0.21.1](https://github.com/dryvist/mlx-benchmarks/compare/v0.21.0...v0.21.1) (2026-08-27)


### Bug Fixes

* avoid cached long-context probes ([#203](https://github.com/dryvist/mlx-benchmarks/issues/203)) ([093c3aa](https://github.com/dryvist/mlx-benchmarks/commit/093c3aa61882efd0a762d7720e5bf4d9d1f43b53))

## [0.21.0](https://github.com/dryvist/mlx-benchmarks/compare/v0.20.0...v0.21.0) (2026-08-27)


### Features

* publish throughput probe evidence ([#204](https://github.com/dryvist/mlx-benchmarks/issues/204)) ([eb393ba](https://github.com/dryvist/mlx-benchmarks/commit/eb393ba00c4aa5e26c34440ed5208b642af72456))

## [0.20.0](https://github.com/dryvist/mlx-benchmarks/compare/v0.19.1...v0.20.0) (2026-08-26)


### Features

* probe synthetic long contexts ([#201](https://github.com/dryvist/mlx-benchmarks/issues/201)) ([4b6d832](https://github.com/dryvist/mlx-benchmarks/commit/4b6d832ec3e05d7ea8e349d366e4b731fd710a12))

## [0.19.1](https://github.com/dryvist/mlx-benchmarks/compare/v0.19.0...v0.19.1) (2026-08-26)


### Bug Fixes

* align Space Gradio SDK version ([cfa53a3](https://github.com/dryvist/mlx-benchmarks/commit/cfa53a3a9770b72308d935c8e21fa1a59ee590a2))
* align Space Gradio SDK version ([74ab334](https://github.com/dryvist/mlx-benchmarks/commit/74ab334fad5e1744db85b2ccd56978bc9f6e34d7))

## [0.19.0](https://github.com/dryvist/mlx-benchmarks/compare/v0.18.8...v0.19.0) (2026-08-26)


### Features

* separate experimental benchmark evidence ([53636d7](https://github.com/dryvist/mlx-benchmarks/commit/53636d782a42479ae004f355fe0d7828bde88357))
* separate experimental benchmark evidence ([6f2440c](https://github.com/dryvist/mlx-benchmarks/commit/6f2440c74f263f0d5afbf3f41118b4ba78ee4a18))


### Documentation

* wrap MTP rollout prompt ([9ac1696](https://github.com/dryvist/mlx-benchmarks/commit/9ac1696d78c330dc60c9112a61e2f623499d7654))

## [0.18.8](https://github.com/dryvist/mlx-benchmarks/compare/v0.18.7...v0.18.8) (2026-08-24)


### Bug Fixes

* **lm-eval:** add arc_challenge_chat overlay without gen_prefix ([73c7054](https://github.com/dryvist/mlx-benchmarks/commit/73c705433804562a26cd02fdcc604e1ad0edbc1b))
* **lm-eval:** arc chat overlay + record 2026-08-23 quick smokes ([5c79825](https://github.com/dryvist/mlx-benchmarks/commit/5c7982520486774359d008e1d5018f29a944bf67))


### Documentation

* **rankings:** fit under 12KB file-size gate ([e3fff21](https://github.com/dryvist/mlx-benchmarks/commit/e3fff21ac433ce56420b4cd2d9cb0f176d282c15))
* **rankings:** record 2026-08-23 jevans-mbp quick smokes ([b74c9f5](https://github.com/dryvist/mlx-benchmarks/commit/b74c9f5841b571d49fd88baae4866210cd3b0f0a))
* **rankings:** slim footnote to fit file-size limit ([750f822](https://github.com/dryvist/mlx-benchmarks/commit/750f8225104c7cbd28106affe420513d33e23877))
* **rankings:** tighten under-load note further below 12KB ([d801e89](https://github.com/dryvist/mlx-benchmarks/commit/d801e89ed70a99d559e08f52659c310b9f144c66))
* **rankings:** trim redundant prose under the 12KB byte gate ([a3c1ef0](https://github.com/dryvist/mlx-benchmarks/commit/a3c1ef038a6d5333c3636fb0786d550d49aae257))

## [0.18.7](https://github.com/dryvist/mlx-benchmarks/compare/v0.18.6...v0.18.7) (2026-08-17)


### Documentation

* add trap on multi-source-of-truth disagreement, fix stale trap ranges ([#188](https://github.com/dryvist/mlx-benchmarks/issues/188)) ([61e6a58](https://github.com/dryvist/mlx-benchmarks/commit/61e6a58b63881223c86f9210aa7a0788618e02ae))

## [0.18.6](https://github.com/dryvist/mlx-benchmarks/compare/v0.18.5...v0.18.6) (2026-08-16)


### Documentation

* add trap on default() swallowing an error condition ([#186](https://github.com/dryvist/mlx-benchmarks/issues/186)) ([e0adc2f](https://github.com/dryvist/mlx-benchmarks/commit/e0adc2fe2425d12ace7b35fd71c1ee8a891c338f))

## [0.18.5](https://github.com/dryvist/mlx-benchmarks/compare/v0.18.4...v0.18.5) (2026-08-16)


### Documentation

* add two traps on file splits and silent data absence ([#184](https://github.com/dryvist/mlx-benchmarks/issues/184)) ([14128fa](https://github.com/dryvist/mlx-benchmarks/commit/14128fa343009af7866e41d8d9ae237fcad78a96))

## [0.18.4](https://github.com/dryvist/mlx-benchmarks/compare/v0.18.3...v0.18.4) (2026-08-16)


### Bug Fixes

* **docs:** split the traps checklist by category instead of raising the file-size limit ([#182](https://github.com/dryvist/mlx-benchmarks/issues/182)) ([f29668a](https://github.com/dryvist/mlx-benchmarks/commit/f29668a65e06108d95aa172d09d651d1e6aa90b3))

## [0.18.3](https://github.com/dryvist/mlx-benchmarks/compare/v0.18.2...v0.18.3) (2026-08-16)


### Documentation

* add four more traps on confident-wrong-answer checks ([#180](https://github.com/dryvist/mlx-benchmarks/issues/180)) ([74b7068](https://github.com/dryvist/mlx-benchmarks/commit/74b7068b2e0aaca93610a054ba90b4a7a1e0884b))

## [0.18.2](https://github.com/dryvist/mlx-benchmarks/compare/v0.18.1...v0.18.2) (2026-08-16)


### Documentation

* add four traps to the benchmark-traps checklist ([#178](https://github.com/dryvist/mlx-benchmarks/issues/178)) ([e75c9ab](https://github.com/dryvist/mlx-benchmarks/commit/e75c9ab0f56048de004094fc7712430d09c99557))

## [0.18.1](https://github.com/dryvist/mlx-benchmarks/compare/v0.18.0...v0.18.1) (2026-08-16)


### Documentation

* **bench-traps:** fix trap 9, add traps 13-16 from Qwen3.8-27B sweep ([#176](https://github.com/dryvist/mlx-benchmarks/issues/176)) ([9307d36](https://github.com/dryvist/mlx-benchmarks/commit/9307d36df26ece2acbd909243b3862dcde1b02cf))

## [0.18.0](https://github.com/dryvist/mlx-benchmarks/compare/v0.17.0...v0.18.0) (2026-08-15)


### Features

* **scripts:** add run-suite.sh end-to-end benchmark runner ([#174](https://github.com/dryvist/mlx-benchmarks/issues/174)) ([5a5b501](https://github.com/dryvist/mlx-benchmarks/commit/5a5b501c213de76cc1cbe1a3d265263f34dbd10d))

## [0.17.0](https://github.com/dryvist/mlx-benchmarks/compare/v0.16.0...v0.17.0) (2026-07-30)


### Features

* **shootout:** agent-brain comparison — factual suite, ranker, and candidate slate ([#164](https://github.com/dryvist/mlx-benchmarks/issues/164)) ([6f2c78c](https://github.com/dryvist/mlx-benchmarks/commit/6f2c78cedfc57e6874bd39cc8ae00e26c1ba85ec))


### Documentation

* **readme:** cut duplicated suite table to clear the file-size gate ([#172](https://github.com/dryvist/mlx-benchmarks/issues/172)) ([40b8b8c](https://github.com/dryvist/mlx-benchmarks/commit/40b8b8ca8c786de6d46a026b833e12a14a9baeac))

## [0.16.0](https://github.com/dryvist/mlx-benchmarks/compare/v0.15.3...v0.16.0) (2026-07-27)


### Features

* **throughput:** make cumulative tok/s the headline metric ([#169](https://github.com/dryvist/mlx-benchmarks/issues/169)) ([a3c49ce](https://github.com/dryvist/mlx-benchmarks/commit/a3c49ced9fc97d98901f77d47af46ebf05d23db3))

## [0.15.3](https://github.com/dryvist/mlx-benchmarks/compare/v0.15.2...v0.15.3) (2026-07-21)


### Documentation

* **benchmarks:** parameterize Doppler selection ([dc9cc05](https://github.com/dryvist/mlx-benchmarks/commit/dc9cc05eba990178065ed3dd5a739b9b8e772289))
* **benchmarks:** parameterize Doppler selection ([#160](https://github.com/dryvist/mlx-benchmarks/issues/160)) ([d95e1b2](https://github.com/dryvist/mlx-benchmarks/commit/d95e1b22473795063f71ee48151d84eb366a8d98))

## [0.15.2](https://github.com/dryvist/mlx-benchmarks/compare/v0.15.1...v0.15.2) (2026-07-19)


### Documentation

* **mlx-llm-configs:** add serving-config parameter reference + parity checklist ([#158](https://github.com/dryvist/mlx-benchmarks/issues/158)) ([0b25819](https://github.com/dryvist/mlx-benchmarks/commit/0b25819ea808f9a0010d03a842eb8a7f5bd41475))

## [0.15.1](https://github.com/dryvist/mlx-benchmarks/compare/v0.15.0...v0.15.1) (2026-07-19)


### Bug Fixes

* **cli:** accept --kind promptstack in mlx-bench-publish ([abfa91c](https://github.com/dryvist/mlx-benchmarks/commit/abfa91caf8cf9deda40fe3af1798df561f3a4437))


### Documentation

* adopt promptfoo-based prompt-eval framework, supersede promptstack ([e1c8a2c](https://github.com/dryvist/mlx-benchmarks/commit/e1c8a2c1a5d8ec9ec603a887a934245271990915))
* rewrap journal list item that markdownlint read as a plus bullet ([7c9731c](https://github.com/dryvist/mlx-benchmarks/commit/7c9731ca9cc1de99f4d1af6d5d143ab1c36d1723))

## [0.15.0](https://github.com/dryvist/mlx-benchmarks/compare/v0.14.0...v0.15.0) (2026-07-19)


### Features

* **converters:** add bench-serve converter for vllm-mlx bench-serve JSON ([#154](https://github.com/dryvist/mlx-benchmarks/issues/154)) ([b599b52](https://github.com/dryvist/mlx-benchmarks/commit/b599b526296516228297bf5cd4963ee4d6de6b14))

## [0.14.0](https://github.com/dryvist/mlx-benchmarks/compare/v0.13.2...v0.14.0) (2026-07-17)


### Features

* **schema:** env-class, concurrency, serving identity, cluster topology ([#150](https://github.com/dryvist/mlx-benchmarks/issues/150)) ([e4afe60](https://github.com/dryvist/mlx-benchmarks/commit/e4afe60d73d6c1def30fb39a31c1fb8b5648f0ce))

## [0.13.2](https://github.com/dryvist/mlx-benchmarks/compare/v0.13.1...v0.13.2) (2026-07-17)


### Bug Fixes

* **agentic:** runner robustness + honest throughput reporting ([#144](https://github.com/dryvist/mlx-benchmarks/issues/144)) ([fa86427](https://github.com/dryvist/mlx-benchmarks/commit/fa86427e87dbc3239f07d1b4e9bcbd2ab74bd775))

## [0.13.1](https://github.com/dryvist/mlx-benchmarks/compare/v0.13.0...v0.13.1) (2026-07-13)


### Documentation

* **rankings:** land 80B token-truncation finding (fits file-size gate) ([#138](https://github.com/dryvist/mlx-benchmarks/issues/138)) ([ff58d90](https://github.com/dryvist/mlx-benchmarks/commit/ff58d90d4945d98f0b74c1b520c445a785406138))

## [0.13.0](https://github.com/dryvist/mlx-benchmarks/compare/v0.12.0...v0.13.0) (2026-07-13)


### Features

* **agentic:** add --temperature and --repetition-penalty flags ([#136](https://github.com/dryvist/mlx-benchmarks/issues/136)) ([4424311](https://github.com/dryvist/mlx-benchmarks/commit/442431127ea2642ccf66fa08e3ceaa11ef0cc5a8))

## [0.12.0](https://github.com/dryvist/mlx-benchmarks/compare/v0.11.3...v0.12.0) (2026-07-12)


### Features

* **promptstack:** add system-prompt-as-independent-variable eval suite ([#130](https://github.com/dryvist/mlx-benchmarks/issues/130)) ([922b44d](https://github.com/dryvist/mlx-benchmarks/commit/922b44d143906b1953a8cd550cd595390ce833f0))

## [0.11.3](https://github.com/dryvist/mlx-benchmarks/compare/v0.11.2...v0.11.3) (2026-07-12)


### Documentation

* **journal:** TB5 cluster session — concurrency curves, link redesign verdicts, batching recommendation ([#128](https://github.com/dryvist/mlx-benchmarks/issues/128)) ([b0850b6](https://github.com/dryvist/mlx-benchmarks/commit/b0850b6c14e75b31e055c9ea4f3913344eaac916))

## [0.11.2](https://github.com/dryvist/mlx-benchmarks/compare/v0.11.1...v0.11.2) (2026-07-11)


### Documentation

* **bench:** concurrency-cascade + cold-start traps, sweep journal, config guards ([#125](https://github.com/dryvist/mlx-benchmarks/issues/125)) ([5814278](https://github.com/dryvist/mlx-benchmarks/commit/5814278db55228912ba2ccb34e8997caecb56ad1))

## [0.11.1](https://github.com/dryvist/mlx-benchmarks/compare/v0.11.0...v0.11.1) (2026-07-10)


### Documentation

* **runbook:** harden managed-window bootout/restore + rotation guidance ([#122](https://github.com/dryvist/mlx-benchmarks/issues/122)) ([1bb1889](https://github.com/dryvist/mlx-benchmarks/commit/1bb1889b790f7772a2255012acde62d5b6c74cc9))

## [0.11.0](https://github.com/dryvist/mlx-benchmarks/compare/v0.10.6...v0.11.0) (2026-07-10)


### Features

* **publish:** emit flat bench-events JSONL feed for log-pipeline ingest ([#121](https://github.com/dryvist/mlx-benchmarks/issues/121)) ([292e731](https://github.com/dryvist/mlx-benchmarks/commit/292e7315a14fa414cab5a87a4a5069f04aaa87bb)), closes [#119](https://github.com/dryvist/mlx-benchmarks/issues/119)

## [0.10.6](https://github.com/dryvist/mlx-benchmarks/compare/v0.10.5...v0.10.6) (2026-07-09)


### Documentation

* **flagship:** 2026-07-09 isolated-window results — no 60-90GB flagship fits ([#114](https://github.com/dryvist/mlx-benchmarks/issues/114)) ([c58da6f](https://github.com/dryvist/mlx-benchmarks/commit/c58da6f55e002cce029c3f251bd67105a590cf9e))

## [0.10.5](https://github.com/dryvist/mlx-benchmarks/compare/v0.10.4...v0.10.5) (2026-07-08)


### Documentation

* **verdict-policy:** document the 1/4 crediting rule for maturity ([#111](https://github.com/dryvist/mlx-benchmarks/issues/111)) ([6922438](https://github.com/dryvist/mlx-benchmarks/commit/69224380fa184c59dccd79b9ae0e0178b2a945e4))

## [0.10.4](https://github.com/dryvist/mlx-benchmarks/compare/v0.10.3...v0.10.4) (2026-07-08)


### Documentation

* **rankings:** mark all current verdicts 1/4 provisional maturity ([#109](https://github.com/dryvist/mlx-benchmarks/issues/109)) ([f7cbd09](https://github.com/dryvist/mlx-benchmarks/commit/f7cbd09438f986b9262b52031ac74f92190e77d2))

## [0.10.3](https://github.com/dryvist/mlx-benchmarks/compare/v0.10.2...v0.10.3) (2026-07-08)


### Documentation

* self-serve benchmarking playbook (RUNBOOK, RANKINGS, examples) ([#107](https://github.com/dryvist/mlx-benchmarks/issues/107)) ([586dd7d](https://github.com/dryvist/mlx-benchmarks/commit/586dd7dbc2128ffa5070b85bae1682e340bbc31c))

## [0.10.2](https://github.com/dryvist/mlx-benchmarks/compare/v0.10.1...v0.10.2) (2026-07-08)


### Documentation

* **agentic:** record brain-selection run + durable per-class lessons ([#105](https://github.com/dryvist/mlx-benchmarks/issues/105)) ([1df057e](https://github.com/dryvist/mlx-benchmarks/commit/1df057eda6eae10ce30fa7dff88f971829dfd818))

## [0.10.1](https://github.com/dryvist/mlx-benchmarks/compare/v0.10.0...v0.10.1) (2026-07-08)


### Documentation

* **configs:** make the qwen3-tasks overlay the coding-suite default ([#103](https://github.com/dryvist/mlx-benchmarks/issues/103)) ([d1fbeac](https://github.com/dryvist/mlx-benchmarks/commit/d1fbeacb64a813285972b222a0def84ea654d574))

## [0.10.0](https://github.com/dryvist/mlx-benchmarks/compare/v0.9.2...v0.10.0) (2026-07-08)


### Features

* **agentic:** many-tool tool-call reliability suite ([#101](https://github.com/dryvist/mlx-benchmarks/issues/101)) ([83dc048](https://github.com/dryvist/mlx-benchmarks/commit/83dc0484915a43ceb16df5fa6367c6c382233c44))

## [0.9.2](https://github.com/dryvist/mlx-benchmarks/compare/v0.9.1...v0.9.2) (2026-07-08)


### Documentation

* per-model-class agentic tool-calling notes (mid-2026 sourced) ([#99](https://github.com/dryvist/mlx-benchmarks/issues/99)) ([ae162c5](https://github.com/dryvist/mlx-benchmarks/commit/ae162c5365ff1fe6af72258f5e780153c1301bd8))

## [0.9.1](https://github.com/dryvist/mlx-benchmarks/compare/v0.9.0...v0.9.1) (2026-07-08)


### Bug Fixes

* **space:** coalesce dual result layouts so the viewer shows all real data ([#96](https://github.com/dryvist/mlx-benchmarks/issues/96)) ([99eb1ab](https://github.com/dryvist/mlx-benchmarks/commit/99eb1abbe39eed358c544fa15776be01d9189625))

## [0.9.0](https://github.com/dryvist/mlx-benchmarks/compare/v0.8.0...v0.9.0) (2026-07-08)


### ⚠ BREAKING CHANGES

* removes --kind promptfoo, the Splunk ship flags (--ship-splunk / --splunk-*), and --log-format. Publishing now covers lm-eval and vllm only.

### Features

* simplify to schema/publish core; add cross-machine hostname ([#92](https://github.com/dryvist/mlx-benchmarks/issues/92)) ([fbd83ba](https://github.com/dryvist/mlx-benchmarks/commit/fbd83baf280a628ad86aeb26e949c1f5d0d6ac6b))

## [0.8.0](https://github.com/dryvist/mlx-benchmarks/compare/v0.7.6...v0.8.0) (2026-07-03)


### Features

* **promptfoo:** model-comparison suites, converter, and optional Splunk ship ([#88](https://github.com/dryvist/mlx-benchmarks/issues/88)) ([bbb6204](https://github.com/dryvist/mlx-benchmarks/commit/bbb6204f5ddf7d462b1caa974851c5fade2a5a70))

## [0.7.6](https://github.com/dryvist/mlx-benchmarks/compare/v0.7.5...v0.7.6) (2026-07-03)


### Bug Fixes

* **ci:** declare jevans-ms self-hosted runner label for actionlint ([#89](https://github.com/dryvist/mlx-benchmarks/issues/89)) ([b8142df](https://github.com/dryvist/mlx-benchmarks/commit/b8142df3ce08a98468504711250666cf0728f694))

## [0.7.5](https://github.com/dryvist/mlx-benchmarks/compare/v0.7.4...v0.7.5) (2026-06-26)


### Documentation

* **readme:** make standalone — drop cross-repo narrative, link the hub ([#79](https://github.com/dryvist/mlx-benchmarks/issues/79)) ([e4a71e4](https://github.com/dryvist/mlx-benchmarks/commit/e4a71e459d348e695148b695eca87a46fb9a6647))

## [0.7.4](https://github.com/dryvist/mlx-benchmarks/compare/v0.7.3...v0.7.4) (2026-06-12)


### Bug Fixes

* **ci:** repoint shared workflows to dryvist hub ([#76](https://github.com/dryvist/mlx-benchmarks/issues/76)) ([3cc02f7](https://github.com/dryvist/mlx-benchmarks/commit/3cc02f7934bf7a0541243c596fbc139f23040a70))

## [0.7.3](https://github.com/dryvist/mlx-benchmarks/compare/v0.7.2...v0.7.3) (2026-06-06)


### Bug Fixes

* **ci:** use HF_TOKEN_WRITE_ALL for Space deployment ([#72](https://github.com/dryvist/mlx-benchmarks/issues/72)) ([1ad58ab](https://github.com/dryvist/mlx-benchmarks/commit/1ad58abad9750ed5d30bb5f033c99a815a3813a2))
* **space:** use format='ISO8601' for mixed timestamp parsing ([b505370](https://github.com/dryvist/mlx-benchmarks/commit/b5053709d6a8b38525fd2c2e09f2f3b86c9aeda2))


### Refactoring

* **logging:** extract _STANDARD_LOG_ATTRS module-level constant ([#71](https://github.com/dryvist/mlx-benchmarks/issues/71)) ([98e0a55](https://github.com/dryvist/mlx-benchmarks/commit/98e0a55e5b65191001cc6395241032312b2acb0e))

## [0.7.2](https://github.com/dryvist/mlx-benchmarks/compare/v0.7.1...v0.7.2) (2026-06-02)


### Bug Fixes

* **ci:** repoint release-please caller to org-native reusable workflow ([#68](https://github.com/dryvist/mlx-benchmarks/issues/68)) ([01ee035](https://github.com/dryvist/mlx-benchmarks/commit/01ee035f9031b0cbe40a3cc5e2b65681e1e3c0c5))
* **ci:** retarget reusable-workflow uses: refs to current org homes ([#66](https://github.com/dryvist/mlx-benchmarks/issues/66)) ([4622ef8](https://github.com/dryvist/mlx-benchmarks/commit/4622ef8a520ce35fae0e4027b9963a8b35b0e291))

## [0.7.1](https://github.com/JacobPEvans/mlx-benchmarks/compare/v0.7.0...v0.7.1) (2026-05-25)


### Bug Fixes

* **ci-gate:** sync pip-audit ignore-vulns with osv-scanner.toml ([#48](https://github.com/JacobPEvans/mlx-benchmarks/issues/48)) ([43c0478](https://github.com/JacobPEvans/mlx-benchmarks/commit/43c04786e5a24cc8d9a8d25dc928b86626dd83b7))
* **deps:** floor huggingface-hub at major-only (&gt;=1.0.0) ([#60](https://github.com/JacobPEvans/mlx-benchmarks/issues/60)) ([750da34](https://github.com/JacobPEvans/mlx-benchmarks/commit/750da340ee3ad7c14049e1c9b256ac700a0e9a50))
* **deps:** update dependency google-adk to &gt;=2.0.0 ([#57](https://github.com/JacobPEvans/mlx-benchmarks/issues/57)) ([9528615](https://github.com/JacobPEvans/mlx-benchmarks/commit/95286158d932666ed9a85c84121c27f8edd480a1))
* **deps:** update dependency pyarrow to v24.0.0 ([#59](https://github.com/JacobPEvans/mlx-benchmarks/issues/59)) ([b10c91b](https://github.com/JacobPEvans/mlx-benchmarks/commit/b10c91bc087a1a381e46597c6789c91a38709ab5))
* **security:** bump pyarrow &gt;=23.0.1 for PYSEC-2026-113 ([#55](https://github.com/JacobPEvans/mlx-benchmarks/issues/55)) ([0c9a047](https://github.com/JacobPEvans/mlx-benchmarks/commit/0c9a0477c7a5f4045fa09d1e320305c3250898f9))
* **security:** ignore unfixable PyPI advisories + bump idna 3.15 ([#45](https://github.com/JacobPEvans/mlx-benchmarks/issues/45)) ([21b1502](https://github.com/JacobPEvans/mlx-benchmarks/commit/21b1502a15b643322f3e050ae92de68d388c21fe))


### Documentation

* add themed architecture diagrams and ecosystem section ([#49](https://github.com/JacobPEvans/mlx-benchmarks/issues/49)) ([7c861cb](https://github.com/JacobPEvans/mlx-benchmarks/commit/7c861cb5e07d9de6a736373bf8480e764512a7c3))

## [0.7.0](https://github.com/JacobPEvans/mlx-benchmarks/compare/v0.6.1...v0.7.0) (2026-05-14)


### Features

* **envelope:** add tokens-per-second metrics ([#38](https://github.com/JacobPEvans/mlx-benchmarks/issues/38)) ([6d65d81](https://github.com/JacobPEvans/mlx-benchmarks/commit/6d65d817d6bd45e897f44d53224cfd7a9a913fd3))


### Bug Fixes

* **deps:** update dependency lm-eval to v0.4.12 ([#41](https://github.com/JacobPEvans/mlx-benchmarks/issues/41)) ([1698c37](https://github.com/JacobPEvans/mlx-benchmarks/commit/1698c373ba0ae25aaf448349c5819d2d05006acd))


### Documentation

* add quick-reset guide for local LLM memory refresh ([#42](https://github.com/JacobPEvans/mlx-benchmarks/issues/42)) ([8e7f348](https://github.com/JacobPEvans/mlx-benchmarks/commit/8e7f34807c52b586a4b891edd32cf8322ffb7f34))

## [0.6.1](https://github.com/JacobPEvans/mlx-benchmarks/compare/v0.6.0...v0.6.1) (2026-05-03)


### Bug Fixes

* **ci:** remove deprecated app-id secret passthrough ([435846d](https://github.com/JacobPEvans/mlx-benchmarks/commit/435846da8063f148d8608cf1292d862abb25a0f8))

## [0.6.0](https://github.com/JacobPEvans/mlx-benchmarks/compare/v0.5.0...v0.6.0) (2026-04-29)


### Features

* add lm-eval reasoning and vllm throughput configs ([#10](https://github.com/JacobPEvans/mlx-benchmarks/issues/10)) ([a9e5608](https://github.com/JacobPEvans/mlx-benchmarks/commit/a9e5608a0ae7c4ac688e60cffeb3dc96d131fb63))
* add uv dev shell with lm-eval as proper dependency ([#8](https://github.com/JacobPEvans/mlx-benchmarks/issues/8)) ([328bb9a](https://github.com/JacobPEvans/mlx-benchmarks/commit/328bb9a283f5e0086d952e8b1bf6e8799725c602))
* add vllm benchmark_serving converter ([6006217](https://github.com/JacobPEvans/mlx-benchmarks/commit/600621795763cd9e8b3101968c957b68280e8ed9))
* **benchmarks:** migrate framework evaluation harness and reports ([#3](https://github.com/JacobPEvans/mlx-benchmarks/issues/3)) ([6e8e03c](https://github.com/JacobPEvans/mlx-benchmarks/commit/6e8e03c4d094a52068d1b4c0742048a33fb5d492))
* Gradio benchmark-viewer Space ([#14](https://github.com/JacobPEvans/mlx-benchmarks/issues/14)) ([5e7cefb](https://github.com/JacobPEvans/mlx-benchmarks/commit/5e7cefb03050d0467d0938047a78683c3bf410a5))
* initial scaffolding for benchmark harness ([#1](https://github.com/JacobPEvans/mlx-benchmarks/issues/1)) ([6fd8afa](https://github.com/JacobPEvans/mlx-benchmarks/commit/6fd8afaff7570a505f12e20f8bf80e64f9d1697f))
* pre-v0.5.0 hardening (CI, security, docs) ([5bf37d4](https://github.com/JacobPEvans/mlx-benchmarks/commit/5bf37d4077947539cdb921126e67e357c0adbfa4))
* production polish — package layout, CI, viewer, docs ([#15](https://github.com/JacobPEvans/mlx-benchmarks/issues/15)) ([0876689](https://github.com/JacobPEvans/mlx-benchmarks/commit/087668915e601613f3b8061e5c8b7b2d754d96ce))
* replace deploy_space.py with huggingface-cli upload ([8764aa8](https://github.com/JacobPEvans/mlx-benchmarks/commit/8764aa80f074b7d32cdd7d210d10baeae3fbfa51))


### Bug Fixes

* **deps:** bump pyarrow + pillow to fix OSV vulnerabilities ([#28](https://github.com/JacobPEvans/mlx-benchmarks/issues/28)) ([5f17230](https://github.com/JacobPEvans/mlx-benchmarks/commit/5f172308ce338f1363b74c031bc20d094e31c76c))
* pass App ID from vars not secrets in release-please ([1589b04](https://github.com/JacobPEvans/mlx-benchmarks/commit/1589b040b52b285b2cd1136161718822e9ab65d1))
* set DEVENV_ROOT and --impure in .envrc for devenv flake ([#9](https://github.com/JacobPEvans/mlx-benchmarks/issues/9)) ([b11881b](https://github.com/JacobPEvans/mlx-benchmarks/commit/b11881ba7bf3b29e68767363989fe436efd80ff1))
* use hf instead of deprecated huggingface-cli in deploy-space ([a27be21](https://github.com/JacobPEvans/mlx-benchmarks/commit/a27be21060e16aff6bef0e86b41c2263cfc25244))


### Documentation

* add project CLAUDE.md and CI badge ([#11](https://github.com/JacobPEvans/mlx-benchmarks/issues/11)) ([846f20d](https://github.com/JacobPEvans/mlx-benchmarks/commit/846f20dda8de717325c6d85aadbecf3215c44dc3))

## [0.5.0](https://github.com/JacobPEvans/mlx-benchmarks/compare/v0.4.0...v0.5.0) (2026-04-27)


### Features

* add lm-eval reasoning and vllm throughput configs ([#10](https://github.com/JacobPEvans/mlx-benchmarks/issues/10)) ([a9e5608](https://github.com/JacobPEvans/mlx-benchmarks/commit/a9e5608a0ae7c4ac688e60cffeb3dc96d131fb63))
* add uv dev shell with lm-eval as proper dependency ([#8](https://github.com/JacobPEvans/mlx-benchmarks/issues/8)) ([328bb9a](https://github.com/JacobPEvans/mlx-benchmarks/commit/328bb9a283f5e0086d952e8b1bf6e8799725c602))
* add vllm benchmark_serving converter ([6006217](https://github.com/JacobPEvans/mlx-benchmarks/commit/600621795763cd9e8b3101968c957b68280e8ed9))
* **benchmarks:** migrate framework evaluation harness and reports ([#3](https://github.com/JacobPEvans/mlx-benchmarks/issues/3)) ([6e8e03c](https://github.com/JacobPEvans/mlx-benchmarks/commit/6e8e03c4d094a52068d1b4c0742048a33fb5d492))
* Gradio benchmark-viewer Space ([#14](https://github.com/JacobPEvans/mlx-benchmarks/issues/14)) ([5e7cefb](https://github.com/JacobPEvans/mlx-benchmarks/commit/5e7cefb03050d0467d0938047a78683c3bf410a5))
* initial scaffolding for benchmark harness ([#1](https://github.com/JacobPEvans/mlx-benchmarks/issues/1)) ([6fd8afa](https://github.com/JacobPEvans/mlx-benchmarks/commit/6fd8afaff7570a505f12e20f8bf80e64f9d1697f))
* pre-v0.5.0 hardening (CI, security, docs) ([5bf37d4](https://github.com/JacobPEvans/mlx-benchmarks/commit/5bf37d4077947539cdb921126e67e357c0adbfa4))
* production polish — package layout, CI, viewer, docs ([#15](https://github.com/JacobPEvans/mlx-benchmarks/issues/15)) ([0876689](https://github.com/JacobPEvans/mlx-benchmarks/commit/087668915e601613f3b8061e5c8b7b2d754d96ce))
* replace deploy_space.py with huggingface-cli upload ([8764aa8](https://github.com/JacobPEvans/mlx-benchmarks/commit/8764aa80f074b7d32cdd7d210d10baeae3fbfa51))


### Bug Fixes

* pass App ID from vars not secrets in release-please ([1589b04](https://github.com/JacobPEvans/mlx-benchmarks/commit/1589b040b52b285b2cd1136161718822e9ab65d1))
* set DEVENV_ROOT and --impure in .envrc for devenv flake ([#9](https://github.com/JacobPEvans/mlx-benchmarks/issues/9)) ([b11881b](https://github.com/JacobPEvans/mlx-benchmarks/commit/b11881ba7bf3b29e68767363989fe436efd80ff1))
* use hf instead of deprecated huggingface-cli in deploy-space ([a27be21](https://github.com/JacobPEvans/mlx-benchmarks/commit/a27be21060e16aff6bef0e86b41c2263cfc25244))


### Documentation

* add project CLAUDE.md and CI badge ([#11](https://github.com/JacobPEvans/mlx-benchmarks/issues/11)) ([846f20d](https://github.com/JacobPEvans/mlx-benchmarks/commit/846f20dda8de717325c6d85aadbecf3215c44dc3))
