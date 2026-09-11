# External benchmark landscape — a guide to what to measure, never a result

Public leaderboards are worth reading and worth borrowing from. They are not
worth trusting as our numbers, and nothing in them may be recorded here as a
measured value.

**What transfers is task design. What never transfers is the score.**

A published figure describes one run, on one serving stack, at one quant, with
one set of sampling parameters, at one reasoning-effort setting, behind one
agent harness, on hardware under some unstated load. Change any of those and
the number moves — which is the entire reason this repo exists. Our own runs
prove the point: the same model and task set scored differently when its
endpoint gained a second concurrent consumer, with no change to the weights.

So an external number may:

- seed a shortlist of arms worth serving,
- suggest which capability a suite should probe,
- form a hypothesis (*"this should beat that here too"*),

and may never:

- appear in `RANKINGS.md` as a value for one of our columns,
- stand in for a run we did not do,
- satisfy any gate in [`verdict-policy.md`](verdict-policy.md), which is
  explicitly about *runs on this fabric over time*.

## Artificial Analysis Intelligence Index v4.3

Ten evaluations in four weighted buckets. Useful mainly because the weighting
moved toward agentic work, which is closer to what we actually delegate than a
code-completion score.

| Bucket | Weight | Components |
| --- | --- | --- |
| Agents | 30% | AA-Briefcase 15%, GDPval-AA v2 10%, AutomationBench-AA 5% |
| Coding | 20% | Terminal-Bench v4.0 10%, SciCode 10% |
| General | 30% | AA-Omniscience 15%, GDP.pdf 10%, AA-LCR v1.1 5% |
| Scientific reasoning | 20% | Humanity's Last Exam 10%, CritPt 10% |

Three are close enough to this estate's work to be worth running **ourselves**:

- **Terminal-Bench v4.0** — 66 tasks, agent drives a real shell in a Docker
  container, success checked programmatically against final filesystem or
  process state. Tier 3 is system administration and security: service
  configuration, log analysis, non-obvious failure modes. Published SOTA sits
  near 59%, so it has real headroom rather than a ceiling.
- **AutomationBench-AA** — 657 workflow-automation tasks driven through REST
  API tools under guardrail constraints. The shape of an execution plane.
- **AA-Omniscience** — 6,000 questions that penalise hallucination and accept
  abstention. For an operations model, "I don't know" is a correct answer and
  most suites score it as a miss.

Reasoning effort is reported as part of a model's identity there
(`27B (xhigh)`, `(max)`, `(high)`) because the same weights score differently
per effort, at different token cost. Treat effort as part of the arm, never as
a detail — an arm is weights *plus* quant *plus* effort *plus* serving config.

## Infrastructure-as-code

Closest published work to the OpenTofu half of our delegation:

- **IaC-Eval v2** — 186 AWS/Terraform tasks scored against Rego intent
  policies rather than string match.
- **SWE-InfraBench** — real modifications to CDK repositories, gated on the
  repository's own tests.
- **TerraRepair** — scanner-verified repair rates (Checkov, Trivy) rather than
  generation quality.

One result from that literature is worth carrying regardless of the scores:
**syntactic validity and security compliance are largely orthogonal in
generated IaC.** A model that emits valid HCL tells you nothing about whether
it emits safe HCL, so `validate` passing is not a quality signal — it is the
weakest possible check wearing the costume of one.

**There is no comparable public Ansible benchmark.** That half of our
configuration surface has no external prior at all, which is worth stating
plainly rather than substituting a Terraform number for it.

## Using this file

When picking arms or designing a suite, read this for *what to probe*. Then
measure it here, on our serving stack, at our quant, with our sampling
parameters, under a declared measurement class — and let
[`verdict-policy.md`](verdict-policy.md) decide when the result is allowed to
mean anything.
