# Label audit: how right is the corpus the model trains on?

For programs of the BigO(Bench) and CodeComplex test splits: the label the corpus gives,
what the symbolic engine answered at first contact, and what an independent agent decided
from the code alone under the project's conventions (worst case, the quantities read as
sizes). The random sample estimates how often each party is right; the disagreement sample
(the engine was 'certain' and the corpus label contradicted it) shows which side the
independent reading takes. Agents' low-confidence answers are left out.

## Random sample: who agrees with an independent reading of the code

| dimension | rows | corpus label | engine (first contact) | both | neither |
|---|---|---|---|---|---|
| time | 49 | 33/49 (67%) | 34/49 (69%) | 26/49 (53%) | 8/49 (16%) |
| space | 49 | 34/49 (69%) | 37/49 (76%) | 28/49 (57%) | 6/49 (12%) |

Corpus label right, by source (random sample):

* time: bigobench 26/42 (62%), synth 5/5 (100%), thealgorithms 2/2 (100%)
* space: bigobench 27/42 (64%), synth 5/5 (100%), thealgorithms 2/2 (100%)

## Disagreements: the engine was 'certain' and the corpus label said otherwise

**time** (58 programs):
* 20/58 (34%) corpus right (the engine is wrong)
* 31/58 (53%) engine right (the corpus label is wrong, or follows another convention)
* 7/58 (12%) neither

**space** (39 programs):
* 7/39 (18%) corpus right (the engine is wrong)
* 27/39 (69%) engine right (the corpus label is wrong, or follows another convention)
* 5/39 (13%) neither

## Reading the numbers

BigO(Bench) labels a program by measuring it against the dimension of its own test
harness: a loop over a small, fixed-size input reads as O(1) there, and anything that
does not scale with that harness dimension does too. The project's convention is the
worst case as a function of the quantities the program reads. The two disagree most on
exactly the programs that dominate the corpus (competitive-programming solutions), so a
model trained on those labels learns the measurement convention, not the one users
expect, and it is capped by the label noise measured above whatever its architecture.
