# GNN readout: dilution, not size

Cleaned training slice (cap 25 per problem, IR near-duplicates collapsed, every non-Python record kept, Python filling up to 12000), 8 epochs, hidden 64, 3 layers, seed 42; scored on the blind corpus (independent labels).

| variant | train rows | blind time | blind space | answers unchanged by 10 dead statements | by 40 | seconds |
|---|---|---|---|---|---|---|
| mean | 12000 | 42/144 (29%) | 62/144 (43%) | 33/144 (23%) | 7/144 (5%) | 80 |
| meanmax | 12000 | 59/144 (41%) | 58/144 (40%) | 112/144 (78%) | 105/144 (73%) | 73 |
| mean+aug | 24000 | 39/144 (27%) | 52/144 (36%) | 53/144 (37%) | 27/144 (19%) | 179 |
| meanmax+aug | 24000 | 53/144 (37%) | 60/144 (42%) | 126/144 (88%) | 122/144 (85%) | 179 |

For comparison the symbolic engine answers 100% of these padded programs identically (the
perturbation suite) and is right on 85% (time) / 88% (space) of the blind corpus at first contact.
