# Engine vs model vs rule: the honest comparison

The honest comparison: symbolic engine, ML model, rule baseline and the product, side by side, on

## blind_v1 (144 programs)

### time, exact class

| predictor | accuracy (95% Wilson) |
|---|---|
| symbolic | 85% (122/144, 78%-90%) |
| gnn | 30% (43/144, 23%-38%) |
| rule | 35% (50/144, 27%-43%) |
| product | 94% (135/144, 89%-97%) |

### space, exact class

| predictor | accuracy (95% Wilson) |
|---|---|
| symbolic | 88% (127/144, 82%-92%) |
| gnn | 39% (56/144, 31%-47%) |
| rule | 33% (48/144, 26%-41%) |
| product | 96% (138/144, 91%-98%) |

### time by language

| language | n | symbolic | gnn | rule | product |
|---|---|---|---|---|---|
| c | 24 | 96% (23/24, 80%-99%) | 33% (8/24, 18%-53%) | 46% (11/24, 28%-65%) | 100% (24/24, 86%-100%) |
| cpp | 24 | 92% (22/24, 74%-98%) | 21% (5/24, 9%-40%) | 33% (8/24, 18%-53%) | 96% (23/24, 80%-99%) |
| go | 24 | 83% (20/24, 64%-93%) | 29% (7/24, 15%-49%) | 46% (11/24, 28%-65%) | 88% (21/24, 69%-96%) |
| java | 24 | 83% (20/24, 64%-93%) | 38% (9/24, 21%-57%) | 25% (6/24, 12%-45%) | 96% (23/24, 80%-99%) |
| javascript | 24 | 71% (17/24, 51%-85%) | 25% (6/24, 12%-45%) | 42% (10/24, 24%-61%) | 92% (22/24, 74%-98%) |
| python | 24 | 83% (20/24, 64%-93%) | 33% (8/24, 18%-53%) | 17% (4/24, 7%-36%) | 92% (22/24, 74%-98%) |

### space by language

| language | n | symbolic | gnn | rule | product |
|---|---|---|---|---|---|
| c | 24 | 100% (24/24, 86%-100%) | 29% (7/24, 15%-49%) | 50% (12/24, 31%-69%) | 100% (24/24, 86%-100%) |
| cpp | 24 | 92% (22/24, 74%-98%) | 46% (11/24, 28%-65%) | 21% (5/24, 9%-40%) | 96% (23/24, 80%-99%) |
| go | 24 | 83% (20/24, 64%-93%) | 29% (7/24, 15%-49%) | 33% (8/24, 18%-53%) | 92% (22/24, 74%-98%) |
| java | 24 | 88% (21/24, 69%-96%) | 38% (9/24, 21%-57%) | 25% (6/24, 12%-45%) | 100% (24/24, 86%-100%) |
| javascript | 24 | 88% (21/24, 69%-96%) | 50% (12/24, 31%-69%) | 50% (12/24, 31%-69%) | 96% (23/24, 80%-99%) |
| python | 24 | 79% (19/24, 60%-91%) | 42% (10/24, 24%-61%) | 21% (5/24, 9%-40%) | 92% (22/24, 74%-98%) |

### time by the engine's first-contact certainty

| certainty | n | symbolic | gnn | rule | product |
|---|---|---|---|---|---|
| assumed | 28 | 64% (18/28, 46%-79%) | 29% (8/28, 15%-47%) | 39% (11/28, 24%-58%) | 79% (22/28, 60%-90%) |
| certain | 112 | 93% (104/112, 87%-96%) | 29% (33/112, 22%-38%) | 35% (39/112, 27%-44%) | 98% (110/112, 94%-100%) |
| unknown | 4 | 0% (0/4, 0%-49%) | 50% (2/4, 15%-85%) | 0% (0/4, 0%-49%) | 75% (3/4, 30%-95%) |

## audit_v1 (49 programs)

### time, exact class

| predictor | accuracy (95% Wilson) |
|---|---|
| symbolic | 69% (34/49, 55%-80%) |
| gnn | 47% (23/49, 34%-61%) |
| rule | 47% (23/49, 34%-61%) |
| product | 69% (34/49, 55%-80%) |

### space, exact class

| predictor | accuracy (95% Wilson) |
|---|---|
| symbolic | 76% (37/49, 62%-85%) |
| gnn | 78% (38/49, 64%-87%) |
| rule | 53% (26/49, 39%-66%) |
| product | 76% (37/49, 62%-85%) |

### time by language

| language | n | symbolic | gnn | rule | product |
|---|---|---|---|---|---|
| java | 7 | 100% (7/7, 65%-100%) | 43% (3/7, 16%-75%) | 43% (3/7, 16%-75%) | 100% (7/7, 65%-100%) |
| python | 42 | 64% (27/42, 49%-77%) | 48% (20/42, 33%-62%) | 48% (20/42, 33%-62%) | 64% (27/42, 49%-77%) |

### space by language

| language | n | symbolic | gnn | rule | product |
|---|---|---|---|---|---|
| java | 7 | 100% (7/7, 65%-100%) | 100% (7/7, 65%-100%) | 86% (6/7, 49%-97%) | 100% (7/7, 65%-100%) |
| python | 42 | 71% (30/42, 56%-83%) | 74% (31/42, 59%-85%) | 48% (20/42, 33%-62%) | 71% (30/42, 56%-83%) |

### time by the engine's first-contact certainty

| certainty | n | symbolic | gnn | rule | product |
|---|---|---|---|---|---|
| assumed | 13 | 54% (7/13, 29%-77%) | 62% (8/13, 36%-82%) | 62% (8/13, 36%-82%) | 54% (7/13, 29%-77%) |
| certain | 35 | 77% (27/35, 61%-88%) | 43% (15/35, 28%-59%) | 43% (15/35, 28%-59%) | 77% (27/35, 61%-88%) |
| unknown | 1 | 0% (0/1, 0%-79%) | 0% (0/1, 0%-79%) | 0% (0/1, 0%-79%) | 0% (0/1, 0%-79%) |

## Promotion (who answers each slice)

| slice | dimension | rows | engine | model | answers | why |
|---|---|---|---|---|---|---|
| c/assumed | time | 4 | 4/4 | 2/4 | engine | fewer than 20 rows: the engine stays the default |
| c/assumed | space | 4 | 4/4 | 0/4 | engine | fewer than 20 rows: the engine stays the default |
| c/certain | time | 20 | 19/20 | 6/20 | engine | model not better beyond the noise of the sample |
| c/certain | space | 20 | 20/20 | 7/20 | engine | model not better beyond the noise of the sample |
| cpp/assumed | time | 3 | 2/3 | 0/3 | engine | fewer than 20 rows: the engine stays the default |
| cpp/assumed | space | 3 | 2/3 | 0/3 | engine | fewer than 20 rows: the engine stays the default |
| cpp/certain | time | 21 | 20/21 | 5/21 | engine | model not better beyond the noise of the sample |
| cpp/certain | space | 21 | 20/21 | 11/21 | engine | model not better beyond the noise of the sample |
| go/assumed | time | 3 | 2/3 | 0/3 | engine | fewer than 20 rows: the engine stays the default |
| go/assumed | space | 3 | 1/3 | 1/3 | engine | fewer than 20 rows: the engine stays the default |
| go/certain | time | 21 | 18/21 | 7/21 | engine | model not better beyond the noise of the sample |
| go/certain | space | 21 | 19/21 | 6/21 | engine | model not better beyond the noise of the sample |
| java/assumed | time | 7 | 6/7 | 3/7 | engine | fewer than 20 rows: the engine stays the default |
| java/assumed | space | 7 | 6/7 | 2/7 | engine | fewer than 20 rows: the engine stays the default |
| java/certain | time | 22 | 21/22 | 7/22 | engine | model not better beyond the noise of the sample |
| java/certain | space | 22 | 22/22 | 14/22 | engine | model not better beyond the noise of the sample |
| java/unknown | time | 2 | 0/2 | 2/2 | engine | fewer than 20 rows: the engine stays the default |
| java/unknown | space | 2 | 0/2 | 0/2 | engine | fewer than 20 rows: the engine stays the default |
| javascript/assumed | time | 7 | 3/7 | 1/7 | engine | fewer than 20 rows: the engine stays the default |
| javascript/assumed | space | 7 | 6/7 | 5/7 | engine | fewer than 20 rows: the engine stays the default |
| javascript/certain | time | 15 | 14/15 | 5/15 | engine | fewer than 20 rows: the engine stays the default |
| javascript/certain | space | 15 | 14/15 | 7/15 | engine | fewer than 20 rows: the engine stays the default |
| javascript/unknown | time | 2 | 0/2 | 0/2 | engine | fewer than 20 rows: the engine stays the default |
| javascript/unknown | space | 2 | 1/2 | 0/2 | engine | fewer than 20 rows: the engine stays the default |
| python/assumed | time | 17 | 8/17 | 10/17 | engine | fewer than 20 rows: the engine stays the default |
| python/assumed | space | 17 | 8/17 | 10/17 | engine | fewer than 20 rows: the engine stays the default |
| python/certain | time | 48 | 39/48 | 18/48 | engine | model not better beyond the noise of the sample |
| python/certain | space | 48 | 40/48 | 30/48 | engine | model not better beyond the noise of the sample |
| python/unknown | time | 1 | 0/1 | 0/1 | engine | fewer than 20 rows: the engine stays the default |
| python/unknown | space | 1 | 1/1 | 1/1 | engine | fewer than 20 rows: the engine stays the default |

