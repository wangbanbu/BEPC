# AR and native ViSpec output comparison

Stored main-phase repeat0 outputs only. No inference, rescoring, or configuration changes were performed. Causes of output differences are not established by this comparison.

| Task | Questions | Returned-token differences | Text differences | Native/shared differences |
|---|---:|---:|---:|---:|
| mme | 200 | 0 | 0 | 0 |
| gqa | 100 | 0 | 0 | 0 |
| textvqa | 100 | 1 | 1 | 0 |
| scienceqa | 100 | 0 | 0 | 0 |
| coco | 100 | 16 | 16 | 0 |

Across 600 questions, AR and native ViSpec differ on returned tokens for 17 questions and output text for 17 questions.

The full difference records and first-divergence histograms are in the adjacent JSON. Token positions are zero based; an ended sequence against a continuing sequence counts as a divergence.

Native ViSpec/strict_shared equivalence is independently audited in FINAL_DELIVERY_VALIDATION.json. This does not imply AR/native equivalence.

Sealed COCO CIDEr: ar=0.410904784047, vispec=0.403488229741, strict_shared=0.403488229741.

## COCO examples

First five token-different COCO questions in frozen ordinal order; no quality-based selection.

### Ordinal 1, question 461278

First token divergence: 3. Returned lengths: AR 18, ViSpec 16.

AR: A wooden bench is situated in a lush garden setting with a white house in the background.

Native ViSpec: A wooden bench sits in a lush garden with a white house in the background.

### Ordinal 2, question 117690

First token divergence: 18. Returned lengths: AR 23, ViSpec 27.

AR: A black dog leaps into the air to catch a frisbee during a Splash Dogs performance at an outdoor event.

Native ViSpec: A black dog leaps into the air to catch a frisbee during a Splash Dogs performance, with a handler observing from a platform.

### Ordinal 5, question 537939

First token divergence: 5. Returned lengths: AR 34, ViSpec 41.

AR: The image shows a workspace setup with a laptop, two monitors, a keyboard, and a mouse on a desk, along with various office supplies and posters on the wall.

Native ViSpec: The image shows a workspace with a desk setup featuring dual monitors, a laptop, a keyboard, a mouse, and various office supplies, with posters and a helpdesk sign on the wall in the background.

### Ordinal 6, question 71929

First token divergence: 30. Returned lengths: AR 32, ViSpec 32.

AR: The image shows a plate of food with grilled meat, rice, and a side salad, accompanied by a glass of water, set against a bright outdoor background.

Native ViSpec: The image shows a plate of food with grilled meat, rice, and a side salad, accompanied by a glass of water, set against a bright outdoor backdrop.

### Ordinal 15, question 361177

First token divergence: 3. Returned lengths: AR 13, ViSpec 16.

AR: A black dog rests peacefully on a bed next to a laptop.

Native ViSpec: A black dog is resting its head on a blue blanket next to a laptop.
