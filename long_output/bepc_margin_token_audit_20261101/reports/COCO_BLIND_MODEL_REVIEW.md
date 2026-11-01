# COCO A/B blind model-assisted factual-difference review

This is one model-assisted image review, not human annotation, not a new image-judge API run, and not frozen Harmful Ground Truth. All 26 supplied A/B pairs were reviewed against their actual local COCO images. Method identity and metric values were not opened. The comparison key must be joined separately after this blind file is saved.

## Scope and interpretation

The 26 pairs were selected for automatic-metric differences in both directions. They are not a random sample of all 100 COCO outputs and cannot estimate a natural protection/harm rate. Each pair is a whole-policy generation comparison: a changed correct phrase does not by itself establish that one specific mismatch decision caused its protection. The event/path audit must establish that connection.

The reviewer separately records changed factual claims and errors shared by both captions. A preferred caption can still contain major hallucinations. Minor wording changes are not automatically protections. Ambiguous low-light objects, partly occluded actions and fine OCR are retained as uncertain. Overall relation is deliberately conservative; attribute-level directional support can exist while the whole-caption relation remains uncertain.

Blind relation counts: {"uncertain": 13, "A_slightly_better": 5, "B_slightly_better": 6, "tie": 2}.

## Case records

### coco_pair_000 / image 136572

Overall: **uncertain**; confidence: medium. Image checked: yes.

- object/attribute: A “a visible windshield and handlebars”; B “a black seat, and a visible front wheel”. The scooter seat and wheel are visible. A separate transparent windshield is not clearly visible; the large white front panel is not itself a windshield. Directional factual support: B.
- scene inference: A “a tiled area”; B “wet in some areas, suggesting recent rain or water spillage”. Tiled paving is visible. Dark pavement patches do not establish rain or a wet surface. Directional factual support: A.

Shared limitations: Both call the foreground bike a BMW, but its tank has a Yamaha-style tuning-fork emblem. Both describe the rear wheel as obscured by a barrier, while the prominent curved barrier is around the front-wheel area.

Mixed changes: B avoids a poorly supported windshield but adds a weather/surface inference. Both retain important pre-existing identification and spatial errors.

### coco_pair_001 / image 136642

Overall: **uncertain**; confidence: high. Image checked: yes.

- action: A “a runner attempting to score”; B “possibly having just hit the ball”. A dropped bat is near home plate and the player runs leftward away from the catcher. B is a better-qualified account; scoring at home is not supported. Directional factual support: B.
- attribute/action: A “black and gray”; B “black and white; crouched in a typical stance”. The umpire has a gray top/dark trousers; the catcher is rising rather than in a deep crouch. Neither uniform description is fully precise; B adds a questionable catcher posture. Directional factual support: neither.

Shared limitations: Statements that home plate is clearly visible and that the catcher is ready to receive a ball are not securely established from the still image.

B improves the runner-action interpretation but introduces posture and uniform inaccuracies, so no clean overall preference is assigned.

### coco_pair_002 / image 167122

Overall: **uncertain**; confidence: low. Image checked: yes.

- object/existence: A “indistinct figures or objects near the building”; B “a fence and what appears to be a doorway; no visible people”. The dark, noisy image supports the car, illuminated wall/building and exterior lamps. Fine background objects and the presence or absence of people cannot be resolved confidently. Directional factual support: uncertain.
- linguistic: A “parked in what appears to be a parking lot”; B “dark, possibly parked car”. Both appropriately hedge location or motion. The wording does not establish a material factual difference. Directional factual support: both.

Shared limitations: Both emphasize the car rear/rear lights; the visible view is predominantly a dark side/body view and this specific orientation/lighting interpretation is uncertain.

Low visibility prevents reliable fine-grained protection judgments; neither added fence/door details nor absolute no-person claims should be promoted to facts.

### coco_pair_003 / image 199403

Overall: **uncertain**; confidence: medium. Image checked: yes.

- attribute: A “short, dark hair styled in a bob”; B “short, dark purple hair”. The hair is short and dark with visible purplish highlights. B adds a plausible supported color qualification, while A is also broadly compatible. Directional factual support: both.
- object: A “a countertop with various objects”; B “a bowl and what looks like a box of cereal”. The background is cluttered but the small food-container identities are not sufficiently legible to verify the cereal claim. Directional factual support: uncertain.

Shared limitations: Both describe the main clothing and kitchen setting accurately. Fine background object names and inferred lighting source remain uncertain.

Mostly compatible wording with an added fine-grained background guess; no clear consequential factual protection.

### coco_pair_004 / image 200567

Overall: **A_slightly_better**; confidence: medium. Image checked: yes.

- action: A “having just thrown the ball”; B “mid-throw, with his arm extended towards the batter”. The pitcher is seen in follow-through with a raised trailing leg; the asserted arm extended toward the batter is not visible. A is more compatible with the visible phase. Directional factual support: A.
- object/relation: A “the dirt of the pitcher's mound”; B “a clear boundary line marking the edge of the field”. Dirt areas are visible; a clear field-boundary line is not evident in this crop. Directional factual support: A.

Shared limitations: Both say the batter is in the act of swinging, although the bat is held upright in a prepared stance. Both include uncertain uniform-number/home-plate details.

A avoids two additional unsupported specifics in B, but it is not an entirely correct caption and shares a central action error.

### coco_pair_005 / image 202913

Overall: **B_slightly_better**; confidence: high. Image checked: yes.

- quantity/state: A “The cake is cut into a triangular slice”; B “A slice has been removed”. The displayed cake has a substantial missing section. B captures the visible missing-section state; A misleadingly describes the remaining large cake as a triangular slice. Directional factual support: B.
- relation: A “rests on the cake”; B “resting on the tray”. The serving utensil lies on the exposed tray with cake/icing on it. B gives the more accurate location. Directional factual support: B.

Shared limitations: The filling includes a pale yellow band; broad all-white interior language should not be read as an exact color annotation.

B improves two clearly checkable state/location claims while preserving the cake and coconut content.

### coco_pair_006 / image 202923

Overall: **uncertain**; confidence: medium. Image checked: yes.

- action: A “mid-swing; The ball is visible in the air”; B “poised to swing”. The bat is held raised in a prepared posture, and no clear airborne ball is identifiable. B avoids these unsupported action/ball assertions. Directional factual support: B.
- attribute: A “wearing protective gear”; B “batter, dressed in a white uniform”. Protective catcher gear is evident; the batter uniform appears gray rather than white. Directional factual support: A.

Shared limitations: Both call the distant player a pitcher in a pitching motion, although the visible distant player looks set in an infield position and a pitch cannot be established. The umpire is low/crouched rather than clearly standing.

B improves the principal batting phase but adds color imprecision; both retain action/role guesses, so the complete-caption ordering remains uncertain.

### coco_pair_007 / image 205183

Overall: **uncertain**; confidence: medium. Image checked: yes.

- object/scene: A “a blurred body of water ... a pond or lake”; B “a park or a similar outdoor setting”. Paving, grass and leaves are visible. A distinct body of water is not securely identifiable; B uses a broader, appropriately qualified setting. Directional factual support: B.
- attribute/inference: A “shadows ... sun ... from the side”; B “indicating it might be autumn”. Visible light/shade is supported, but season is only a qualified inference and not directly determined by scattered leaves. Directional factual support: uncertain.

Shared limitations: Both preserve the goose and walking posture. Brown is concentrated along the head/top of neck, rather than uniformly covering the neck.

B removes an uncertain water feature but the changes are small and include additional inference; no confident overall quality winner.

### coco_pair_008 / image 23117

Overall: **A_slightly_better**; confidence: medium. Image checked: yes.

- object/action: A “one of them is holding a pipe”; B “both are holding their umbrellas ... presenting them”. A conspicuous curved yellow pipe is visible at the right person's mouth/hand. A includes a supported distinctive object omitted by B. Directional factual support: A.
- attribute: A “umbrellas ... blue”; B “umbrellas ... appear to have a cloud-like pattern”. The umbrellas have dark outer canopies and illuminated blue undersides; the obvious cloud pattern is on the suits, not clearly on the umbrellas. Directional factual support: A.

Shared limitations: Both call the setting outdoors at night; darkness and event surroundings do not conclusively establish indoors versus outdoors.

A has a modest supported-information advantage, not proof of an entirely factual caption.

### coco_pair_009 / image 252573

Overall: **uncertain**; confidence: medium. Image checked: yes.

- object/action: A “chewing on a black cord ... a black electrical outlet”; B “resting ... head resting on a black cable ... possibly a power adapter or charger”. The cat's mouth is against a flat black object beside a cable. No wall electrical outlet is visible. The precise object identity and chewing versus resting cannot be reliably resolved. Directional factual support: uncertain.

Shared limitations: Both assert closed eyes and a collar/tag; the eye is largely obscured and the small blue item near the paw does not unequivocally establish a neck tag. The small remote at lower right is visible.

B is more hedged about device identity, but neither caption securely resolves the interaction; avoid labeling this as definite action protection.

### coco_pair_010 / image 282224

Overall: **B_slightly_better**; confidence: medium. Image checked: yes.

- object/action: A “gesturing with his hand”; B “holding a phone to his ear”. The far-left player has a hand at his ear and a small dark object consistent with a phone. B captures the distinctive apparent phone interaction, but the partly occluded device warrants moderate confidence. Directional factual support: B.
- linguistic/coverage: A “four individuals ... smiling ... conversation”; B “four individuals ... friendly interaction”. Most remaining changes restate the same visible group, rackets and court scene. Directional factual support: both.

Shared limitations: Inferred friendship, conversation and match phase cannot be proven from the image alone.

B plausibly adds a visible action/object detail; the claim is sufficiently small/occluded that human confirmation is still useful.

### coco_pair_011 / image 28377

Overall: **A_slightly_better**; confidence: high. Image checked: yes.

- attribute/binding: A “right ... bright yellow; left ... muted gray”; B “buildings with a warm, yellowish hue”. The nearest left building is gray and the right buildings yellow. A gives the more accurate color-to-building binding. Directional factual support: A.
- object: A “bell tower with a clock”; B “clock tower ... with a flag at the top”. Both identify the clock/tower; a very small flag-like feature is visible above it. This addition does not negate B's overbroad building-color claim. Directional factual support: both.

Shared limitations: The precise geographic region is inferred, not established by pixels.

A preserves a clear left/right color distinction that B collapses.

### coco_pair_012 / image 289260

Overall: **uncertain**; confidence: high. Image checked: yes.

- object/relation: A “foreground ... woven basket ... round plate with a lid”; B “table ... small bowl of fruit, a plate of eggs ... loaf of bread or a cake”. Fruit and eggs are visible on the rear-left work table, while two covered-looking dishes/bowls are on the foreground table. A's woven basket is below/rear of that table, and B ambiguously conflates table locations. Directional factual support: uncertain.
- attribute: A “well-lit”; B “dimly-lit”. The indoor scene is subdued but sufficiently lit; these are subjective and not a strong factual distinction. Directional factual support: uncertain.

Shared limitations: Both infer smoke/steam and active cooking from a blackened wall/stove region; visible soot is not evidence of currently rising smoke. The sign begins with wording ending in modern kitchens?, so quoting only modern kitchens loses the questioning/contextual wording.

Several objects exist but are relocated or overinterpreted; shared smoke hallucination is more important than the changed style.

### coco_pair_013 / image 298699

Overall: **A_slightly_better**; confidence: high. Image checked: yes.

- quantity/binding: A “two tall, modern buildings ... one ... Encore”; B “a tall ... building with a curved facade ... Encore visible on it”. Two major building volumes are visible. Encore is on the wider right building, rather than the narrow tall central tower; B risks binding the label to the wrong building. Directional factual support: A.
- object: A “traffic lights and street signs”; B “a pedestrian crossing sign; a palm tree”. Traffic lights and a palm tree are visible. A specifically identifiable pedestrian-crossing sign is not clear in the supplied view. Directional factual support: uncertain.

Shared limitations: Motion of the monorail is inferred from a still image.

A better preserves building count and label binding; B adds a real palm tree but loses the two-building distinction.

### coco_pair_014 / image 301837

Overall: **uncertain**; confidence: medium. Image checked: yes.

- coverage/object: A “stovetop ... range hood ... partially visible”; B “white ... cabinets; built-in sink ... modern faucet; plate with food”. The white cabinets, sink/faucet and left-counter fruit/food plate are visible and add supported coverage. A separate range hood is not clearly exposed. Directional factual support: B.
- object/inference: A “no visible clutter”; B “dish rack; towel ... suggesting recent use”. The small metallic counter appliance cannot confidently be called a dish rack; a hanging towel does not establish recent use. Directional factual support: A.

Shared limitations: Both overstate a refrigerator top control panel and describe the floor as dark gray; the floor is relatively light gray and the refrigerator-top row appears to be small items.

B adds useful visible content alongside additional object/use inferences. Recall and precision trade off, so overall is uncertain.

### coco_pair_015 / image 314370

Overall: **uncertain**; confidence: medium. Image checked: yes.

- object/coverage: A “more groceries ... carton of milk and a box of eggs”; B “right ... refrigerator; ... geometric design”. A small refrigerator at lower right and a geometric tile pattern are visible; B adds supported background details. Milk and egg cartons are also visible. Directional factual support: both.
- object: A “cheese ... various packaged goods”; B “bag of flour ... box of pasta”. Cheese and packaged groceries are visible. The particular flour/pasta identities are not clearly legible enough to verify. Directional factual support: A.

Shared limitations: Both mention a microwave on the left; no microwave is visible in the crop. A's description of open upper cabinets is imprecise for the visible open shelving.

B improves some background coverage but adds uncertain ingredient identities; shared microwave error remains.

### coco_pair_016 / image 317244

Overall: **B_slightly_better**; confidence: high. Image checked: yes.

- relation/coverage: A “a few other vehicles and a distant building”; B “front facing the left ... yellow line ... orange and white cylindrical objects”. The truck faces left; parking lines and orange cylindrical barriers in the left background are visible. B adds mostly supported spatial/context details. Directional factual support: B.

Shared limitations: Both describe a single large front camper window and a smaller side window. The camper has two visible front windows plus side windows, so exact window count/layout is not fully correct in either.

B offers modest additional supported coverage while retaining the same main vehicle description and shared window imprecision.

### coco_pair_017 / image 35738

Overall: **tie**; confidence: high. Image checked: yes.

- object/coverage: A “a small, rectangular object ... possibly a book or a box”; B “boy's tie is loosely tied”. A small red book-like object is visible on the TV and the tie has a loose collar arrangement. Both captions retain the same salient boy/tie/TV/lamp content. Directional factual support: both.
- linguistic: A “smiling; soft and even”; B “cheerful expression; soft”. Compatible paraphrases without a material factual change. Directional factual support: both.

Shared limitations: The floor lamp shade is gray/metallic rather than clearly white; the lighting-source interpretation is inferential.

A is longer and has a minor supported extra object; there is no clear consequential improvement or regression.

### coco_pair_018 / image 358664

Overall: **B_slightly_better**; confidence: high. Image checked: yes.

- action/relation: A “possibly a grind, with their skateboard on the edge of the ramp”; B “actively engaged in skateboarding ... concrete bowl”. The active skater is riding in the lower bowl, away from the rim; no grind on an edge is visible. B removes that unsupported action/location. Directional factual support: B.
- coverage: A “left ... walking ... holding a skateboard”; B “foreground ... holding a skateboard”. Both correctly describe the nearer standing/walking person holding a skateboard. Directional factual support: both.

Shared limitations: The shirt colors are light gray/white and the exact maneuver cannot be determined from a single image.

B preserves the two people while avoiding a meaningful action and spatial hallucination.

### coco_pair_019 / image 3716

Overall: **uncertain**; confidence: medium. Image checked: yes.

- posture/relation: A “legs crossed”; B “edge of the bed, with one leg crossed over the other”. One bent, raised knee is visible; the other leg and precise edge/crossed relation are occluded, so neither stronger posture claim is established. Directional factual support: uncertain.
- linguistic: A “light-colored bedding ... blue blanket”; B “light-colored blanket ... blue blanket ... foreground”. Both capture the same visible furnishings and laptop activity. Directional factual support: both.

Shared limitations: Both call the map the Americas, but the displayed map is principally South America. Bed versus sofa/daybed is not unambiguous from the visible upholstered backing. A stove is not clearly identifiable in the left kitchen area.

The changed posture language is not verifiable and the main content/shared errors are unchanged.

### coco_pair_020 / image 431570

Overall: **B_slightly_better**; confidence: medium. Image checked: yes.

- text/object: A “logos, including Rizla and Gulf Oil”; B “sponsor logos and the number 14”. Rizla and a small 14 are visible. The oil logo reads like Q8 Oils rather than Gulf Oil. B avoids the wrong brand and includes a visible number, although prominently displayed exaggerates its size. Directional factual support: B.
- scene: A “trees and a clear sky”; B “grass ... trees ... motion blur”. Trees and grass are visible; no substantial clear sky is shown, and motion blur/speed is not clearly established. Directional factual support: uncertain.

Shared limitations: Both say the rider is leaning into a turn; the motorcycle appears relatively upright in this side view, and a turn/high speed cannot be securely inferred.

B avoids a checkable wrong logo while retaining unsupported action interpretation; the preference is limited.

### coco_pair_021 / image 497438

Overall: **A_slightly_better**; confidence: high. Image checked: yes.

- attribute/object: A “child on the right ... pink outfit”; B “headband with cat ears”. The right child has a pointed party-hat-like piece/hat above the head. A cat-ear headband is not supported by the visible shape; A avoids this specific misidentification. Directional factual support: A.
- gaze/relation: A “looking towards the left”; B “child in the center ... looking at the camera”. The center child's gaze is directed toward the left of the frame, rather than clearly at the camera. Directional factual support: A.

Shared limitations: Expressions of joy and active celebration are inferred; several children have neutral expressions. The image is strongly color-cast and age/style interpretations should remain cautious.

A avoids B's new fine-grained hat and gaze claims while preserving the same central children/party scene.

### coco_pair_022 / image 49984

Overall: **uncertain**; confidence: medium. Image checked: yes.

- attribute/object: A “light wooden countertop ... soap dispenser”; B “light-colored countertop ... hairdryer”. The counter is light-colored but not visibly wooden. A corded wall-side device is compatible with a hairdryer, although small and partially occluded; soap/pump bottles are also present. Directional factual support: B.
- coverage: A “mirror ... above the vanity”; B “no mirror mention”. A large mirror is clearly visible; A retains this salient fixture that B omits. Directional factual support: A.

Shared limitations: Both place the white towel on the toilet tank, but the white hand towel hangs on the wall rack beside the beige towel.

B improves uncertain material wording, while A includes the visible mirror. The precision/coverage tradeoff does not yield a clear overall preference.

### coco_pair_023 / image 535602

Overall: **tie**; confidence: high. Image checked: yes.

- object/coverage: A “clear plastic bag with a red object inside”; B “partially used roll of paper towels ... small plastic container”. A paper-towel roll is clearly visible at left, and transparent packaging/food containers are also on the floor. These are different mostly compatible secondary details. Directional factual support: both.
- linguistic: A “single slice of pizza”; B “small slice of pizza”. Both capture the remaining irregular pizza portion in the open box; precise slice geometry is not important to the changed claim. Directional factual support: both.

Shared limitations: Both infer a casual/lived-in scene, which is descriptive rather than a verified activity or time of day.

The cat, open pizza box, laptop/map and clutter are maintained; no clear factual or salient-content loss is evident.

### coco_pair_024 / image 537574

Overall: **uncertain**; confidence: high. Image checked: yes.

- attribute/binding: A “wearing a black harness”; B “a black collar ... leash ... attached to the collar”. A black neck collar with an attached leash is visible, rather than a body harness. B corrects a checkable equipment identification. Directional factual support: B.
- text/relation: A “digital display showing the date and time”; B “date 14 ... 12:05 PM ... 32°F ... leash is secured to the car's interior”. Small display text is not fully reliable; the leading display symbol could be a track indicator, not a date. The far end of the leash is not visible, so its securing point is unverified. Directional factual support: uncertain.

Shared limitations: Both place the dog near the driver's seat despite the dog being on the passenger-side floor/seat area; exact seating assignment is not fully shown. A's claim that harness and bottle are the only objects contradicts the visible dashboard/controls and its own preceding text.

B has a clear collar-over-harness factual improvement, but extra OCR/security assertions prevent treating the full-caption difference as unambiguously beneficial.

### coco_pair_025 / image 578314

Overall: **B_slightly_better**; confidence: high. Image checked: yes.

- object/existence/coverage: A “no visible clutter or additional objects”; B “a trash bin is visible to the left of the toilet”. A lidded trash bin is clearly visible on the left. B adds the real object and avoids A's overbroad no-additional-objects statement. Directional factual support: B.

Shared limitations: Both claim a flush handle on the left; the tank instead shows a top-right rectangular flush panel. Both place the brush holder on the wall, but it stands on the floor at right.

B improves a visible object omission while retaining important shared accessory-location and flush-control errors.

## Reliability limits

No inter-annotator agreement, Cohen kappa, independent human reliability, swapped-order consistency or three-run judge stability can be calculated from this single review. A/B randomization masks the method identity but cannot remove single-reviewer bias. No labels, thresholds, metrics or generation outputs were changed. All 26 records, including ties and uncertain cases, remain in the output.

Provenance: input SHA256 `8fab9f0f5446a4c494654601b432b36c2476e6ea10f54cbc18ee0322b86eb175`; structured JSONL SHA256 `c32db12997b9ab2e7acf447d24a9191b34b31eb04dc15ba7506fb43284235abb`. Image SHA256 values are stored per record. Two initially attempted wrong image IDs were not found; these were not reviewed or substituted, and the correct case image paths were subsequently opened for every case.
