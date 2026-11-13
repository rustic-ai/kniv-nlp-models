"""M3 (v8 plan r2): end-to-end negation and hedging set.

    uv run python -m v6.eval.m3_negation build
    uv run python -m v6.eval.m3_negation predict --set hand|pairs [--slice 0:500]
    uv run python -m v6.eval.m3_negation score

Two parts, both scored the way the plan asks: is the fact returned, and with
the right polarity and modality.

- ``hand``: about 150 statements written for this set, outside business text
  (M4 has that): general relations from the label bank (employed by, resides
  in, founded, spouse, member of, born in, educated at, owns, travels to,
  speaks, ...), in the forms the outside evaluation and the P5 hand set
  failed on (did not, never, no longer, denied, plans to, might, rumoured,
  expected to, reported, questions, conditionals, wants to), with asserted
  controls (about a third), a fifth of them conversation turns where the
  speaker is an entity. Statements come in groups: one fact, an asserted
  form and its rewrites, so a group is the unit of the bootstrap.
- ``pairs``: the short-text minimal pairs whose base text the attribute
  retrain held out (``negation_pilot.heldout_base``), each rewrite (negated,
  reported, planned, desired, questioned, hypothetical) and its base fact's
  evidence sentence as the asserted control (``negation_pilot.load_pairs``).

Every statement is run with gold entities (the relation step alone) and end
to end (entities found with the schema's types). Per expected fact:

- **returned**: a fact above the deployed bar with its relation and
  arguments (either direction for symmetric relations);
- **right when returned**: polarity and modality, the attribute head's
  argmax, are the expected ones (several modalities accepted where the text
  allows more than one reading);
- **returned and right**: both, over all expected facts;
- **forced**: the attribute head on the expected fact wherever it ranks in
  its pair's top 4, whether or not it passes the bar.

``predict`` runs the relation step once (v7b8ev) and reads two attribute
heads on its features: v7.1's and ``-attrv8b``'s (the retrain froze the
relation path, so its relation outputs are the same; checked on the first
statements). Detectable differences (80% power, 5% level) come from the
paired bootstrap standard error of the two heads' difference.
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from ..config import DATA_DIR, RUNS_DIR

DATA = DATA_DIR / "v6-eval" / "m3_negation.json"
OUT = RUNS_DIR / "m3-negation"
SHIFT = 20.0
RUN = "ultrabert-380m-v7b8ev"
HEADS = {"v7.1": RUN, "attrv8b": "ultrabert-380m-v7b8ev-attrv8b"}

REL_NAMES = ["employed by", "resides in", "founded", "spouse", "located in", "member of", "born in",
             "educated at", "owns", "travels to", "friend of", "sibling of", "likes", "speaks",
             "member of sports team"]
SYMMETRIC = {"spouse", "friend of", "sibling of"}
TYPE_NAMES = ["person", "organization", "location", "language", "miscellaneous"]
ABBR = {"per": "person", "org": "organization", "loc": "location", "lang": "language", "misc": "miscellaneous"}
MODS = {"asserted", "hypothetical", "planned", "desired", "questioned", "reported"}

# (group, id suffix, forms, text, entities "type:Name|alias; ...", facts "H > relation > T [attrs]; ...")
# Conversation: turns "speaker| text || speaker| text"; a speaker is the argument "@speaker".
# attrs: "neg", "opt", and accepted modalities joined by "/" (default asserted).
SPEC = [
    # ── employed by ─────────────────────────────────────────────────
    ("emp-01", "a", "asserted", "Lena Hartmann works for Corvel Analytics.",
     "per:Lena Hartmann; org:Corvel Analytics", "Lena Hartmann > employed by > Corvel Analytics"),
    ("emp-01", "b", "not", "Lena Hartmann does not work for Corvel Analytics.",
     "per:Lena Hartmann; org:Corvel Analytics", "Lena Hartmann > employed by > Corvel Analytics [neg]"),
    ("emp-01", "c", "nolonger", "Lena Hartmann no longer works for Corvel Analytics.",
     "per:Lena Hartmann; org:Corvel Analytics", "Lena Hartmann > employed by > Corvel Analytics [neg]"),
    ("emp-01", "d", "might", "Lena Hartmann might start working for Corvel Analytics in the autumn.",
     "per:Lena Hartmann; org:Corvel Analytics",
     "Lena Hartmann > employed by > Corvel Analytics [hypothetical/planned]"),
    ("emp-02", "a", "asserted", "Tomas Reyes has been an engineer at Silverpine Labs since 2019.",
     "per:Tomas Reyes; org:Silverpine Labs", "Tomas Reyes > employed by > Silverpine Labs"),
    ("emp-02", "b", "denied", "Tomas Reyes denied that he works for Silverpine Labs.",
     "per:Tomas Reyes|he; org:Silverpine Labs", "Tomas Reyes > employed by > Silverpine Labs [neg reported/asserted]"),
    ("emp-02", "c", "expected", "Tomas Reyes is expected to join Silverpine Labs next month.",
     "per:Tomas Reyes; org:Silverpine Labs", "Tomas Reyes > employed by > Silverpine Labs [planned/hypothetical]"),
    ("emp-03", "a", "asserted conv", "user| I work for a small studio called Fernhill Games.",
     "org:Fernhill Games", "@user > employed by > Fernhill Games"),
    ("emp-03", "b", "nolonger conv", "user| I used to work for Fernhill Games, but not anymore.",
     "org:Fernhill Games", "@user > employed by > Fernhill Games [neg]"),
    ("emp-03", "c", "desired conv", "user| Is Fernhill Games still hiring? I would love to work for them.",
     "org:Fernhill Games|them", "@user > employed by > Fernhill Games [desired]"),
    ("emp-03", "d", "conditional conv",
     "user| If the interview goes well, I will be working for Fernhill Games.",
     "org:Fernhill Games", "@user > employed by > Fernhill Games [hypothetical/planned]"),
    ("emp-04", "a", "asserted", "Ines Moreau joined the Orchard Street Clinic as a nurse in 2020.",
     "per:Ines Moreau; org:Orchard Street Clinic", "Ines Moreau > employed by > Orchard Street Clinic"),
    ("emp-04", "b", "question", "Was Ines Moreau ever employed by the Orchard Street Clinic?",
     "per:Ines Moreau; org:Orchard Street Clinic",
     "Ines Moreau > employed by > Orchard Street Clinic [questioned]"),
    ("emp-04", "c", "reported", "Ines Moreau was reported to have taken a job at the Orchard Street Clinic.",
     "per:Ines Moreau; org:Orchard Street Clinic", "Ines Moreau > employed by > Orchard Street Clinic [reported]"),
    ("emp-05", "a", "asserted", "Rafael Okafor teaches mathematics at Brookmere Academy.",
     "per:Rafael Okafor; org:Brookmere Academy", "Rafael Okafor > employed by > Brookmere Academy"),
    ("emp-05", "b", "never", "Rafael Okafor never taught at Brookmere Academy.",
     "per:Rafael Okafor; org:Brookmere Academy", "Rafael Okafor > employed by > Brookmere Academy [neg]"),
    ("emp-05", "c", "planned", "Rafael Okafor plans to apply for a teaching post at Brookmere Academy.",
     "per:Rafael Okafor; org:Brookmere Academy",
     "Rafael Okafor > employed by > Brookmere Academy [planned/desired]"),
    # ── resides in ──────────────────────────────────────────────────
    ("res-01", "a", "asserted", "Maja Kowalczyk lives in Gdansk.",
     "per:Maja Kowalczyk; loc:Gdansk", "Maja Kowalczyk > resides in > Gdansk"),
    ("res-01", "b", "not", "Maja Kowalczyk does not live in Gdansk; she only visits her parents there.",
     "per:Maja Kowalczyk|she; loc:Gdansk",
     "Maja Kowalczyk > resides in > Gdansk [neg]; Maja Kowalczyk > travels to > Gdansk [opt]"),
    ("res-01", "c", "planned", "Maja Kowalczyk is moving to Gdansk next year.",
     "per:Maja Kowalczyk; loc:Gdansk", "Maja Kowalczyk > resides in > Gdansk [planned]"),
    ("res-01", "d", "might", "Maja Kowalczyk might live in Gdansk, but nobody is sure.",
     "per:Maja Kowalczyk; loc:Gdansk", "Maja Kowalczyk > resides in > Gdansk [hypothetical]"),
    ("res-02", "a", "asserted conv", "user| I live in Porto with my two cats.",
     "loc:Porto", "@user > resides in > Porto"),
    ("res-02", "b", "nolonger conv", "user| I do not live in Porto anymore, I moved away in March.",
     "loc:Porto", "@user > resides in > Porto [neg]"),
    ("res-02", "c", "desired conv", "user| I am thinking about moving to Porto.",
     "loc:Porto", "@user > resides in > Porto [desired/planned/hypothetical]"),
    ("res-02", "d", "question conv", "assistant| Do you live in Porto? || user| Why do you ask?",
     "loc:Porto", "@user > resides in > Porto [questioned]"),
    ("res-03", "a", "asserted", "Since the war, Anton Petrov has lived in Tbilisi.",
     "per:Anton Petrov; loc:Tbilisi", "Anton Petrov > resides in > Tbilisi"),
    ("res-03", "b", "rumoured", "Anton Petrov is rumoured to be living in Tbilisi under another name.",
     "per:Anton Petrov; loc:Tbilisi", "Anton Petrov > resides in > Tbilisi [reported]"),
    ("res-03", "c", "never", "Anton Petrov has never lived in Tbilisi.",
     "per:Anton Petrov; loc:Tbilisi", "Anton Petrov > resides in > Tbilisi [neg]"),
    ("res-04", "a", "asserted", "Grace Nwosu has settled in Kampala with her family.",
     "per:Grace Nwosu; loc:Kampala", "Grace Nwosu > resides in > Kampala"),
    ("res-04", "b", "desired", "Grace Nwosu wants to settle in Kampala.",
     "per:Grace Nwosu; loc:Kampala", "Grace Nwosu > resides in > Kampala [desired]"),
    ("res-04", "c", "conditional", "If her visa is approved, Grace Nwosu will settle in Kampala.",
     "per:Grace Nwosu|her; loc:Kampala", "Grace Nwosu > resides in > Kampala [hypothetical/planned]"),
    ("res-05", "a", "asserted conv", "user| My sister and I both live in Leeds.",
     "loc:Leeds", "@user > resides in > Leeds"),
    ("res-05", "b", "planned conv", "user| Next spring I am moving to Leeds for work.",
     "loc:Leeds", "@user > resides in > Leeds [planned]"),
    # ── founded ─────────────────────────────────────────────────────
    ("fnd-01", "a", "asserted", "Hana Sato founded the Bluefin Rowing Club in 2004.",
     "per:Hana Sato; org:Bluefin Rowing Club", "Hana Sato > founded > Bluefin Rowing Club"),
    ("fnd-01", "b", "not", "Hana Sato did not found the Bluefin Rowing Club, although she is often "
     "credited with it.",
     "per:Hana Sato|she; org:Bluefin Rowing Club|it", "Hana Sato > founded > Bluefin Rowing Club [neg]"),
    ("fnd-01", "c", "reported", "Hana Sato is said to have founded the Bluefin Rowing Club.",
     "per:Hana Sato; org:Bluefin Rowing Club", "Hana Sato > founded > Bluefin Rowing Club [reported]"),
    ("fnd-02", "a", "asserted", "The Tallgrass Theatre Company was founded by Owen Brandt.",
     "org:Tallgrass Theatre Company; per:Owen Brandt", "Owen Brandt > founded > Tallgrass Theatre Company"),
    ("fnd-02", "b", "question", "Did Owen Brandt found the Tallgrass Theatre Company?",
     "org:Tallgrass Theatre Company; per:Owen Brandt",
     "Owen Brandt > founded > Tallgrass Theatre Company [questioned]"),
    ("fnd-02", "c", "denied", "Owen Brandt denied founding the Tallgrass Theatre Company.",
     "org:Tallgrass Theatre Company; per:Owen Brandt",
     "Owen Brandt > founded > Tallgrass Theatre Company [neg reported/asserted]"),
    ("fnd-03", "a", "asserted", "Amira Haddad started the Cedar Lane Food Bank in her garage.",
     "per:Amira Haddad|her; org:Cedar Lane Food Bank", "Amira Haddad > founded > Cedar Lane Food Bank"),
    ("fnd-03", "b", "planned", "Amira Haddad plans to found a charity called the Cedar Lane Food Bank.",
     "per:Amira Haddad; org:Cedar Lane Food Bank", "Amira Haddad > founded > Cedar Lane Food Bank [planned]"),
    ("fnd-03", "c", "desired", "Amira Haddad hopes to start the Cedar Lane Food Bank one day.",
     "per:Amira Haddad; org:Cedar Lane Food Bank", "Amira Haddad > founded > Cedar Lane Food Bank [desired]"),
    ("fnd-04", "a", "asserted conv", "user| I founded a chess club, the Riverside Knights, at my school.",
     "org:Riverside Knights", "@user > founded > Riverside Knights; @user > member of > Riverside Knights [opt]"),
    ("fnd-04", "b", "not conv", "user| I did not found the Riverside Knights, I just joined it later.",
     "org:Riverside Knights|it", "@user > founded > Riverside Knights [neg]; @user > member of > Riverside Knights"),
    ("fnd-04", "c", "desired conv", "user| I would like to start a chess club called the Riverside Knights.",
     "org:Riverside Knights", "@user > founded > Riverside Knights [desired]"),
    # ── spouse ──────────────────────────────────────────────────────
    ("sp-01", "a", "asserted", "Daniel Fischer is married to Clara Weiss.",
     "per:Daniel Fischer; per:Clara Weiss", "Daniel Fischer > spouse > Clara Weiss"),
    ("sp-01", "b", "not", "Daniel Fischer is not married to Clara Weiss; they are business partners.",
     "per:Daniel Fischer; per:Clara Weiss", "Daniel Fischer > spouse > Clara Weiss [neg]"),
    ("sp-01", "c", "planned", "Daniel Fischer and Clara Weiss are engaged and plan to marry in June.",
     "per:Daniel Fischer; per:Clara Weiss", "Daniel Fischer > spouse > Clara Weiss [planned]"),
    ("sp-01", "d", "rumoured", "Rumours say Daniel Fischer secretly married Clara Weiss.",
     "per:Daniel Fischer; per:Clara Weiss", "Daniel Fischer > spouse > Clara Weiss [reported]"),
    ("sp-02", "a", "asserted", "Priyanka Shah's husband, Arjun Mehta, is a pilot.",
     "per:Priyanka Shah; per:Arjun Mehta", "Priyanka Shah > spouse > Arjun Mehta"),
    ("sp-02", "b", "nolonger", "Priyanka Shah divorced Arjun Mehta in 2015.",
     "per:Priyanka Shah; per:Arjun Mehta", "Priyanka Shah > spouse > Arjun Mehta [neg]"),
    ("sp-02", "c", "question", "Is Priyanka Shah married to Arjun Mehta?",
     "per:Priyanka Shah; per:Arjun Mehta", "Priyanka Shah > spouse > Arjun Mehta [questioned]"),
    ("sp-03", "a", "asserted conv", "user| My wife Elena and I just celebrated our tenth anniversary.",
     "per:Elena", "@user > spouse > Elena"),
    ("sp-03", "b", "not conv", "user| Elena and I are not married, we just live together.",
     "per:Elena", "@user > spouse > Elena [neg]"),
    ("sp-03", "c", "desired conv", "user| I want to marry Elena next summer.",
     "per:Elena", "@user > spouse > Elena [desired/planned]"),
    ("sp-04", "a", "asserted", "Kwame Asante married Abena Owusu in Kumasi.",
     "per:Kwame Asante; per:Abena Owusu; loc:Kumasi", "Kwame Asante > spouse > Abena Owusu"),
    ("sp-04", "b", "might", "Kwame Asante might marry Abena Owusu if her family agrees.",
     "per:Kwame Asante; per:Abena Owusu|her", "Kwame Asante > spouse > Abena Owusu [hypothetical]"),
    ("sp-04", "c", "never", "Kwame Asante never married Abena Owusu.",
     "per:Kwame Asante; per:Abena Owusu", "Kwame Asante > spouse > Abena Owusu [neg]"),
    # ── located in ──────────────────────────────────────────────────
    ("loc-01", "a", "asserted", "The Moorfield Observatory is located in Cumbria.",
     "loc:Moorfield Observatory; loc:Cumbria", "Moorfield Observatory > located in > Cumbria"),
    ("loc-01", "b", "not", "The Moorfield Observatory is not in Cumbria; it is in Northumberland.",
     "loc:Moorfield Observatory|it; loc:Cumbria; loc:Northumberland",
     "Moorfield Observatory > located in > Cumbria [neg]; Moorfield Observatory > located in > Northumberland"),
    ("loc-01", "c", "planned", "The Moorfield Observatory will be moved to Cumbria next year.",
     "loc:Moorfield Observatory; loc:Cumbria", "Moorfield Observatory > located in > Cumbria [planned]"),
    ("loc-02", "a", "asserted", "Lake Varga lies in eastern Hungary.",
     "loc:Lake Varga; loc:Hungary", "Lake Varga > located in > Hungary"),
    ("loc-02", "b", "question", "Is Lake Varga in Hungary?",
     "loc:Lake Varga; loc:Hungary", "Lake Varga > located in > Hungary [questioned]"),
    ("loc-02", "c", "nolonger", "Lake Varga is no longer in Hungary since the border moved in 1947.",
     "loc:Lake Varga; loc:Hungary", "Lake Varga > located in > Hungary [neg]"),
    ("loc-03", "a", "asserted", "The Ashcombe Library sits in the centre of Bristol.",
     "org:Ashcombe Library; loc:Bristol", "Ashcombe Library > located in > Bristol"),
    ("loc-03", "b", "expected", "The Ashcombe Library is expected to open in Bristol in 2027.",
     "org:Ashcombe Library; loc:Bristol", "Ashcombe Library > located in > Bristol [planned/hypothetical]"),
    ("loc-03", "c", "reported", "According to the council, the Ashcombe Library is in Bristol.",
     "org:Ashcombe Library; loc:Bristol", "Ashcombe Library > located in > Bristol [reported]"),
    ("loc-04", "a", "asserted", "Kestrel Island is part of the Hebrides.",
     "loc:Kestrel Island; loc:Hebrides", "Kestrel Island > located in > Hebrides"),
    ("loc-04", "b", "might", "Kestrel Island may lie within the Hebrides; the old maps disagree.",
     "loc:Kestrel Island; loc:Hebrides", "Kestrel Island > located in > Hebrides [hypothetical]"),
    # ── member of ───────────────────────────────────────────────────
    ("mem-01", "a", "asserted", "Sofia Lindqvist is a member of the Northgate Choir.",
     "per:Sofia Lindqvist; org:Northgate Choir", "Sofia Lindqvist > member of > Northgate Choir"),
    ("mem-01", "b", "nolonger", "Sofia Lindqvist left the Northgate Choir in 2021.",
     "per:Sofia Lindqvist; org:Northgate Choir", "Sofia Lindqvist > member of > Northgate Choir [neg]"),
    ("mem-01", "c", "desired", "Sofia Lindqvist wants to join the Northgate Choir.",
     "per:Sofia Lindqvist; org:Northgate Choir", "Sofia Lindqvist > member of > Northgate Choir [desired]"),
    ("mem-01", "d", "denied", "Sofia Lindqvist denied being a member of the Northgate Choir.",
     "per:Sofia Lindqvist; org:Northgate Choir", "Sofia Lindqvist > member of > Northgate Choir [neg reported/asserted]"),
    ("mem-02", "a", "asserted", "Felix Baumann belongs to the Alpine Rescue League.",
     "per:Felix Baumann; org:Alpine Rescue League", "Felix Baumann > member of > Alpine Rescue League"),
    ("mem-02", "b", "never", "Felix Baumann has never been a member of the Alpine Rescue League.",
     "per:Felix Baumann; org:Alpine Rescue League", "Felix Baumann > member of > Alpine Rescue League [neg]"),
    ("mem-02", "c", "rumoured", "Felix Baumann is rumoured to have joined the Alpine Rescue League.",
     "per:Felix Baumann; org:Alpine Rescue League", "Felix Baumann > member of > Alpine Rescue League [reported]"),
    ("mem-03", "a", "asserted conv", "user| I am in a book club called the Tuesday Readers.",
     "org:Tuesday Readers", "@user > member of > Tuesday Readers"),
    ("mem-03", "b", "nolonger conv", "user| I quit the Tuesday Readers last month.",
     "org:Tuesday Readers", "@user > member of > Tuesday Readers [neg]"),
    ("mem-03", "c", "question conv", "user| Should I join the Tuesday Readers?",
     "org:Tuesday Readers", "@user > member of > Tuesday Readers [questioned/desired]"),
    ("mem-04", "a", "asserted", "Yusuf Demir is a member of the Green Valley Party.",
     "per:Yusuf Demir; org:Green Valley Party", "Yusuf Demir > member of > Green Valley Party"),
    ("mem-04", "b", "expected", "Yusuf Demir is expected to join the Green Valley Party before the election.",
     "per:Yusuf Demir; org:Green Valley Party",
     "Yusuf Demir > member of > Green Valley Party [planned/hypothetical]"),
    ("mem-04", "c", "not", "Yusuf Demir said he did not join the Green Valley Party.",
     "per:Yusuf Demir|he; org:Green Valley Party", "Yusuf Demir > member of > Green Valley Party [neg]"),
    # ── born in ─────────────────────────────────────────────────────
    ("brn-01", "a", "asserted", "Lucia Romano was born in Naples.",
     "per:Lucia Romano; loc:Naples", "Lucia Romano > born in > Naples"),
    ("brn-01", "b", "not", "Lucia Romano was not born in Naples, but she grew up there.",
     "per:Lucia Romano|she; loc:Naples",
     "Lucia Romano > born in > Naples [neg]; Lucia Romano > resides in > Naples [opt]"),
    ("brn-01", "c", "reported", "Lucia Romano was reportedly born in Naples.",
     "per:Lucia Romano; loc:Naples", "Lucia Romano > born in > Naples [reported]"),
    ("brn-02", "a", "asserted", "Born in Accra, Kofi Mensah moved abroad at nineteen.",
     "per:Kofi Mensah; loc:Accra", "Kofi Mensah > born in > Accra"),
    ("brn-02", "b", "question", "Was Kofi Mensah born in Accra?",
     "per:Kofi Mensah; loc:Accra", "Kofi Mensah > born in > Accra [questioned]"),
    ("brn-02", "c", "might", "Kofi Mensah may have been born in Accra; the records are incomplete.",
     "per:Kofi Mensah; loc:Accra", "Kofi Mensah > born in > Accra [hypothetical]"),
    ("brn-03", "a", "asserted conv", "user| I was born in Cork, but I have lived in Dublin for years.",
     "loc:Cork; loc:Dublin", "@user > born in > Cork; @user > resides in > Dublin"),
    ("brn-03", "b", "not conv", "user| I was not born in Cork, my brother was.",
     "loc:Cork", "@user > born in > Cork [neg]"),
    # ── educated at ─────────────────────────────────────────────────
    ("edu-01", "a", "asserted", "Nadia Petrova studied physics at Uppsala University.",
     "per:Nadia Petrova; org:Uppsala University", "Nadia Petrova > educated at > Uppsala University"),
    ("edu-01", "b", "never", "Nadia Petrova never studied at Uppsala University.",
     "per:Nadia Petrova; org:Uppsala University", "Nadia Petrova > educated at > Uppsala University [neg]"),
    ("edu-01", "c", "planned", "Nadia Petrova plans to study at Uppsala University.",
     "per:Nadia Petrova; org:Uppsala University", "Nadia Petrova > educated at > Uppsala University [planned]"),
    ("edu-01", "d", "conditional", "Nadia Petrova might study at Uppsala University if she gets the scholarship.",
     "per:Nadia Petrova|she; org:Uppsala University",
     "Nadia Petrova > educated at > Uppsala University [hypothetical]"),
    ("edu-02", "a", "asserted", "Marcus Bell graduated from Hollins College in 2012.",
     "per:Marcus Bell; org:Hollins College", "Marcus Bell > educated at > Hollins College"),
    ("edu-02", "b", "denied", "Marcus Bell denied that he attended Hollins College.",
     "per:Marcus Bell|he; org:Hollins College", "Marcus Bell > educated at > Hollins College [neg reported/asserted]"),
    ("edu-02", "c", "reported", "Marcus Bell is said to have studied at Hollins College.",
     "per:Marcus Bell; org:Hollins College", "Marcus Bell > educated at > Hollins College [reported]"),
    ("edu-03", "a", "asserted conv", "user| I studied architecture at Delft University of Technology.",
     "org:Delft University of Technology", "@user > educated at > Delft University of Technology"),
    ("edu-03", "b", "desired conv", "user| I would love to study at Delft University of Technology.",
     "org:Delft University of Technology", "@user > educated at > Delft University of Technology [desired]"),
    ("edu-03", "c", "question conv", "assistant| Did you study at Delft University of Technology? || user| Good question.",
     "org:Delft University of Technology", "@user > educated at > Delft University of Technology [questioned]"),
    # ── owns ────────────────────────────────────────────────────────
    ("own-01", "a", "asserted", "Oliver Grant owns a sailboat named the Marigold.",
     "per:Oliver Grant; misc:Marigold", "Oliver Grant > owns > Marigold"),
    ("own-01", "b", "nolonger", "Oliver Grant sold the Marigold last summer.",
     "per:Oliver Grant; misc:Marigold", "Oliver Grant > owns > Marigold [neg]"),
    ("own-01", "c", "desired", "Oliver Grant wants to buy the Marigold.",
     "per:Oliver Grant; misc:Marigold", "Oliver Grant > owns > Marigold [desired]"),
    ("own-01", "d", "question", "Does Oliver Grant still own the Marigold?",
     "per:Oliver Grant; misc:Marigold", "Oliver Grant > owns > Marigold [questioned]"),
    ("own-02", "a", "asserted", "Rosa Delgado owns the Blue Heron Cafe in Seville.",
     "per:Rosa Delgado; org:Blue Heron Cafe; loc:Seville",
     "Rosa Delgado > owns > Blue Heron Cafe; Blue Heron Cafe > located in > Seville [opt]"),
    ("own-02", "b", "not", "Rosa Delgado does not own the Blue Heron Cafe; she manages it for her uncle.",
     "per:Rosa Delgado|she; org:Blue Heron Cafe|it", "Rosa Delgado > owns > Blue Heron Cafe [neg]"),
    ("own-02", "c", "rumoured", "Rosa Delgado is rumoured to own the Blue Heron Cafe.",
     "per:Rosa Delgado; org:Blue Heron Cafe", "Rosa Delgado > owns > Blue Heron Cafe [reported]"),
    ("own-03", "a", "asserted conv", "user| I own a cottage called Wrenfield near the coast.",
     "loc:Wrenfield", "@user > owns > Wrenfield"),
    ("own-03", "b", "conditional conv", "user| If I win the lottery, I will buy Wrenfield.",
     "loc:Wrenfield", "@user > owns > Wrenfield [hypothetical]"),
    # ── travels to ──────────────────────────────────────────────────
    ("trv-01", "a", "asserted", "Ellen Park flew to Nairobi on Monday.",
     "per:Ellen Park; loc:Nairobi", "Ellen Park > travels to > Nairobi"),
    ("trv-01", "b", "not", "Ellen Park did not travel to Nairobi after all; the trip was cancelled.",
     "per:Ellen Park; loc:Nairobi", "Ellen Park > travels to > Nairobi [neg]"),
    ("trv-01", "c", "planned", "Ellen Park will travel to Nairobi next week.",
     "per:Ellen Park; loc:Nairobi", "Ellen Park > travels to > Nairobi [planned]"),
    ("trv-01", "d", "might", "Ellen Park might travel to Nairobi in June.",
     "per:Ellen Park; loc:Nairobi", "Ellen Park > travels to > Nairobi [hypothetical]"),
    ("trv-02", "a", "asserted conv", "user| I went to Montreal last weekend.",
     "loc:Montreal", "@user > travels to > Montreal"),
    ("trv-02", "b", "never conv", "user| I never went to Montreal, even though I meant to.",
     "loc:Montreal", "@user > travels to > Montreal [neg]"),
    ("trv-02", "c", "desired conv", "user| I really want to go to Montreal.",
     "loc:Montreal", "@user > travels to > Montreal [desired]"),
    ("trv-02", "d", "question conv", "assistant| Are you travelling to Montreal this year? || user| Let me check my calendar.",
     "loc:Montreal", "@user > travels to > Montreal [questioned]"),
    # ── friend of / sibling of ──────────────────────────────────────
    ("frd-01", "a", "asserted", "Jonas Berg is a close friend of Mila Novak.",
     "per:Jonas Berg; per:Mila Novak", "Jonas Berg > friend of > Mila Novak"),
    ("frd-01", "b", "nolonger", "Jonas Berg and Mila Novak are no longer friends.",
     "per:Jonas Berg; per:Mila Novak", "Jonas Berg > friend of > Mila Novak [neg]"),
    ("frd-01", "c", "denied", "Jonas Berg denied being friends with Mila Novak.",
     "per:Jonas Berg; per:Mila Novak", "Jonas Berg > friend of > Mila Novak [neg reported/asserted]"),
    ("frd-02", "a", "asserted conv", "user| My best friend, Tariq, is visiting me tomorrow.",
     "per:Tariq", "@user > friend of > Tariq"),
    ("frd-02", "b", "nolonger conv", "user| Tariq and I stopped being friends years ago.",
     "per:Tariq", "@user > friend of > Tariq [neg]"),
    ("sib-01", "a", "asserted", "Hugo Laurent is the brother of Camille Laurent.",
     "per:Hugo Laurent; per:Camille Laurent", "Hugo Laurent > sibling of > Camille Laurent"),
    ("sib-01", "b", "not", "Hugo Laurent is not related to Camille Laurent, despite the shared surname.",
     "per:Hugo Laurent; per:Camille Laurent", "Hugo Laurent > sibling of > Camille Laurent [neg]"),
    ("sib-01", "c", "question", "Is Hugo Laurent the brother of Camille Laurent?",
     "per:Hugo Laurent; per:Camille Laurent", "Hugo Laurent > sibling of > Camille Laurent [questioned]"),
    # ── likes / speaks ──────────────────────────────────────────────
    ("lik-01", "a", "asserted conv", "user| I really like jazz.", "misc:jazz", "@user > likes > jazz"),
    ("lik-01", "b", "not conv", "user| I do not like jazz at all.", "misc:jazz", "@user > likes > jazz [neg]"),
    ("lik-01", "c", "nolonger conv", "user| I used to like jazz, but not anymore.",
     "misc:jazz", "@user > likes > jazz [neg]"),
    ("lik-02", "a", "asserted", "Beatrix Cole likes Thai food.",
     "per:Beatrix Cole; misc:Thai food", "Beatrix Cole > likes > Thai food"),
    ("lik-02", "b", "not", "Beatrix Cole does not like Thai food.",
     "per:Beatrix Cole; misc:Thai food", "Beatrix Cole > likes > Thai food [neg]"),
    ("spk-01", "a", "asserted", "Ana Ribeiro speaks Portuguese and Italian.",
     "per:Ana Ribeiro; lang:Portuguese; lang:Italian",
     "Ana Ribeiro > speaks > Portuguese; Ana Ribeiro > speaks > Italian"),
    ("spk-01", "b", "not", "Ana Ribeiro does not speak Italian.",
     "per:Ana Ribeiro; lang:Italian", "Ana Ribeiro > speaks > Italian [neg]"),
    ("spk-01", "c", "desired", "Ana Ribeiro is taking lessons and hopes to speak Italian fluently.",
     "per:Ana Ribeiro; lang:Italian", "Ana Ribeiro > speaks > Italian [desired]"),
    ("spk-02", "a", "asserted conv", "user| I speak fluent Japanese.", "lang:Japanese", "@user > speaks > Japanese"),
    ("spk-02", "b", "not conv", "user| I cannot speak Japanese, sadly.",
     "lang:Japanese", "@user > speaks > Japanese [neg]"),
    ("spk-02", "c", "might conv", "user| I might take lessons so I can speak Japanese on my trip.",
     "lang:Japanese", "@user > speaks > Japanese [hypothetical/desired/planned]"),
    # ── member of sports team ───────────────────────────────────────
    ("spt-01", "a", "asserted", "Diego Ferreira plays for Atletico Ribeira.",
     "per:Diego Ferreira; org:Atletico Ribeira", "Diego Ferreira > member of sports team > Atletico Ribeira"),
    ("spt-01", "b", "nolonger", "Diego Ferreira no longer plays for Atletico Ribeira.",
     "per:Diego Ferreira; org:Atletico Ribeira",
     "Diego Ferreira > member of sports team > Atletico Ribeira [neg]"),
    ("spt-01", "c", "expected", "Diego Ferreira is expected to sign for Atletico Ribeira.",
     "per:Diego Ferreira; org:Atletico Ribeira",
     "Diego Ferreira > member of sports team > Atletico Ribeira [planned/hypothetical]"),
    ("spt-01", "d", "reported", "According to the local press, Diego Ferreira plays for Atletico Ribeira.",
     "per:Diego Ferreira; org:Atletico Ribeira",
     "Diego Ferreira > member of sports team > Atletico Ribeira [reported]"),
    ("spt-02", "a", "asserted", "Sam Whitaker joined the Harbor City Hawks this season.",
     "per:Sam Whitaker; org:Harbor City Hawks", "Sam Whitaker > member of sports team > Harbor City Hawks"),
    ("spt-02", "b", "never", "Sam Whitaker was never a member of the Harbor City Hawks.",
     "per:Sam Whitaker; org:Harbor City Hawks",
     "Sam Whitaker > member of sports team > Harbor City Hawks [neg]"),
    ("spt-02", "c", "conditional", "If he recovers, Sam Whitaker could play for the Harbor City Hawks.",
     "per:Sam Whitaker|he; org:Harbor City Hawks",
     "Sam Whitaker > member of sports team > Harbor City Hawks [hypothetical]"),
]
# Part 2 (plan r4 C7): groups numbered from 11, in their own module.
from .m3_hand_extra import SPEC_EXTRA  # noqa: E402
SPEC = SPEC + SPEC_EXTRA

# Form tag -> the family it is reported under.
FAMILY = {"asserted": "asserted", "not": "negated", "never": "negated", "nolonger": "negated",
          "denied": "negated", "planned": "planned", "expected": "planned", "desired": "desired",
          "might": "hypothetical", "conditional": "hypothetical", "rumoured": "reported",
          "reported": "reported", "question": "questioned",
          # part 2 forms
          "refused": "negated", "neither": "negated", "false": "negated", "yet": "negated",
          "may": "hypothetical", "probably": "hypothetical", "would": "hypothetical",
          "considering": "planned", "hopes": "desired", "allegedly": "reported", "claims": "reported"}


# ── build ──────────────────────────────────────────────────────────
def _parse_ents(s: str) -> list[tuple[str, list[str]]]:
    out = []
    for part in filter(None, (p.strip() for p in s.split(";"))):
        ty, names = part.split(":", 1)
        out.append((ABBR[ty], [n.strip() for n in names.split("|")]))
    return out


def _parse_facts(s: str) -> list[dict]:
    out = []
    for part in filter(None, (p.strip() for p in s.split(";"))):
        attrs = []
        if part.endswith("]"):
            part, a = part[:-1].split("[")
            attrs = a.split()
        h, r, t = (x.strip() for x in part.split(">"))
        mods = [m for a in attrs for m in a.split("/") if m in MODS] or ["asserted"]
        out.append({"head": h, "relation": r, "tail": t, "polarity": "negative" if "neg" in attrs else "positive",
                    "modality": mods, "optional": "opt" in attrs})
    return out


def _schema():
    from ..label_bank import load as load_bank
    bank = load_bank()
    rel = {r["name"]: r for r in bank["relations"]}
    typ = {t["name"]: t for t in bank.get("entity_types", bank.get("types", []))}
    missing = [n for n in REL_NAMES if n not in rel]
    assert not missing, missing
    rels = [{"name": n, "description": rel[n].get("description", "")} for n in REL_NAMES]
    types = [{"name": n, "description": typ.get(n, {}).get("description", "")} for n in TYPE_NAMES]
    return rels, types


def build() -> int:
    from ..infer import Extractor
    from ..windows import tokenize
    from .m4_business import _find
    rels, types = _schema()
    names = {r["name"] for r in rels}
    items, problems, ids = [], [], set()
    for gid, suf, tags, text, ents_s, facts_s in SPEC:
        sid = f"{gid}{suf}"
        assert sid not in ids, sid
        ids.add(sid)
        conv = "conv" in tags.split()
        if conv:
            turns = [tuple(x.strip() for x in t.split("|", 1)) for t in text.split("||")]
            ws = Extractor.windows(turns=turns)
        else:
            ws = Extractor.windows(text)
        assert len(ws) == 1, sid
        w = ws[0]
        toks, sp = w["tokens"], w["sentence_spans"]
        speakers = w.get("sentence_speakers") if conv else None
        ents = _parse_ents(ents_s)
        order = sorted(((len(tokenize(n)), k, n) for k, (_, ns) in enumerate(ents) for n in ns), reverse=True)
        spans: dict[int, list] = defaultdict(list)
        taken: set = set()
        for _, k, n in order:
            found = _find(toks, tokenize(n), taken)
            if not found and n == ents[k][1][0] and not spans[k]:
                problems.append(f"{sid}: '{n}' not in text")
            spans[k] += found
        idx = {ns[0]: k for k, (_, ns) in enumerate(ents)}
        facts, opt = [], []
        for f in _parse_facts(facts_s):
            if f["relation"] not in names:
                problems.append(f"{sid}: relation '{f['relation']}' not in the schema")
            arg = []
            for x in (f["head"], f["tail"]):
                if x.startswith("@"):
                    if not conv or x[1:] not in (speakers or []):
                        problems.append(f"{sid}: speaker '{x}' not a speaker")
                    arg.append(x)
                elif x in idx:
                    arg.append(idx[x])
                else:
                    problems.append(f"{sid}: fact argument '{x}' is not an entity")
                    arg.append(None)
            if None in arg:
                continue
            g = {"head": arg[0], "relation": f["relation"], "tail": arg[1], "polarity": f["polarity"],
                 "modality": f["modality"]}
            (opt if f["optional"] else facts).append(g)
        tg = tags.split()
        form = next(t for t in tg if t in FAMILY)
        items.append({
            "id": f"m3:{sid}", "group": gid, "form": form, "family": FAMILY[form], "conv": conv,
            "text": text, "tokens": toks, "sentence_spans": sp, "sentence_speakers": speakers,
            "entities": [{"type": ty, "spans": sorted(list(s) for s in spans[k]), "name": ns[0]}
                         for k, (ty, ns) in enumerate(ents)],
            "facts": facts, "optional_facts": opt, "relations": rels, "types": types})
    if problems:
        print("\n".join(problems))
        return 1
    DATA.write_text(json.dumps({"relations": rels, "types": types, "items": items}, ensure_ascii=False, indent=0))
    c = counts(items)
    print(json.dumps(c, indent=1))
    print(f"{len(items)} statements, {c['facts']} expected facts -> {DATA}")
    return 0


def kind_of(f: dict) -> str:
    if f["polarity"] == "negative":
        return "negated"
    return f["modality"][0]


def counts(items: list[dict]) -> dict:
    form, fam, kind, rel = Counter(), Counter(), Counter(), Counter()
    for it in items:
        form[it["form"]] += 1
        fam[it["family"]] += 1
        for f in it["facts"]:
            kind[kind_of(f)] += 1
            rel[f["relation"]] += 1
    return {"statements": len(items), "groups": len({it["group"] for it in items}),
            "conversation": sum(it["conv"] for it in items),
            "facts": sum(len(it["facts"]) for it in items),
            "per_form": dict(form), "per_family": dict(fam), "per_kind": dict(kind), "per_relation": dict(rel)}


def load_hand() -> list[dict]:
    return json.loads(DATA.read_text())["items"]


def load_pairs() -> list[dict]:
    """Held-out minimal pairs (every rewrite of a held-out base) and their
    asserted base sentences, with the item's entity types for end to end."""
    from ..label_bank import load as load_bank
    from .heldout_zs import _lit
    from .negation_pilot import load_pairs as lp
    bank = load_bank()
    typ = {t["name"]: t for t in bank.get("entity_types", bank.get("types", []))}
    d = json.loads((DATA_DIR / "v6-corpus" / "kg_short.json").read_text())
    tys = {x["id"]: _lit(x["types"]) for x in d["pairs"]}
    tys.update({x["id"]: _lit(x["types"]) for x in d["train"]})
    out = []
    for it in lp(10 ** 6, heldout=True):
        h, rel, t = it["fact"]
        ep, em = it["expect"]
        kind = "asserted" if it["kind"].startswith("asserted") else it["kind"]
        base = it["id"].split(":var:")[0]
        out.append({"id": f"pairs:{it['id']}" + (":base" if kind == "asserted" else ""), "group": base,
                    "form": kind, "family": kind, "conv": False, "tokens": it["tokens"],
                    "sentence_spans": it["sentence_spans"], "sentence_speakers": None,
                    "entities": [{"type": e["type"], "spans": e["spans"]} for e in it["entities"]],
                    "facts": [{"head": h, "relation": rel, "tail": t, "polarity": ep, "modality": [em]}],
                    "optional_facts": [], "relations": it["relations"],
                    "types": [{"name": n, "description": typ.get(n, {}).get("description", "")}
                              for n in tys.get(it["id"], [])]})
    return out


# ── predict ────────────────────────────────────────────────────────
def predict(items: list[dict], modes: list[str], device: str = "cpu", run: str | None = None) -> dict:
    """Per statement and mode, every candidate the decoder ranks (top 4 per
    pair, margins kept below the bar) with each attribute head's
    polarity / modality probabilities. With ``run``, that run's own relation
    step and attribute head alone (the head is named after the run)."""
    import torch
    from ..infer import Extractor
    from ..train.gold_eval import CKPT_DIR
    if run:
        ex = exb = Extractor(run, device=device)
        exs = (ex,)
    else:
        ex = Extractor(RUN, device=device, local=CKPT_DIR)
        exb = Extractor(HEADS["attrv8b"], device=device, local=CKPT_DIR)
        exs = (ex, exb)
    for e in exs:
        e.rel_offsets = {k: v + SHIFT for k, v in e.rel_offsets.items()}
        e.unseen_offset -= SHIFT
        e.top = max(e.top, 4)
    assert ex.rel_offsets == exb.rel_offsets and ex.unseen_offset == exb.unseen_offset
    heads = {run: ex.model.attr} if run else {"v7.1": ex.model.attr, "attrv8b": exb.model.attr}
    m = ex.model
    cap = {}
    fwd = m.forward

    def capture(b, **kw):
        cap["b"] = b
        return fwd(b, **kw)
    m.forward = capture
    rows = {mode: [] for mode in modes}
    check = []
    t0 = time.time()
    try:
        for n, it in enumerate(items):
            w = {"tokens": it["tokens"], "sentence_spans": it["sentence_spans"],
                 "sentence_speakers": it.get("sentence_speakers")}
            for mode in modes:
                if mode == "gold":
                    ents = [{"type": e["type"], "mentions": sorted(tuple(s) for s in e["spans"])}
                            for e in it["entities"] if e["spans"]]
                else:
                    ents = ex._entities(w, it["types"] or None)
                cap.clear()
                nodes, facts = ex._relations(w, ents, list(it["relations"]), ex.top)
                if n < 5 and mode == "gold" and not run:   # the attrv8b run's own relation step agrees
                    _, fb = exb._relations(w, ents, list(it["relations"]), exb.top)
                    check.append(max([abs(a["margin"] - b["margin"]) for a, b in zip(facts, fb)] or [0.0])
                                 if len(facts) == len(fb) else float("inf"))
                if mode == "gold":
                    first = {tuple(mm): i for i, e in enumerate(ents) for mm in e["mentions"]}
                    gidx = [i for i, e in enumerate(it["entities"]) if e["spans"]]
                    keys = ["@" + nd["text"] if nd.get("speaker") else gidx[first[tuple(nd["spans"][0])]]
                            for nd in nodes]
                else:
                    keys = ["@" + nd["text"] if nd.get("speaker") else sorted({x.lower() for x in nd["mentions"]})
                            for nd in nodes]
                cand = []
                if facts:
                    b = cap["b"]
                    pairs = list(zip(b["pair_h"].tolist(), b["pair_t"].tolist()))
                    names = [r["name"] for r in it["relations"]]
                    P = torch.tensor([pairs.index((f["head"], f["tail"])) for f in facts], device=ex.device)
                    K = torch.tensor([names.index(f["relation"]) for f in facts], device=ex.device)
                    att = {}
                    with torch.no_grad():
                        for hn, hd in heads.items():
                            a = hd(m.relation._pair, m.relation._labels, m.relation._ent, b, P, K)
                            att[hn] = (torch.softmax(a["polarity"], -1).cpu().numpy(),
                                       torch.softmax(a["modality"], -1).cpu().numpy())
                    for j, f in enumerate(facts):
                        cand.append({"h": keys[f["head"]], "r": f["relation"], "t": keys[f["tail"]],
                                     "m": f["margin"] - SHIFT, "unseen": f["relation"] not in ex.rel_offsets,
                                     "attr": {hn: (v[0][j].tolist(), v[1][j].tolist()) for hn, v in att.items()}})
                rows[mode].append({"id": it["id"], "cand": cand})
            if n % 100 == 0:
                print(f"  {n}/{len(items)} [{time.time() - t0:.0f}s]", flush=True)
    finally:
        m.forward = fwd
    return {"rows": rows, "check": check}


# ── score ──────────────────────────────────────────────────────────
def _mentions(it: dict, k) -> set:
    if isinstance(k, str):
        return {k}
    return {" ".join(it["tokens"][a:b]).lower() for a, b in it["entities"][k]["spans"]}


def _arg(it, mode, key, k) -> bool:
    if isinstance(k, str) or isinstance(key, str):
        return key == k
    if mode == "gold":
        return key == k
    return bool(set(key) & _mentions(it, k))


def _match(it, mode, c, f) -> bool:
    if c["r"] != f["relation"]:
        return False
    if _arg(it, mode, c["h"], f["head"]) and _arg(it, mode, c["t"], f["tail"]):
        return True
    return f["relation"] in SYMMETRIC and _arg(it, mode, c["h"], f["tail"]) and _arg(it, mode, c["t"], f["head"])


def fact_rows(items: list[dict], rows: list[dict], mode: str) -> list[dict]:
    """One row per expected fact: returned, rank-4 found, and per head
    polarity / modality right (on the returned candidate, and forced)."""
    from ..schemas import MODALITIES, POLARITIES
    out = []
    for it, row in zip(items, rows):
        for f in it["facts"]:
            ms = [c for c in row["cand"] if _match(it, mode, c, f)]
            best = max(ms, key=lambda c: c["m"]) if ms else None
            ret = best is not None and best["m"] > 0
            r = {"id": it["id"], "group": it["group"], "form": it["form"], "family": it["family"],
                 "conv": it["conv"], "kind": kind_of(f), "relation": f["relation"],
                 "ranked": best is not None, "returned": ret, "margin": best["m"] if best else None}
            for hn in HEADS:
                if best is None:
                    r[hn] = {"pol": False, "mod": False}
                    continue
                pol, mod = best["attr"][hn]
                r[hn] = {"pol": POLARITIES[int(np.argmax(pol))] == f["polarity"],
                         "mod": MODALITIES[int(np.argmax(mod))] in f["modality"],
                         "pred": [POLARITIES[int(np.argmax(pol))], MODALITIES[int(np.argmax(mod))]],
                         "p_neg": float(pol[POLARITIES.index("negative")])}
            out.append(r)
    return out


def fact_prf(items: list[dict], rows: list[dict], mode: str) -> np.ndarray:
    """Per statement [tp, predicted, expected] at the deployed bar."""
    out = []
    for it, row in zip(items, rows):
        pred, seen = [], set()
        for c in sorted(row["cand"], key=lambda c: -c["m"]):
            k = (json.dumps(c["h"]), c["r"], json.dumps(c["t"]))
            if c["m"] > 0 and k not in seen:
                seen.add(k)
                pred.append(c)
        used, tp, fp = set(), 0, 0
        for c in pred:
            j = next((j for j, f in enumerate(it["facts"]) if j not in used and _match(it, mode, c, f)), None)
            if j is not None:
                used.add(j)
                tp += 1
            elif not any(_match(it, mode, c, f) for f in it["optional_facts"]):
                fp += 1
        out.append([tp, tp + fp, len(it["facts"])])
    return np.array(out, float)


METRICS = ("returned", "right_when_returned", "returned_and_right", "forced_polarity", "forced_modality",
           "forced_both")


def metric(rs: list[dict], hn: str, name: str) -> float:
    if not rs:
        return float("nan")
    if name == "returned":
        return float(np.mean([r["returned"] for r in rs]))
    if name == "right_when_returned":
        ret = [r for r in rs if r["returned"]]
        return float(np.mean([r[hn]["pol"] and r[hn]["mod"] for r in ret])) if ret else float("nan")
    if name == "returned_and_right":
        return float(np.mean([r["returned"] and r[hn]["pol"] and r[hn]["mod"] for r in rs]))
    rk = [r for r in rs if r["ranked"]]          # forced: the fact wherever it ranks in its pair's top 4
    if not rk:
        return float("nan")
    if name == "forced_polarity":
        return float(np.mean([r[hn]["pol"] for r in rk]))
    if name == "forced_modality":
        return float(np.mean([r[hn]["mod"] for r in rk]))
    return float(np.mean([r[hn]["pol"] and r[hn]["mod"] for r in rk]))


def paired(rs: list[dict], name: str, n: int = 2000, seed: int = 13):
    """attrv8b - v7.1 with a bootstrap over groups; the detectable difference
    (80% power, 5% level) is 2.8 standard errors of the paired difference."""
    g = defaultdict(list)
    for r in rs:
        g[r["group"]].append(r)
    keys = list(g)
    rng = np.random.default_rng(seed)
    d0 = metric(rs, "attrv8b", name) - metric(rs, "v7.1", name)
    ds = []
    for _ in range(n):
        s = [r for k in rng.choice(len(keys), len(keys)) for r in g[keys[k]]]
        ds.append(metric(s, "attrv8b", name) - metric(s, "v7.1", name))
    ds = np.array([x for x in ds if np.isfinite(x)])
    se = float(ds.std()) if len(ds) else float("nan")
    return d0, float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5)), 2.8 * se


def single_se(rs: list[dict], hn: str, name: str, n: int = 2000, seed: int = 13) -> float:
    """2.8 standard errors of one head's metric (bootstrap over groups): the
    detectable difference against a fixed target."""
    g = defaultdict(list)
    for r in rs:
        g[r["group"]].append(r)
    keys = list(g)
    rng = np.random.default_rng(seed)
    v = [metric([r for k in rng.choice(len(keys), len(keys)) for r in g[keys[k]]], hn, name) for _ in range(n)]
    v = np.array([x for x in v if np.isfinite(x)])
    return 2.8 * float(v.std()) if len(v) else float("nan")


def table(rs: list[dict], by: str, order: list[str] | None = None) -> list[str]:
    grp = defaultdict(list)
    for r in rs:
        grp[r[by]].append(r)
    L = [f"| {by} | n | ranked top 4 | returned | v7.1 right when returned | attrv8b | v7.1 returned and right | attrv8b | "
         "v7.1 forced polarity / modality | attrv8b |", "|---|---|---|---|---|---|---|---|---|---|"]
    for k in (order or sorted(grp)):
        if k not in grp:
            continue
        x = grp[k]
        f = lambda hn, nm: metric(x, hn, nm)
        L.append(f"| {k} | {len(x)} | {np.mean([r['ranked'] for r in x]):.2f} | {f('v7.1', 'returned'):.2f} | {f('v7.1', 'right_when_returned'):.2f} | "
                 f"{f('attrv8b', 'right_when_returned'):.2f} | {f('v7.1', 'returned_and_right'):.2f} | "
                 f"{f('attrv8b', 'returned_and_right'):.2f} | {f('v7.1', 'forced_polarity'):.2f} / "
                 f"{f('v7.1', 'forced_modality'):.2f} | {f('attrv8b', 'forced_polarity'):.2f} / "
                 f"{f('attrv8b', 'forced_modality'):.2f} |")
    return L


KIND_ORDER = ["asserted", "negated", "reported", "planned", "desired", "questioned", "hypothetical"]
FORM_ORDER = ["asserted", "not", "never", "nolonger", "denied", "planned", "expected", "desired", "might",
              "conditional", "rumoured", "reported", "question"]


def score() -> int:
    res, md = {}, []
    for name in ("hand", "pairs"):
        files = sorted(OUT.glob(f"preds-{name}-*.pkl"))
        if not files:
            continue
        items = load_hand() if name == "hand" else load_pairs()
        byid = {it["id"]: it for it in items}
        rows = defaultdict(dict)
        check = []
        for p in files:
            d = pickle.loads(p.read_bytes())
            check += d["check"]
            for mode, rr in d["rows"].items():
                for r in rr:
                    rows[mode][r["id"]] = r
        res[name] = {"check_max_margin_diff": max(check) if check else None}
        md += [f"## {name} ({len(items)} statements; relation step check, max margin difference "
               f"v7b8ev vs attrv8b run on the first statements: {max(check) if check else 'n/a'})", ""]
        for mode in ("gold", "e2e"):
            if not rows.get(mode):
                continue
            its = [byid[i] for i in rows[mode] if i in byid]
            rr = [rows[mode][it["id"]] for it in its]
            fr = fact_rows(its, rr, mode)
            prf = fact_prf(its, rr, mode)
            s = prf.sum(0)
            p, r_, f1 = s[0] / max(s[1], 1), s[0] / max(s[2], 1), 2 * s[0] / max(s[1] + s[2], 1)
            out = {"statements": len(its), "facts": len(fr), "fact_prf": [p, r_, f1]}
            md += [f"### {name}, {mode} entities: {len(its)} statements, {len(fr)} expected facts; "
                   f"fact P / R / F1 {p:.3f} / {r_:.3f} / {f1:.3f}", ""]
            md += table(fr, "kind", KIND_ORDER) + [""]
            if name == "hand":
                md += table(fr, "form", FORM_ORDER) + [""]
                md += table([dict(x, conv="conversation" if x["conv"] else "text") for x in fr], "conv") + [""]
            # detectable differences
            md += ["| subset | metric | v7.1 | attrv8b | attrv8b - v7.1 [95%] | detectable paired | "
                   "detectable vs a fixed target (v7.1) |", "|---|---|---|---|---|---|---|"]
            out["mde"] = {}
            subsets = {"all": fr, "negated": [x for x in fr if x["kind"] == "negated"],
                       "hedged": [x for x in fr if x["kind"] not in ("asserted", "negated")],
                       "asserted": [x for x in fr if x["kind"] == "asserted"]}
            for sn, sub in subsets.items():
                for nm in ("returned_and_right", "forced_polarity", "forced_modality"):
                    if not sub:
                        continue
                    d, lo, hi, mde = paired(sub, nm)
                    mde1 = single_se(sub, "v7.1", nm)
                    out["mde"][f"{sn}|{nm}"] = {"v7.1": metric(sub, "v7.1", nm), "attrv8b": metric(sub, "attrv8b", nm),
                                                 "diff": [d, lo, hi], "mde_paired": mde, "mde_single": mde1}
                    md.append(f"| {sn} ({len(sub)}) | {nm} | {metric(sub, 'v7.1', nm):.3f} | "
                              f"{metric(sub, 'attrv8b', nm):.3f} | {d * 100:+.1f} [{lo * 100:+.1f}, {hi * 100:+.1f}] | "
                              f"{mde * 100:.1f} | {mde1 * 100:.1f} |")
            md.append("")
            if name == "hand":
                wrong = Counter()
                for x in fr:
                    for hn in HEADS:
                        if "pred" in x[hn] and not (x[hn]["pol"] and x[hn]["mod"]):
                            wrong[(hn, x["kind"], tuple(x[hn]["pred"]))] += 1
                md += ["forced readings that are wrong (head, expected kind, predicted polarity / modality: n):", ""]
                for (hn, k, pr), c in sorted(wrong.items(), key=lambda z: (z[0][0], z[0][1], -z[1])):
                    md.append(f"- {hn}, {k}: {pr[0]} / {pr[1]}: {c}")
                md.append("")
            out["facts_rows"] = fr
            res[name][mode] = out
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "score.json").write_text(json.dumps(res, indent=1, default=float))
    (OUT / "score.md").write_text("\n".join(md))
    print("\n".join(md))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    sub.add_parser("counts")
    p = sub.add_parser("predict")
    p.add_argument("--set", choices=["hand", "pairs"], required=True)
    p.add_argument("--slice", default="", help="a:b, statements of the set")
    p.add_argument("--modes", default="gold,e2e")
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--run", help="one run's relation step and attribute head (preds-run-<run>-*.pkl)")
    sub.add_parser("score")
    a = ap.parse_args()
    if a.cmd == "build":
        return build()
    if a.cmd == "counts":
        print(json.dumps(counts(load_hand()), indent=1))
        print(json.dumps(counts(load_pairs()), indent=1))
        return 0
    if a.cmd == "score":
        return score()
    if a.threads:
        import torch
        torch.set_num_threads(a.threads)
    items = load_hand() if a.set == "hand" else load_pairs()
    lo, hi = (int(x) if x else None for x in (a.slice.split(":") if a.slice else ("", "")))
    items = items[lo:hi]
    d = predict(items, a.modes.split(","), a.device, a.run)
    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"{a.set}-{lo or 0}-{hi if hi is not None else 'end'}"
    tag = f"run-{a.run}-{tag}" if a.run else tag
    (OUT / f"preds-{tag}.pkl").write_bytes(pickle.dumps(d))
    print(f"written {OUT / f'preds-{tag}.pkl'}; relation check {d['check']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
