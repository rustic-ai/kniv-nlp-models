"""M4 (v8 plan r2): business relation suite.

    uv run python -m v6.eval.m4_business build
    uv run python -m v6.eval.m4_business score --run ultrabert-380m-v7b8ev --local [--naming user|bank]

About 150 short business texts written for this suite (fictional companies
and people, real places), each with the full set of facts its schema
expresses: supplier of, customer of, acquired, holds stake in, sued, board
member of, priced at, with works for and based in as controls. Forms vary:
active, passive, nominal ("the acquisition of X by Y"), lists, several
sentences, negated ("did not", "no longer", "denied"), hedged (plans to,
might, wants to, questions, reports), type traps (a person beside a
company-to-company fact: the outside evaluation found "a person supplier of a
company") and distractors with no target fact.

Every item has one schema (the nine relations, five entity types). Expected
facts carry polarity and modality (several modalities may be accepted where
the text allows more than one reading); facts a reader could defend but the
text does not require are ``optional``: neither counted as found nor as
false. "supplier of" and "customer of" are inverses, so each expected one
also expects the other.

``score`` runs the extractor on CPU, with gold entities (the relation step
alone) and end to end (entities found with the schema's types), at the run's
deployed calibration, and reports:

- fact F1 (head, relation, tail; any polarity), per relation and per form;
- out-of-bank relations: which of the schema's names the label bank does not
  have, and whether each is found;
- type violations: predicted facts whose argument types break the relation's
  constraint (gold types with gold entities, predicted types end to end);
- polarity and modality accuracy on the expected facts that were returned,
  and "returned and right" per attribute kind.

``--naming bank`` swaps each relation for its nearest label-bank relation
(name and description), to compare the names users write with the bank's.

Items are stored in the short-dev item format of ``heldout_zs`` (tokens,
sentence_spans, entities, facts, relations, types), so M1 can predict and
judge them unchanged.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict

import numpy as np

from ..config import DATA_DIR, RUNS_DIR

DATA = DATA_DIR / "v6-eval" / "m4_business.json"
OUT = RUNS_DIR / "m4-business"
SHIFT = 20.0

RELATIONS = [
    {"name": "supplier of", "description": "the head organization supplies goods or services to the tail organization"},
    {"name": "customer of", "description": "the head organization buys goods or services from the tail organization"},
    {"name": "acquired", "description": "the head organization bought or took over the tail organization"},
    {"name": "holds stake in", "description": "the head person or organization owns shares or a part of the tail organization"},
    {"name": "sued", "description": "the head person or organization brought a lawsuit against the tail"},
    {"name": "board member of", "description": "the head person sits on the board of directors of the tail organization"},
    {"name": "priced at", "description": "the head product or service is sold or offered at the tail price"},
    {"name": "works for", "description": "the head person is employed by or works for the tail organization"},
    {"name": "based in", "description": "the head organization is based or headquartered in the tail location"},
]
# Nearest label-bank relation per name (``--naming bank``); "sued" has none.
BANK_NAME = {"supplier of": "supplies to", "customer of": "purchases from", "acquired": "acquires",
             "holds stake in": "invests in", "board member of": "board member of",
             "priced at": "has price", "works for": "employed by", "based in": "headquartered in"}
TYPES = [
    {"name": "person", "description": "a named person"},
    {"name": "organization", "description": "a company, bank, fund, newspaper or other named organization"},
    {"name": "location", "description": "a named place: country, city or region"},
    {"name": "product", "description": "a named product, model or service offered for sale"},
    {"name": "money", "description": "an amount of money, such as a price"},
]
ORG, PER, LOC, PROD, MON = "organization", "person", "location", "product", "money"
CONSTRAINT = {"supplier of": ({ORG}, {ORG}), "customer of": ({ORG}, {ORG}), "acquired": ({ORG}, {ORG}),
              "holds stake in": ({ORG, PER}, {ORG}), "sued": ({ORG, PER}, {ORG, PER}),
              "board member of": ({PER}, {ORG}), "priced at": ({PROD}, {MON}),
              "works for": ({PER}, {ORG}), "based in": ({ORG}, {LOC})}
INVERSE = {"supplier of": "customer of", "customer of": "supplier of"}
ABBR = {"org": ORG, "per": PER, "loc": LOC, "prod": PROD, "mon": MON}
MODS = {"asserted", "hypothetical", "planned", "desired", "questioned", "reported"}

# (id, form tags, text, entities "type:Name|alias; ...", facts "H > relation > T [attrs]; ...")
# attrs: "neg", "opt", and accepted modalities joined by "/" (default asserted).
SPEC = [
    # ── supplier of / customer of ───────────────────────────────────
    ("sup-01", "active", "Orrin Metals supplies cold-rolled steel to Wexley Motors.",
     "org:Orrin Metals; org:Wexley Motors", "Orrin Metals > supplier of > Wexley Motors"),
    ("sup-02", "passive", "The battery cells in Wexley Motors' new hatchback are supplied by Kalvik Cells.",
     "org:Wexley Motors; org:Kalvik Cells", "Kalvik Cells > supplier of > Wexley Motors"),
    ("sup-03", "active", "Pinecrest Foods buys all of its packaging from Moraine Paper.",
     "org:Pinecrest Foods; org:Moraine Paper", "Pinecrest Foods > customer of > Moraine Paper"),
    ("sup-04", "nominal", "Halvard Systems' supply contract with Brightwater Grid runs until 2028.",
     "org:Halvard Systems; org:Brightwater Grid", "Halvard Systems > supplier of > Brightwater Grid"),
    ("sup-05", "multi", "Tamsin Logistics is based in Rotterdam. Since 2021 it has handled warehousing for "
     "Velloway Retail across the Benelux.",
     "org:Tamsin Logistics; loc:Rotterdam; org:Velloway Retail; loc:Benelux",
     "Tamsin Logistics > based in > Rotterdam; Tamsin Logistics > supplier of > Velloway Retail"),
    ("sup-06", "list", "Kestrel Avionics counts Zephyr Air and Norland Cargo among its biggest customers.",
     "org:Kestrel Avionics; org:Zephyr Air; org:Norland Cargo",
     "Zephyr Air > customer of > Kestrel Avionics; Norland Cargo > customer of > Kestrel Avionics"),
    ("sup-07", "active", "Rookwood Insurance became a client of Ironleaf Software in March, moving its claims "
     "system onto Ironleaf's platform.",
     "org:Rookwood Insurance; org:Ironleaf Software|Ironleaf", "Rookwood Insurance > customer of > Ironleaf Software"),
    ("sup-08", "negated", "Fenwick Glass does not supply Wexley Motors.",
     "org:Fenwick Glass; org:Wexley Motors", "Fenwick Glass > supplier of > Wexley Motors [neg]"),
    ("sup-09", "nolonger", "Moraine Paper no longer supplies cartons to Saltmarsh Brewing, which switched to a "
     "local mill last year.",
     "org:Moraine Paper; org:Saltmarsh Brewing", "Moraine Paper > supplier of > Saltmarsh Brewing [neg]"),
    ("sup-10", "denied", "Kalvik Cells denied reports that it supplies batteries to Arcwell Robotics.",
     "org:Kalvik Cells; org:Arcwell Robotics", "Kalvik Cells > supplier of > Arcwell Robotics [neg]"),
    ("sup-11", "planned", "Orrin Metals plans to start supplying aluminium to Kestrel Avionics next spring.",
     "org:Orrin Metals; org:Kestrel Avionics", "Orrin Metals > supplier of > Kestrel Avionics [planned]"),
    ("sup-12", "hypothetical", "If the trial goes well, Verdant Agro could become a customer of Lumio Labs.",
     "org:Verdant Agro; org:Lumio Labs", "Verdant Agro > customer of > Lumio Labs [hypothetical]"),
    ("sup-13", "reported", "According to two people familiar with the deal, Tessaly Chips supplies the "
     "processors used by Arcwell Robotics.",
     "org:Tessaly Chips; org:Arcwell Robotics", "Tessaly Chips > supplier of > Arcwell Robotics [reported]"),
    ("sup-14", "questioned", "Does Halvard Systems still supply Brightwater Grid?",
     "org:Halvard Systems; org:Brightwater Grid", "Halvard Systems > supplier of > Brightwater Grid [questioned]"),
    ("sup-15", "typetrap planned", "Declan Moore, who negotiated the contract, said Fenwick Glass will supply "
     "Aurelle Cosmetics with bottles from March.",
     "per:Declan Moore; org:Fenwick Glass; org:Aurelle Cosmetics",
     "Fenwick Glass > supplier of > Aurelle Cosmetics [planned/asserted]"),
    ("sup-16", "multi", "Priya Raman runs procurement at Velloway Retail. She signed the retailer's first order "
     "with Saltmarsh Brewing in June.",
     "per:Priya Raman; org:Velloway Retail; org:Saltmarsh Brewing",
     "Priya Raman > works for > Velloway Retail; Velloway Retail > customer of > Saltmarsh Brewing"),
    ("sup-17", "distractor", "Fenwick Glass and Moraine Paper both bid for the Pinecrest Foods packaging "
     "contract, which went to neither.",
     "org:Fenwick Glass; org:Moraine Paper; org:Pinecrest Foods",
     "Fenwick Glass > supplier of > Pinecrest Foods [opt neg]; Moraine Paper > supplier of > Pinecrest Foods [opt neg]"),
    ("sup-18", "passive", "Wexley Motors is supplied with seat fabric by Dunvale Textiles of Leeds.",
     "org:Wexley Motors; org:Dunvale Textiles; loc:Leeds",
     "Dunvale Textiles > supplier of > Wexley Motors; Dunvale Textiles > based in > Leeds"),
    ("sup-19", "active", "Ironleaf Software is the payroll vendor for Dunmore Bank.",
     "org:Ironleaf Software; org:Dunmore Bank", "Ironleaf Software > supplier of > Dunmore Bank"),
    ("sup-20", "nominal", "Deliveries of copper concentrate from Cobalt Ridge Mining to Ostrava Steelworks "
     "resumed on Monday.",
     "org:Cobalt Ridge Mining; org:Ostrava Steelworks", "Cobalt Ridge Mining > supplier of > Ostrava Steelworks"),
    ("sup-21", "active", "Quellant Bio has been the sole supplier of reagents to Yarrow Health since 2019.",
     "org:Quellant Bio; org:Yarrow Health", "Quellant Bio > supplier of > Yarrow Health"),
    ("sup-22", "multi nolonger", "Zephyr Air used to buy its catering from Saltmarsh Brewing. This year it moved "
     "the contract to Pinecrest Foods.",
     "org:Zephyr Air; org:Saltmarsh Brewing; org:Pinecrest Foods",
     "Zephyr Air > customer of > Saltmarsh Brewing [neg]; Zephyr Air > customer of > Pinecrest Foods"),
    ("sup-23", "desired", "Arcwell Robotics wants Tessaly Chips to become its main chip supplier.",
     "org:Arcwell Robotics; org:Tessaly Chips", "Tessaly Chips > supplier of > Arcwell Robotics [desired]"),
    ("sup-24", "reported", "Analysts said Kalvik Cells supplies Wexley Motors, but Wexley declined to comment.",
     "org:Kalvik Cells; org:Wexley Motors|Wexley", "Kalvik Cells > supplier of > Wexley Motors [reported]"),
    ("sup-25", "typetrap", "Lucas Ferreira, an engineer at Kalvik Cells, delivered the first batteries to "
     "Wexley Motors himself.",
     "per:Lucas Ferreira; org:Kalvik Cells; org:Wexley Motors",
     "Lucas Ferreira > works for > Kalvik Cells; Kalvik Cells > supplier of > Wexley Motors [opt]"),

    # ── acquired ────────────────────────────────────────────────────
    ("acq-01", "active", "Halvard Systems acquired Lumio Labs for $210 million in 2023.",
     "org:Halvard Systems; org:Lumio Labs; mon:$210 million", "Halvard Systems > acquired > Lumio Labs"),
    ("acq-02", "passive", "Juniper Lane Media was bought by Pelham Capital in a deal that closed in May.",
     "org:Juniper Lane Media; org:Pelham Capital", "Pelham Capital > acquired > Juniper Lane Media"),
    ("acq-03", "nominal", "The acquisition of Fenwick Glass by Aurelle Cosmetics was approved by regulators on "
     "Tuesday.",
     "org:Fenwick Glass; org:Aurelle Cosmetics", "Aurelle Cosmetics > acquired > Fenwick Glass"),
    ("acq-04", "nominal", "Wexley Motors' takeover of Kalvik Cells gives the carmaker control of its battery "
     "supply.",
     "org:Wexley Motors; org:Kalvik Cells",
     "Wexley Motors > acquired > Kalvik Cells; Kalvik Cells > supplier of > Wexley Motors [opt]"),
    ("acq-05", "distractor", "Rookwood Insurance and Dunmore Bank agreed to merge into a new company called "
     "Rookmore.",
     "org:Rookwood Insurance; org:Dunmore Bank; org:Rookmore", ""),
    ("acq-06", "negated multi", "Pelham Capital did not buy Ironleaf Software after all. The talks collapsed in "
     "August.",
     "org:Pelham Capital; org:Ironleaf Software", "Pelham Capital > acquired > Ironleaf Software [neg]"),
    ("acq-07", "denied", "Cobalt Ridge Mining denied that it had taken over Brask Minerals.",
     "org:Cobalt Ridge Mining; org:Brask Minerals", "Cobalt Ridge Mining > acquired > Brask Minerals [neg]"),
    ("acq-08", "planned", "Sandoval Freight intends to acquire Tamsin Logistics by the end of the year.",
     "org:Sandoval Freight; org:Tamsin Logistics", "Sandoval Freight > acquired > Tamsin Logistics [planned]"),
    ("acq-09", "hypothetical", "Verdant Agro may buy a smaller seed company, possibly Hollin Seeds, if "
     "financing allows.",
     "org:Verdant Agro; org:Hollin Seeds", "Verdant Agro > acquired > Hollin Seeds [hypothetical]"),
    ("acq-10", "reported", "Zephyr Air is rumoured to be buying Norland Cargo.",
     "org:Zephyr Air; org:Norland Cargo", "Zephyr Air > acquired > Norland Cargo [reported/planned]"),
    ("acq-11", "questioned", "Will Arcwell Robotics buy Grisham Motion?",
     "org:Arcwell Robotics; org:Grisham Motion", "Arcwell Robotics > acquired > Grisham Motion [questioned]"),
    ("acq-12", "multi", "Greyfield Partners announced on Monday that it had completed its purchase of Yarrow "
     "Health. Yarrow, based in Calgary, will keep its name.",
     "org:Greyfield Partners; org:Yarrow Health|Yarrow; loc:Calgary",
     "Greyfield Partners > acquired > Yarrow Health; Yarrow Health > based in > Calgary"),
    ("acq-13", "planned typetrap", "Ingrid Solberg, chief executive of Halvard Systems, said the purchase of "
     "Brightwater Grid would close in the second quarter.",
     "per:Ingrid Solberg; org:Halvard Systems; org:Brightwater Grid",
     "Ingrid Solberg > works for > Halvard Systems; Halvard Systems > acquired > Brightwater Grid [planned/asserted]"),
    ("acq-14", "active", "Ironleaf Software snapped up the start-up Quillon AI for an undisclosed sum.",
     "org:Ironleaf Software; org:Quillon AI", "Ironleaf Software > acquired > Quillon AI"),
    ("acq-15", "nominal", "Since its takeover in 2020, Moraine Paper has been a subsidiary of Dunvale "
     "Textiles.",
     "org:Moraine Paper; org:Dunvale Textiles", "Dunvale Textiles > acquired > Moraine Paper"),
    ("acq-16", "distractor", "Sandoval Freight's bid for Norland Cargo was rejected by Norland's board.",
     "org:Sandoval Freight; org:Norland Cargo|Norland", "Sandoval Freight > acquired > Norland Cargo [opt neg]"),
    ("acq-17", "list", "In the past decade Pelham Capital has bought Brenner & Holt, Saltmarsh Brewing and "
     "Ostrava Steelworks.",
     "org:Pelham Capital; org:Brenner & Holt; org:Saltmarsh Brewing; org:Ostrava Steelworks",
     "Pelham Capital > acquired > Brenner & Holt; Pelham Capital > acquired > Saltmarsh Brewing; "
     "Pelham Capital > acquired > Ostrava Steelworks"),
    ("acq-18", "active", "The founders of Lumio Labs sold the company to Tessaly Chips in 2022.",
     "org:Lumio Labs; org:Tessaly Chips", "Tessaly Chips > acquired > Lumio Labs"),
    ("acq-19", "desired", "Aurelle Cosmetics wants to acquire Fenwick Glass, but Fenwick's owners have refused "
     "to sell.",
     "org:Aurelle Cosmetics; org:Fenwick Glass|Fenwick", "Aurelle Cosmetics > acquired > Fenwick Glass [desired]"),
    ("acq-20", "reported", "The Financial Ledger reported that Greyfield Partners bought Quellant Bio.",
     "org:The Financial Ledger; org:Greyfield Partners; org:Quellant Bio",
     "Greyfield Partners > acquired > Quellant Bio [reported]"),
    ("acq-21", "negated", "Despite years of speculation, Halvard Systems never acquired Orrin Metals.",
     "org:Halvard Systems; org:Orrin Metals", "Halvard Systems > acquired > Orrin Metals [neg]"),
    ("acq-22", "multi", "Arcwell Robotics, which is headquartered in Nagoya, completed the acquisition of "
     "Grisham Motion in April. Grisham's founder, Hana Kobayashi, will join Arcwell as head of research.",
     "org:Arcwell Robotics|Arcwell; loc:Nagoya; org:Grisham Motion|Grisham; per:Hana Kobayashi",
     "Arcwell Robotics > based in > Nagoya; Arcwell Robotics > acquired > Grisham Motion; "
     "Hana Kobayashi > works for > Arcwell Robotics [planned]; Hana Kobayashi > works for > Grisham Motion [opt]"),

    # ── holds stake in ──────────────────────────────────────────────
    ("stk-01", "active", "Pelham Capital holds a 12% stake in Ironleaf Software.",
     "org:Pelham Capital; org:Ironleaf Software", "Pelham Capital > holds stake in > Ironleaf Software"),
    ("stk-02", "active", "Rafael Duarte owns about a fifth of Saltmarsh Brewing.",
     "per:Rafael Duarte; org:Saltmarsh Brewing", "Rafael Duarte > holds stake in > Saltmarsh Brewing"),
    ("stk-03", "active", "Greyfield Partners bought a 30 percent stake in Velloway Retail on Thursday.",
     "org:Greyfield Partners; org:Velloway Retail", "Greyfield Partners > holds stake in > Velloway Retail"),
    ("stk-04", "passive", "A minority stake in Quellant Bio is held by Dunmore Bank's investment arm.",
     "org:Quellant Bio; org:Dunmore Bank", "Dunmore Bank > holds stake in > Quellant Bio"),
    ("stk-05", "nominal", "Zephyr Air's shareholding in Norland Cargo has grown to 9 percent.",
     "org:Zephyr Air; org:Norland Cargo", "Zephyr Air > holds stake in > Norland Cargo"),
    ("stk-06", "nolonger", "Sandoval Freight sold its entire stake in Tamsin Logistics last month.",
     "org:Sandoval Freight; org:Tamsin Logistics", "Sandoval Freight > holds stake in > Tamsin Logistics [neg]"),
    ("stk-07", "planned", "Wexley Motors is in talks to buy a stake in Kalvik Cells.",
     "org:Wexley Motors; org:Kalvik Cells", "Wexley Motors > holds stake in > Kalvik Cells [planned/hypothetical]"),
    ("stk-08", "hypothetical", "Elena Volkova could take a stake in Juniper Lane Media if its share price falls "
     "further.",
     "per:Elena Volkova; org:Juniper Lane Media", "Elena Volkova > holds stake in > Juniper Lane Media [hypothetical]"),
    ("stk-09", "denied", "Dunmore Bank denied holding any shares in Cobalt Ridge Mining.",
     "org:Dunmore Bank; org:Cobalt Ridge Mining", "Dunmore Bank > holds stake in > Cobalt Ridge Mining [neg]"),
    ("stk-10", "reported", "Filings show that Kwame Asante is the largest shareholder in Hollis Pharma.",
     "per:Kwame Asante; org:Hollis Pharma", "Kwame Asante > holds stake in > Hollis Pharma [reported/asserted]"),
    ("stk-11", "multi", "Amara Okonkwo has invested in Lumio Labs since its seed round and now owns 8 percent of "
     "the company. She also sits on its board.",
     "per:Amara Okonkwo; org:Lumio Labs",
     "Amara Okonkwo > holds stake in > Lumio Labs; Amara Okonkwo > board member of > Lumio Labs"),
    ("stk-12", "active", "Orrin Metals and Fenwick Glass hold stakes in each other.",
     "org:Orrin Metals; org:Fenwick Glass",
     "Orrin Metals > holds stake in > Fenwick Glass; Fenwick Glass > holds stake in > Orrin Metals"),
    ("stk-13", "questioned", "Does Pelham Capital still hold shares in Juniper Lane Media?",
     "org:Pelham Capital; org:Juniper Lane Media", "Pelham Capital > holds stake in > Juniper Lane Media [questioned]"),
    ("stk-14", "negated", "Brightwater Grid has no ownership interest in Halvard Systems.",
     "org:Brightwater Grid; org:Halvard Systems", "Brightwater Grid > holds stake in > Halvard Systems [neg]"),
    ("stk-15", "list", "The investors in Lumio Labs include Greyfield Partners and Pelham Capital.",
     "org:Lumio Labs; org:Greyfield Partners; org:Pelham Capital",
     "Greyfield Partners > holds stake in > Lumio Labs; Pelham Capital > holds stake in > Lumio Labs"),
    ("stk-16", "desired", "Kwame Asante would like to increase his stake in Hollis Pharma.",
     "per:Kwame Asante; org:Hollis Pharma", "Kwame Asante > holds stake in > Hollis Pharma [asserted/desired]"),

    # ── sued ────────────────────────────────────────────────────────
    ("sue-01", "active", "Fenwick Glass sued Aurelle Cosmetics for breach of contract.",
     "org:Fenwick Glass; org:Aurelle Cosmetics", "Fenwick Glass > sued > Aurelle Cosmetics"),
    ("sue-02", "passive", "Ironleaf Software was sued by Rookwood Insurance over a failed system migration.",
     "org:Ironleaf Software; org:Rookwood Insurance", "Rookwood Insurance > sued > Ironleaf Software"),
    ("sue-03", "nominal", "The lawsuit brought by Hollis Pharma against Quellant Bio was dismissed in June.",
     "org:Hollis Pharma; org:Quellant Bio", "Hollis Pharma > sued > Quellant Bio"),
    ("sue-04", "active", "Elena Volkova filed suit against Juniper Lane Media, claiming the paper had defamed "
     "her.",
     "per:Elena Volkova; org:Juniper Lane Media", "Elena Volkova > sued > Juniper Lane Media"),
    ("sue-05", "active", "Saltmarsh Brewing took Velloway Retail to court over unpaid invoices.",
     "org:Saltmarsh Brewing; org:Velloway Retail", "Saltmarsh Brewing > sued > Velloway Retail"),
    ("sue-06", "nominal", "Kestrel Avionics faces a lawsuit from Zephyr Air over delayed parts.",
     "org:Kestrel Avionics; org:Zephyr Air", "Zephyr Air > sued > Kestrel Avionics"),
    ("sue-07", "nominal", "Tamsin Logistics denied the allegations in the lawsuit Sandoval Freight filed "
     "against it in May.",
     "org:Tamsin Logistics; org:Sandoval Freight", "Sandoval Freight > sued > Tamsin Logistics"),
    ("sue-08", "denied", "Greyfield Partners denied suing Dunmore Bank, saying the case was brought by a "
     "former partner.",
     "org:Greyfield Partners; org:Dunmore Bank", "Greyfield Partners > sued > Dunmore Bank [neg]"),
    ("sue-09", "negated", "Wexley Motors has not sued Kalvik Cells, despite the recall.",
     "org:Wexley Motors; org:Kalvik Cells", "Wexley Motors > sued > Kalvik Cells [neg]"),
    ("sue-10", "planned", "Orrin Metals plans to sue Cobalt Ridge Mining over the contaminated shipment.",
     "org:Orrin Metals; org:Cobalt Ridge Mining", "Orrin Metals > sued > Cobalt Ridge Mining [planned]"),
    ("sue-11", "hypothetical", "Lumio Labs may sue Tessaly Chips if the patent talks fail.",
     "org:Lumio Labs; org:Tessaly Chips", "Lumio Labs > sued > Tessaly Chips [hypothetical]"),
    ("sue-12", "hypothetical", "Arcwell Robotics threatened to sue Grisham Motion.",
     "org:Arcwell Robotics; org:Grisham Motion", "Arcwell Robotics > sued > Grisham Motion [hypothetical/planned]"),
    ("sue-13", "reported", "According to court filings, Brenner & Holt sued Pinecrest Foods in March.",
     "org:Brenner & Holt; org:Pinecrest Foods", "Brenner & Holt > sued > Pinecrest Foods [reported/asserted]"),
    ("sue-14", "nominal", "Halvard Systems settled the lawsuit Brightwater Grid had filed against it.",
     "org:Halvard Systems; org:Brightwater Grid", "Brightwater Grid > sued > Halvard Systems"),
    ("sue-15", "multi nolonger", "Declan Moore left Ironleaf Software in 2022. A year later he sued his former "
     "employer for unpaid bonuses.",
     "per:Declan Moore; org:Ironleaf Software",
     "Declan Moore > sued > Ironleaf Software; Declan Moore > works for > Ironleaf Software [neg]"),
    ("sue-16", "multi", "After Pelham Capital sued Juniper Lane Media, Juniper Lane countersued Pelham for fraud.",
     "org:Pelham Capital|Pelham; org:Juniper Lane Media|Juniper Lane",
     "Pelham Capital > sued > Juniper Lane Media; Juniper Lane Media > sued > Pelham Capital"),
    ("sue-17", "questioned", "Did Yarrow Health sue Hollis Pharma over the trial data?",
     "org:Yarrow Health; org:Hollis Pharma", "Yarrow Health > sued > Hollis Pharma [questioned]"),
    ("sue-18", "distractor", "Regulators in Brussels opened an investigation into Velloway Retail's pricing.",
     "loc:Brussels; org:Velloway Retail", ""),

    # ── board member of ─────────────────────────────────────────────
    ("brd-01", "active", "Ingrid Solberg sits on the board of Dunmore Bank.",
     "per:Ingrid Solberg; org:Dunmore Bank", "Ingrid Solberg > board member of > Dunmore Bank"),
    ("brd-02", "passive", "Tomasz Wierzbicki was appointed to the board of directors of Ostrava Steelworks.",
     "per:Tomasz Wierzbicki; org:Ostrava Steelworks", "Tomasz Wierzbicki > board member of > Ostrava Steelworks"),
    ("brd-03", "nominal", "Mei-Lin Chau's seat on the Tessaly Chips board expires next year.",
     "per:Mei-Lin Chau; org:Tessaly Chips", "Mei-Lin Chau > board member of > Tessaly Chips"),
    ("brd-04", "active", "Owen Pritchard, a non-executive director at Rookwood Insurance, chairs its audit "
     "committee.",
     "per:Owen Pritchard; org:Rookwood Insurance", "Owen Pritchard > board member of > Rookwood Insurance"),
    ("brd-05", "nolonger", "Farah Haddad resigned from the board of Aurelle Cosmetics in January.",
     "per:Farah Haddad; org:Aurelle Cosmetics", "Farah Haddad > board member of > Aurelle Cosmetics [neg]"),
    ("brd-06", "planned", "Jonas Lindqvist will join the board of Brightwater Grid in September.",
     "per:Jonas Lindqvist; org:Brightwater Grid", "Jonas Lindqvist > board member of > Brightwater Grid [planned]"),
    ("brd-07", "hypothetical", "Shareholders may elect Nadia Benali to the board of Velloway Retail at the "
     "annual meeting.",
     "per:Nadia Benali; org:Velloway Retail", "Nadia Benali > board member of > Velloway Retail [hypothetical]"),
    ("brd-08", "denied", "Lucas Ferreira denied that he had ever served on the board of Cobalt Ridge Mining.",
     "per:Lucas Ferreira; org:Cobalt Ridge Mining", "Lucas Ferreira > board member of > Cobalt Ridge Mining [neg]"),
    ("brd-09", "multi", "Sofia Marchetti is the chief financial officer of Hollis Pharma. She is also a "
     "director of Yarrow Health.",
     "per:Sofia Marchetti; org:Hollis Pharma; org:Yarrow Health",
     "Sofia Marchetti > works for > Hollis Pharma; Sofia Marchetti > board member of > Yarrow Health"),
    ("brd-10", "active", "Kwame Asante chairs the board of Saltmarsh Brewing.",
     "per:Kwame Asante; org:Saltmarsh Brewing", "Kwame Asante > board member of > Saltmarsh Brewing"),
    ("brd-11", "negated", "Priya Raman works for Velloway Retail but is not on its board.",
     "per:Priya Raman; org:Velloway Retail",
     "Priya Raman > works for > Velloway Retail; Priya Raman > board member of > Velloway Retail [neg]"),
    ("brd-12", "list", "The board of Kestrel Avionics includes Amara Okonkwo, Rafael Duarte and Elena Volkova.",
     "org:Kestrel Avionics; per:Amara Okonkwo; per:Rafael Duarte; per:Elena Volkova",
     "Amara Okonkwo > board member of > Kestrel Avionics; Rafael Duarte > board member of > Kestrel Avionics; "
     "Elena Volkova > board member of > Kestrel Avionics"),
    ("brd-13", "reported", "Ingrid Solberg is expected to join the board of Pelham Capital, people close to the "
     "firm said.",
     "per:Ingrid Solberg; org:Pelham Capital",
     "Ingrid Solberg > board member of > Pelham Capital [reported/planned]"),
    ("brd-14", "questioned", "Is Hana Kobayashi still a director of Grisham Motion?",
     "per:Hana Kobayashi; org:Grisham Motion", "Hana Kobayashi > board member of > Grisham Motion [questioned]"),

    # ── priced at ───────────────────────────────────────────────────
    ("prc-01", "active", "The Halvard H7 controller sells for $1,250.",
     "prod:Halvard H7; mon:$1,250", "Halvard H7 > priced at > $1,250"),
    ("prc-02", "active", "Wexley Motors priced the Strata hatchback at €31,900.",
     "org:Wexley Motors; prod:Strata; mon:€31,900", "Strata > priced at > €31,900"),
    ("prc-03", "active", "The Lumio Pocket costs 129 dollars in the United States.",
     "prod:Lumio Pocket; mon:129 dollars; loc:United States", "Lumio Pocket > priced at > 129 dollars"),
    ("prc-04", "nominal", "At $49 a month, Ironleaf Ledger is the cheapest payroll package on the market.",
     "mon:$49; prod:Ironleaf Ledger", "Ironleaf Ledger > priced at > $49"),
    ("prc-05", "active", "Aurelle Cosmetics cut the price of its Night Serum from $85 to $62.",
     "org:Aurelle Cosmetics; prod:Night Serum; mon:$85; mon:$62",
     "Night Serum > priced at > $62; Night Serum > priced at > $85 [opt]"),
    ("prc-06", "planned", "Arcwell Robotics will price the R2 arm at $18,000 when it launches in May.",
     "org:Arcwell Robotics; prod:R2 arm; mon:$18,000", "R2 arm > priced at > $18,000 [planned]"),
    ("prc-07", "hypothetical", "The Tessaly T9 chip could cost as little as $40 in volume.",
     "prod:Tessaly T9; mon:$40", "Tessaly T9 > priced at > $40 [hypothetical]"),
    ("prc-08", "reported", "Retailers say the Zephyr Flex fare is now sold at £89.",
     "prod:Zephyr Flex; mon:£89", "Zephyr Flex > priced at > £89 [reported]"),
    ("prc-09", "negated", "Contrary to earlier reports, the Kestrel K4 headset is not priced at $999.",
     "prod:Kestrel K4; mon:$999", "Kestrel K4 > priced at > $999 [neg]"),
    ("prc-10", "multi", "Kalvik Cells sells its K-Pack battery for $4,200 per unit. Wexley Motors is its largest "
     "customer.",
     "org:Kalvik Cells; prod:K-Pack; mon:$4,200; org:Wexley Motors",
     "K-Pack > priced at > $4,200; Wexley Motors > customer of > Kalvik Cells"),
    ("prc-11", "list", "The basic Halvard Hub costs $199 and the Hub Pro costs $349.",
     "prod:Halvard Hub; prod:Hub Pro; mon:$199; mon:$349",
     "Halvard Hub > priced at > $199; Hub Pro > priced at > $349"),
    ("prc-12", "questioned", "Is the Lumio Pocket still priced at $129?",
     "prod:Lumio Pocket; mon:$129", "Lumio Pocket > priced at > $129 [questioned]"),
    ("prc-13", "desired", "Velloway Retail hopes to sell the Brightline kettle at under $30.",
     "org:Velloway Retail; prod:Brightline; mon:$30", "Brightline > priced at > $30 [desired/planned]"),
    ("prc-14", "distractor", "Analysts at Pelham Capital expect phone prices to fall next year.",
     "org:Pelham Capital", ""),

    # ── works for / based in (controls) ─────────────────────────────
    ("wrk-01", "active", "Lucas Ferreira is a software engineer at Ironleaf Software.",
     "per:Lucas Ferreira; org:Ironleaf Software", "Lucas Ferreira > works for > Ironleaf Software"),
    ("wrk-02", "active", "Hollis Pharma hired Nadia Benali as head of regulatory affairs.",
     "org:Hollis Pharma; per:Nadia Benali", "Nadia Benali > works for > Hollis Pharma"),
    ("wrk-03", "active", "Tomasz Wierzbicki, who leads sales at Orrin Metals, met buyers in Gdansk.",
     "per:Tomasz Wierzbicki; org:Orrin Metals; loc:Gdansk", "Tomasz Wierzbicki > works for > Orrin Metals"),
    ("wrk-04", "nolonger", "Elena Volkova left Juniper Lane Media in 2021 to start her own firm.",
     "per:Elena Volkova; org:Juniper Lane Media", "Elena Volkova > works for > Juniper Lane Media [neg]"),
    ("wrk-05", "planned", "Owen Pritchard will join Sandoval Freight as finance director in July.",
     "per:Owen Pritchard; org:Sandoval Freight", "Owen Pritchard > works for > Sandoval Freight [planned]"),
    ("wrk-06", "active", "Mei-Lin Chau has worked at Tessaly Chips for eleven years.",
     "per:Mei-Lin Chau; org:Tessaly Chips", "Mei-Lin Chau > works for > Tessaly Chips"),
    ("wrk-07", "questioned", "Does Jonas Lindqvist still work for Brightwater Grid?",
     "per:Jonas Lindqvist; org:Brightwater Grid", "Jonas Lindqvist > works for > Brightwater Grid [questioned]"),
    ("wrk-08", "multi", "Farah Haddad runs the Lyon office of Aurelle Cosmetics. The company is based in Paris.",
     "per:Farah Haddad; loc:Lyon; org:Aurelle Cosmetics; loc:Paris",
     "Farah Haddad > works for > Aurelle Cosmetics; Aurelle Cosmetics > based in > Paris; "
     "Aurelle Cosmetics > based in > Lyon [opt]"),
    ("bas-01", "active", "Quellant Bio is headquartered in Basel.",
     "org:Quellant Bio; loc:Basel", "Quellant Bio > based in > Basel"),
    ("bas-02", "nominal", "Verdant Agro, a seed company from Pune, sells to farmers across India.",
     "loc:Pune; org:Verdant Agro; loc:India", "Verdant Agro > based in > Pune"),
    ("bas-03", "nolonger", "Dunmore Bank moved its headquarters from Leeds to Manchester in 2019.",
     "org:Dunmore Bank; loc:Leeds; loc:Manchester",
     "Dunmore Bank > based in > Manchester; Dunmore Bank > based in > Leeds [neg]"),
    ("bas-04", "nominal", "Sandoval Freight, a Monterrey trucking company, operates 900 lorries.",
     "org:Sandoval Freight; loc:Monterrey", "Sandoval Freight > based in > Monterrey"),
    ("bas-05", "planned", "Lumio Labs plans to move its headquarters to Lisbon next year.",
     "org:Lumio Labs; loc:Lisbon", "Lumio Labs > based in > Lisbon [planned]"),
    ("bas-06", "negated", "Despite its name, Ostrava Steelworks is not based in Ostrava. Its head office is in "
     "Brno.",
     "org:Ostrava Steelworks; loc:Ostrava; loc:Brno",
     "Ostrava Steelworks > based in > Ostrava [neg]; Ostrava Steelworks > based in > Brno"),
    ("bas-07", "active", "Norland Cargo operates from Tromsø, where it was founded.",
     "org:Norland Cargo; loc:Tromsø", "Norland Cargo > based in > Tromsø"),
    ("bas-08", "questioned", "Is Kestrel Avionics still based in Toulouse?",
     "org:Kestrel Avionics; loc:Toulouse", "Kestrel Avionics > based in > Toulouse [questioned]"),

    # ── distractors: no target fact ─────────────────────────────────
    ("dis-01", "distractor", "Halvard Systems and Lumio Labs both exhibited at the trade fair in Hanover.",
     "org:Halvard Systems; org:Lumio Labs; loc:Hanover", ""),
    ("dis-02", "distractor", "Kestrel Avionics compared bids from Orrin Metals and Fenwick Glass but has not "
     "chosen a supplier.",
     "org:Kestrel Avionics; org:Orrin Metals; org:Fenwick Glass",
     "Orrin Metals > supplier of > Kestrel Avionics [opt neg]; Fenwick Glass > supplier of > Kestrel Avionics [opt neg]"),
    ("dis-03", "distractor", "Ingrid Solberg praised Dunvale Textiles in a speech at a business forum in Leeds.",
     "per:Ingrid Solberg; org:Dunvale Textiles; loc:Leeds", ""),
    ("dis-04", "distractor", "Pelham Capital published a report on the battery market, naming Kalvik Cells as "
     "one to watch.",
     "org:Pelham Capital; org:Kalvik Cells", ""),
    ("dis-05", "distractor", "Rookwood Insurance sponsored the Calgary marathon for the third year.",
     "org:Rookwood Insurance; loc:Calgary", ""),
    ("dis-06", "distractor", "A new lager from Saltmarsh Brewing won a medal at a competition in Porto.",
     "org:Saltmarsh Brewing; loc:Porto", ""),
    ("dis-07", "distractor", "Sofia Marchetti met Kwame Asante in Nairobi to discuss a joint research grant.",
     "per:Sofia Marchetti; per:Kwame Asante; loc:Nairobi", ""),
    ("dis-08", "distractor", "Zephyr Air and Norland Cargo signed a code-share agreement covering flights to "
     "Brisbane.",
     "org:Zephyr Air; org:Norland Cargo; loc:Brisbane", ""),
    ("dis-09", "distractor", "The price of copper rose sharply after Cobalt Ridge Mining reported a flood at its "
     "main site.",
     "org:Cobalt Ridge Mining", ""),
    ("dis-10", "distractor", "Lawyers for Yarrow Health reviewed the draft contract with Hollis Pharma before "
     "talks stalled.",
     "org:Yarrow Health; org:Hollis Pharma", ""),
    ("dis-11", "distractor", "Juniper Lane Media interviewed Elena Volkova about her childhood in Gdansk.",
     "org:Juniper Lane Media; per:Elena Volkova; loc:Gdansk", ""),
    ("dis-12", "distractor", "Turbines owned by Brightwater Grid were damaged in a storm near Aberdeen.",
     "org:Brightwater Grid; loc:Aberdeen", ""),
    ("dis-13", "distractor", "Nadia Benali wrote a book about the history of Tamsin Logistics.",
     "per:Nadia Benali; org:Tamsin Logistics", ""),
    ("dis-14", "distractor", "Velloway Retail and Pinecrest Foods compete for shoppers in Lyon.",
     "org:Velloway Retail; org:Pinecrest Foods; loc:Lyon", ""),
    ("dis-15", "distractor", "Arcwell Robotics showed the R2 arm to visitors from Wexley Motors.",
     "org:Arcwell Robotics; prod:R2 arm; org:Wexley Motors", ""),

    # ── mixed paragraphs ────────────────────────────────────────────
    ("mix-01", "multi", "Brask Minerals, a mining company based in Perth, agreed on Monday to sell its nickel "
     "unit to Orrin Metals for $640 million. Orrin, which already supplies nickel to Kalvik Cells, said the deal "
     "would secure its supply for a decade.",
     "org:Brask Minerals; loc:Perth; org:Orrin Metals|Orrin; mon:$640 million; org:Kalvik Cells",
     "Brask Minerals > based in > Perth; Orrin Metals > supplier of > Kalvik Cells"),
    ("mix-02", "multi", "Hollis Pharma, headquartered in Boston, has sued Quellant Bio over a licensing dispute. "
     "Quellant, which makes reagents for the labs of Hollis, called the suit baseless.",
     "org:Hollis Pharma|Hollis; loc:Boston; org:Quellant Bio|Quellant",
     "Hollis Pharma > based in > Boston; Hollis Pharma > sued > Quellant Bio; Quellant Bio > supplier of > Hollis Pharma"),
    ("mix-03", "multi list", "Greyfield Partners owns 40% of Tamsin Logistics and has two seats on its board, "
     "held by Rafael Duarte and Hana Kobayashi. Both work for Greyfield.",
     "org:Greyfield Partners|Greyfield; org:Tamsin Logistics; per:Rafael Duarte; per:Hana Kobayashi",
     "Greyfield Partners > holds stake in > Tamsin Logistics; Rafael Duarte > board member of > Tamsin Logistics; "
     "Hana Kobayashi > board member of > Tamsin Logistics; Rafael Duarte > works for > Greyfield Partners; "
     "Hana Kobayashi > works for > Greyfield Partners"),
    ("mix-04", "multi planned", "Velloway Retail will start selling the Lumio Pocket in its stores next month "
     "at $119, ten dollars below the list price. Lumio Labs, based in Oslo, makes the device.",
     "org:Velloway Retail; prod:Lumio Pocket; mon:$119; mon:ten dollars; org:Lumio Labs; loc:Oslo",
     "Lumio Pocket > priced at > $119 [planned/asserted]; Lumio Labs > based in > Oslo; "
     "Lumio Labs > supplier of > Velloway Retail [opt planned]"),
    ("mix-05", "multi nolonger", "Ironleaf Software lost its biggest customer this year when Dunmore Bank moved "
     "its payroll to Halvard Systems. Ironleaf is now being sued by Dunmore over data that was not returned.",
     "org:Ironleaf Software|Ironleaf; org:Dunmore Bank|Dunmore; org:Halvard Systems",
     "Dunmore Bank > customer of > Ironleaf Software [neg]; Dunmore Bank > customer of > Halvard Systems; "
     "Dunmore Bank > sued > Ironleaf Software"),
    ("mix-06", "multi nolonger", "Kwame Asante, the founder of Saltmarsh Brewing, sold his remaining shares in "
     "the brewer to Pelham Capital in 2024. He stayed on as chairman of the board.",
     "per:Kwame Asante; org:Saltmarsh Brewing; org:Pelham Capital",
     "Kwame Asante > holds stake in > Saltmarsh Brewing [neg]; Pelham Capital > holds stake in > Saltmarsh Brewing; "
     "Kwame Asante > board member of > Saltmarsh Brewing; Kwame Asante > works for > Saltmarsh Brewing [opt]"),
    ("mix-07", "multi denied", "Wexley Motors denied a newspaper report that it plans to buy Kalvik Cells. A "
     "spokesperson said the carmaker remains a customer of Kalvik and has no interest in owning it.",
     "org:Wexley Motors; org:Kalvik Cells|Kalvik",
     "Wexley Motors > acquired > Kalvik Cells [neg asserted/planned]; Wexley Motors > customer of > Kalvik Cells"),
    ("mix-08", "multi", "Tessaly Chips makes the T9 processor in Taiwan and sells it to Arcwell Robotics for "
     "about $38 a unit. Arcwell also holds a small stake in Tessaly.",
     "org:Tessaly Chips|Tessaly; prod:T9; loc:Taiwan; org:Arcwell Robotics|Arcwell; mon:$38",
     "T9 > priced at > $38; Tessaly Chips > supplier of > Arcwell Robotics; Arcwell Robotics > holds stake in > Tessaly Chips; "
     "Tessaly Chips > based in > Taiwan [opt]"),
    ("mix-09", "multi negated", "Shareholders of Juniper Lane Media voted against the sale of the company to "
     "Greyfield Partners. Greyfield, which owns 15 percent of Juniper Lane, said it would not raise its offer.",
     "org:Juniper Lane Media|Juniper Lane; org:Greyfield Partners|Greyfield",
     "Greyfield Partners > holds stake in > Juniper Lane Media; Greyfield Partners > acquired > Juniper Lane Media [opt neg]"),
    ("mix-10", "multi nolonger", "Owen Pritchard joined Rookwood Insurance as chief risk officer after twelve "
     "years at Dunmore Bank. Rookwood is based in Edinburgh.",
     "per:Owen Pritchard; org:Rookwood Insurance|Rookwood; org:Dunmore Bank; loc:Edinburgh",
     "Owen Pritchard > works for > Rookwood Insurance; Owen Pritchard > works for > Dunmore Bank [neg]; "
     "Rookwood Insurance > based in > Edinburgh"),
    ("mix-11", "multi negated", "Aurelle Cosmetics accused Fenwick Glass of shipping defective bottles but has "
     "not filed a lawsuit. Fenwick says it will keep supplying Aurelle under the current contract.",
     "org:Aurelle Cosmetics|Aurelle; org:Fenwick Glass|Fenwick",
     "Aurelle Cosmetics > sued > Fenwick Glass [neg]; Fenwick Glass > supplier of > Aurelle Cosmetics [asserted/reported]"),
    ("mix-12", "multi list", "Verdant Agro, the Pune seed company, buys its fertiliser from Cobalt Ridge Mining "
     "and its packaging from Moraine Paper. Its chairman, Amara Okonkwo, also sits on the board of Moraine.",
     "org:Verdant Agro; loc:Pune; org:Cobalt Ridge Mining; org:Moraine Paper|Moraine; per:Amara Okonkwo",
     "Verdant Agro > based in > Pune; Verdant Agro > customer of > Cobalt Ridge Mining; "
     "Verdant Agro > customer of > Moraine Paper; Amara Okonkwo > board member of > Verdant Agro; "
     "Amara Okonkwo > board member of > Moraine Paper"),
    ("mix-13", "multi", "Zephyr Air sells its Flex fare between Oslo and Tromsø for 690 kroner. The airline, "
     "based in Bergen, is part-owned by Norland Cargo.",
     "org:Zephyr Air; prod:Flex fare; loc:Oslo; loc:Tromsø; mon:690 kroner; loc:Bergen; org:Norland Cargo",
     "Flex fare > priced at > 690 kroner; Zephyr Air > based in > Bergen; Norland Cargo > holds stake in > Zephyr Air"),
    ("mix-14", "multi typetrap", "Rafael Duarte, a buyer at Pinecrest Foods, said the company had stopped "
     "ordering flour from Verdant Agro after a quality dispute.",
     "per:Rafael Duarte; org:Pinecrest Foods; org:Verdant Agro",
     "Rafael Duarte > works for > Pinecrest Foods; Pinecrest Foods > customer of > Verdant Agro [neg]"),
    ("mix-15", "multi", "Brenner & Holt, the law firm, represented Saltmarsh Brewing in its lawsuit against "
     "Velloway Retail. The firm is based in Dublin.",
     "org:Brenner & Holt; org:Saltmarsh Brewing; org:Velloway Retail; loc:Dublin",
     "Saltmarsh Brewing > sued > Velloway Retail; Brenner & Holt > based in > Dublin; "
     "Brenner & Holt > supplier of > Saltmarsh Brewing [opt]"),
]


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


def _find(tokens: list[str], sub: list[str], taken: set) -> list[tuple[int, int]]:
    n, out = len(sub), []
    for i in range(len(tokens) - n + 1):
        if tokens[i:i + n] == sub and not any(j in taken for j in range(i, i + n)):
            out.append((i, i + n))
            taken.update(range(i, i + n))
    return out


def build() -> int:
    from ..infer import Extractor
    from ..windows import tokenize
    items, problems = [], []
    for sid, tags, text, ents_s, facts_s in SPEC:
        ws = Extractor.windows(text)
        assert len(ws) == 1, sid
        toks, sp = ws[0]["tokens"], ws[0]["sentence_spans"]
        ents = _parse_ents(ents_s)
        # longest surface first, so "Ostrava Steelworks" claims its words before "Ostrava"
        order = sorted(((len(tokenize(n)), k, n) for k, (_, ns) in enumerate(ents) for n in ns), reverse=True)
        spans: dict[int, list] = defaultdict(list)
        taken: set = set()
        for _, k, n in order:
            found = _find(toks, tokenize(n), taken)
            if not found and n == ents[k][1][0] and not spans[k]:
                problems.append(f"{sid}: '{n}' not in text")
            spans[k] += found
        for k, (_, ns) in enumerate(ents):
            if not spans[k]:
                problems.append(f"{sid}: entity '{ns[0]}' has no mention")
        idx = {ns[0]: k for k, (_, ns) in enumerate(ents)}
        facts = []
        for f in _parse_facts(facts_s):
            for x in (f["head"], f["tail"]):
                if x not in idx:
                    problems.append(f"{sid}: fact argument '{x}' is not an entity")
            if f["head"] not in idx or f["tail"] not in idx:
                continue
            h, t = idx[f["head"]], idx[f["tail"]]
            ch, ct = CONSTRAINT[f["relation"]]
            if ents[h][0] not in ch or ents[t][0] not in ct:
                problems.append(f"{sid}: expected fact breaks its own type constraint: {f}")
            g = {"head": h, "relation": f["relation"], "tail": t, "polarity": f["polarity"],
                 "modality": f["modality"], "optional": f["optional"]}
            facts.append(g)
            if f["relation"] in INVERSE:
                facts.append({**g, "head": t, "tail": h, "relation": INVERSE[f["relation"]], "inverse": True})
        # an inverse never overrides a fact written out
        seen, uniq = set(), []
        for g in sorted(facts, key=lambda g: g.get("inverse", False)):
            key = (g["head"], g["relation"], g["tail"])
            if key not in seen:
                seen.add(key)
                uniq.append(g)
        req = [g for g in uniq if not g["optional"]]
        opt = [g for g in uniq if g["optional"]]
        items.append({
            "id": f"m4:{sid}", "tags": tags.split(), "text": text, "tokens": toks, "sentence_spans": sp,
            "participants": [], "speakers": None, "domain": "business",
            "entities": [{"type": ty, "spans": sorted(list(s) for s in spans[k]), "name": ns[0]}
                         for k, (ty, ns) in enumerate(ents)],
            "facts": [{k: g[k] for k in ("head", "tail", "relation", "polarity", "modality")} for g in req],
            "optional_facts": [{k: g[k] for k in ("head", "tail", "relation", "polarity", "modality")} for g in opt],
            "relations": RELATIONS, "types": TYPES})
    if problems:
        print("\n".join(problems))
        return 1
    DATA.write_text(json.dumps({"relations": RELATIONS, "types": TYPES, "constraints":
                                {k: [sorted(a), sorted(b)] for k, (a, b) in CONSTRAINT.items()},
                                "bank_names": BANK_NAME, "items": items}, ensure_ascii=False, indent=0))
    c = counts(items)
    print(f"{len(items)} items, {c['facts']} expected facts ({c['optional']} optional) -> {DATA}")
    return 0


def load(limit: int | None = None) -> list[dict]:
    """The items in ``heldout_zs``'s short-dev format (M1 reads these)."""
    items = json.loads(DATA.read_text())["items"]
    return items[:limit] if limit else items


def counts(items: list[dict]) -> dict:
    rel, kind, form = Counter(), Counter(), Counter()
    for it in items:
        for t in it["tags"]:
            form[t] += 1
        for f in it["facts"]:
            rel[f["relation"]] += 1
            kind[attr_kind(f)] += 1
    return {"items": len(items), "facts": sum(rel.values()),
            "optional": sum(len(it["optional_facts"]) for it in items),
            "per_relation": dict(rel), "per_kind": dict(kind), "per_form": dict(form)}


def attr_kind(f: dict) -> str:
    if f["polarity"] == "negative":
        return "negated"
    return f["modality"][0] if len(f["modality"]) == 1 or f["modality"][0] != "asserted" else "asserted"


# ── predict ────────────────────────────────────────────────────────
def schema(naming: str) -> tuple[list[dict], dict]:
    """The relations to pass and a map from passed name to the suite's name."""
    if naming == "user":
        return RELATIONS, {r["name"]: r["name"] for r in RELATIONS}
    from ..label_bank import load as load_bank
    bank = {r["name"]: r for r in load_bank()["relations"]}
    rels, back = [], {}
    for r in RELATIONS:
        b = BANK_NAME.get(r["name"])
        if b and b in bank:
            rels.append({"name": b, "description": bank[b].get("description", "")})
            back[b] = r["name"]
        else:
            rels.append(r)
            back[r["name"]] = r["name"]
    return rels, back


def predict(ex, items: list[dict], mode: str, naming: str) -> list[dict]:
    """Per item, every candidate fact the decoder ranks (top-k per pair) with
    its margin above the deployed bar (``m``), its argument types and the
    attribute head's polarity / modality probabilities."""
    import torch
    rels, back = schema(naming)
    cap, cap_u = {}, {}
    fwd = ex.model.forward

    def capture(b, **kw):
        cap["b"] = b
        return fwd(b, **kw)
    ex.model.forward = capture
    m = ex.model
    u = ex.unseen_ex                            # KNIV_UNSEEN_FROM: routed relations' batch is the other model's
    if u is not None:
        fwd_u = u.model.forward

        def capture_u(b, **kw):
            cap_u["b"] = b
            return fwd_u(b, **kw)
        u.model.forward = capture_u
    rows = []
    try:
        for it in items:
            w = {"tokens": it["tokens"], "sentence_spans": it["sentence_spans"]}
            if mode == "gold":
                ents = [{"type": e["type"], "mentions": sorted(tuple(s) for s in e["spans"])}
                        for e in it["entities"]]
            else:
                ents = ex._entities(w, it["types"])
            cap.clear(); cap_u.clear()
            nodes, facts = ex._relations(w, ents, list(rels), ex.top)
            first = {tuple(mm): i for i, e in enumerate(ents) for mm in e["mentions"]}
            keys = []
            for n in nodes:
                if mode == "gold":
                    keys.append(first[tuple(n.get("model_spans", n["spans"])[0])])
                else:
                    keys.append(frozenset(x.lower() for x in n["mentions"]))
            cand = []
            if facts:
                routed = set()
                if u is not None and cap_u:
                    routed = {r["name"] for r in rels if r["name"] not in ex.ver_trained
                              and r["name"] not in ex.rel_offsets}
                att = {}
                for mod_, cb, names in ((m, cap.get("b"), [r["name"] for r in rels if r["name"] not in routed]),
                                        (u.model if u is not None else None, cap_u.get("b"),
                                         [r["name"] for r in rels if r["name"] in routed])):
                    js = [j for j, f in enumerate(facts) if (f["relation"] in routed) == (mod_ is not m)]
                    if not js or cb is None or getattr(mod_, "attr", None) is None:
                        continue
                    pairs = list(zip(cb["pair_h"].tolist(), cb["pair_t"].tolist()))
                    P = torch.tensor([pairs.index((facts[j]["head"], facts[j]["tail"])) for j in js], device=ex.device)
                    K = torch.tensor([names.index(facts[j]["relation"]) for j in js], device=ex.device)
                    with torch.no_grad():
                        a_ = mod_.attr(mod_.relation._pair, mod_.relation._labels, mod_.relation._ent, cb, P, K)
                    for i, j in enumerate(js):
                        att[j] = (a_["polarity"][i], a_["modality"][i])
                a = att or None
                for j, f in enumerate(facts):
                    c = {"key": (keys[f["head"]], back[f["relation"]], keys[f["tail"]]),
                         "rel": back[f["relation"]], "m": f["margin"] - SHIFT,
                         "unseen": f["relation"] not in ex.rel_offsets,
                         "types": (nodes[f["head"]]["type"], nodes[f["tail"]]["type"]),
                         "texts": (nodes[f["head"]]["text"], nodes[f["tail"]]["text"])}
                    if a is not None and j in a:
                        c["pol"] = torch.softmax(a[j][0], -1).cpu().tolist()
                        c["mod"] = torch.softmax(a[j][1], -1).cpu().tolist()
                    cand.append(c)
            rows.append({"id": it["id"], "cand": cand,
                         "ents": [(e.get("type"), [tuple(x) for x in e["mentions"]]) for e in ents]})
    finally:
        ex.model.forward = fwd
        if u is not None:
            u.model.forward = fwd_u
    return rows


# ── score ──────────────────────────────────────────────────────────
def _mentions(it: dict, k: int) -> set[str]:
    return {" ".join(it["tokens"][a:b]).lower() for a, b in it["entities"][k]["spans"]}


def _match(it: dict, mode: str, key, f: dict) -> bool:
    h, r, t = key
    if r != f["relation"]:
        return False
    if mode == "gold":
        return h == f["head"] and t == f["tail"]
    return bool(h & _mentions(it, f["head"])) and bool(t & _mentions(it, f["tail"]))


def score_rows(items: list[dict], rows: list[dict], mode: str, delta: float = 0.0,
               bank: set | None = None) -> dict:
    """Facts above the bar (new-relation path moved by ``delta``) against the
    expected facts; optional facts are ignored either way."""
    from ..schemas import MODALITIES, POLARITIES
    tot = np.zeros(3)
    per_rel = defaultdict(lambda: np.zeros(3))
    per_form = defaultdict(lambda: np.zeros(3))
    per_kind = defaultdict(lambda: np.zeros(4))   # expected, returned, polarity right, both right
    viol = np.zeros(2)
    per_item = []
    misses, false = [], []
    for it, row in zip(items, rows):
        pred, seen = [], set()
        for c in sorted(row["cand"], key=lambda c: -c["m"]):
            if c["m"] > (delta if c["unseen"] else 0.0) and c["key"] not in seen:
                seen.add(c["key"])
                pred.append(c)
        used = set()
        hit = {}
        fp = 0
        for c in pred:
            ch, ct = CONSTRAINT[c["rel"]]
            if mode == "gold":
                ty = (it["entities"][c["key"][0]]["type"], it["entities"][c["key"][2]]["type"])
            else:
                ty = c["types"]
            viol += [ty[0] not in ch or ty[1] not in ct, 1]
            j = next((j for j, f in enumerate(it["facts"]) if j not in used and _match(it, mode, c["key"], f)), None)
            if j is not None:
                used.add(j)
                hit[j] = c
                continue
            if any(_match(it, mode, c["key"], f) for f in it["optional_facts"]):
                continue
            fp += 1
            per_rel[c["rel"]] += [0, 1, 0]
            false.append((it["id"], c["texts"][0], c["rel"], c["texts"][1], round(c["m"], 2)))
        tp = len(hit)
        c3 = np.array([tp, tp + fp, len(it["facts"])])
        tot += c3
        per_item.append(c3)
        for t in it["tags"]:
            per_form[t] += c3
        for j, f in enumerate(it["facts"]):
            k = attr_kind(f)
            per_rel[f["relation"]] += [j in hit, j in hit, 1]
            if j not in hit:
                per_kind[k] += [1, 0, 0, 0]
                misses.append((it["id"], it["entities"][f["head"]]["name"], f["relation"],
                               it["entities"][f["tail"]]["name"], k))
                continue
            c = hit[j]
            pr = mr = False
            if "pol" in c:
                pr = POLARITIES[int(np.argmax(c["pol"]))] == f["polarity"]
                mr = MODALITIES[int(np.argmax(c["mod"]))] in f["modality"]
            per_kind[k] += [1, 1, pr, pr and mr]
    f1 = lambda c: (c[0] / max(c[1], 1), c[0] / max(c[2], 1),
                    2 * c[0] / max(c[1] + c[2], 1))
    oob = sorted(r["name"] for r in RELATIONS if bank is not None and r["name"] not in bank)
    return {"prf": f1(tot), "counts": tot.tolist(),
            "per_relation": {r: {"prf": f1(c), "counts": c.tolist()} for r, c in sorted(per_rel.items())},
            "per_form": {r: {"prf": f1(c), "counts": c.tolist()} for r, c in sorted(per_form.items())},
            "per_kind": {k: {"n": int(c[0]), "returned": c[1] / max(c[0], 1),
                             "polarity_right_when_returned": c[2] / max(c[1], 1),
                             "both_right_when_returned": c[3] / max(c[1], 1),
                             "returned_and_right": c[3] / max(c[0], 1)} for k, c in sorted(per_kind.items())},
            "type_violations": viol[0] / max(viol[1], 1), "n_pred": int(viol[1]),
            "out_of_bank": {r: {"found": int(per_rel[r][0]), "of": int(per_rel[r][2])} for r in oob},
            "per_item": [c.tolist() for c in per_item], "misses": misses, "false": false}


def bootstrap(a: list, b: list, n: int = 2000, seed: int = 13):
    a, b = np.array(a), np.array(b)
    rng = np.random.default_rng(seed)
    f = lambda c: 2 * c[:, 0].sum() / max(c[:, 1].sum() + c[:, 2].sum(), 1)
    d = []
    for _ in range(n):
        i = rng.integers(0, len(a), len(a))
        d.append(f(a[i]) - f(b[i]))
    return float(f(a) - f(b)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def report(name: str, s: dict) -> str:
    p, r, f = s["prf"]
    L = [f"### {name}", "",
         f"fact P / R / F1 {p:.3f} / {r:.3f} / {f:.3f} ({int(s['counts'][0])} of {int(s['counts'][2])} found, "
         f"{int(s['counts'][1])} predicted); type violations {s['type_violations']:.3f} of {s['n_pred']} "
         "predicted facts", "",
         "| relation | P | R | F1 | expected |", "|---|---|---|---|---|"]
    for k, v in s["per_relation"].items():
        L.append(f"| {k} | {v['prf'][0]:.2f} | {v['prf'][1]:.2f} | {v['prf'][2]:.2f} | {int(v['counts'][2])} |")
    L += ["", "| expected kind | n | returned | polarity right (returned) | both right (returned) | "
          "returned and right |", "|---|---|---|---|---|---|"]
    for k, v in s["per_kind"].items():
        L.append(f"| {k} | {v['n']} | {v['returned']:.2f} | {v['polarity_right_when_returned']:.2f} | "
                 f"{v['both_right_when_returned']:.2f} | {v['returned_and_right']:.2f} |")
    L += ["", "| form | P | R | F1 | expected |", "|---|---|---|---|---|"]
    for k, v in s["per_form"].items():
        L.append(f"| {k} | {v['prf'][0]:.2f} | {v['prf'][1]:.2f} | {v['prf'][2]:.2f} | {int(v['counts'][2])} |")
    if s["out_of_bank"]:
        L += ["", "out of bank: " + ", ".join(f"{k} {v['found']}/{v['of']}" for k, v in s["out_of_bank"].items())]
    return "\n".join(L)


def score(run: str, local: bool, naming: str, modes: list[str], device: str) -> int:
    from ..infer import Extractor
    from ..label_bank import load as load_bank
    from ..train.gold_eval import CKPT_DIR
    items = load()
    ex = Extractor(run, device=device, local=CKPT_DIR if local else None)
    ex.rel_offsets = {k: v + SHIFT for k, v in ex.rel_offsets.items()}
    ex.unseen_offset -= SHIFT
    ex.top = max(ex.top, 4)
    import os
    tag = ""
    if os.environ.get("KNIV_VERIFIER"):     # plan r4 stage 3: the verifier decides the relations it was trained on
        ex.use_verifier(os.environ["KNIV_VERIFIER"], threshold=float(os.environ.get("KNIV_VER_THRESHOLD", "0.48")))
        ex.margin_shift = SHIFT
        tag = "-ver" if os.environ["KNIV_VERIFIER"].endswith("-ver1") else "-" + os.environ["KNIV_VERIFIER"].split("-")[-1]
    if ex.trim_det:                          # KNIV_TRIM_DET=1: leading articles trimmed from mentions
        tag += "-det"
    if ex.unseen_ex is not None:             # KNIV_UNSEEN_FROM=RUN: never-trained relations through RUN
        tag += "-u" + ex.unseen_ex.run.split("-")[-1]
    run_file = run + tag
    rels, _ = schema(naming)
    bank = {r["name"] for r in load_bank()["relations"]}
    bank_suite = {r["name"] for r in RELATIONS if (BANK_NAME.get(r["name"]) if naming == "bank" else r["name"]) in bank}
    OUT.mkdir(parents=True, exist_ok=True)
    res = {"run": run, "naming": naming, "relations": rels, "unseen_offset": ex.unseen_offset + SHIFT,
           "counts": counts(items)}
    for mode in modes:
        t0 = time.time()
        rows = predict(ex, items, mode, naming)
        s = score_rows(items, rows, mode, bank=bank_suite)
        sweep = {f"{d:+.1f}": score_rows(items, rows, mode, delta=d)["prf"][2] for d in np.arange(-3, 3.01, 0.5)}
        s["sweep_new_relation_bar"] = sweep
        s["seconds"] = time.time() - t0
        res[mode] = s
        (OUT / f"rows-{run_file}-{naming}-{mode}.json").write_text(json.dumps(
            [{"id": r["id"], "cand": [{**c, "key": [sorted(x) if isinstance(x, frozenset) else x for x in c["key"]]}
                                      for c in r["cand"]], "ents": r["ents"]} for r in rows], default=list))
        print(report(f"{run}, {naming} names, {mode} entities", s), flush=True)
        print("bar sweep (F1 by new-relation offset delta): "
              + " ".join(f"{k}:{v:.3f}" for k, v in sweep.items()), flush=True)
        print(f"[{s['seconds']:.0f}s]\n", flush=True)
    (OUT / f"score-{run_file}-{naming}.json").write_text(json.dumps(res, indent=1, default=list))
    print(f"written {OUT / f'score-{run_file}-{naming}.json'}")
    return 0


def compare(a: str, b: str) -> int:
    """Paired bootstrap of fact F1 between two score files (same items)."""
    A, B = (json.loads((OUT / f"score-{x}.json").read_text()) for x in (a, b))
    for mode in ("gold", "e2e"):
        if mode in A and mode in B:
            d, lo, hi = bootstrap(A[mode]["per_item"], B[mode]["per_item"])
            print(f"{mode}: {a} - {b} F1 {d * 100:+.1f} [{lo * 100:+.1f}, {hi * 100:+.1f}]")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    sub.add_parser("counts")
    s = sub.add_parser("score")
    s.add_argument("--run", default="ultrabert-380m-v7b8ev")
    s.add_argument("--local", action="store_true", help="checkpoint from data/v6-eval/checkpoints")
    s.add_argument("--naming", default="user", choices=["user", "bank"])
    s.add_argument("--modes", default="gold,e2e")
    s.add_argument("--device", default="cpu")
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    a = ap.parse_args()
    if a.cmd == "build":
        return build()
    if a.cmd == "counts":
        print(json.dumps(counts(load()), indent=1))
        return 0
    if a.cmd == "compare":
        return compare(a.a, a.b)
    return score(a.run, a.local, a.naming, a.modes.split(","), a.device)


if __name__ == "__main__":
    raise SystemExit(main())
