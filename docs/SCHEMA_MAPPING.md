# Schema mapping: the relation registry against outside standards

The relation registry (`RELATION_REGISTRY` in `src/entity_graph/extract/ontology.py`,
D40) lists each relation's terms in outside standards. This page lays the
registry out against those standards, and lists what the standards define that the
registry does not have. That list replaces finding gaps through failed evaluations
(D41).

**Standards used**

- **ACE 2005**: relation types and event types from the ACE 2005 annotation
  guidelines. They were built for extracting relations from news and broadcast text.
- **TAC KBP**: the slot-filling relations, per:* and org:*.
- **Wikidata**: property IDs. All the IDs below were checked against the Wikidata
  API on 2026-11-07.
- **schema.org**: properties of Person, Organization and Event.
- **FIBO**: the EDM Council's financial ontology.
- **SEM / CIDOC-CRM**: event models. SEM is the Simple Event Model. CIDOC-CRM is
  ISO 21127.

`^` marks the inverse: the standard's property runs tail -> head. For example,
`founded` is `^P112` because Wikidata says *organization* "founder" *person*. A dash
means the standard has no clear match.

## 1. The registry

| Relation | Module | Group | ACE 2005 | TAC KBP | Wikidata | schema.org | Other |
|---|---|---|---|---|---|---|---|
| works_for | core | affiliation | ORG-AFF.Employment | per:employee_or_member_of | P108 employer | worksFor | |
| located_in | core | location | PHYS.Located; GEN-AFF.Org-Location | org:city/stateorprovince/country_of_headquarters | P276 location; P159 headquarters; P131 admin. territory | location | SEM hasPlace; CIDOC P7 took place at (events) |
| founded | core | affiliation | ORG-AFF.Founder | ^org:founded_by | ^P112 founder | ^founder | |
| leads | core | affiliation | ORG-AFF.Employment | ^org:top_members_employees | ^P169 CEO; ^P488 chairperson; ^P1037 director/manager | | |
| partnered_with | core | cooperation | | | P2652 partnership with | | |
| acquired | core | ownership | event Business.Merge-Org | | (^P127 owned by, after the fact) | | FIBO BE ownership and control |
| invested_in | core | ownership | ORG-AFF.Investor-Shareholder | ^org:shareholders | (^P127 owned by, partial) | | FIBO BE ownership and control |
| member_of | core | affiliation | ORG-AFF.Membership | per:employee_or_member_of | P463 member of; ^P3320 board member | memberOf | FOAF ^member |
| subsidiary_of | core | ownership | PART-WHOLE.Subsidiary | org:parents | P749 parent organization (^P355) | parentOrganization | |
| supplies | commerce | commerce | | | | | |
| sued | legal | conflict | event Justice.Sue | | | | |
| underwriter_of | finance | commerce | | | | | FIBO securities-issuance underwriting (class not verified) |
| auditor_of | finance | commerce | | | ^P8571 external auditor | | SEC XBRL dei AuditorName |
| advisor_to | commerce | commerce | | | | | |
| develops | commerce | creation | ART.User-Owner-Inventor-Manufacturer | | ^P178 developer; ^P176 manufacturer | ^manufacturer | |
| lends_to | finance | commerce | (event Transaction.Transfer-Money) | | | | FIBO FBC Debt: lender -> borrower |
| allied_with | politics | cooperation | | | | | |
| opposed | politics | conflict | | | | | |

Terms in parentheses are partial matches. They are not in the registry's `maps`.

Each relation outside the core, and each family and education relation, also lists
cue words: the words a text uses to state it. With `RELATION_CUES` on, a window's
prompt lists such a relation only when the window holds one of them (D47).

**Coverage**

- **Core module:** all 9 core relations map to ACE, TAC KBP or Wikidata. So the core
  is already the general relation set those standards share.
- **Domain modules:**
  - `supplies`, `advisor_to`, `underwriter_of` and the `politics` relations have no
    match in ACE, TAC KBP or Wikidata.
  - That is expected: they are domain relations, and they belong in modules rather
    than in the core.
  - They still need definitions of their own. The meanings in the registry, which
    the gold sets check, are those definitions.

## 2. In the standards but not in the registry

These are ordered by how widely they are shared. A relation in ACE, TAC KBP *and*
Wikidata is a core candidate, because a general text of any kind can state it.

None of these is adopted. Each would change the prompt, so each needs a probe on
every gold set before it goes in (D36 rules).

### Core candidates

| Proposed | Head -> tail | ACE 2005 | TAC KBP | Wikidata | schema.org | Why it matters |
|---|---|---|---|---|---|---|
| `spouse_of` (symmetric), `parent_of`, `sibling_of` (symmetric) | person -> person | PER-SOC.Family | per:spouse, per:children, per:parents, per:siblings, per:other_family | P26, P40, P22, P3373 | spouse, children, parent, sibling | Histories, biographies and news. The registry has no person-to-person relation at all. |
| `educated_at` | person -> organization | ORG-AFF.Student-Alum | per:schools_attended | P69 | alumniOf | Every director's and officer's biography in an S-1. No relation in the registry holds it today. |
| `born_in` | person -> location | GEN-AFF.Citizen-Resident-Religion-Ethnicity | per:city/stateorprovince/country_of_birth | P19 | birthPlace | Biographies and histories. `located_in` covers residence only. |
| widen `member_of` to an organization head | organization -> organization | ORG-AFF.Membership | org:member_of | P463 | memberOf | Consortia, trade associations, alliances and indexes. Today the head of `member_of` must be a person. |
| widen `founded` to an organization head | organization -> organization | ORG-AFF.Founder | org:founded_by | P112 | founder | Joint ventures, foundations and spin-offs. |
| `owns` | person/organization -> organization/product | ORG-AFF.Ownership | | P127 / P1830 | owns | Control that is not a stake bought as an investment. ACE keeps Ownership apart from Investor-Shareholder. |

### Module candidates

| Proposed module | Relations | Standards | For |
|---|---|---|---|
| `events` | `participated_in` (person/organization -> event), `organized` (person/organization -> event); events already take `located_in` | SEM hasActor / hasPlace; CIDOC P11 had participant, P14 carried out by, P7 took place at; Wikidata P1344 / P710, P664, P276; schema.org attendee, organizer, location; ACE event types | Histories, news, minutes. It makes the event a node rather than a scatter of pairwise edges. |
| `research` | `authored` (person -> work), `funded` (person/organization -> organization) | Wikidata P50, P8324; schema.org author, funder; Dublin Core creator | Papers, grants, reports, nonprofits |
| `politics` (add) | `succeeded` (person -> person, in an office) | Wikidata P1365 / P1366; ACE Personnel.Start-Position / End-Position | Histories |
| `commerce` (add) | person-to-person business ties | ACE PER-SOC.Business; Wikidata P1327 | Lower priority |

### Not relations: attributes of an edge or a document

- **Time on edges.** Wikidata qualifies statements with P580 start time, P582 end
  time and P585 point in time. ACE has Personnel.Start-Position and
  End-Position.
  - The graph keeps no time on an edge, so "former director" and "director" are the
    same edge.
  - DATE entities are excluded as nodes by default, so dates can't be linked to
    edges as nodes either.
  - A date belongs on the edge, as its `valid_from` and `valid_to`, rather than in a
    new relation.
- **Document metadata.** Dublin Core's title, creator, date, publisher and source
  describe the document, not the network. They fit the `documents` table, not the
  relation registry.
- **FOAF** overlaps schema.org: `member` and `knows`. It adds nothing new here.

## 3. What follows from this

1. **The core is close to the shared standard.** The largest general gaps are
   person-to-person relations (family) and education. Both are common in any
   biographical text, filings included.
2. **The cheapest change is widening types.** Letting `member_of` and `founded` take
   an organization head is a type change, not a new relation. It is still a prompt
   change, so it needs a probe.
3. **Events are a module and a modelling choice.** They are not a missing label. See
   the D40 design.
4. **Domain relations have no standard to borrow** (`supplies`, `advisor_to`,
   `underwriter_of`). Their meanings, checked on the gold sets, are their
   definitions.
5. **A gold set outside filings is needed to measure any of this.** Wikidata IDs
   make a public set such as DocRED, which is labelled with Wikidata properties, a
   candidate. Which of these properties it covers, and its licence, still need
   checking.
