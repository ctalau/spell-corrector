#!/usr/bin/env python3
"""Stage 1 of the unmarked-markup audit: regex candidates for missing <uicontrol>, <filepath>, <codeph>.

Walks every .dita topic, linearizes each block element (p, li, cmd, entry, ...) into plain text, and runs one set of
regexes per target element over the text that is *not* already inside semantic inline markup. Presentational wrappers
(<b>, <i>, <u>, <ph>) are transparent: text in them counts as unmarked, because `<b>Save</b>` is exactly the kind of
miss this is looking for.

Output: one JSONL row per candidate -- file, line, element the regex argues for, the matched text, and the block's
plain text with the match marked [[like this]] for the judge (stage 2, judge_unmarked.py).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from lxml import etree

# Text inside these is already marked up (or is not prose: links, index terms, code blocks, comments).
SKIP = {
    "uicontrol", "menucascade", "wintitle", "codeph", "codeblock", "pre", "filepath", "userinput", "systemoutput",
    "cmdname", "option", "parmname", "apiname", "varname", "msgph", "keyword", "xref", "indexterm", "index-see",
    "index-see-also", "draft-comment", "required-cleanup", "data", "image", "alt", "screen", "msgblock", "synph",
    "kwd", "term", "cite", "prolog", "metadata", "shortcut", "tm", "object", "foreign", "svg-container",
    "xmlelement", "xmlatt", "xmlpi", "xmlnsname", "textentity", "numcharref", "parameterentity", "abbreviated-form",
    "glossterm", "sep", "syntaxdiagram", "groupseq", "param",
    # Titles are left out on purpose: the guide's house style writes UI names in titles as plain text.
    "title", "navtitle", "searchtitle", "linktext",
}
BLOCKS = {
    "p", "li", "cmd", "info", "stepresult", "stepxmp", "note", "shortdesc", "entry", "stentry", "dd", "dt",
    "context", "result", "postreq", "prereq", "lq", "sli", "choice", "choption", "chdesc", "abstract", "section",
    "example", "tutorialinfo", "steps-informal", "desc", "q",
}

# ---- regexes -------------------------------------------------------------------------------------------------------
CAP = r"[A-Z0-9](?:[\w.+#/&'-]*[\w+#])?"            # a capitalized label word ("Save", "XML", "Find/Replace")
LABEL = rf"{CAP}(?:\s+(?:{CAP}|(?:and|or|of|to|for|as|in|on|with|the|a|an|by|from)(?=\s+{CAP})))*"
UI_VERB = r"(?i:click|double-click|right-click|select|press|choose|check|uncheck|clear|enable|disable|tick|toggle|" \
          r"expand|collapse|go\s+to|switch\s+to|navigate\s+to)"
UI_NOUN = r"button|buttons|tab|tabs|menu|menus|submenu|sub-menu|menu\s+item|option|options|check\s*box|checkboxes|" \
          r"field|fields|dialog(?:\s+box)?|window|wizard|view|panel|pane|side-pane|toolbar|drop-down(?:\s+(?:menu|list|box))?|" \
          r"combo\s*box|list|radio\s+button|icon|action|page|preferences?\s+page|link|column|section|area|editor|mode|perspective"

UI_PATTERNS = [
    # "click Save", "select the Format and Indent action", "press OK"
    ("verb+label", re.compile(rf"(?<![\w-])(?:{UI_VERB})\s+(?:the\s+)?(?P<m>{LABEL})")),
    # "the Save button", "Advanced tab", "Preferences dialog box"
    ("label+noun", re.compile(rf"(?P<m>{LABEL})\s+(?:{UI_NOUN})\b")),
    # "Options > Preferences" style cascades written as text
    ("cascade", re.compile(rf"(?P<m>{CAP}(?:\s+{CAP})*(?:\s*(?:>|→|->)\s*{CAP}(?:\s+{CAP})*)+)")),
]

EXT = r"xml|dita|ditamap|bookmap|ditaval|xsl|xslt|xsd|dtd|rng|rnc|nvdl|sch|xpr|xpl|xproc|xquery|xq|xql|wsdl|css|less|" \
      r"scss|js|mjs|json|jsonld|yaml|yml|html?|xhtml|txt|md|properties|jar|zip|war|ear|exe|bat|cmd|sh|ps1|vmoptions|" \
      r"framework|opt|epub|pdf|docx?|xlsx?|pptx?|odt|ods|odp|png|gif|jpe?g|svg|ico|log|ini|conf|cfg|java|class|py|" \
      r"catalog|scenarios|xspec|xlf|xliff|ttf|otf|fo|xpl|lic|key|pem|p12|jks|csv|tsv|sql|dmg|pkg|app|plist|kts?|gradle|pom|lock|tmx"
FILE_PATTERNS = [
    # "catalog.xml", "oxygen.sh", "plugin.xml" (a word with a known extension)
    ("filename", re.compile(rf"(?<![\w@/.-])(?P<m>[\w$~{{}}.-]*[\w}}]\.(?:{EXT}))(?![\w/-]|\.\w)", re.I)),
    # "C:\\Program Files\\Oxygen", "/usr/local/Oxygen", "~/.oxygen", "frameworks/dita/DITA-OT", "${oxygenHome}/lib"
    ("path", re.compile(r"(?<![\w<>@:/.-])(?P<m>(?:[A-Za-z]:\\|\\\\|~[/\\]|\$\{\w+\}[/\\]|\[[A-Z_]+\][/\\]|/(?=[\w.])|\.{1,2}/)[^\s,;()'\"]*[\w/\\])")),
    ("path", re.compile(r"(?<![\w<>@:/.\\-])(?P<m>[\w.-]+(?:[/\\][\w.-]+){2,}[/\\]?)(?![\w/\\])")),
]

CODE_PATTERNS = [
    # an element named in prose: "the <topicref> element", "<xsl:template>"
    ("tag", re.compile(r"(?P<m></?[A-Za-z_][\w.:-]*(?:\s+[\w:-]+=\"[^\"]*\")*\s*/?>)")),
    # attributes / XPath: "@href", "the @class attribute"
    ("attr", re.compile(r"(?<![\w@])(?P<m>@[A-Za-z_][\w:-]*(?:=\S+)?)")),
    # "xsl:template", "xs:string", "fn:doc"
    ("qname", re.compile(r"(?<![\w:/.])(?P<m>(?:xsl|xs|fn|xsd|xi|xlink|oxy|oxyd|sch|sqf|dita|ditaarch|map|math|saxon|xhtml|svg|mml|ant|xmlns)(?::[A-Za-z_][\w.-]*)+(?:\(\))?)(?![\w:])")),
    # calls and editor variables: "getRange()", "${cfd}", "${pdu}", "-Dfoo=bar", "--help"
    ("call", re.compile(r"(?<![\w.])(?P<m>[A-Za-z_][\w.]*\(\s*[^()\s]{0,40}\))")),
    ("variable", re.compile(r"(?P<m>\$\{[\w:.*-]+(?:\([^)]*\))?\}|\$[A-Z_][A-Z0-9_]+|%[A-Z_][A-Z0-9_]+%)")),
    ("flag", re.compile(r"(?:(?<=^)|(?<=[\s(\[]))(?P<m>--?[A-Za-z][\w.-]*(?:=[^\s,;]+)?)(?=[\s,;.)]|$)")),
    # identifiers that are not words: camelCase, snake_case, dotted.package.Names
    ("identifier", re.compile(r"(?<![\w.$/-])(?P<m>[a-z]+[A-Z][A-Za-z0-9]*(?:\.[A-Za-z_]\w*)*|[a-z0-9]+_[a-z0-9_]+|[a-z]\w*(?:\.[a-z]\w*){2,}(?:\.[A-Z]\w*)?)(?![\w/-])")),
    # an attribute written as a value assignment: class="- topic/p ", format="dita"
    ("assignment", re.compile(r"(?<![\w-])(?P<m>[a-z][\w:-]*=\"[^\"]{0,60}\")")),
]

# <b>/<i>/<u> around a short span: the writer marked it as special, just not semantically ("<b>Create link</b>").
EMPHASIS = {"b", "i", "u"}
EMPHASIS_MAX_WORDS = 6

PATTERNS = {"uicontrol": UI_PATTERNS, "filepath": FILE_PATTERNS, "codeph": CODE_PATTERNS}

# Capitalized words after a UI verb that are almost never labels. Cheap and deliberately short: kev does the real work.
UI_STOP = {"I", "A", "An", "The", "This", "That", "These", "Those", "It", "If", "When", "For", "To", "In", "On", "You",
           "Oxygen", "Note", "Tip", "Important", "Attention", "Remember", "See",
           "Windows", "Linux", "Mac", "OS", "macOS", "Java", "Eclipse", "Web"}


def local(tag) -> str:
    return etree.QName(tag).localname if isinstance(tag, str) else ""


def linearize(block):
    """-> (text, spans) where spans are (start, end) of characters that are free prose (not in SKIP markup) and belong
    to `block` itself rather than to a nested block."""
    parts, spans, marked, pos = [], [], [], 0

    def emit(s, free):
        nonlocal pos
        if not s: return
        if free: spans.append((pos, pos + len(s)))
        parts.append(s); pos += len(s)

    def walk(el, free, own):
        name = local(el.tag)
        if name == "":  # comment / PI
            return
        in_free = free and name not in SKIP
        in_own = own and (el is block or name not in BLOCKS)
        start = pos
        emit(el.text, in_free and in_own)
        for c in el:
            walk(c, in_free, in_own)
            if local(c.tag) or isinstance(c, etree._Comment):
                emit(c.tail, in_free and in_own)
        if free and in_own and (name in PATTERNS or name in EMPHASIS) and el is not block:
            marked.append((name, start, pos))
        if el is not block and name in BLOCKS:
            emit(" ", False)

    walk(block, True, True)
    return "".join(parts), spans, marked


def norm(text, spans):
    """Collapse whitespace, remapping (start, end) spans."""
    out, remap, prev_ws = [], [], False
    for i, ch in enumerate(text):
        remap.append(len(out))
        if ch.isspace():
            if not prev_ws and out: out.append(" ")
            prev_ws = True
        else:
            out.append(ch); prev_ws = False
    remap.append(len(out))
    s = "".join(out)
    return s, [(remap[a], remap[b]) for a, b in spans if remap[b] > remap[a]]


def candidates_in(text, spans):
    seen = set()  # (start, end) in `text`: a span nominated by two rules is judged once
    for elem, pats in PATTERNS.items():
        for a, b in spans:
            chunk = text[a:b]
            for kind, rx in pats:
                for m in rx.finditer(chunk):
                    s, e = m.span("m")
                    val = m.group("m").rstrip(".:,")
                    e = s + len(val)
                    if (a + s, a + e) in seen or not val.strip(): continue
                    if elem == "uicontrol":
                        words = val.split()
                        while words and words[0] in UI_STOP: words.pop(0)
                        if not words: continue
                        if len(val) < 2: continue
                        val = " ".join(words); s = e - len(val)
                        if (a + s, a + e) in seen: continue
                    if elem == "codeph" and kind == "flag" and len(val) < 3: continue
                    seen.add((a + s, a + e))
                    yield elem, kind, val, a + s, a + e


def neighbour_text(block) -> str:
    """Context for short blocks: a bare <dt> or table cell says little on its own, so the judge also sees the <dd> it
    heads or the rest of its row. Appended after the block text, so match offsets are unchanged."""
    name = local(block.tag)
    if name == "dt":
        dd = block.getparent().find("dd") if block.getparent() is not None else None
        others = [dd] if dd is not None else []
    elif name in ("entry", "stentry"):
        others = [c for c in block.getparent() if c is not block]
    else:
        return ""
    return " | ".join(norm("".join(o.itertext()), [])[0].strip() for o in others)[:400]


def scan_file(path: Path, root: Path, gold: bool = False):
    try:
        tree = etree.parse(str(path), etree.XMLParser(load_dtd=False, resolve_entities=False, no_network=True, recover=True))
    except etree.XMLSyntaxError:
        return
    # A <dl> whose terms are mostly <uicontrol> documents UI options; a term in it with no <uicontrol> breaks the pattern.
    dt_convention = set()
    for dl in tree.iter("dl"):
        dts = [dt for dt in dl.iter("dt")]
        ui = [dt for dt in dts if any(local(c.tag) in ("uicontrol", "menucascade") for c in dt.iter())]
        if len(ui) >= 2 and len(ui) >= 0.5 * len(dts):
            dt_convention.update(dt for dt in dts if dt not in ui and not any(local(c.tag) in SKIP for c in dt.iterdescendants()))
    for block in tree.iter():
        if local(block.tag) not in BLOCKS: continue
        if any(local(a.tag) in SKIP for a in block.iterancestors()): continue
        raw, spans, marked = linearize(block)
        extra = neighbour_text(block)
        extra = f" || {extra}" if extra else ""
        if gold:
            # Validation rows: an element the writers *did* mark up, shown to the judge as plain text.
            marked = [m for m in marked if m[0] in PATTERNS]
            text, mspans = norm(raw, [(a, b) for _, a, b in marked])
            names = [n for n, a, b in marked if norm(raw, [(a, b)])[1]]
            for name, (s, e) in zip(names, mspans):
                val = text[s:e].strip()
                if not val: continue
                yield {"file": str(path.relative_to(root)), "line": block.sourceline, "block": local(block.tag),
                       "element": name, "rule": "gold", "match": val, "context": text[:s] + "[[" + val + "]]" + text[e:] + extra}
            continue
        emph = [(a, b) for n, a, b in marked if n in EMPHASIS]
        text, spans_emph = norm(raw, spans + emph)
        spans, emph = spans_emph[:len(spans_emph) - len(emph)], spans_emph[len(spans_emph) - len(emph):]
        if not spans: continue
        rows = list(candidates_in(text, spans))
        taken = {(s, e) for *_, s, e in rows}
        for s, e in emph:
            val = text[s:e].strip(" .,:;")
            s = text.index(val, s) if val else s; e = s + len(val)
            if not val or len(val.split()) > EMPHASIS_MAX_WORDS or (s, e) in taken: continue
            rows.append(("any", "emphasis", val, s, e))
        if local(block.tag) == "dt" and block in dt_convention:
            val = re.split(r"\s+(?:\(|-|–|—)\s*", text.strip(), maxsplit=1)[0].strip(" .:")
            val = re.sub(rf"\s+(?:{UI_NOUN})$", "", val)  # "Chat Mode drop-down menu" -> "Chat Mode"
            s = text.find(val)
            if val and len(val.split()) <= 8 and s >= 0 and (s, s + len(val)) not in taken:
                rows.append(("uicontrol", "dt-convention", val, s, s + len(val)))
        for elem, kind, val, s, e in rows:
            yield {
                "file": str(path.relative_to(root)), "line": block.sourceline, "block": local(block.tag),
                "element": elem, "rule": kind, "match": val,
                "context": text[:s] + "[[" + val + "]]" + text[e:] + extra,
            }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="DITA folder to scan")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gold", action="store_true",
                    help="emit the existing <uicontrol>/<filepath>/<codeph> as plain-text rows labelled with their element")
    a = ap.parse_args()
    root = Path(a.root)
    n = 0
    with open(a.out, "w", encoding="utf-8") as f:
        for p in sorted(root.rglob("*.dita")):
            for row in scan_file(p, root, a.gold):
                f.write(json.dumps(row, ensure_ascii=False) + "\n"); n += 1
    print(f"{n} candidates -> {a.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
