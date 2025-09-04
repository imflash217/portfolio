# address_extractor.py
import re
from rich import print
import difflib
from typing import List, Dict, Tuple, Optional

class AddressExtractor:
    """
    Extract US addresses from messy single-line text, returning a list of dicts:
      { street, apartment, city, state, zip, single_line, zip_state_warning }

    Modes:
      - libpostal  (best, needs native libpostal + pip install postal)
      - usaddress  (good, pure Python: pip install usaddress)
      - regex      (no deps; robust baseline)

    Usage:
      ae = AddressExtractor()
      recs = ae.extract(text, method="auto")
    """

    # ----------------- USPS state data -----------------
    USPS_ABBR = {
        "AL":"Alabama","AK":"Alaska","AZ":"Arizona","AR":"Arkansas","CA":"California","CO":"Colorado","CT":"Connecticut",
        "DE":"Delaware","DC":"District of Columbia","FL":"Florida","GA":"Georgia","HI":"Hawaii","ID":"Idaho","IL":"Illinois",
        "IN":"Indiana","IA":"Iowa","KS":"Kansas","KY":"Kentucky","LA":"Louisiana","ME":"Maine","MD":"Maryland","MA":"Massachusetts",
        "MI":"Michigan","MN":"Minnesota","MS":"Mississippi","MO":"Missouri","MT":"Montana","NE":"Nebraska","NV":"Nevada",
        "NH":"New Hampshire","NJ":"New Jersey","NM":"New Mexico","NY":"New York","NC":"North Carolina","ND":"North Dakota",
        "OH":"Ohio","OK":"Oklahoma","OR":"Oregon","PA":"Pennsylvania","RI":"Rhode Island","SC":"South Carolina","SD":"South Dakota",
        "TN":"Tennessee","TX":"Texas","UT":"Utah","VT":"Vermont","VA":"Virginia","WA":"Washington","WV":"West Virginia",
        "WI":"Wisconsin","WY":"Wyoming","PR":"Puerto Rico","VI":"Virgin Islands","GU":"Guam","AS":"American Samoa","MP":"Northern Mariana Islands"
    }
    STATE_NAME_TO_ABBR = {v.upper(): k for k, v in USPS_ABBR.items()}

    # coarse ZIP prefix ranges for quick state-vs-ZIP sanity check
    ZIP_PREFIX_RANGES = {
        "AL": ("35","36"), "AK": ("99","99"), "AZ": ("85","86"), "AR": ("71","72"),
        "CA": ("90","96"), "CO": ("80","81"), "CT": ("06","06"), "DC": ("20","20"),
        "DE": ("19","19"), "FL": ("32","34"), "GA": ("30","31"), "HI": ("96","96"),
        "ID": ("83","83"), "IL": ("60","62"), "IN": ("46","47"), "IA": ("50","52"),
        "KS": ("66","67"), "KY": ("40","42"), "LA": ("70","71"), "ME": ("04","04"),
        "MD": ("20","21"), "MA": ("01","02"), "MI": ("48","49"), "MN": ("55","56"),
        "MS": ("39","39"), "MO": ("63","65"), "MT": ("59","59"), "NE": ("68","69"),
        "NV": ("89","89"), "NH": ("03","03"), "NJ": ("07","08"), "NM": ("87","88"),
        "NY": ("10","14"), "NC": ("27","28"), "ND": ("58","58"), "OH": ("43","45"),
        "OK": ("73","74"), "OR": ("97","97"), "PA": ("15","19"), "RI": ("02","02"),
        "SC": ("29","29"), "SD": ("57","57"), "TN": ("37","38"), "TX": ("75","79"),
        "UT": ("84","84"), "VT": ("05","05"), "VA": ("22","24"), "WA": ("98","99"),
        "WV": ("24","26"), "WI": ("53","54"), "WY": ("82","83"),
    }

    # separators/noise
    SEP_STRONG = re.compile(r"[|;]+")
    PHONE_FAX_TAIL = re.compile(r"(?i)\b(?:attn|attention|fax|ph|phone|tel|telephone|mobile|mob|cell)\b.*$")
    UNIT_TOKEN_RE = re.compile(r"(?i)\b(?:Apt|Apartment|Unit|Suite|Ste|#|Fl|Floor|Bldg|Building|Rm|Room)\b\s*([\w\-\.]+)")

    def __init__(self, fuzzy_state_cutoff: float = 0.8):
        self.fuzzy_state_cutoff = fuzzy_state_cutoff
        # Try optional deps
        self._have_usaddress = False
        self._have_libpostal = False
        try:
            import usaddress  # type: ignore
            self._have_usaddress = True
            self._usaddress = usaddress
        except Exception:
            self._usaddress = None
        try:
            from postal.parser import parse_address  # type: ignore
            self._have_libpostal = True
            self._libpostal_parse = parse_address
        except Exception:
            self._libpostal_parse = None

        states_abbr = "|".join(sorted(self.USPS_ABBR.keys()))
        # Compact, non-VERBOSE window to reduce parser edge cases.
        self.ADDRESS_WINDOW = re.compile(
            r"(?P<street>(?:\d{1,6}(?:[A-Za-z]|\-\d+)?\s+[\w\.\'&/()\-\#]+(?:\s+[\w\.\'&/()\-\#]+)*|P\.?\s*O\.?\s*Box\s+\d+)"
            r"(?:\s*,?\s*(?:(?:Apt|Apartment|Unit|Suite|Ste|#|Fl|Floor|Bldg|Building|Rm|Room)\s*[\w\-\.]+))*)"
            r"[\s,]+(?P<city>[A-Za-z][A-Za-z\.\-'\s]+?)\s*,?\s*"
            r"(?P<state>(" + states_abbr + r")|[A-Za-z][A-Za-z\.\s]+?)\s*,?\s+"
            r"(?P<zip>\d{5}(?:-\d{4})?)",
            re.IGNORECASE
        )

    # ----------------- public API -----------------
    def extract(self, text: str, method: str = "auto") -> List[Dict[str, str]]:
        """
        method ∈ {"auto","libpostal","usaddress","regex"}
        """
        m = method.lower()
        if m == "auto":
            if self._have_libpostal:
                return self._extract_libpostal(text)
            if self._have_usaddress:
                return self._extract_usaddress(text)
            return self._extract_regex(text)
        elif m == "libpostal":
            if not self._have_libpostal:
                raise RuntimeError("libpostal not available. Install native lib + `pip install postal`.")
            return self._extract_libpostal(text)
        elif m == "usaddress":
            if not self._have_usaddress:
                raise RuntimeError("usaddress not available. `pip install usaddress`.")
            return self._extract_usaddress(text)
        elif m == "regex":
            return self._extract_regex(text)
        else:
            raise ValueError("method must be one of: auto, libpostal, usaddress, regex")

    # ----------------- helpers -----------------
    def _preclean_chunks(self, s: str) -> List[str]:
        # First split on strong separators to avoid a span eating the next address.
        chunks = self.SEP_STRONG.split(s)
        return [re.sub(r"\s+", " ", c).strip(" ,") for c in chunks if c.strip()]

    def _cap_city(self, s: str) -> str:
        s = s.strip(" ,")
        return " ".join(w if w.isupper() else w.capitalize() for w in s.split())

    def _fuzzy_state_to_abbr(self, token: str) -> str:
        t = token.strip().upper().replace(".", "")
        if re.fullmatch(r"[A-Z]{2}", t) and t in self.USPS_ABBR:
            return t
        best = difflib.get_close_matches(t, self.STATE_NAME_TO_ABBR.keys(),
                                         n=1, cutoff=self.fuzzy_state_cutoff)
        return self.STATE_NAME_TO_ABBR[best[0]] if best else ""

    def _zip_state_mismatch(self, state: str, zipc: str) -> bool:
        if not (state and zipc and re.match(r"^\d{5}", zipc)): return False
        p = zipc[:2]
        rng = self.ZIP_PREFIX_RANGES.get(state.upper())
        return bool(rng) and not (rng[0] <= p <= rng[1])

    def _split_street_and_apartment(self, street_raw: str) -> Tuple[str, str]:
        street = street_raw.strip(" ,")
        apartment = ""
        m = self.UNIT_TOKEN_RE.search(street)
        if m:
            apartment = street[m.start():].strip(" ,")
            street = street[:m.start()].strip(" ,")
            return street, apartment
        m2 = re.search(r"#\s*([\w\-\.]+)$", street)
        if m2:
            apartment = f"#{m2.group(1)}"
            street = street[:m2.start()].strip(" ,")
        return street, apartment

    def _dedupe(self, recs: List[Dict[str, str]]) -> List[Dict[str, str]]:
        seen, out = set(), []
        for r in recs:
            key = (r.get("street","").lower(), r.get("apartment","").lower(),
                   r.get("city","").lower(), r.get("state","").upper(), r.get("zip",""))
            if key not in seen:
                seen.add(key); out.append(r)
        return out

    # ----------------- method: regex (no deps) -----------------
    def _extract_regex(self, text: str) -> List[Dict[str,str]]:
        results: List[Dict[str,str]] = []
        for chunk in self._preclean_chunks(text):
            for m in self.ADDRESS_WINDOW.finditer(chunk):
                # prune trailing phone/fax tail if it leaks in
                span = self.PHONE_FAX_TAIL.sub("", chunk[m.start():m.end()]).strip(" ,")
                street_raw = m.group("street")
                street, apartment = self._split_street_and_apartment(street_raw)
                city = self._cap_city(m.group("city"))
                state = self._fuzzy_state_to_abbr(m.group("state"))
                zipc = m.group("zip")

                rec = {
                    "street": street,
                    "apartment": apartment,
                    "city": city,
                    "state": state,
                    "zip": zipc,
                    "single_line": f"{street}{', ' + apartment if apartment else ''}, {city}, {state} {zipc}".replace("  ", " ").strip(),
                    "zip_state_warning": bool(state and self._zip_state_mismatch(state, zipc)),
                }
                results.append(rec)
        return self._dedupe(results)

    # ----------------- method: usaddress -----------------
    def _extract_usaddress(self, text: str) -> List[Dict[str,str]]:
        usaddress = self._usaddress
        results: List[Dict[str,str]] = []
        for chunk in self._preclean_chunks(text):
            for m in self.ADDRESS_WINDOW.finditer(chunk):
                span = self.PHONE_FAX_TAIL.sub("", chunk[m.start():m.end()]).strip(" ,")
                try:
                    tagged, _ = usaddress.tag(span)  # type: ignore
                except Exception:
                    tagged = {}

                # street base
                number = (tagged.get("AddressNumber","") if tagged else "").strip()
                street_name = " ".join(filter(None, [
                    tagged.get("StreetNamePreDirectional","") if tagged else "",
                    tagged.get("StreetName","") if tagged else "",
                    tagged.get("StreetNamePostType","") if tagged else "",
                    tagged.get("StreetNamePostDirectional","") if tagged else "",
                ])).strip()
                street = (number + " " + street_name).strip() or m.group("street")

                # apartment/unit (Occupancy only; exclude USPS box for apartment)
                apartment = " ".join(filter(None, [
                    (tagged.get("OccupancyType","")) if tagged else "",
                    (tagged.get("OccupancyIdentifier","")) if tagged else "",
                ])).strip()
                if not apartment:
                    # rescue from raw street if needed
                    street, apartment = self._split_street_and_apartment(street)

                city  = self._cap_city(tagged.get("PlaceName","") if tagged else m.group("city"))
                state = self._fuzzy_state_to_abbr(tagged.get("StateName","") if tagged else m.group("state")) or ""
                zipc  = (tagged.get("ZipCode","") if tagged else m.group("zip"))
                zipc  = (re.findall(r"\d{5}(?:-\d{4})?", zipc) or [zipc])[0]

                rec = {
                    "street": street,
                    "apartment": apartment,
                    "city": city,
                    "state": state,
                    "zip": zipc,
                    "single_line": f"{street}{', ' + apartment if apartment else ''}, {city}, {state} {zipc}".replace("  ", " ").strip(),
                    "zip_state_warning": bool(state and self._zip_state_mismatch(state, zipc)),
                }
                results.append(rec)
        return self._dedupe(results)

    # ----------------- method: libpostal -----------------
    def _extract_libpostal(self, text: str) -> List[Dict[str,str]]:
        parse_address = self._libpostal_parse
        results: List[Dict[str,str]] = []
        for chunk in self._preclean_chunks(text):
            for m in self.ADDRESS_WINDOW.finditer(chunk):
                span = self.PHONE_FAX_TAIL.sub("", chunk[m.start():m.end()]).strip(" ,")
                parts = dict(parse_address(span))  # type: ignore

                # street vs apartment
                number = parts.get("house_number","")
                road   = parts.get("road","") or parts.get("house","")
                street = " ".join(p for p in [number, road] if p).strip() or m.group("street")

                apartment = ""
                if parts.get("unit"):
                    apartment = f"Unit {parts['unit']}"
                elif parts.get("level"):
                    apartment = f"Fl {parts['level']}"
                else:
                    # rescue #Unit patterns
                    s2, apt2 = self._split_street_and_apartment(street)
                    if apt2:
                        street, apartment = s2, apt2

                city  = self._cap_city(parts.get("city") or parts.get("town") or parts.get("village") or parts.get("suburb") or m.group("city"))
                state = self._fuzzy_state_to_abbr(parts.get("state","") or m.group("state")) or ""
                zipc  = parts.get("postcode") or m.group("zip")

                rec = {
                    "street": street,
                    "apartment": apartment,
                    "city": city,
                    "state": state,
                    "zip": zipc,
                    "single_line": f"{street}{', ' + apartment if apartment else ''}, {city}, {state} {zipc}".replace("  ", " ").strip(),
                    "zip_state_warning": bool(state and self._zip_state_mismatch(state, zipc)),
                }
                results.append(rec)
        return self._dedupe(results)


sample_texts = [

    # 1. Apartment with typo in city + ZIP mismatch
    "Flash AI Inc., 415 barton creeek dr, apt 6e, charlote, NC, 20262, united states attn: fax +12234567, ph: (+1) 919-523-6546",

    # 2. Street with hash-style apartment
    "Warehouse: 55-57 W 39th St #12B, New York, NY 10018-1234; call: 212-555-0100",

    # 3. PO Box with full state name
    "Mailing: PO Box 12345, Austin, Texas 78711 USA",

    # 4. Normal address with Suite
    "Customer Care, 123 Main Street Suite 400, Springfield, IL 62704 ph: 555-111-2222",

    # 5. Multiple addresses chained
    "ACME Corp 77 Massachusetts Ave, Cambridge, MA 02139 | Data Center: 1600 Amphitheatre Parkway, Mountain View, CA 94043; Satellite Office: 1 Microsoft Way, Redmond WA 98052",

    # 6. Typos in state/city
    "Global HQ: 350 Fifth Ave, Ste 5900, New Yrok, Neww York 10118",

    # 7. Extra noise and country
    "Order Returns → 500 Market St Floor 3, San Francisco, CA 94105, United States of America (Open 9am-5pm)",

    # 8. Short ZIP only
    "Lab: 200 Elm St, Denver, CO 802; Backup: 1400 Broadway, New York NY 10018",

    # 9. Address without unit
    "Remote Office: 1 Infinite Loop, Cupertino, CA 95014",

    # 10. Address with hyphenated ZIP+4
    "Payment Center: 700 Pennsylvania Avenue NW, Washington, DC 20408-0001"
]



if __name__ == "__main__":
    ae = AddressExtractor()
    for i, text in enumerate(sample_texts, 1):
        print(f"\n=== SAMPLE {i} ===")
        print("AUTO:", ae.extract(text, method="auto"))
        print("REGX:", ae.extract(text, method="regex"))
        print("USAD:", ae.extract(text, method="usaddress"))
