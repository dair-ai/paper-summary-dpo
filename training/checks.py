"""The automatic style and support checks behind the results table. Used by evaluate.py and show_results.py."""
import re

NUM = re.compile(r"(?<![\w.])\d[\d,]*\.?\d*")
nums = lambda t: {m.replace(",", "").rstrip(".") for m in NUM.findall(t) if len(m.replace(",", "").rstrip(".")) >= 2}
take = lambda o: o.split("Key points", 1)[0]
points = lambda o: o.split("Key points", 1)[-1]
CHECKS = {
    "em or en dash": lambda o, s: bool(re.search("[—–]", o)),
    "Section/Figure/Table ref": lambda o, s: bool(re.search(r"\b(Section|Figure|Fig\.|Table|Definition|Appendix)\s*\d", o)),
    "et al outside credit": lambda o, s: o.count("et al") > 1 or "et al" in points(o),
    "missing Why it matters": lambda o, s: "Why it matters" not in o,
    "bullet not '**Label.** text'": lambda o, s: any(not re.match(r"- \*\*[^*]+\.\*\* \S", l) for l in points(o).splitlines()
                                                      if l.startswith(("- ", "* ", "• "))),
    "no 'First author et al.' credit": lambda o, s: not re.search(r"\bet al\.|\bThe authors\b", take(o)),
    "numbers not in input": lambda o, s: bool(nums(o) - nums(s)),
    "results in curator's take": lambda o, s: bool(re.search(r"\d+(\.\d+)?\s?%|\d+(\.\d+)?x\b", take(o))),
    "opens 'The paper tackles/addresses'": lambda o, s: bool(re.search(r"(?i)the paper (tackles|addresses|investigates)", o)),
}
