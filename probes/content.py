"""Probe content: substantive material per topic, plus per-language framing.

Structure
---------
``TOPICS[topic]`` holds two independent *stems*. Each stem has:
  ``payload`` — the constant material. For syntax-heavy topics this is code or
      notation and is **byte-identical across every language cell**. That
      invariance is the experimental control (see probes/README.md).
  ``ask``     — a dict of per-language request sentences wrapping the payload.
  ``schema``  — JSON keys used when ``format=json``.
  ``points``  — bullet headings used when ``format=bulleted``.

Keeping the framing sentences short and concrete bounds translation risk. All
non-English text is model-assisted and marked ``translation_reviewed: false``
in the emitted YAML.
"""

LANGS = ("en", "zh", "de", "ja")
REGISTERS = ("formal", "casual")
FORMATS = ("prose", "json", "bulleted")

# --- Constant payloads. Identical bytes in every language cell. --------------

FIB = """def fib(n):
    if n <= 1:
        return n
    return fib(n - 1) + fib(n - 2)"""

DEDUPE = """def dedupe(items):
    seen = set()
    return [x for x in items if not (x in seen or seen.add(x))]"""

RUST_BORROW = """fn longest<'a>(a: &'a str, b: &'a str) -> &'a str {
    if a.len() > b.len() { a } else { b }
}"""

RUST_ITER = """let total: u32 = values
    .iter()
    .filter(|v| **v % 2 == 0)
    .sum();"""

SQL_JOIN = """SELECT c.name, COUNT(o.id) AS n_orders
FROM customers c
LEFT JOIN orders o ON o.customer_id = c.id
GROUP BY c.name
HAVING COUNT(o.id) > 3;"""

SQL_WINDOW = """SELECT id, amount,
       SUM(amount) OVER (PARTITION BY user_id ORDER BY ts) AS running
FROM payments;"""

PROOF_IRRATIONAL = """Suppose sqrt(2) = p/q in lowest terms.
Then p^2 = 2 q^2, so p is even; write p = 2k.
Then 4k^2 = 2q^2, so q^2 = 2k^2 and q is even.
But then p and q share a factor 2."""

PROOF_INDUCTION = """Claim: 1 + 2 + ... + n = n(n+1)/2.
Base: n = 1 gives 1 = 1(2)/2.
Step: assume for n; then adding (n+1) gives
n(n+1)/2 + (n+1) = (n+1)(n+2)/2."""


def _t(en, zh, de, ja):
    return {"en": en, "zh": zh, "de": de, "ja": ja}


TOPICS: dict[str, list[dict]] = {
    "python": [
        {
            "payload": FIB,
            "ask": _t(
                "Explain the time complexity of this function and how memoisation changes it.",
                "请解释这个函数的时间复杂度，以及记忆化会如何改变它。",
                "Erkläre die Zeitkomplexität dieser Funktion und wie Memoisierung sie verändert.",
                "この関数の時間計算量と、メモ化によってどう変わるかを説明してください。",
            ),
            "schema": ["complexity", "with_memoisation", "reason"],
            "points": ["complexity", "why", "fix"],
        },
        {
            "payload": DEDUPE,
            "ask": _t(
                "Describe why this preserves order, and one case where it fails.",
                "说明它为什么能保持顺序，以及一种会失败的情况。",
                "Beschreibe, warum die Reihenfolge erhalten bleibt, und einen Fehlerfall.",
                "順序が保たれる理由と、失敗する場合を一つ説明してください。",
            ),
            "schema": ["order_preserved", "failure_case"],
            "points": ["mechanism", "failure"],
        },
    ],
    "rust": [
        {
            "payload": RUST_BORROW,
            "ask": _t(
                "Explain what the lifetime annotation guarantees here.",
                "请解释这里的生命周期标注保证了什么。",
                "Erkläre, was die Lifetime-Annotation hier garantiert.",
                "ここでのライフタイム注釈が何を保証するのか説明してください。",
            ),
            "schema": ["guarantee", "without_annotation"],
            "points": ["guarantee", "counterexample"],
        },
        {
            "payload": RUST_ITER,
            "ask": _t(
                "Explain why this compiles without an explicit loop and what iter() yields.",
                "请解释为什么它不需要显式循环即可编译，以及 iter() 产生什么。",
                "Erkläre, warum das ohne explizite Schleife kompiliert und was iter() liefert.",
                "明示的なループなしでコンパイルできる理由と iter() が返すものを説明してください。",
            ),
            "schema": ["iterator_chain", "yielded_type"],
            "points": ["chain", "type"],
        },
    ],
    "sql": [
        {
            "payload": SQL_JOIN,
            "ask": _t(
                "Explain what rows this returns and why LEFT JOIN matters here.",
                "请解释它返回哪些行，以及为什么这里 LEFT JOIN 很重要。",
                "Erkläre, welche Zeilen zurückkommen und warum LEFT JOIN hier wichtig ist.",
                "どの行が返るか、なぜ LEFT JOIN が重要かを説明してください。",
            ),
            "schema": ["rows_returned", "join_rationale"],
            "points": ["result", "join_choice"],
        },
        {
            "payload": SQL_WINDOW,
            "ask": _t(
                "Explain the window function and what PARTITION BY changes.",
                "请解释这个窗口函数，以及 PARTITION BY 改变了什么。",
                "Erkläre die Fensterfunktion und was PARTITION BY ändert.",
                "このウィンドウ関数と PARTITION BY が変えるものを説明してください。",
            ),
            "schema": ["window_meaning", "partition_effect"],
            "points": ["window", "partition"],
        },
    ],
    "math_proof": [
        {
            "payload": PROOF_IRRATIONAL,
            "ask": _t(
                "Identify the proof technique and state where the contradiction arises.",
                "指出这个证明使用的方法，并说明矛盾出现在哪里。",
                "Nenne die Beweistechnik und wo der Widerspruch entsteht.",
                "証明技法を示し、矛盾がどこで生じるか述べてください。",
            ),
            "schema": ["technique", "contradiction_point"],
            "points": ["technique", "contradiction"],
        },
        {
            "payload": PROOF_INDUCTION,
            "ask": _t(
                "State the induction hypothesis and verify the inductive step.",
                "写出归纳假设并验证归纳步骤。",
                "Nenne die Induktionsannahme und prüfe den Induktionsschritt.",
                "帰納法の仮定を述べ、帰納ステップを検証してください。",
            ),
            "schema": ["hypothesis", "step_valid"],
            "points": ["hypothesis", "step"],
        },
    ],
    "law": [
        {
            "payload": "A tenant remains in a flat after the lease expires. The landlord "
                       "continues to accept monthly rent without signing a new agreement.",
            "ask": _t(
                "Explain what tenancy status this creates and the notice implications.",
                "请解释这会形成何种租赁关系，以及在通知方面的含义。",
                "Erkläre, welches Mietverhältnis entsteht und was das für Kündigungsfristen heißt.",
                "どのような賃貸関係が生じるか、解約通知への影響を説明してください。",
            ),
            "schema": ["tenancy_type", "notice_period"],
            "points": ["status", "notice"],
        },
        {
            "payload": "Two parties sign a contract. One clause says disputes go to "
                       "arbitration; another says courts of a named city have exclusive jurisdiction.",
            "ask": _t(
                "Explain how such conflicting clauses are typically resolved.",
                "请解释通常如何处理这种相互冲突的条款。",
                "Erkläre, wie solche widersprüchlichen Klauseln üblicherweise aufgelöst werden.",
                "このように矛盾する条項が通常どう解決されるか説明してください。",
            ),
            "schema": ["conflict", "resolution_principle"],
            "points": ["conflict", "principle"],
        },
    ],
    "medicine": [
        {
            "payload": "A patient reports sudden one-sided facial weakness, unable to "
                       "raise the eyebrow on that side, with normal limb strength.",
            "ask": _t(
                "Explain what the forehead involvement indicates about the lesion site.",
                "请解释前额受累说明病变位置在哪里。",
                "Erkläre, was die Beteiligung der Stirn über den Läsionsort aussagt.",
                "前額部の関与が病変部位について何を示すか説明してください。",
            ),
            "schema": ["lesion_site", "reasoning"],
            "points": ["localisation", "reasoning"],
        },
        {
            "payload": "A patient on a new medication develops a dry persistent cough "
                       "several weeks after starting treatment for high blood pressure.",
            "ask": _t(
                "Explain the likely drug class involved and the mechanism.",
                "请解释可能涉及的药物类别及其机制。",
                "Erkläre die wahrscheinliche Wirkstoffklasse und den Mechanismus.",
                "関与しうる薬剤クラスとその機序を説明してください。",
            ),
            "schema": ["drug_class", "mechanism"],
            "points": ["class", "mechanism"],
        },
    ],
    "music_theory": [
        {
            "payload": "A progression moves ii - V - I in C major, then repeats the "
                       "same shape a whole step lower.",
            "ask": _t(
                "Explain the function of each chord and what the transposition implies.",
                "请解释每个和弦的功能，以及移调意味着什么。",
                "Erkläre die Funktion jedes Akkords und was die Transposition bedeutet.",
                "各和音の機能と、移調が意味するものを説明してください。",
            ),
            "schema": ["chord_functions", "transposition_effect"],
            "points": ["functions", "transposition"],
        },
        {
            "payload": "A melody uses the notes of A minor but consistently raises the "
                       "seventh degree when ascending.",
            "ask": _t(
                "Identify the scale and explain why the seventh is raised.",
                "指出这是什么音阶，并解释为何升高第七级。",
                "Nenne die Tonleiter und erkläre, warum die Septime erhöht wird.",
                "この音階を特定し、第七音が上げられる理由を説明してください。",
            ),
            "schema": ["scale_name", "raised_seventh_reason"],
            "points": ["scale", "reason"],
        },
    ],
    "cooking": [
        {
            "payload": "Onions are cooked slowly in fat for forty minutes until deep "
                       "brown, stirred often, with a pinch of salt added early.",
            "ask": _t(
                "Explain the chemical process and why salt is added early.",
                "请解释其中的化学过程，以及为什么早加盐。",
                "Erkläre den chemischen Prozess und warum das Salz früh zugegeben wird.",
                "この化学的過程と、塩を早く加える理由を説明してください。",
            ),
            "schema": ["process", "salt_role"],
            "points": ["chemistry", "salt"],
        },
        {
            "payload": "A bread dough is left to rise overnight in a cold place rather "
                       "than for two hours at room temperature.",
            "ask": _t(
                "Explain how the slow cold rise changes flavour and texture.",
                "请解释低温慢发酵如何改变风味和口感。",
                "Erkläre, wie die langsame kalte Gare Geschmack und Textur verändert.",
                "低温長時間発酵が風味と食感をどう変えるか説明してください。",
            ),
            "schema": ["flavour_change", "texture_change"],
            "points": ["flavour", "texture"],
        },
    ],
    "history": [
        {
            "payload": "Between 1845 and 1852 Ireland's population fell by roughly a "
                       "quarter through death and emigration.",
            "ask": _t(
                "Explain the principal causes and one long-term consequence.",
                "请解释主要原因和一个长期后果。",
                "Erkläre die Hauptursachen und eine langfristige Folge.",
                "主な原因と長期的な帰結を一つ説明してください。",
            ),
            "schema": ["causes", "long_term_consequence"],
            "points": ["causes", "consequence"],
        },
        {
            "payload": "The printing press spread across Europe in the decades after 1450, "
                       "reaching most major cities within fifty years.",
            "ask": _t(
                "Explain one social effect and one economic effect of this spread.",
                "请解释这一传播的一个社会影响和一个经济影响。",
                "Erkläre je eine soziale und eine wirtschaftliche Folge dieser Verbreitung.",
                "この普及の社会的影響と経済的影響を一つずつ説明してください。",
            ),
            "schema": ["social_effect", "economic_effect"],
            "points": ["social", "economic"],
        },
    ],
    "poetry": [
        {
            "payload": "The sea is calm tonight.\nThe tide is full, the moon lies fair\n"
                       "Upon the straits.",
            "ask": _t(
                "Describe the metre and the mood established by these lines.",
                "请描述这几行的格律和所营造的情绪。",
                "Beschreibe das Versmaß und die Stimmung dieser Zeilen.",
                "これらの行の韻律と、生み出される雰囲気を述べてください。",
            ),
            "schema": ["metre", "mood"],
            "points": ["metre", "mood"],
        },
        {
            "payload": "A poem repeats the same end-word at the close of each stanza, "
                       "rotating through six such words in a fixed pattern.",
            "ask": _t(
                "Identify this form and explain the effect of the repetition.",
                "指出这种诗体，并解释重复所产生的效果。",
                "Nenne diese Form und erkläre die Wirkung der Wiederholung.",
                "この詩形を特定し、反復の効果を説明してください。",
            ),
            "schema": ["form_name", "repetition_effect"],
            "points": ["form", "effect"],
        },
    ],
}

# --- Framing. Register and format wrappers, per language. --------------------

FORMAL_OPEN = _t(
    "Consider the following:", "请看以下内容：",
    "Betrachten Sie Folgendes:", "以下を検討してください。",
)
CASUAL_OPEN = _t(
    "ok so i've got this:", "我这有个东西：",
    "also ich hab hier das:", "これなんだけど：",
)
CASUAL_TAIL = _t(
    "any idea?", "有什么想法吗？", "irgendeine Idee?", "どう思う？",
)
JSON_INSTR = _t(
    "Answer as a JSON object with keys:", "请用带以下键的 JSON 对象回答：",
    "Antworte als JSON-Objekt mit den Schlüsseln:", "次のキーを持つ JSON オブジェクトで答えてください。",
)
BULLET_INSTR = _t(
    "Answer as a bulleted list covering:", "请用要点列表回答，涵盖：",
    "Antworte als Aufzählung zu:", "次の項目を箇条書きで答えてください。",
)


def render(topic: str, stem_idx: int, lang: str, register: str, fmt: str) -> str:
    """Render one probe. The payload is invariant across lang/register/format."""
    stem = TOPICS[topic][stem_idx]
    opener = (FORMAL_OPEN if register == "formal" else CASUAL_OPEN)[lang]
    ask = stem["ask"][lang]

    if fmt == "json":
        keys = ", ".join(f'"{k}"' for k in stem["schema"])
        instruction = f"{ask}\n{JSON_INSTR[lang]} {keys}"
    elif fmt == "bulleted":
        pts = ", ".join(stem["points"])
        instruction = f"{ask}\n{BULLET_INSTR[lang]} {pts}"
    else:
        instruction = ask

    body = f"{opener}\n\n{stem['payload']}\n\n{instruction}"
    if register == "casual":
        body = f"{body}\n{CASUAL_TAIL[lang]}"
    return body
