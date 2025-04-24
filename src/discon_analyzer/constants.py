# Extended comparison keywords for Discussion sections
DISCUSSION_COMPARISON_KEYWORDS = [
    "similar to", "consistent with", "contrary to", "in contrast to", 
    "aligned with", "previous research", "prior studies", "other studies", 
    "compared to", "compared with", "comparable to", "in line with",
    "in agreement with", "differs from", "different from", "unlike",
    "contradicts", "supports", "extends", "builds on", "expands upon",
    "previous work", "earlier work", "previous findings", "earlier findings", 
    "existing literature", "existing research", "literature suggests", 
    "corroborates", "confirms", "reaffirms", "refutes", "challenges",
    "in accordance with", "resonates with", "echoes", "diverges from",
    "parallels", "aligns with", "deviates from", "converges with",
    "corresponds to", "contrasts with", "reinforces", "validates",
    "consistent to", "as reported by", "as found by", "as observed by",
    "as noted by", "similar results", "dissimilar to", "disagrees with",
    "congruent with", "incongruent with", "matches with", "coincides with",
    "concordant with", "discordant with", "as demonstrated by", "mirrors",
    "resembles", "analogous to", "akin to", "compatible with", "incompatible with",
    "concurrent with", "in harmony with", "conflicts with", "at odds with",
    "converges on", "diverges from", "runs counter to", "in opposition to",
    "coincidental with", "complementary to", "supplements", "augments",
    "in keeping with", "inconsistent with", "matches findings from", 
    "contradictory to", "aligns to", "as suggested by", "conforms with",
    "previous investigations", "established findings", "accepted theory",
    "consensus in literature", "prevailing view", "scholarly consensus",
    "differs significantly from", "highly similar to", "closely related to",
    "markedly different from", "significantly different from",
    "partially aligns with", "partially contradicts", "somewhat consistent with",
    "draws parallels to", "echoes findings from", "elaborates on",
    "substantiates", "empirically supports", "challenges conventional wisdom",
    "compares favorably with", "compares unfavorably with", 
    "extends beyond", "builds upon prior work", "broadly agrees with",
    "moderately consistent with", "provides an alternative perspective to",
    "revisits the claims of", "refines the approach of", "further supports",
    "introduces nuances to", "partly confirms", "slightly deviates from",
    "findings align partially with", "interpreted differently from",
    "differs methodologically from", "corroborates evidence from",
    "demonstrates convergence with", "contrasts starkly with",
    "suggests an alternative view to", "questions the conclusions of",
    "partially refutes", "provides additional support for",
    "raises doubts about", "further elaborates on", "strengthens prior claims",
    "reconciles conflicting views", "modifies the perspective on",
    "adds depth to", "introduces a competing perspective"
]


# Extended contribution keywords for Conclusion sections
CONCLUSION_CONTRIBUTION_KEYWORDS = [
    "this study contributes", "our main contribution", "we contribute",
    "the contribution of this paper", "the contributions are",
    "this paper presents a contribution", "this research provides a contribution",
    "this work advances", "this research advances", "this paper advances",
    "this study advances", "we advance", "our work advances",
    "this study offers", "our findings suggest", "our results show",
    "this paper demonstrates", "we have shown", "we have demonstrated",
    "this study highlights", "our study presents", "we introduce",
    "we have proposed", "we propose", "this paper proposes",
    "key finding", "significant finding", "novel approach", 
    "innovative method", "new perspective", "new insight",
    "implications for", "practical applications", "theoretical implications",
    "extends knowledge", "fills a gap", "addresses a gap",
    "this work presents", "our analysis reveals", "our investigation shows",
    "we developed", "we designed", "we implemented", "we created",
    "first study to", "first attempt to", "first investigation of",
    "this research establishes", "this paper offers", "this paper contributes to",
    "significance of this study", "importance of this work", "value of this research",
    "breakthrough", "pioneering study", "lays the foundation for",
    "expands upon", "enhances understanding", "provides a framework",
    "introduces a new", "revolutionary method", "transforms the way",
    "challenges existing", "bridges the gap between", "establishes a basis for",
    "clarifies", "demonstrates effectiveness", "paves the way for",
    "opens new directions", "yields new insights", "offers a novel perspective",
    "presents a refined approach", "provides empirical evidence",
    "contributes significantly to", "proposes a refined framework",
    "validates previous work", "builds upon", "overcomes limitations of",
    "enhances prior research", "fosters better understanding",
    "leads to better", "provides a robust framework",
    "introduces innovative", "addresses an unresolved issue",
    "introduces an improved model", "provides a comprehensive overview",
    "sets a new standard", "presents a compelling case for",
    "pioneers a new approach", "delivers new contributions",
    "outlines a promising direction", "develops a systematic approach",
    "presents a unique contribution", "achieves novel results",
    "redefines the problem", "brings a fresh perspective"
]


# Citation patterns (to detect comparisons with specific papers)
CITATION_PATTERNS = [
    r'$$[^)]*\d{4}[^)]*$$',  # Captures (Author, 2023) or (Author et al., 2023)
    r'\[[^\]]*\d+[^\]]*\]',  # Captures [1], [2-4]
    r'et al\.?',              # Captures "et al." mentions
    r'[A-Z][a-z]+ (?:and|&) [A-Z][a-z]+', # Captures "Smith and Jones"
    r'[A-Z][a-z]+ et al\.?'   # Captures "Smith et al."
]

# ...existing code...

# Comparison prepositions that often indicate comparison
COMPARISON_PREPOSITIONS = ["like", "unlike", "as", "than"]

# Terms that indicate different kinds of research or prior work
COMPARISON_REFERENCE_TERMS = ["previous", "prior", "existing", "other"]

# Terms that indicate reference to academic literature
LITERATURE_TERMS = ["literature", "studies", "research", "researchers", "papers", "findings"]

# Define semantic patterns for contribution claims
CONTRIBUTION_VERBS = [
    # Original verbs
    "contribute", "advance", "extend", "offer", "provide", "concludes",
    "demonstrate", "show", "reveal", "suggest", "indicate",
    "present", "demonstrate", "introduce", "propose", "highlight", 
    "show", "establish", "identify", "develop", "create", "fill", 
    "address", "expand", "enhance", "improve", "discover", 
    "innovate", "refine", "augment", "pioneer", "investigate", 
    "redefine", "design", "implement", "unveil", "introduce a novel",
    "shed light on", "illustrate", "evaluate", "frame", "outline", 
    "reconceptualize", "revisit", "clarify", "offer insights into", 
    "put forward", "reformulate", "propose a framework for",
    "establish a new perspective", "underscore", "articulate",
    "synthesize", "redefine theoretical foundations", "empirically validate",
    "bridge the gap", "break new ground", "construct", "craft", 
    "devise", "document", "elucidate", "formulate", "generate",
    "initiate", "lay groundwork for", "map out", "modernize",
    "operationalize", "optimize", "originate", "pave the way for",
    "pinpoint", "pioneer", "prove", "revolutionize", "spearhead",
    "transform", "uncover", "validate", "verify", "yield",
    "bring to light", "conceptualize", "distinguish", "elaborate",
    "elevate understanding of", "exemplify", "forge new paths in",
    "foster advancement in", "illuminate", "innovatively apply",
    "integrate perspectives on", "lay the foundation for",
    "make significant inroads into", "open new avenues for",
    "present first evidence of", "provide empirical support for",
    "reconceive", "resolve longstanding questions about",
    "systematically analyze", "unify theories of", "verify hypotheses about",
    "solve problems in", "tackle challenges in", "reveal mechanisms of",
    "demonstrate the feasibility of", "confirm the importance of",
    "challenge assumptions about", "deepen knowledge of", "catalyze research on"
]

CONTRIBUTION_TERMS = [
    "contribution", "advancement", "innovation", "novel", "new",
    "significant", "important", "key", "major", "primary", "main"
]

# Define semantic patterns for comparison with prior research
COMPARISON_VERBS = [
    # Original verbs
    "compare", "contrast", "align", "differ", "support", 
    "contradict", "confirm", "validate", "challenge", "refute", 
    "extend", "build on", "resonate", "echo", "diverge", "agree",
    "corroborate", "reaffirm", "parallel", "correspond", 
    "reinforce", "replicate", "deviate", "supplement", "oppose",
    "amplify", "reinterpret", "elaborate on", "nuance", 
    "synthesize", "harmonize with", "reconcile", "complicate",
    "substantiate", "mirror", "expand upon", "converge with", 
    "deconstruct", "modify", "reinterpret in light of", 
    "refine existing findings", "contextualize", "frame differently", 
    "critique", "problematize", "examine similarities with", 
    "draw connections between", "reexamine", "update", "reposition",
    "juxtapose", "analyze against", "view in light of", "position relative to",
    "evaluate in context of", "situate within", "assess in relation to",
    "weigh against", "measure against", "distinguish from", "differentiate from",
    "scrutinize alongside", "consider in relation to", "put in dialogue with",
    "read alongside", "interpret through lens of", "examine in reference to",
    "analyze in connection with", "investigate in comparison to",
    "place in conversation with", "trace connections to", "highlight differences from",
    "identify similarities with", "underscore distinctions from", "emphasize parallels with",
    "note departures from", "articulate relationship with", "demonstrate links to",
    "show departures from", "illustrate connections between", "map onto",
    "locate within tradition of", "contest findings of", "affirm conclusions of",
    "dispute results of", "engage critically with", "respond to arguments in",
    "counter perspectives in", "elaborate upon insights from", "qualify conclusions of",
    "complement analysis in", "supplement understanding from", "build upon framework of",
    "adapt methodology from", "extend theoretical approach of", "enhance model proposed by",
    "reconsider assumptions of", "revisit conclusions drawn by", "reassess evidence from"
]

RESEARCH_TERMS = [
    "study", "research", "paper", "work", "finding", "result",
    "approach", "method", "literature", "previous", "prior", "existing"
]

# Ownership terms for semantic analysis
OWNERSHIP_TERMS = ["our", "this", "the", "these"]