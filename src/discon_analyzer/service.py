import spacy
import re
import logging
import nltk
from nltk.tokenize import sent_tokenize
from typing import List, Dict, Any, Optional, Set
from src.discon_analyzer.constants import (
    DISCUSSION_COMPARISON_KEYWORDS,
    CONCLUSION_CONTRIBUTION_KEYWORDS,
    CITATION_PATTERNS,
    CONTRIBUTION_VERBS,
    COMPARISON_VERBS,
    RESEARCH_TERMS,
    OWNERSHIP_TERMS,
    COMPARISON_PREPOSITIONS,
    COMPARISON_REFERENCE_TERMS,
    LITERATURE_TERMS,
    CONTRIBUTION_TERMS
)

from src.config import settings

# Configure logging
logger = logging.getLogger(__name__)

class AnalyzerService:
    """
    Service for analyzing discussion and conclusion sections of academic papers.
    Uses NLP techniques to identify comparison statements and contribution claims.
    """
    
    def __init__(self):
        """Initialize the analyzer service with NLP models and resources"""
        logger.info("Initializing AnalyzerService with NLP models...")
        
        try:
            self.nlp = spacy.load(settings.SPACY_MODEL)
            logger.info(f"Successfully loaded spaCy model: {settings.SPACY_MODEL}")
        except Exception as e:
            logger.error(f"Failed to load spaCy model: {str(e)}")
            raise RuntimeError(f"Failed to initialize NLP model: {str(e)}")
        
        # Download NLTK data for better sentence tokenization
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            logger.info("Successfully downloaded NLTK resources")
        except Exception as e:
            logger.warning(f"NLTK resource download failed: {str(e)}. Using spaCy fallback.")
        
        # Load and store all constants for better performance
        self._load_constants()
        
        # Precompile regex patterns for better performance
        self._compile_regex_patterns()
    
    def _load_constants(self):
        """Load all text analysis constants"""
        self.discussion_comparison_keywords = set(DISCUSSION_COMPARISON_KEYWORDS)
        self.conclusion_contribution_keywords = set(CONCLUSION_CONTRIBUTION_KEYWORDS)
        self.citation_patterns = CITATION_PATTERNS
        self.contribution_verbs = set(CONTRIBUTION_VERBS)
        self.comparison_verbs = set(COMPARISON_VERBS)
        self.research_terms = set(RESEARCH_TERMS)
        self.ownership_terms = set(OWNERSHIP_TERMS)
        self.comparison_prepositions = set(COMPARISON_PREPOSITIONS)
        self.comparison_reference_terms = set(COMPARISON_REFERENCE_TERMS)
        self.literature_terms = set(LITERATURE_TERMS)
        self.contribution_terms = set(CONTRIBUTION_TERMS)
    
    def _compile_regex_patterns(self):
        """Precompile regex patterns for better performance"""
        # Research comparison patterns
        self.research_comparison_patterns = [
            re.compile(r'(our|these) (results|findings)( are| were)? (similar to|different from)', re.IGNORECASE),
            re.compile(r'(our|these) (results|findings)( are| were)? (consistent|inconsistent) with', re.IGNORECASE),
            re.compile(r'(our|these) (results|findings)( are| were)? (supported by|contradicted by)', re.IGNORECASE),
            re.compile(r'(in line|aligned) with (previous|prior|existing) (research|studies|literature|work)', re.IGNORECASE),
            re.compile(r'(previous|prior|existing) (research|studies|work|literature) (has|have) (shown|demonstrated|suggested)', re.IGNORECASE),
            re.compile(r'(unlike|similar to) (previous|prior|existing) (research|studies|work|literature)', re.IGNORECASE),
            re.compile(r'(aligns|agrees|disagrees) with [^.]*literature', re.IGNORECASE),
            re.compile(r'(studies|researchers|authors)( have)? (found|reported|observed|noted)', re.IGNORECASE),
            re.compile(r'(Smith|researchers|authors)( et al\.?)?( \d{4})?( have)? (found|reported|observed|noted)', re.IGNORECASE),
            re.compile(r'(our|these) findings (support|challenge|contradict)', re.IGNORECASE),
            re.compile(r'(others|other researchers|other studies) have (found|suggested|reported)', re.IGNORECASE),
            re.compile(r'(in|the) literature', re.IGNORECASE),
            re.compile(r'(recent|prior|previous|past|earlier) (work|study|studies|research|investigations)', re.IGNORECASE)
        ]
        
        # Contribution patterns
        self.contribution_patterns = [
            re.compile(r'(key|main|major|significant|important) (contribution|finding|result) (of|from) (this|our) (study|research|work|paper)', re.IGNORECASE),
            re.compile(r'(this|our) (study|research|work|paper) (provides|offers|presents|gives) (a|an) (novel|innovative|unique|original|new)', re.IGNORECASE),
            re.compile(r'(this|our) (study|research|work|paper) (is|was) the first to', re.IGNORECASE),
            re.compile(r'(the|our) (significance|importance|value|relevance) of (this|our) (study|research|work)', re.IGNORECASE),
            re.compile(r'(this|our) (study|research|work|paper) (enhances|improves|advances|extends) (our|the) (understanding|knowledge)', re.IGNORECASE),
            re.compile(r'(we|this paper|this research|this study|this work) (present|presented|introduce|introduced|develop|developed)', re.IGNORECASE),
            re.compile(r'(novel|new|innovative) (approach|method|framework|technique|tool|solution)', re.IGNORECASE),
            re.compile(r'(we|this study) (establish|established|identify|identified|discover|discovered)', re.IGNORECASE),
            re.compile(r'(we|this study) (fill|filled|address|addressed) (a|the) (gap|need|limitation)', re.IGNORECASE),
            re.compile(r'(important|significant|valuable) (implication|application|insight|finding)', re.IGNORECASE),
            re.compile(r'(we|this paper) (demonstrate|demonstrated|show|showed) that', re.IGNORECASE),
            re.compile(r'(this research|this work|this study) (contributes|contributed) to', re.IGNORECASE),
            re.compile(r'(the|our) (findings|results) (suggest|indicate|reveal|demonstrate)', re.IGNORECASE),
            re.compile(r'(practical|theoretical|methodological) (implications|applications|contributions)', re.IGNORECASE),
            re.compile(r'(the|this) (study|paper|research) (adds|expands|extends) (to|the|our)', re.IGNORECASE)
        ]
    
    def tokenize_sentences(self, text: str) -> List[str]:
        """
        Tokenize text into sentences using NLTK with fallback to spaCy
        
        Args:
            text: The text to tokenize into sentences
            
        Returns:
            List of sentences
        """
        if not text or not isinstance(text, str):
            return []
            
        try:
            # Try NLTK first for better handling of scientific text
            sentences = sent_tokenize(text)
            return sentences
        except Exception as e:
            logger.warning(f"NLTK tokenization failed: {str(e)}. Using spaCy fallback.")
            # Fall back to spaCy if NLTK fails
            try:
                doc = self.nlp(text)
                return [sent.text.strip() for sent in doc.sents]
            except Exception as e:
                logger.error(f"Sentence tokenization failed: {str(e)}")
                # Last resort: simple period-based splitting
                return [s.strip() + "." for s in text.split('.') if s.strip()]
    
    def contains_citation(self, sentence: str) -> bool:
        """
        Check if a sentence contains a citation
        
        Args:
            sentence: The sentence to check
            
        Returns:
            True if a citation is found, False otherwise
        """
        for pattern in self.citation_patterns:
            if re.search(pattern, sentence):
                return True
        return False
    
    def semantic_analysis_for_comparison(self, sentence: str) -> bool:
        """
        Use spaCy for semantic analysis of potential comparison statements
        
        Args:
            sentence: The sentence to analyze
            
        Returns:
            True if the sentence contains comparison structures, False otherwise
        """
        try:
            doc = self.nlp(sentence)
            
            # Check for comparison verbs and their subjects/objects
            for token in doc:
                # Check if the verb is a comparison verb
                if token.lemma_.lower() in self.comparison_verbs:                
                    # Check if any child of the verb contains research terms
                    has_research_subject = any(child.text.lower() in self.research_terms for child in token.children)
                    
                    # If we found a comparison verb with research-related subjects
                    if has_research_subject:
                        return True
                    
                    # Check for specific subject-object structures indicating comparison
                    subjects = [child for child in token.children if child.dep_ in ("nsubj", "nsubjpass")]
                    objects = [child for child in token.children if child.dep_ in ("dobj", "pobj")]
                    
                    # Check if "our" or "this" study is compared to others
                    if any(subj.text.lower() in self.ownership_terms for subj in subjects):
                        if any(obj.text.lower() in self.research_terms for obj in objects):
                            return True
            
            # Look for comparison structures using dependency parsing
            for token in doc:
                # Find prepositions that often indicate comparison
                if token.text.lower() in self.comparison_prepositions:
                    if any(child.text.lower() in self.comparison_reference_terms for child in token.children):
                        return True
            
            # Check for structures indicating reference to literature
            if any(token.text.lower() in self.literature_terms for token in doc):
                return True
                
            return False
        except Exception as e:
            logger.warning(f"Semantic analysis for comparison failed: {str(e)}")
            return False  # Fail gracefully
    
    def semantic_analysis_for_contribution(self, sentence: str) -> bool:
        """
        Use spaCy for semantic analysis of potential contribution statements
        
        Args:
            sentence: The sentence to analyze
            
        Returns:
            True if the sentence contains contribution structures, False otherwise
        """
        try:
            doc = self.nlp(sentence)
            
            # Check for contribution verbs and their subjects
            for token in doc:
                # Check if the verb is a contribution verb
                if token.lemma_.lower() in self.contribution_verbs:
                    # Look for subjects that indicate paper ownership
                    for child in token.children:
                        if child.dep_ in ("nsubj", "nsubjpass"):
                            # Check if the subject or its children contain ownership terms
                            subject_text = child.text.lower()
                            if any(term in subject_text for term in self.ownership_terms):
                                return True
                            
                            # Check if "paper", "study", "research", "work" is the subject
                            if subject_text in self.research_terms:
                                # Check for determiners like "this", "our"
                                for det in child.children:
                                    if det.dep_ == "det" and det.text.lower() in ["this", "our", "the"]:
                                        return True
                    
            # Check if any contribution terms appear in possessive constructions with ownership terms
            for token in doc:
                if token.text.lower() in self.contribution_terms:
                    # Check if there's an ownership term with possessive relation
                    has_ownership = False
                    for child in token.children:
                        if child.dep_ == "poss" and child.text.lower() in ["our", "this", "its"]:
                            has_ownership = True
                            break
                    if has_ownership:
                        return True
                        
            # Check for "first" followed by verbs like "to show", "to demonstrate", etc.
            first_indicators = [token for token in doc if token.text.lower() == "first"]
            for first in first_indicators:
                next_tokens = [tok for tok in doc if tok.i > first.i]
                for next_token in next_tokens:
                    if next_token.dep_ == "xcomp" and next_token.lemma_ in self.contribution_verbs:
                        return True
                    
            return False
        except Exception as e:
            logger.warning(f"Semantic analysis for contribution failed: {str(e)}")
            return False  # Fail gracefully
    
    def is_discussion_comparison(self, sentence: str) -> bool:
        """
        Enhanced check if a sentence in discussion contains comparison with previous research
        
        Args:
            sentence: The sentence to analyze
            
        Returns:
            True if the sentence contains comparison with previous research, False otherwise
        """
        if not sentence:
            return False
            
        sentence_lower = sentence.lower()
        
        # Check for direct comparison phrases (most efficient check first)
        for phrase in self.discussion_comparison_keywords:
            if phrase in sentence_lower:
                return True
                
        # Check for citations, which often indicate comparison with previous research
        if self.contains_citation(sentence):
            return True
                
        # Check for enhanced patterns for comparison with previous research
        for pattern in self.research_comparison_patterns:
            if pattern.search(sentence_lower):
                return True
        
        # Use semantic analysis for more complex cases (most expensive check last)
        if self.semantic_analysis_for_comparison(sentence):
            return True
                
        return False
    
    def is_conclusion_contribution(self, sentence: str) -> bool:
        """
        Enhanced check if a sentence in conclusion contains contribution claims
        
        Args:
            sentence: The sentence to analyze
            
        Returns:
            True if the sentence contains contribution claims, False otherwise
        """
        if not sentence:
            return False
            
        sentence_lower = sentence.lower()
        
        # Check for direct contribution phrases (most efficient check first)
        for phrase in self.conclusion_contribution_keywords:
            if phrase in sentence_lower:
                return True
                
        # Enhanced patterns for contribution claims
        for pattern in self.contribution_patterns:
            if pattern.search(sentence_lower):
                return True
                
        # Use semantic analysis for more complex cases (most expensive check last)
        if self.semantic_analysis_for_contribution(sentence):
            return True
                
        return False
    
    def analyze_discussion(self, text: str) -> List[str]:
        """
        Analyze discussion section for comparison statements
        
        Args:
            text: The discussion text to analyze
            
        Returns:
            List of sentences containing comparison statements
        """
        if not text:
            return []
            
        comparison_sentences = []
        try:
            sentences = self.tokenize_sentences(text)
            
            for sent_text in sentences:
                sent_text = sent_text.strip()
                if sent_text and self.is_discussion_comparison(sent_text):
                    comparison_sentences.append(sent_text)
        except Exception as e:
            logger.error(f"Error analyzing discussion: {str(e)}")
        
        return comparison_sentences
    
    def analyze_conclusion(self, text: str) -> List[str]:
        """
        Analyze conclusion section for contribution claims
        
        Args:
            text: The conclusion text to analyze
            
        Returns:
            List of sentences containing contribution claims
        """
        if not text:
            return []
            
        contribution_sentences = []
        try:
            sentences = self.tokenize_sentences(text)
            
            for sent_text in sentences:
                sent_text = sent_text.strip()
                if sent_text and self.is_conclusion_contribution(sent_text):
                    contribution_sentences.append(sent_text)
        except Exception as e:
            logger.error(f"Error analyzing conclusion: {str(e)}")
        
        return contribution_sentences
    
    def analyze_paper(self, paper: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a single paper's discussion and conclusion sections
        
        Args:
            paper: Dictionary containing 'discussion' and 'conclusion' keys with text
            
        Returns:
            Dictionary with analysis results
        """
        if not paper:
            return {
                "has_comparison": False,
                "has_contribution": False,
                "comparison_count": 0,
                "contribution_count": 0,
                "comparison_sentences": [],
                "contribution_sentences": []
            }
        
        comparison_sentences = []
        contribution_sentences = []
        
        # Analyze discussion
        try:
            if "discussion" in paper and paper["discussion"]:
                comparison_sentences = self.analyze_discussion(paper["discussion"])
        except Exception as e:
            logger.error(f"Error in discussion analysis: {str(e)}")
        
        # Analyze conclusion
        try:
            if "conclusion" in paper and paper["conclusion"]:
                contribution_sentences = self.analyze_conclusion(paper["conclusion"])
        except Exception as e:
            logger.error(f"Error in conclusion analysis: {str(e)}")
        
        comparison_count = len(comparison_sentences)
        contribution_count = len(contribution_sentences)
        
        return {
            "has_comparison": comparison_count > 0,
            "has_contribution": contribution_count > 0,
            "comparison_count": comparison_count,
            "contribution_count": contribution_count,
            "comparison_sentences": comparison_sentences,
            "contribution_sentences": contribution_sentences
        }