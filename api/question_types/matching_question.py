from typing import Dict, Any, List
import json
import re
from .base_question import BaseQuestion

class MatchingQuestion(BaseQuestion):
    """Matching question class"""
    
    def __init__(self, question_data: Dict[str, Any]):
        super().__init__(question_data)
        self.concepts = question_data.get("concepts", [])
        self.descriptions = question_data.get("descriptions", [])
        self.correct_mapping = question_data.get("correct_mapping", {})
        self.scoring = question_data.get("scoring", {
            "method": "exact_match",
            "points_per_correct": 1,
            "total_possible": len(self.concepts)
        })
    
    def build_prompt(self) -> str:
        """Build matching question prompt"""
        concepts_text = "\n".join([f"{i+1}. {concept}" for i, concept in enumerate(self.concepts)])
        descriptions_text = "\n".join([f"{chr(65+i)}. {desc}" for i, desc in enumerate(self.descriptions)])
        
        return f"""As a blockchain domain expert, please match the following concepts with their corresponding descriptions.

Concept list:
{concepts_text}

Description list:
{descriptions_text}

{self.instructions}

Please match each concept with the corresponding description letter, and only output the numbered relationships in the following format:
1 -> A
2 -> B
...

Do not explain, do not output anything else.
"""
    
    def evaluate_response(self, response: str) -> Dict:
        """Evaluate the model's answer"""
        try:
            # Parse the model's answer
            matches = {}
            model_mapping = {}  # Used to store the original model answers
            
            # Try to extract answers in expected format: Number -> Letter
            lines = response.strip().split('\n')
            for line in lines:
                # Handle standard format: "1 -> A" or "1->A"
                if '->' in line:
                    parts = line.split('->')
                    if len(parts) == 2:
                        try:
                            concept_idx_str = parts[0].strip()
                            # Extract just the number from text like "Starting with ETH (concept 1)"
                            numbers = re.findall(r'\b\d+\b', concept_idx_str)
                            if numbers:
                                concept_idx = int(numbers[0]) - 1  # Use the first number found
                            else:
                                concept_idx = int(concept_idx_str) - 1  # Try direct conversion
                                
                            desc_letter = parts[1].strip()
                            # Extract just the letter if there's additional text
                            letters = re.findall(r'\b[A-Z]\b', desc_letter.upper())
                            if letters:
                                desc_letter = letters[0]
                                
                            if 0 <= concept_idx < len(self.concepts):
                                concept = self.concepts[concept_idx]
                                # Save original answer
                                model_mapping[desc_letter] = concept
                                # If the letter already exists, there's a duplicate match, record error
                                if desc_letter in matches:
                                    print(f"Warning: Letter {desc_letter} has duplicate matches")
                                    continue
                                matches[desc_letter] = concept
                        except ValueError as e:
                            print(f"Error parsing line '{line}': {e}")
                            continue
                
                # Try to match alternative formats like "1: A" or "1. A" or "1 - A"
                elif re.search(r'\d+[\s]*[:.-][\s]*[A-Z]', line, re.IGNORECASE):
                    try:
                        # Extract number and letter
                        match = re.search(r'(\d+)[\s]*[:.-][\s]*([A-Z])', line, re.IGNORECASE)
                        if match:
                            concept_idx = int(match.group(1)) - 1
                            desc_letter = match.group(2).upper()
                            
                            if 0 <= concept_idx < len(self.concepts):
                                concept = self.concepts[concept_idx]
                                model_mapping[desc_letter] = concept
                                if desc_letter in matches:
                                    print(f"Warning: Letter {desc_letter} has duplicate matches")
                                    continue
                                matches[desc_letter] = concept
                    except ValueError as e:
                        print(f"Error parsing line '{line}': {e}")
                        continue
            
            # If no matches found with standard formats, try to extract any number-letter pairs
            if not matches:
                print("No standard format matches found, trying to extract concept-letter pairs...")
                # Look for patterns like "Concept X goes with Letter Y"
                for i, concept in enumerate(self.concepts):
                    concept_mentions = re.findall(rf'{re.escape(concept)}[\s\S]{{1,30}}?([A-Z])\b', response, re.IGNORECASE)
                    if concept_mentions:
                        desc_letter = concept_mentions[0].upper()
                        model_mapping[desc_letter] = concept
                        if desc_letter not in matches:  # Avoid duplicates
                            matches[desc_letter] = concept
                
                # Look for patterns like "Description Y matches with Concept X"
                for i, desc in enumerate(self.descriptions):
                    letter = chr(65 + i)  # A, B, C, ...
                    desc_mentions = re.findall(rf'{re.escape(desc)}[\s\S]{{1,50}}?({"|".join(re.escape(c) for c in self.concepts)})', response, re.IGNORECASE)
                    if desc_mentions:
                        concept = desc_mentions[0]
                        # Find exact match from concepts list (case-insensitive)
                        for c in self.concepts:
                            if c.lower() == concept.lower():
                                concept = c
                                break
                        model_mapping[letter] = concept
                        if letter not in matches:  # Avoid duplicates
                            matches[letter] = concept
            
            # Create description text to letter mapping
            desc_to_letter = {}
            for i, desc in enumerate(self.descriptions):
                letter = chr(65 + i)  # A, B, C, ...
                desc_to_letter[desc] = letter
            
            # Calculate number of correct matches
            correct_matches = 0
            for desc, expected_concept in self.correct_mapping.items():
                letter = desc_to_letter[desc]
                if letter in matches and matches[letter] == expected_concept:
                    correct_matches += 1
            
            # Calculate score
            score = correct_matches * self.scoring["points_per_correct"]
            
            # Debug information
            print("\n=== Scoring Details ===")
            print(f"Description to letter mapping: {desc_to_letter}")
            print(f"Model's original answer: {model_mapping}")
            print(f"Processed answer: {matches}")
            print(f"Correct answer: {self.correct_mapping}")
            print(f"Number of correct matches: {correct_matches}")
            print("===============\n")
            
            return {
                "score": score,
                "total_possible": self.scoring["total_possible"],
                "correct_matches": correct_matches,
                "total_matches": len(self.correct_mapping),
                "matches": matches,
                "model_mapping": model_mapping,  # Save original answer
                "has_duplicate_matches": len(matches) < len(model_mapping)  # Use original answer length to determine if there are duplicates
            }
        except Exception as e:
            print(f"Error while evaluating answer: {e}")
            return {
                "score": 0,
                "total_possible": self.scoring["total_possible"],
                "correct_matches": 0,
                "total_matches": len(self.correct_mapping),
                "matches": {},
                "model_mapping": {},
                "error": str(e)
            }
    
    def get_result_fields(self) -> Dict[str, Any]:
        """Get matching question result fields"""
        return {
            "question_type": "matching",
            "concepts": self.concepts,
            "descriptions": self.descriptions,
            "correct_mapping": self.correct_mapping,
            "scoring": self.scoring
        } 