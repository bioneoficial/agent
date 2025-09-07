"""
JSON parsing utilities with retry logic for LLM responses.
Handles malformed JSON and provides fallback strategies.
"""

import json
import re
from typing import Dict, Any, Optional, Union, Type
from pydantic import BaseModel, ValidationError
import logging

logger = logging.getLogger(__name__)


def extract_json_from_text(text: str) -> Optional[str]:
    """Extract JSON from text that might contain extra content."""
    text = text.strip()
    
    # Try to find JSON within code blocks
    json_patterns = [
        r'```json\s*(.*?)\s*```',
        r'```\s*(.*?)\s*```',
        r'\{.*\}',
        r'\[.*\]'
    ]
    
    for pattern in json_patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            candidate = matches[0].strip()
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                continue
    
    # If no pattern matches, try the whole text
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass
    
    return None


def clean_json_string(json_str: str) -> str:
    """Clean common JSON formatting issues."""
    # Remove comments
    json_str = re.sub(r'//.*?\n', '\n', json_str)
    json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)
    
    # Fix trailing commas
    json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
    
    # Fix single quotes to double quotes
    json_str = re.sub(r"'([^']*)':", r'"\1":', json_str)
    json_str = re.sub(r":\s*'([^']*)'", r': "\1"', json_str)
    
    return json_str


def force_json(text: str, max_attempts: int = 3) -> Dict[str, Any]:
    """
    Force extract valid JSON from LLM response with multiple strategies.
    
    Args:
        text: Raw text response from LLM
        max_attempts: Maximum number of cleaning attempts
        
    Returns:
        Parsed JSON dictionary
        
    Raises:
        ValueError: If JSON cannot be extracted after all attempts
    """
    original_text = text
    
    for attempt in range(max_attempts):
        try:
            # First attempt: direct parsing
            if attempt == 0:
                return json.loads(text)
            
            # Extract JSON from text
            json_candidate = extract_json_from_text(text)
            if json_candidate is None:
                raise ValueError(f"No JSON found in text: {text[:200]}...")
            
            # Clean and parse
            cleaned = clean_json_string(json_candidate)
            return json.loads(cleaned)
            
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse attempt {attempt + 1} failed: {e}")
            if attempt == max_attempts - 1:
                # Last attempt: try to salvage what we can
                return _salvage_json(original_text)
            continue
    
    raise ValueError(f"Could not parse JSON after {max_attempts} attempts: {original_text[:200]}...")


def _salvage_json(text: str) -> Dict[str, Any]:
    """Last resort: try to extract key-value pairs manually."""
    result = {}
    
    # Look for key-value patterns
    patterns = [
        r'"([^"]+)":\s*"([^"]*)"',
        r'"([^"]+)":\s*(\d+)',
        r'"([^"]+)":\s*(true|false)',
        r'"([^"]+)":\s*\[(.*?)\]'
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            if len(match) == 2:
                key, value = match
                if value.isdigit():
                    result[key] = int(value)
                elif value in ('true', 'false'):
                    result[key] = value == 'true'
                elif value.startswith('[') and value.endswith(']'):
                    # Try to parse array
                    try:
                        result[key] = json.loads(value)
                    except:
                        result[key] = value.strip('[]').split(',')
                else:
                    result[key] = value
    
    return result if result else {"error": "Could not parse JSON", "raw_text": text[:500]}


def parse_json_with_retry(
    text: str, 
    model_class: Type[BaseModel], 
    max_retries: int = 3
) -> Union[BaseModel, Dict[str, Any]]:
    """
    Parse JSON and validate against Pydantic model with retry logic.
    
    Args:
        text: Raw text from LLM
        model_class: Pydantic model class to validate against
        max_retries: Maximum retry attempts
        
    Returns:
        Validated model instance or fallback dictionary
    """
    
    for attempt in range(max_retries):
        try:
            # Extract and parse JSON
            json_data = force_json(text, max_attempts=2)
            
            # Validate with Pydantic model
            return model_class(**json_data)
            
        except (ValueError, ValidationError) as e:
            logger.warning(f"Parse attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                # Return fallback dictionary
                try:
                    return force_json(text)
                except ValueError:
                    return {
                        "error": "Parse failed",
                        "original_text": text[:500],
                        "model_class": model_class.__name__
                    }
            continue
    
    return {"error": "Max retries exceeded"}


def validate_json_schema(data: Dict[str, Any], required_fields: list) -> bool:
    """Validate that JSON contains required fields."""
    return all(field in data for field in required_fields)


def repair_json_structure(data: Dict[str, Any], model_class: Type[BaseModel]) -> Dict[str, Any]:
    """Attempt to repair JSON structure to match expected schema."""
    try:
        # Get model fields
        model_fields = model_class.__fields__ if hasattr(model_class, '__fields__') else {}
        
        repaired = {}
        for field_name, field_info in model_fields.items():
            if field_name in data:
                repaired[field_name] = data[field_name]
            else:
                # Provide default values based on field type
                if hasattr(field_info, 'default'):
                    repaired[field_name] = field_info.default
                elif field_info.annotation == str:
                    repaired[field_name] = ""
                elif field_info.annotation == list:
                    repaired[field_name] = []
                elif field_info.annotation == dict:
                    repaired[field_name] = {}
                elif field_info.annotation == bool:
                    repaired[field_name] = False
                elif field_info.annotation == int:
                    repaired[field_name] = 0
                elif field_info.annotation == float:
                    repaired[field_name] = 0.0
        
        return repaired
        
    except Exception as e:
        logger.warning(f"Could not repair JSON structure: {e}")
        return data
