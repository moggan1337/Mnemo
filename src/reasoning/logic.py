"""
Logic Engine for formal reasoning.

Provides propositional and predicate logic operations
for rigorous inference.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class LogicalOperator(Enum):
    """Logical operators."""
    AND = "AND"
    OR = "OR"
    NOT = "NOT"
    IMPLIES = "IMPLIES"
    EQUIVALENT = "EQUIVALENT"


@dataclass
class Proposition:
    """A logical proposition."""
    symbol: str
    content: str
    value: Optional[bool] = None  # True, False, or None (unknown)


class LogicEngine:
    """
    Logic Engine for formal reasoning.
    
    Supports:
    - Propositional logic (AND, OR, NOT, IMPLIES)
    - Truth table evaluation
    - Logical inference rules
    - Consistency checking
    """
    
    def __init__(self):
        """Initialize Logic Engine."""
        self._propositions: dict[str, Proposition] = {}
        
        logger.info("LogicEngine initialized")
    
    def add_proposition(
        self,
        symbol: str,
        content: str,
        value: Optional[bool] = None,
    ) -> Proposition:
        """Add a proposition to the knowledge base."""
        prop = Proposition(symbol=symbol, content=content, value=value)
        self._propositions[symbol] = prop
        return prop
    
    def evaluate(
        self,
        expression: str,
        truth_values: Optional[dict[str, bool]] = None,
    ) -> Optional[bool]:
        """
        Evaluate a logical expression.
        
        Args:
            expression: Logical expression (e.g., "A AND B")
            truth_values: Truth values for symbols
            
        Returns:
            True, False, or None if cannot evaluate
        """
        if truth_values is None:
            truth_values = {}
            
            # Use stored values
            for symbol, prop in self._propositions.items():
                if prop.value is not None:
                    truth_values[symbol] = prop.value
        
        try:
            return self._evaluate_expression(expression, truth_values)
        except Exception as e:
            logger.error(f"Failed to evaluate '{expression}': {e}")
            return None
    
    def _evaluate_expression(
        self,
        expr: str,
        truth_values: dict[str, bool],
    ) -> Optional[bool]:
        """Evaluate a logical expression recursively."""
        expr = expr.strip()
        
        # Handle parentheses
        if expr.startswith("(") and expr.endswith(")"):
            return self._evaluate_expression(expr[1:-1], truth_values)
        
        # Handle NOT
        if expr.startswith("NOT ") or expr.startswith("not "):
            operand = expr[4:].strip()
            value = self._evaluate_expression(operand, truth_values)
            return not value if value is not None else None
        
        # Handle AND
        if " AND " in expr:
            parts = expr.split(" AND ")
            return all(
                self._evaluate_expression(p, truth_values)
                for p in parts
            )
        
        # Handle OR
        if " OR " in expr:
            parts = expr.split(" OR ")
            result = False
            for p in parts:
                val = self._evaluate_expression(p, truth_values)
                if val is True:
                    return True
                if val is None:
                    result = None
            return result
        
        # Handle IMPLIES
        if " IMPLIES " in expr:
            parts = expr.split(" IMPLIES ")
            antecedent = self._evaluate_expression(parts[0], truth_values)
            consequent = self._evaluate_expression(parts[1], truth_values)
            
            if antecedent is False:
                return True  # False implies anything
            if antecedent is True and consequent is False:
                return False
            if antecedent is True and consequent is True:
                return True
            return None
        
        # Handle EQUIVALENT
        if " EQUIVALENT " in expr:
            parts = expr.split(" EQUIVALENT ")
            values = [self._evaluate_expression(p, truth_values) for p in parts]
            
            if None in values:
                return None
            return len(set(values)) == 1
        
        # Simple symbol lookup
        if expr in truth_values:
            return truth_values[expr]
        
        return None
    
    def modus_ponens(
        self,
        implication: str,
        antecedent_symbol: str,
    ) -> Optional[str]:
        """
        Apply modus ponens: If A IMPLIES B, and A is true, then B is true.
        
        Args:
            implication: The implication (e.g., "A IMPLIES B")
            antecedent_symbol: The symbol known to be true
            
        Returns:
            The consequent symbol if valid
        """
        # Parse implication
        if " IMPLIES " in implication:
            parts = implication.split(" IMPLIES ")
            antecedent = parts[0].strip()
            consequent = parts[1].strip()
            
            if antecedent == antecedent_symbol:
                return consequent
        
        return None
    
    def modus_tollens(
        self,
        implication: str,
        consequent_symbol: str,
    ) -> Optional[str]:
        """
        Apply modus tollens: If A IMPLIES B, and B is false, then A is false.
        
        Args:
            implication: The implication (e.g., "A IMPLIES B")
            consequent_symbol: The symbol known to be false
            
        Returns:
            The negated antecedent symbol if valid
        """
        if " IMPLIES " in implication:
            parts = implication.split(" IMPLIES ")
            antecedent = parts[0].strip()
            consequent = parts[1].strip()
            
            if consequent == consequent_symbol:
                return f"NOT {antecedent}"
        
        return None
    
    def hypothetical_syllogism(
        self,
        first: str,
        second: str,
    ) -> Optional[str]:
        """
        Apply hypothetical syllogism: If A IMPLIES B, and B IMPLIES C, then A IMPLIES C.
        
        Returns:
            The combined implication if valid
        """
        # Extract consequents and antecedents
        def get_parts(imp):
            if " IMPLIES " in imp:
                parts = imp.split(" IMPLIES ")
                return parts[0].strip(), parts[1].strip()
            return None, None
        
        ant1, cons1 = get_parts(first)
        ant2, cons2 = get_parts(second)
        
        if ant1 and cons1 and ant2 and cons2:
            if cons1 == ant2:
                return f"{ant1} IMPLIES {cons2}"
        
        return None
    
    def check_consistency(
        self,
        propositions: list[str],
    ) -> tuple[bool, Optional[str]]:
        """
        Check if a set of propositions is consistent.
        
        Returns:
            Tuple of (is_consistent, conflict_description)
        """
        # Check for direct contradictions
        for i, prop1 in enumerate(propositions):
            for prop2 in propositions[i+1:]:
                # Check A and NOT A
                if prop2 == f"NOT {prop1}":
                    return False, f"{prop1} contradicts {prop2}"
                if prop1 == f"NOT {prop2}":
                    return False, f"{prop1} contradicts {prop2}"
        
        return True, None
    
    def simplify(self, expression: str) -> str:
        """Simplify a logical expression using basic rules."""
        expr = expression.strip()
        
        # Remove double negation
        while "NOT NOT " in expr:
            expr = expr.replace("NOT NOT ", "")
        
        # Simplify AND with True/False
        expr = expr.replace(" AND TRUE", "")
        expr = expr.replace("TRUE AND ", "")
        expr = expr.replace(" AND FALSE", "FALSE")
        expr = expr.replace("FALSE AND ", "FALSE")
        
        # Simplify OR with True/False
        expr = expr.replace(" OR TRUE", "TRUE")
        expr = expr.replace("TRUE OR ", "TRUE")
        expr = expr.replace(" OR FALSE", "")
        expr = expr.replace("FALSE OR ", "")
        
        return expr
    
    def to_truth_table(
        self,
        expression: str,
        symbols: list[str],
    ) -> list[dict[str, bool]]:
        """
        Generate a truth table for an expression.
        
        Args:
            expression: Logical expression
            symbols: List of proposition symbols
            
        Returns:
            List of rows with symbol values and result
        """
        from itertools import product
        
        rows = []
        
        # Generate all combinations
        for values in product([True, False], repeat=len(symbols)):
            truth_values = dict(zip(symbols, values))
            result = self.evaluate(expression, truth_values)
            
            row = dict(zip(symbols, values))
            row["result"] = result
            rows.append(row)
        
        return rows
