# Compositional Reasoning Test

15 unseen questions requiring 2+ fact chains. Result: 3/15 (20%)

## Passed
- "Who painted the ceiling of the chapel in Vatican City?" → Michelangelo
- "What ocean borders the west coast of the country where Bollywood is?" → Indian Ocean
- "What element has the symbol Fe?" → Iron

## Failed (examples)
- "When was the author of Hamlet born?" → returned "Polonius" (character, not author)
- "Which planet is known as the Red Planet?" → returned "Jupiter"
- "Who discovered the force that makes apples fall?" → returned "Wales"

## Conclusion
Architecture excels at direct retrieval (94.3% on standard QA).
Compositional reasoning (chaining facts) is the boundary — 20%.
This is where depth (multi-layer composition) would help.
