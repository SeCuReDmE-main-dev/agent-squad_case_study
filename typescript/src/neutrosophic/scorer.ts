import { Triplet } from "./triplet";

const HEDGING_PATTERNS = [
  "maybe",
  "might",
  "possibly",
  "probably",
  "not sure",
  "unclear",
  "unknown",
  "ambiguous",
  "insufficient",
  "could be",
  "it depends",
];

const ERROR_PATTERNS = [
  "error",
  "failed",
  "failure",
  "cannot",
  "can't",
  "unable",
  "invalid",
  "contradiction",
  "conflict",
  "not possible",
  "i do not know",
  "i don't know",
];

const REFUSAL_PATTERNS = [
  "i can't help",
  "i cannot help",
  "i cannot comply",
  "i can't comply",
  "sorry, but i can't",
  "sorry, but i cannot",
];

export function scoreTextResponse(text: string): Triplet {
  if (typeof text !== "string") {
    throw new TypeError("text must be a string");
  }

  const normalized = text.toLowerCase().split(/\s+/).filter(Boolean).join(" ");
  if (!normalized) {
    return new Triplet(0.0, 1.0, 0.0);
  }

  const hedgeCount = countMatches(normalized, HEDGING_PATTERNS);
  const errorCount = countMatches(normalized, ERROR_PATTERNS);
  const refusalCount = countMatches(normalized, REFUSAL_PATTERNS);
  const wordCount = normalized.match(/\b\w+\b/g)?.length || 0;
  const substance = Math.min(wordCount / 80, 1.0);
  const directness = hedgeCount === 0 ? 1.0 : Math.max(0.0, 1.0 - hedgeCount * 0.2);

  const truth = clamp(0.35 + 0.45 * substance + 0.20 * directness - 0.20 * errorCount);
  const indeterminacy = clamp((wordCount >= 8 ? 0.15 : 0.45) + 0.18 * hedgeCount);
  const falsity = clamp(0.22 * errorCount + 0.35 * refusalCount);

  return new Triplet(truth, indeterminacy, falsity);
}

export function scoreClassifierConfidence(confidence: number, selected: boolean): Triplet {
  if (typeof confidence !== "number") {
    throw new TypeError("confidence must be a number");
  }
  if (!Number.isFinite(confidence)) {
    throw new RangeError("confidence must be finite");
  }

  const confidenceValue = clamp(confidence);
  if (selected) {
    return new Triplet(confidenceValue, 1 - confidenceValue, 0);
  }

  return new Triplet(0, Math.max(1 - confidenceValue, 0.7), 0);
}

function countMatches(text: string, patterns: string[]): number {
  return patterns.reduce((count, pattern) => {
    const escaped = pattern.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    return count + (text.match(new RegExp(`(?<!\\w)${escaped}(?!\\w)`, "g"))?.length || 0);
  }, 0);
}

function clamp(value: number): number {
  return Math.max(0.0, Math.min(1.0, value));
}
